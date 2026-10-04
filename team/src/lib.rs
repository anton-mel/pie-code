//! A team of agents in one pie process, run like an operating system runs
//! processes. The harness (`../team.py`) decides who does what; this program
//! gives it the system calls:
//!
//! - `read`: a new agent reads a text (`{"op":"read","k":0,"text":..}`).
//! - `fork`: a new agent is a copy-on-write fork of another, sharing every
//!   page it has read, then reads its own text (`{"op":"fork","from":0,"k":1,..}`).
//! - `send`: an agent reads a message, a question or a tool's output.
//! - `pipe`: an agent reads what other agents last wrote, token for token,
//!   without leaving the engine (`{"op":"pipe","from":[1,2],"k":0,"prefix":..,"text":..}`).
//!
//! Every op ends with the agent writing `max_tokens` tokens, which stay in its
//! memory. Ops on different agents run at once; replies come back as
//! `{"k","text","fed","ms"}`, in the order they finish.

use futures::StreamExt;
use futures::future::{Either, LocalBoxFuture, select};
use futures::stream::FuturesUnordered;
use inferlet::eta::hybrid::prelude::*;
use inferlet::session;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
struct Op {
    op: String,
    k: u32,
    #[serde(default)]
    from: Vec<u32>,
    #[serde(default)]
    text: String,
    /// For pipes: read before what the other agents wrote.
    #[serde(default)]
    prefix: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
}

fn default_max_tokens() -> usize {
    16
}

struct Agent {
    ws: WorkingSet,
    rs: Vec<RsWorkingSet>,
    len: u32,
    penalty: Option<Vec<f32>>,
}

/// An agent and what it has not read yet: its own last token, which is not in
/// its KV until its next turn, and what it last wrote, for pipes.
struct Member {
    agent: Agent,
    pending: Vec<u32>,
    last: Vec<u32>,
}

/// The next token: the best one the rule allows. Takes first, puts last.
fn pick(rule: &Option<Channel>) -> (Tensor, Option<Tensor>) {
    match rule {
        Some(rule) => {
            let penalty = rule.take();
            let token = reduce_argmax(&intrinsics::logits() - &penalty);
            (reshape(token, [1]), Some(penalty))
        }
        None => (reshape(reduce_argmax(intrinsics::logits()), [1]), None),
    }
}

impl Agent {
    fn bind<'a>(
        &'a self,
        fwd: &ForwardPass,
        geometry: KvGeometry<'a, std::ops::RangeFull, std::ops::RangeFrom<u32>>,
    ) -> Result<()> {
        fwd.attention(
            Some(KvBinding {
                working_set: &self.ws,
                geometry,
            }),
            &self.rs,
            RsGeometry {
                fold_len: None,
                buffer: 0..0,
            },
        )?;
        Ok(())
    }

    /// A copy-on-write twin: it shares everything read so far.
    fn fork(&self, on: &Pipeline) -> Result<Agent> {
        Ok(Agent {
            ws: self.ws.fork(on)?,
            rs: self.rs.iter().map(|rs| rs.fork(on)).collect::<Result<_>>()?,
            len: self.len,
            penalty: self.penalty.clone(),
        })
    }

    fn reserve(&self, tokens: u32) -> Result<u32> {
        let pages = tokens.div_ceil(kv_page_size()).max(1);
        let have = self.ws.page_len();
        if pages > have {
            self.ws.reserve(pages - have).context("reserve KV")?;
        }
        Ok(pages)
    }

    /// Prefills `tokens` after what the agent already holds; returns the next token.
    async fn append(&mut self, on: &Pipeline, tokens: &[u32]) -> Result<i32> {
        let page = kv_page_size();
        let mut next = 0;
        for chunk in tokens.chunks(prefill_chunk_hint()) {
            let (start, n) = (self.len, chunk.len() as u32);
            let total = start + n;
            self.reserve(total)?;
            let input = Channel::from_iter(chunk.iter().map(|&t| t as i32));
            let embed_indptr = Channel::from([0u32, n]);
            let positions = Channel::from_iter(start..total);
            let pages = Channel::from_iter(0..self.ws.page_len());
            let page_indptr = Channel::from([0u32, total.div_ceil(page)]);
            let w_slot = Channel::from_iter((start..total).map(|p| p / page));
            let w_off = Channel::from_iter((start..total).map(|p| p % page));
            let kv_len = Channel::from([total]);
            let out = Channel::new([1], dtype::i32);
            let rule = self.penalty.clone().map(Channel::from);

            let fwd = ForwardPass::new();
            fwd.embed(&input, &embed_indptr)?;
            self.bind(
                &fwd,
                KvGeometry {
                    readable_pages: ..,
                    writable_pages: (start / page)..,
                    kv_len: &kv_len,
                    pages: &pages,
                    page_indptr: &page_indptr,
                    w_slot: &w_slot,
                    w_off: &w_off,
                    positions: &positions,
                    mask: None,
                },
            )?;
            fwd.epilogue(move || out.put(pick(&rule).0));
            fwd.submit(on).context("append")?;
            next = out.take_host::<i32>().await?;
            self.len = total;
        }
        Ok(next)
    }

    /// `generate`, ending early after a stop token.
    async fn generate_until(&mut self, on: &Pipeline, first: i32, count: usize, stops: &[u32]) -> Result<Vec<u32>> {
        let mut generated = vec![first as u32];
        if count == 0 || stops.contains(&(first as u32)) {
            on.close();
            return Ok(generated);
        }
        let (page, start) = (kv_page_size(), self.len);
        let max_pages = self.reserve(start + count as u32 + 1)?;
        let token_in = Channel::from([first]);
        let embed_indptr = Channel::from([0u32, 1]);
        let positions = Channel::from([start]);
        let pages = Channel::from_iter(0..max_pages);
        let page_indptr = Channel::from([0u32, (start + 1).div_ceil(page)]);
        let w_slot = Channel::from([start / page]);
        let w_off = Channel::from([start % page]);
        let kv_len = Channel::from([start + 1]);
        let out = Channel::new([1], dtype::i32).capacity(channel_capacity() as u32);
        let rule = self.penalty.clone().map(Channel::from);

        let fwd = ForwardPass::new();
        fwd.embed(&token_in, &embed_indptr)?;
        self.bind(
            &fwd,
            KvGeometry {
                readable_pages: ..,
                writable_pages: (start / page)..,
                kv_len: &kv_len,
                pages: &pages,
                page_indptr: &page_indptr,
                w_slot: &w_slot,
                w_off: &w_off,
                positions: &positions,
                mask: None,
            },
        )?;
        fwd.epilogue(move || {
            let length = kv_len.take();
            let (token, penalty) = pick(&rule);
            let next_length = &length + 1u32;
            token_in.put(&token);
            kv_len.put(&next_length);
            positions.put(&length);
            w_slot.put(&length / page);
            w_off.put(&length % page);
            page_indptr.put(indptr(1, &next_length.div_ceil(page)));
            out.put(&token);
            if let (Some(rule), Some(penalty)) = (&rule, penalty) {
                rule.put(&penalty);
            }
        });
        run_ahead(on, &fwd, count, async || {
            let token = out.take_host::<i32>().await? as u32;
            generated.push(token);
            Ok(if stops.contains(&token) { ControlFlow::Break(()) } else { ControlFlow::Continue(()) })
        })
        .await?;
        self.len = start + generated.len() as u32 - 1;
        Ok(generated)
    }
}


fn fresh() -> Result<Agent> {
    let rs = match model::pass_kind() {
        model::ForwardKind::Attention => Vec::new(),
        model::ForwardKind::Hybrid => vec![RsWorkingSet::new()],
        _ => return Err("a team needs a model with a KV cache".into()),
    };
    Ok(Agent { ws: WorkingSet::new(), rs, len: 0, penalty: None })
}

type Done = (u32, Member, Result<(String, usize, f64)>);

/// One turn: read `tokens` after what `m` holds, then write `count` tokens.
async fn turn(k: u32, mut m: Member, tokens: Vec<u32>, count: usize) -> Done {
    let clock = std::time::Instant::now();
    let result: Result<(String, usize, f64)> = async {
        let mut feed = std::mem::take(&mut m.pending);
        feed.extend(tokens);
        let on = Pipeline::new();
        let first = m.agent.append(&on, &feed).await?;
        let wrote = m.agent.generate_until(&on, first, count.saturating_sub(1), &[]).await?;
        m.pending = vec![*wrote.last().unwrap_or(&0)];
        m.last = wrote.clone();
        Ok((model::decode(&wrote)?, feed.len(), clock.elapsed().as_secs_f64() * 1e3))
    }
    .await;
    (k, m, result)
}

fn reply(k: u32, result: &Result<(String, usize, f64)>) {
    let line = match result {
        Ok((text, fed, ms)) => format!(
            r#"{{"k":{k},"fed":{fed},"ms":{ms:.1},"text":{}}}"#,
            inferlet::serde_json::to_string(text).unwrap_or_default()
        ),
        Err(e) => format!(r#"{{"k":{k},"error":{}}}"#, inferlet::serde_json::to_string(&e.to_string()).unwrap_or_default()),
    };
    session::send(&line);
}

#[inferlet::main]
async fn main(_: inferlet::serde_json::Value) -> Result<String> {
    let mut members: HashMap<u32, Member> = HashMap::new();
    let mut running: FuturesUnordered<LocalBoxFuture<'static, Done>> = FuturesUnordered::new();
    let mut inbox = Box::pin(session::receive());
    session::send(r#"{"ready":true}"#);
    loop {
        let message = if running.is_empty() {
            let message = (&mut inbox).await;
            inbox = Box::pin(session::receive());
            message
        } else {
            match select(&mut inbox, running.next()).await {
                Either::Left((message, rest)) => {
                    drop(rest);
                    inbox = Box::pin(session::receive());
                    message
                }
                Either::Right((done, _)) => {
                    if let Some((k, member, result)) = done {
                        reply(k, &result);
                        members.insert(k, member);
                    }
                    continue;
                }
            }
        };
        let Some(message) = message else { break };
        if message == "quit" {
            break;
        }
        let op: Op = match inferlet::serde_json::from_str(&message) {
            Ok(op) => op,
            Err(e) => {
                session::send(&format!(r#"{{"error":{}}}"#, inferlet::serde_json::to_string(&e.to_string()).unwrap_or_default()));
                continue;
            }
        };
        let text = model::encode(&op.text);
        let started: Result<(Member, Vec<u32>)> = (|| match op.op.as_str() {
            "read" => Ok((Member { agent: fresh()?, pending: Vec::new(), last: Vec::new() }, text)),
            "fork" => {
                let parent = op.from.first().and_then(|a| members.get(a)).ok_or("fork: no such idle agent")?;
                let on = Pipeline::new();
                let agent = parent.agent.fork(&on)?;
                on.close();
                Ok((Member { agent, pending: parent.pending.clone(), last: Vec::new() }, text))
            }
            "send" => Ok((members.remove(&op.k).ok_or("send: no such idle agent")?, text)),
            "pipe" => {
                let mut tokens = model::encode(&op.prefix);
                for a in &op.from {
                    tokens.extend(members.get(a).ok_or("pipe: no such idle agent")?.last.iter().copied());
                }
                tokens.extend(text);
                Ok((members.remove(&op.k).ok_or("pipe: no such idle agent")?, tokens))
            }
            other => Err(format!("unknown op {other}").into()),
        })();
        match started {
            Ok((member, tokens)) => running.push(Box::pin(turn(op.k, member, tokens, op.max_tokens))),
            Err(e) => reply(op.k, &Err(e)),
        }
    }
    Ok(String::new())
}
