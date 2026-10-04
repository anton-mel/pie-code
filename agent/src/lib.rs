//! A long-lived agent: reads its context once, then answers turns the client
//! sends. Between turns it holds no pipeline, so pie may move its KV off the
//! device; the next turn only appends the new tokens.
//!
//! `ban` is a rule that runs on every token, inside the engine: no token
//! containing one of those letters is ever chosen.
//!
//! A turn that is a JSON list of questions fans out: the agent forks itself
//! once per question, copy-on-write, and the forks answer side by side. The
//! context is neither copied nor read again, and the forks are dropped after.
//!
//! `append` and `generate` are pie's `prefix-tree-kv-cache` helpers, bound to
//! the hybrid pass so one program runs on attention and hybrid models.

use inferlet::eta::hybrid::prelude::*;
use inferlet::session;
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    context: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    ban: String,
}

#[derive(serde::Deserialize)]
struct Turn {
    text: String,
    max_tokens: usize,
}

fn default_max_tokens() -> usize {
    16
}

struct Agent {
    ws: WorkingSet,
    rs: Vec<RsWorkingSet>,
    len: u32,
    /// One entry per vocabulary token: what to subtract from its logit.
    penalty: Option<Vec<f32>>,
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

    /// Appends `tokens` and decodes `count` more, on a pipeline of its own.
    async fn answer(mut self, tokens: Vec<u32>, count: usize) -> Result<Vec<u32>> {
        let on = Pipeline::new();
        let first = self.append(&on, &tokens).await?;
        self.generate(&on, first, count).await
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

    /// Greedy-decodes `count` tokens after `first`; closes `on`.
    async fn generate(&mut self, on: &Pipeline, first: i32, count: usize) -> Result<Vec<u32>> {
        let mut generated = vec![first as u32];
        if count == 0 {
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
            generated.push(out.take_host::<i32>().await? as u32);
            Ok(ControlFlow::Continue(()))
        })
        .await?;
        self.len = start + count as u32;
        Ok(generated)
    }
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let rs = match model::pass_kind() {
        model::ForwardKind::Attention => Vec::new(),
        model::ForwardKind::Hybrid => vec![RsWorkingSet::new()],
        _ => return Err("an agent needs a model with a KV cache".into()),
    };
    let banned = |bytes: &[u8]| bytes.iter().any(|byte| input.ban.as_bytes().contains(&byte.to_ascii_lowercase()));
    let penalty = (!input.ban.is_empty()).then(|| {
        let mut penalty = vec![0.0f32; model::output_vocab_size() as usize];
        for token in model::vocabs().into_iter().filter(|token| banned(&token.bytes)) {
            penalty[token.id as usize] = 1.0e9;
        }
        penalty
    });
    let mut agent = Agent {
        ws: WorkingSet::new(),
        rs,
        len: 0,
        penalty,
    };
    let mut pending = model::encode(&input.context);
    let mut turns = 0u32;
    let mut limit = input.max_tokens;
    loop {
        let clock = std::time::Instant::now();
        let on = Pipeline::new();
        let fed = pending.len();
        let first = agent.append(&on, &pending).await?;
        let ttft = clock.elapsed();
        let tokens = agent.generate(&on, first, limit.saturating_sub(1)).await?;
        let total = clock.elapsed();
        // The last generated token is not in the KV yet; it leads the next turn.
        pending = vec![*tokens.last().unwrap_or(&0)];
        session::send(&format!(
            r#"{{"turn":{turns},"fed":{fed},"context":{},"ttft_ms":{:.1},"total_ms":{:.1},"text":{}}}"#,
            agent.len,
            ttft.as_secs_f64() * 1e3,
            total.as_secs_f64() * 1e3,
            inferlet::serde_json::to_string(&model::decode(&tokens)?).map_err(|e| e.to_string())?,
        ));
        turns += 1;
        loop {
            let message = match session::receive().await {
                Some(message) if message != "quit" => message,
                _ => return Ok(format!("{turns} turns")),
            };
            // {"text": ..., "max_tokens": n}: one turn with its own length.
            if let Ok(turn) = inferlet::serde_json::from_str::<Turn>(&message) {
                pending.extend(model::encode(&turn.text));
                limit = turn.max_tokens;
                break;
            }
            let Ok(questions) = inferlet::serde_json::from_str::<Vec<String>>(&message) else {
                pending.extend(model::encode(&message));
                limit = input.max_tokens;
                break;
            };
            let clock = std::time::Instant::now();
            let setup = Pipeline::new();
            let mut forks = Vec::with_capacity(questions.len());
            for question in &questions {
                let mut tokens = pending.clone();
                tokens.extend(model::encode(question));
                forks.push(agent.fork(&setup)?.answer(tokens, input.max_tokens.saturating_sub(1)));
            }
            setup.close();
            let mut texts = Vec::with_capacity(forks.len());
            for tokens in futures::future::join_all(forks).await {
                texts.push(model::decode(&tokens?)?);
            }
            session::send(&format!(
                r#"{{"fanout":{},"total_ms":{:.1},"texts":{}}}"#,
                texts.len(),
                clock.elapsed().as_secs_f64() * 1e3,
                inferlet::serde_json::to_string(&texts).map_err(|e| e.to_string())?,
            ));
        }
    }
}
