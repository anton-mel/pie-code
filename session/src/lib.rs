//! One chat session kept alive inside pie, for an OpenAI-style front end
//! (`../serve.py`). The front end resends the whole conversation on every
//! request; the session keeps the KV of everything it has read, so it only
//! reads what is new. Replies are generated on a fork that is dropped after,
//! so the kept KV is exactly the conversation as the front end will send it
//! back, and a recurrent state never has to be rewound.
//!
//! Shared memory: sessions that start with the same tokens (an agent app's
//! system prompt and tools) read them once. The first publishes its KV and
//! recurrent state at that point under a name (`share.key`); later ones map
//! that snapshot copy-on-write, the cross-process twin of `fork`, and read
//! only what follows.
//!
//! Messages in: `{"tokens": [...], "max_tokens": n, "share": {"key", "at"}}`
//! (tokens after those already read; `share` on the first only) or `"quit"`.
//! Messages out: `{"text", "tokens", "fed", "mapped", "ttft_ms", "total_ms"}`.

use inferlet::chat;
use inferlet::eta::hybrid::prelude::*;
use inferlet::session;
use serde::Deserialize;

#[derive(Deserialize)]
struct Turn {
    tokens: Vec<u32>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    share: Option<Share>,
}

/// The first `at` tokens (a whole number of pages) are common to many
/// sessions and live under `key`.
#[derive(Deserialize)]
struct Share {
    key: String,
    at: u32,
}

fn default_max_tokens() -> usize {
    1024
}

struct Agent {
    ws: WorkingSet,
    rs: Vec<RsWorkingSet>,
    len: u32,
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

#[inferlet::main]
async fn main(first: Turn) -> Result<String> {
    let rs = match model::pass_kind() {
        model::ForwardKind::Attention => Vec::new(),
        model::ForwardKind::Hybrid => vec![RsWorkingSet::new()],
        _ => return Err("a session needs a model with a KV cache".into()),
    };
    let stops = chat::stop_tokens();
    let mut main = Agent {
        ws: WorkingSet::new(),
        rs,
        len: 0,
        penalty: None,
    };
    let mut turn = first;
    let mut mapped = 0u32;
    let mut missed = String::new();
    if let Some(share) = turn.share.take()
        && share.at % kv_page_size() == 0
        && share.at > 0
        && (share.at as usize) < turn.tokens.len()
    {
        let key = share.key.as_bytes();
        let kv = WorkingSet::from_index(key)?;
        let rs = if main.rs.is_empty() { Some(Vec::new()) } else { RsWorkingSet::from_index(key)?.map(|rs| vec![rs]) };
        missed = format!("{}{}", if kv.is_none() { "kv" } else { "" }, if rs.is_none() { "rs" } else { "" });
        if let (Some(ws), Some(rs)) = (kv, rs) {
            // Mapped: the shared tokens are already read.
            main = Agent { ws, rs, len: share.at, penalty: None };
            mapped = share.at;
        } else {
            // Read them, then publish them for the sessions after this one.
            let read = Pipeline::new();
            main.append(&read, &turn.tokens[..share.at as usize]).await?;
            read.close();
            let publish = Pipeline::new();
            let pages = main.ws.slice(&publish, 0, share.at / kv_page_size())?;
            let states = main.rs.iter().map(|rs| rs.fork(&publish)).collect::<Result<Vec<_>>>()?;
            publish.close();
            pages.update_index(key)?;
            for state in &states {
                state.update_index(key)?;
            }
            session::send(r#"{"published":true}"#);
        }
        turn.tokens.drain(..share.at as usize);
    }
    loop {
        let clock = std::time::Instant::now();
        let fed = turn.tokens.len();
        let read = Pipeline::new();
        let next = main.append(&read, &turn.tokens).await?;
        read.close();
        let ttft = clock.elapsed();
        let reply = Pipeline::new();
        let mut fork = main.fork(&reply)?;
        let tokens = fork
            .generate_until(&reply, next, turn.max_tokens.saturating_sub(1), &stops)
            .await?;
        drop(fork);
        let shown: Vec<u32> = tokens.iter().copied().filter(|t| !stops.contains(t)).collect();
        session::send(&format!(
            r#"{{"fed":{fed},"mapped":{mapped},"missed":"{missed}","context":{},"tokens":{},"ttft_ms":{:.1},"total_ms":{:.1},"text":{}}}"#,
            main.len,
            tokens.len(),
            ttft.as_secs_f64() * 1e3,
            clock.elapsed().as_secs_f64() * 1e3,
            inferlet::serde_json::to_string(&model::decode(&shown)?).map_err(|e| e.to_string())?,
        ));
        turn = loop {
            match session::receive().await {
                Some(message) if message != "quit" => {
                    if let Ok(turn) = inferlet::serde_json::from_str::<Turn>(&message) {
                        mapped = 0;
                        break turn;
                    }
                }
                _ => return Ok(String::new()),
            }
        };
    }
}
