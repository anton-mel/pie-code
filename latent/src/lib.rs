//! LatentMAS on pie: agents that think in the model's hidden states and hand
//! their thoughts to the next agent through one shared KV context
//! (arXiv 2511.20639). One process runs one question's whole team.
//!
//! A latent step feeds the last layer's hidden state back in as the next
//! input instead of sampling a token: `e = h * scale / |h|`, `scale` the mean
//! norm of the input embeddings (the paper's realignment without its matrix).
//! The loop runs on device: each step's epilogue writes the next input into the
//! channel the following step reads, so `m` steps cost `m` fires and no host trip.
//!
//! Messages in (JSON), each answered with one JSON message:
//! - `{"op": "think", "tokens": [...], "steps": m, "scale": s}`: read `tokens`
//!   after the context, then take `m` latent steps. Replies `{"len", "ms"}`.
//! - `{"op": "say", "tokens": [...], "max_tokens", "temperature", "top_p",
//!   "seed", "stops": [...], "fresh": bool}`: read `tokens` (on a new, empty
//!   context when `fresh`) and sample a reply. Replies `{"tokens", "len", "ms"}`.
//! - `"quit"`.

use inferlet::eta::hybrid::prelude::*;
use inferlet::session;

use serde::Deserialize;

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "lowercase")]
enum Request {
    Think {
        tokens: Vec<u32>,
        steps: u32,
        scale: f32,
    },
    /// Like `think` on a fresh context, one host trip per step, returning every latent input:
    /// the realigned hidden state after the prompt and after each step (a parity check).
    Probe {
        tokens: Vec<u32>,
        steps: u32,
        scale: f32,
    },
    Say {
        tokens: Vec<u32>,
        max_tokens: usize,
        #[serde(default)]
        temperature: f32,
        #[serde(default = "one")]
        top_p: f32,
        #[serde(default)]
        seed: u32,
        #[serde(default)]
        stops: Vec<u32>,
        #[serde(default)]
        fresh: bool,
    },
}

fn one() -> f32 {
    1.0
}

/// The sampled next token from the read-out row: greedy at temperature 0, else top-p.
fn sample(rng: &Tensor, temperature: f32, top_p: f32) -> Tensor {
    let logits = intrinsics::logits();
    if temperature == 0.0 {
        return reshape(reduce_argmax(&logits), [1]);
    }
    reshape(nucleus_sample(&logits / temperature, top_p, rng), [1])
}

/// The next latent input from the read-out hidden row: `h * scale / |h|`.
fn realign(width: u32, scale: f32) -> Tensor {
    let h = intrinsics::hidden(width);
    let inv = rsqrt(reduce_sum(&h * &h));
    &h * &broadcast(&reshape(inv, [1, 1]) * scale, [1, width])
}

struct Mind {
    ws: WorkingSet,
    rs: Vec<RsWorkingSet>,
    len: u32,
}

/// One pass's view of the context's pages, for rows `start..start + n`.
struct Geometry {
    positions: Channel,
    pages: Channel,
    page_indptr: Channel,
    w_slot: Channel,
    w_off: Channel,
    kv_len: Channel,
}

impl Mind {
    fn new() -> Result<Mind> {
        Ok(Mind {
            ws: WorkingSet::new(),
            rs: match model::pass_kind() {
                model::ForwardKind::Hybrid => vec![RsWorkingSet::new()],
                _ => Vec::new(),
            },
            len: 0,
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

    fn geometry(&self, start: u32, n: u32, pages: u32) -> Geometry {
        let page = kv_page_size();
        let total = start + n;
        Geometry {
            positions: Channel::from_iter(start..total),
            pages: Channel::from_iter(0..pages),
            page_indptr: Channel::from([0u32, total.div_ceil(page)]),
            w_slot: Channel::from_iter((start..total).map(|p| p / page)),
            w_off: Channel::from_iter((start..total).map(|p| p % page)),
            kv_len: Channel::from([total]),
        }
    }

    fn bind(&self, fwd: &ForwardPass, g: &Geometry, start: u32) -> Result<()> {
        fwd.attention(
            Some(KvBinding {
                working_set: &self.ws,
                geometry: KvGeometry {
                    readable_pages: ..,
                    writable_pages: (start / kv_page_size())..,
                    kv_len: &g.kv_len,
                    pages: &g.pages,
                    page_indptr: &g.page_indptr,
                    w_slot: &g.w_slot,
                    w_off: &g.w_off,
                    positions: &g.positions,
                    mask: None,
                },
            }),
            &self.rs,
            RsGeometry {
                fold_len: None,
                buffer: 0..0,
            },
        )?;
        Ok(())
    }

    /// Reads `tokens` after the context. The last chunk's pass runs `last`, which binds its
    /// reading and readout and plants its epilogue.
    async fn read(
        &mut self,
        on: &Pipeline,
        tokens: &[u32],
        last: impl FnOnce(&ForwardPass, u32) -> Result<Channel>,
    ) -> Result<Channel> {
        let chunks: Vec<&[u32]> = tokens.chunks(prefill_chunk_hint().max(1)).collect();
        let mut last = Some(last);
        let mut done = None;
        for (i, chunk) in chunks.iter().enumerate() {
            let (start, n) = (self.len, chunk.len() as u32);
            let pages = self.reserve(start + n)?;
            let g = self.geometry(start, n, pages);
            let input = Channel::from_iter(chunk.iter().map(|&t| t as i32));
            let indptr_ch = Channel::from([0u32, n]);
            let fwd = ForwardPass::new();
            let out = if i + 1 == chunks.len() {
                last.take().expect("one last chunk")(&fwd, n)?
            } else {
                let out = Channel::new([1], dtype::i32);
                let put = out.clone();
                fwd.epilogue(move || put.put(reshape(reduce_argmax(intrinsics::logits()), [1])));
                out
            };
            fwd.embed(&input, &indptr_ch)?;
            self.bind(&fwd, &g, start)?;
            fwd.submit(on).context("read")?;
            self.len = start + n;
            done = Some(out);
        }
        done.ok_or_else(|| "nothing to read".into())
    }

    /// Reads `tokens`, then takes `steps` latent steps; the context ends `steps` rows longer.
    async fn think(&mut self, on: &Pipeline, tokens: &[u32], steps: u32, scale: f32) -> Result<()> {
        let width = model::reading("latent")
            .ok_or("this model has no `latent` reading")?
            .readout_width;
        let read = self
            .read(on, tokens, move |fwd, n| {
                fwd.reading("hidden")?;
                fwd.readout(&Channel::from([n - 1]))?;
                let first = Channel::new([1, width], dtype::f32);
                let put = first.clone();
                fwd.epilogue(move || put.put(realign(width, scale)));
                Ok(first)
            })
            .await?;
        // The first latent input crosses to the host once (one row); the steps after stay on device.
        let first: Vec<f32> = read.take_host().await?;
        if steps == 0 {
            return Ok(());
        }
        let lat = Channel::from_shaped([1, width], first);

        let (page, start) = (kv_page_size(), self.len);
        let pages = self.reserve(start + steps)?;
        let g = self.geometry(start, 1, pages);
        let token_in = Channel::from([0i32]);
        let indptr_ch = Channel::from([0u32, 1]);
        let ack = Channel::new([1], dtype::u32).capacity(channel_capacity() as u32);
        let fwd = ForwardPass::new();
        fwd.reading("latent")?;
        fwd.embed(&token_in, &indptr_ch)?;
        fwd.input("latents", &lat)?;
        self.bind(&fwd, &g, start)?;
        let (kv_len, positions, w_slot, w_off, page_indptr) =
            (g.kv_len.clone(), g.positions.clone(), g.w_slot.clone(), g.w_off.clone(), g.page_indptr.clone());
        let put = ack.clone();
        fwd.epilogue(move || {
            let length = kv_len.take();
            let next_length = &length + 1u32;
            lat.take();
            lat.put(realign(width, scale));
            let token = token_in.take();
            token_in.put(&token);
            kv_len.put(&next_length);
            positions.put(&length);
            w_slot.put(&length / page);
            w_off.put(&length % page);
            page_indptr.put(indptr(1, &next_length.div_ceil(page)));
            put.put(&next_length);
        });
        let mut taken = 0;
        run_ahead(on, &fwd, steps as usize, async || {
            ack.take_host::<u32>().await?;
            taken += 1;
            Ok(ControlFlow::Continue(()))
        })
        .await?;
        self.len = start + taken;
        Ok(())
    }

    /// `think`, one pass and one host trip per step, returning each step's latent input.
    async fn probe(&mut self, on: &Pipeline, tokens: &[u32], steps: u32, scale: f32) -> Result<Vec<Vec<f32>>> {
        let width = model::reading("latent")
            .ok_or("this model has no `latent` reading")?
            .readout_width;
        let read = self
            .read(on, tokens, move |fwd, n| {
                fwd.reading("hidden")?;
                fwd.readout(&Channel::from([n - 1]))?;
                let first = Channel::new([1, width], dtype::f32);
                let put = first.clone();
                fwd.epilogue(move || put.put(realign(width, scale)));
                Ok(first)
            })
            .await?;
        let mut inputs: Vec<Vec<f32>> = vec![read.take_host().await?];
        for _ in 0..steps {
            let start = self.len;
            let pages = self.reserve(start + 1)?;
            let g = self.geometry(start, 1, pages);
            let lat = Channel::from_shaped([1, width], inputs.last().cloned().unwrap_or_default());
            let out = Channel::new([1, width], dtype::f32);
            let put = out.clone();
            let fwd = ForwardPass::new();
            fwd.reading("latent")?;
            fwd.embed(&Channel::from([0i32]), &Channel::from([0u32, 1]))?;
            fwd.input("latents", &lat)?;
            self.bind(&fwd, &g, start)?;
            fwd.epilogue(move || put.put(realign(width, scale)));
            fwd.submit(on).context("probe step")?;
            inputs.push(out.take_host().await?);
            self.len = start + 1;
        }
        Ok(inputs)
    }

    /// Reads `tokens` and samples up to `count` tokens, ending after a stop token.
    async fn say(
        &mut self,
        on: &Pipeline,
        tokens: &[u32],
        count: usize,
        temperature: f32,
        top_p: f32,
        seed: u32,
        stops: &[u32],
    ) -> Result<Vec<u32>> {
        // Each pass gets its own random state: a seeded channel attaches to one pass only.
        let rng_first = Channel::from([seed, 0u32]);
        let rng = Channel::from([seed, 1u32 << 20]);
        let first = self
            .read(on, tokens, move |fwd, n| {
                fwd.readout(&Channel::from([n - 1]))?;
                let out = Channel::new([1], dtype::i32);
                let put = out.clone();
                fwd.epilogue(move || {
                    let r = rng_first.take();
                    put.put(sample(&r, temperature, top_p));
                    rng_first.put(&r + iota(2));
                });
                Ok(out)
            })
            .await?;
        let first = first.take_host::<i32>().await?;
        let mut generated = vec![first as u32];
        if count <= 1 || stops.contains(&(first as u32)) {
            return Ok(generated);
        }

        let (page, start) = (kv_page_size(), self.len);
        let pages = self.reserve(start + count as u32 + 1)?;
        let g = self.geometry(start, 1, pages);
        let token_in = Channel::from([first]);
        let indptr_ch = Channel::from([0u32, 1]);
        let out = Channel::new([1], dtype::i32).capacity(channel_capacity() as u32);
        let fwd = ForwardPass::new();
        fwd.embed(&token_in, &indptr_ch)?;
        self.bind(&fwd, &g, start)?;
        let (kv_len, positions, w_slot, w_off, page_indptr) =
            (g.kv_len.clone(), g.positions.clone(), g.w_slot.clone(), g.w_off.clone(), g.page_indptr.clone());
        let put = out.clone();
        fwd.epilogue(move || {
            let length = kv_len.take();
            let r = rng.take();
            let token = sample(&r, temperature, top_p);
            let next_length = &length + 1u32;
            token_in.put(&token);
            kv_len.put(&next_length);
            positions.put(&length);
            w_slot.put(&length / page);
            w_off.put(&length % page);
            page_indptr.put(indptr(1, &next_length.div_ceil(page)));
            rng.put(&r + iota(2));
            put.put(&token);
        });
        run_ahead(on, &fwd, count - 1, async || {
            let token = out.take_host::<i32>().await? as u32;
            generated.push(token);
            Ok(if stops.contains(&token) {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            })
        })
        .await?;
        self.len = start + generated.len() as u32 - 1;
        Ok(generated)
    }
}

#[inferlet::main]
async fn main(first: Request) -> Result<String> {
    let mut mind = Mind::new()?;
    let mut request = first;
    loop {
        let began = std::time::Instant::now();
        let pipe = Pipeline::new();
        let reply = match request {
            Request::Think { tokens, steps, scale } => {
                mind.think(&pipe, &tokens, steps, scale).await?;
                format!(r#"{{"len":{},"ms":{:.1}}}"#, mind.len, began.elapsed().as_secs_f64() * 1e3)
            }
            Request::Probe { tokens, steps, scale } => {
                mind = Mind::new()?;
                let inputs = mind.probe(&pipe, &tokens, steps, scale).await?;
                format!(r#"{{"inputs":{}}}"#, inferlet::serde_json::to_string(&inputs).unwrap_or_default())
            }
            Request::Say { tokens, max_tokens, temperature, top_p, seed, stops, fresh } => {
                if fresh {
                    mind = Mind::new()?;
                }
                let made = mind.say(&pipe, &tokens, max_tokens, temperature, top_p, seed, &stops).await?;
                format!(
                    r#"{{"tokens":{},"len":{},"ms":{:.1}}}"#,
                    inferlet::serde_json::to_string(&made).unwrap_or_default(),
                    mind.len,
                    began.elapsed().as_secs_f64() * 1e3
                )
            }
        };
        pipe.close();
        session::send(&reply);
        request = loop {
            match session::receive().await {
                Some(message) if message != "quit" => {
                    match inferlet::serde_json::from_str::<Request>(&message) {
                        Ok(next) => break next,
                        Err(error) => session::send(&format!(r#"{{"error":{:?}}}"#, error.to_string())),
                    }
                }
                _ => return Ok(String::new()),
            }
        };
    }
}
