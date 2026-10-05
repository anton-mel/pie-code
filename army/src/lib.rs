//! An agent army in one pie process: the main agent never solves alone, it delegates.
//!
//! The question is read once. The main agent splits the work into tasks that can run at the
//! same time, and each task gets a worker: a copy-on-write fork that pays only for its own tokens
//! and decodes in the same fires as every other live agent. A finding is shared the moment it
//! lands: appended once to a blackboard that every later agent forks from, and handed as a note
//! to every running worker. The main agent plans the next wave from the blackboard, filling only
//! gaps, until it says the team is ready; then solvers write the whole solution from it, two that
//! agree end the run (the rest are killed and their memory freed at once), and a referee settles a
//! disagreement. Each round, surer agents get more tokens and doubtful ones fewer.
//!
//! Modes: `team` (the above), and the baselines on the same engine: `single` (one agent) and
//! `vote` (n independent agents run to completion, majority answer).
//!
//! In: one JSON `Job`. Out: JSON events, `board` (every agent, each round), `trace` (an agent's
//! new text), `say` (what the team decided), and one `done`.

use futures::future::join_all;
use inferlet::eta::hybrid::prelude::*;
use inferlet::session;
use serde::Deserialize;
use std::collections::HashSet;

#[derive(Deserialize)]
struct Job {
    /// The conversation up to the end of the question's text, rendered by the caller.
    head: Vec<u32>,
    /// Chat-template pieces: what opens a user turn, what closes it and opens the reply (without
    /// thinking), and what seals an open reply. The caller renders them for its model.
    turn_open: Vec<u32>,
    turn_close: Vec<u32>,
    seal: Vec<u32>,
    stops: Vec<u32>,
    #[serde(default = "default_mode")]
    mode: String,
    /// Most agents of one wave (`vote`: exactly this many).
    #[serde(default = "eight")]
    width: usize,
    /// Tokens the whole run may write.
    #[serde(default = "budget")]
    budget: usize,
    #[serde(default)]
    seed: u32,
    /// `tree`: entropy (nats) above which a chunk counts as a hesitation worth branching on.
    #[serde(default = "doubt")]
    doubt: f32,
}

fn doubt() -> f32 {
    2.0
}

fn default_mode() -> String {
    "team".into()
}
fn eight() -> usize {
    8
}
fn budget() -> usize {
    65_536
}

const LEAD_FIRST: &str = "You are the main agent and you lead a team of workers who run IN PARALLEL, at \
the same time. Never solve the problem yourself: your job is to delegate. Split the problem into {min} to \
{width} sub-tasks that can all run at once and together cover the whole solution, for example: restate \
the key structure or equation; compute each independent quantity separately; one task per case or \
sub-case; derive a needed formula; an independent check by another method. Tasks must not overlap, must \
not depend on each other, and each must name exactly what result to return. Prefer more, smaller tasks: \
parallel work finishes sooner. Write one per line as `T1: <task>`.";
const LEAD_NEXT: &str = "You are the main agent. Your workers' findings are above. If a part is missing, a \
finding looks wrong, or two findings conflict, write up to {width} new parallel sub-tasks that fill exactly \
those gaps (one per line, `T1: <task>`, never repeating finished work). If the findings already let a \
solver finish the whole problem, write only READY.";
const WORK: &str = "You are one worker in a parallel team. Do only this task: {task}\nDo not solve the \
whole problem. Be brief and exact. End with one line `RESULT: <your finding in one sentence>`.";
const CONCLUDE: &str =
    "Stop working now. Write only one line: `RESULT: <what you established, in one sentence>`.";
const SOLVE: &str = "Using the problem and every team finding above (check them: a finding can be wrong), \
write the complete solution. End with the final answer in \\boxed{}.";
const CODE_RULES: &str = "You can run Python 3 (the standard library, sympy, numpy). To run code, write \
exactly one short ```python block that prints what you need, then end your turn: its output comes back to \
you. Keep code short and put your reasoning in plain text, not in comments. Compute instead of reasoning at \
length: brute force, enumerate, use sympy, or estimate by simulation to check a formula. When you are sure, \
write the final answer in \\boxed{}.";
const SPAWN_RULES: &str = "You can also delegate, and should when the work has independent parts. If the \
remaining work splits into pieces that can run at the same time (separate cases, separate quantities or \
probabilities, a simulation to check your formula while you derive it), write one ```spawn block with one task per line, then end your turn. Helpers do all the tasks at once, each \
knowing everything you know so far, and their results come back to you together. Spawn only for work that \
truly runs in parallel and that you will not do yourself: never to repeat work.";
const HELPER: &str = "You are a helper the agent above spawned. Do only this task: {task}\nYou can run Python \
as described. Be brief. End with one line `RESULT: <your result>`.";
/// Helpers one spawn block may start.
const SPAWN_MAX: usize = 8;
const NUDGE: &str =
    "Continue: run code to check your reasoning, or give the final answer in \\boxed{}.";
/// Tool calls one agent may make.
const TOOL_TURNS: usize = 8;
/// `code`: tokens the agent may write across all its turns.
const CODE_MAX: usize = 12_288;
const SOLVE_ALONE: &str = "Solve it step by step. End with the final answer in \\boxed{}.";
const REFEREE: &str = "The team's solvers disagree: they got {answers}. Using the problem and the team \
findings above, determine carefully which is right, re-deriving the step where they differ. End with the \
final answer in \\boxed{}.";

const ROUND: usize = 256;
/// `tree`: tokens between checkpoints. Shorter finds hesitations sooner, at a fork per chunk.
const TREE_CHUNK: usize = 64;
const WORKER_MAX: usize = 1024;
const SOLVER_MAX: usize = 6144;
const WAVES: usize = 3;
/// Tasks the first wave must have: the main agent always delegates.
const FIRST_WAVE_MIN: usize = 3;
const SOLVERS: usize = 3;
/// Direct attempts racing the team from the start.
const SCOUTS: usize = 2;
const QUORUM: usize = 2;
const TEMPERATURE: f32 = 0.7;
const TOP_P: f32 = 0.95;

fn send(event: serde_json::Value) {
    session::send(&event.to_string());
}

/// The next token sampled from the read-out row, and how unsure the model was (entropy, nats).
fn step(rng: &Tensor) -> (Tensor, Tensor) {
    let scaled = &intrinsics::logits() / TEMPERATURE;
    let token = reshape(nucleus_sample(&scaled, TOP_P, rng), [1]);
    let unsure = reshape(entropy(softmax(&scaled)), [1]);
    (token, unsure)
}

/// `step`'s token with one token ruled out: a fork must not take the road its parent took.
fn step_banning(rng: &Tensor, ban: u32) -> Tensor {
    let vocab = intrinsics::vocab();
    let scaled = reshape(&intrinsics::logits() / TEMPERATURE, [vocab]);
    let masked = select(eq(iota(vocab), ban), -1.0e9f32, &scaled);
    reshape(nucleus_sample(reshape(masked, [1, vocab]), TOP_P, rng), [1])
}

struct Mind {
    ws: WorkingSet,
    rs: Vec<RsWorkingSet>,
    len: u32,
}

impl Mind {
    fn new() -> Mind {
        Mind {
            ws: WorkingSet::new(),
            rs: vec![RsWorkingSet::new()],
            len: 0,
        }
    }

    fn fork(&self) -> Result<Mind> {
        let on = Pipeline::new();
        let twin = Mind {
            ws: self.ws.fork(&on)?,
            rs: self
                .rs
                .iter()
                .map(|rs| rs.fork(&on))
                .collect::<Result<_>>()?,
            len: self.len,
        };
        on.close();
        Ok(twin)
    }

    fn reserve(&self, tokens: u32) -> Result<u32> {
        let pages = tokens.div_ceil(kv_page_size()).max(1);
        let have = self.ws.page_len();
        if pages > have {
            self.ws.reserve(pages - have)?;
        }
        Ok(pages)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "one pass's geometry, bound in one place"
    )]
    fn bind(
        &self,
        fwd: &ForwardPass,
        kv_len: &Channel,
        pages: &Channel,
        page_indptr: &Channel,
        w_slot: &Channel,
        w_off: &Channel,
        positions: &Channel,
        start: u32,
    ) -> Result<()> {
        fwd.attention(
            Some(KvBinding {
                working_set: &self.ws,
                geometry: KvGeometry {
                    readable_pages: ..,
                    writable_pages: (start / kv_page_size())..,
                    kv_len,
                    pages,
                    page_indptr,
                    w_slot,
                    w_off,
                    positions,
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

    /// Reads `tokens` after the context; returns the token sampled after the last one.
    async fn read(&mut self, tokens: &[u32], seed: u32) -> Result<i32> {
        self.read_banning(tokens, seed, None).await
    }

    /// `read`, with the token sampled after the last one never `ban`: the other road at a fork.
    async fn read_banning(&mut self, tokens: &[u32], seed: u32, ban: Option<u32>) -> Result<i32> {
        let page = kv_page_size();
        let mut next = 0;
        for chunk in tokens.chunks(prefill_chunk_hint().max(1) as usize) {
            let (start, n) = (self.len, chunk.len() as u32);
            let total = start + n;
            let pages = self.reserve(total)?;
            let input = Channel::from_iter(chunk.iter().map(|&t| t as i32));
            let embed_indptr = Channel::from([0u32, n]);
            let positions = Channel::from_iter(start..total);
            let page_ids = Channel::from_iter(0..pages);
            let page_indptr = Channel::from([0u32, total.div_ceil(page)]);
            let w_slot = Channel::from_iter((start..total).map(|p| p / page));
            let w_off = Channel::from_iter((start..total).map(|p| p % page));
            let kv_len = Channel::from([total]);
            let out = Channel::new([1], dtype::i32);
            let fwd = ForwardPass::new();
            fwd.embed(&input, &embed_indptr)?;
            fwd.readout(&Channel::from([n - 1]))?;
            self.bind(
                &fwd,
                &kv_len,
                &page_ids,
                &page_indptr,
                &w_slot,
                &w_off,
                &positions,
                start,
            )?;
            let rng = Channel::from([seed, 7u32]);
            let put = out;
            fwd.epilogue(move || {
                let r = rng.take();
                match ban {
                    Some(ban) => put.put(step_banning(&r, ban)),
                    None => put.put(step(&r).0),
                }
                rng.put(&r + iota(2));
            });
            let on = Pipeline::new();
            fwd.submit(&on)?;
            next = out.take_host::<i32>().await?;
            on.close();
            self.len = total;
        }
        Ok(next)
    }

    /// Feeds `first`, then decodes up to `count` tokens, ending after a stop token. Returns the
    /// tokens and their entropies; the last token is not fed yet.
    async fn decode(
        &mut self,
        first: i32,
        count: usize,
        seed: u32,
        stops: &[u32],
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let (page, start) = (kv_page_size(), self.len);
        let pages = self.reserve(start + count as u32 + 1)?;
        let token_in = Channel::from([first]);
        let embed_indptr = Channel::from([0u32, 1]);
        let positions = Channel::from([start]);
        let page_ids = Channel::from_iter(0..pages);
        let page_indptr = Channel::from([0u32, (start + 1).div_ceil(page)]);
        let w_slot = Channel::from([start / page]);
        let w_off = Channel::from([start % page]);
        let kv_len = Channel::from([start + 1]);
        let cap = channel_capacity() as u32;
        let out = Channel::new([1], dtype::i32).capacity(cap);
        let unsure = Channel::new([1], dtype::f32).capacity(cap);
        let rng = Channel::from([seed, start]);
        let fwd = ForwardPass::new();
        fwd.embed(&token_in, &embed_indptr)?;
        self.bind(
            &fwd,
            &kv_len,
            &page_ids,
            &page_indptr,
            &w_slot,
            &w_off,
            &positions,
            start,
        )?;
        let (o, u) = (out, unsure);
        fwd.epilogue(move || {
            let length = kv_len.take();
            let r = rng.take();
            let (token, h) = step(&r);
            let next_length = &length + 1u32;
            token_in.put(&token);
            kv_len.put(&next_length);
            positions.put(&length);
            w_slot.put(&length / page);
            w_off.put(&length % page);
            page_indptr.put(indptr(1, next_length.div_ceil(page)));
            rng.put(&r + iota(2));
            o.put(&token);
            u.put(&h);
        });
        let on = Pipeline::new();
        let mut tokens = Vec::new();
        let mut hs = Vec::new();
        run_ahead(&on, &fwd, count, async || {
            let token = out.take_host::<i32>().await? as u32;
            hs.push(unsure.take_host::<f32>().await?);
            tokens.push(token);
            Ok(if stops.contains(&token) {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            })
        })
        .await?;
        self.len = start + tokens.len() as u32;
        Ok((tokens, hs))
    }
}

#[derive(Clone, Copy, PartialEq)]
enum State {
    Live,
    Done,
    Killed,
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    /// The main agent: splits the remaining work into parallel tasks, or says the team is ready.
    Main,
    /// One sub-task; ends with a one-line finding posted to the blackboard.
    Worker,
    /// Writes the whole solution; its boxed answer is a vote.
    Solver,
    /// Settles solvers that disagree; its vote counts twice.
    Referee,
}

struct Agent {
    id: usize,
    kind: Kind,
    /// T-number for a worker, else empty.
    label: String,
    task: String,
    mind: Mind,
    next: i32,
    tokens: Vec<u32>,
    entropy: Vec<f32>,
    state: State,
    max: usize,
    answer: Option<String>,
    finding: Option<String>,
    /// Teammate findings it was handed while running.
    heard: usize,
    parent: Option<usize>,
    fate: &'static str,
    /// Tokens of its text already streamed.
    sent: usize,
    /// Where its current turn starts in `tokens`: the text a tool call is read from.
    turn_from: usize,
    /// Tool calls it has made.
    tools: usize,
    /// Helpers it spawned that have not reported yet; it is paused until they all have.
    waiting: usize,
    /// What its helpers reported, delivered together when the last one does.
    reports: Vec<String>,
}

impl Agent {
    fn unsure(&self, last: usize) -> f32 {
        let tail = &self.entropy[self.entropy.len().saturating_sub(last)..];
        if tail.is_empty() {
            0.0
        } else {
            tail.iter().sum::<f32>() / tail.len() as f32
        }
    }

    /// Ends it and frees its memory at once: its pages go back to the pool for the others.
    fn retire(&mut self, state: State, fate: &'static str) {
        self.state = state;
        self.fate = fate;
        self.mind = Mind {
            ws: WorkingSet::new(),
            rs: Vec::new(),
            len: 0,
        };
    }

    fn text(&self) -> String {
        model::decode(&self.tokens)
            .unwrap_or_default()
            .replace("<|im_end|>", "")
    }

    fn role(&self) -> String {
        match self.kind {
            Kind::Main => "main".into(),
            Kind::Worker => self.label.clone(),
            Kind::Solver => "solver".into(),
            Kind::Referee => "referee".into(),
        }
    }
}

struct Run {
    job: Job,
    /// Tokens a live agent decodes per round.
    chunk: usize,
    agents: Vec<Agent>,
    written: usize,
    began: u64,
    findings: usize,
    wave: usize,
}

impl Run {
    fn turn(&self, text: &str) -> Vec<u32> {
        let mut tokens = self.job.turn_open.clone();
        tokens.extend(model::encode(text));
        tokens.extend(&self.job.turn_close);
        tokens
    }

    fn say(&self, text: impl Into<String>) {
        send(serde_json::json!({ "event": "say", "text": text.into() }));
    }

    fn seconds(&self) -> f64 {
        (inferlet::monotonic_now_ns().saturating_sub(self.began)) as f64 / 1e9
    }

    /// Streams what each agent wrote since the last call.
    fn traces(&mut self) {
        for a in &mut self.agents {
            if a.sent < a.tokens.len() {
                let text = model::decode(&a.tokens[a.sent..]).unwrap_or_default();
                a.sent = a.tokens.len();
                send(serde_json::json!({ "event": "trace", "id": a.id + 1, "text": text }));
            }
        }
    }

    /// One snapshot of the team: every agent, what it does, and the run so far.
    fn board(&mut self, winner: Option<usize>) {
        self.traces();
        let agents: Vec<serde_json::Value> = self.agents.iter().map(|a| serde_json::json!({
            "id": a.id + 1,
            "parent": a.parent.map(|p| p + 1),
            "role": a.role(),
            "task": a.task,
            "state": match a.state { State::Live => "live", State::Done => "done", State::Killed => "killed" },
            "fate": a.fate,
            "tokens": a.tokens.len(),
            "max": a.max,
            "unsure": (a.unsure(ROUND) * 100.0).round() / 100.0,
            "answer": a.answer,
            "finding": a.finding,
            "heard": a.heard,
            "winner": winner == Some(a.id),
        })).collect();
        send(serde_json::json!({
            "event": "board", "wave": self.wave, "written": self.written,
            "seconds": (self.seconds() * 10.0).round() / 10.0, "findings": self.findings, "agents": agents,
        }));
    }

    /// A new agent forked from `from`: it holds everything `from` read and pays only for its turn.
    #[allow(
        clippy::too_many_arguments,
        reason = "an agent's whole identity, stated where it is made"
    )]
    async fn spawn(
        &mut self,
        from: &Mind,
        kind: Kind,
        label: String,
        task: String,
        ask: &str,
        max: usize,
        parent: Option<usize>,
    ) -> Result<usize> {
        let mut mind = from.fork()?;
        let id = self.agents.len();
        let next = mind
            .read(&self.turn(ask), self.job.seed.wrapping_add(100 + id as u32))
            .await?;
        self.agents.push(Agent {
            id,
            kind,
            label,
            task,
            mind,
            next,
            tokens: vec![next as u32],
            entropy: Vec::new(),
            state: State::Live,
            max,
            answer: None,
            finding: None,
            heard: 0,
            parent,
            fate: "",
            sent: 0,
            turn_from: 0,
            tools: 0,
            waiting: 0,
            reports: Vec::new(),
        });
        Ok(id)
    }

    /// Tokens each live agent gets this round. In a team, surer agents get twice the share and
    /// the most doubtful a quarter: they all run at once, so a short share leaves the fires sooner
    /// and the fires left are smaller and faster for the agents doing well.
    fn shares(&self) -> Vec<(usize, usize)> {
        let mut live: Vec<(usize, f32)> = self
            .agents
            .iter()
            .filter(|a| a.state == State::Live)
            .map(|a| {
                (
                    a.id,
                    if a.entropy.len() < ROUND {
                        f32::NAN
                    } else {
                        a.unsure(ROUND)
                    },
                )
            })
            .collect();
        let ranked: Vec<usize> = {
            let mut scored: Vec<(usize, f32)> =
                live.iter().copied().filter(|(_, u)| !u.is_nan()).collect();
            scored.sort_by(|a, b| a.1.total_cmp(&b.1));
            scored.into_iter().map(|(id, _)| id).collect()
        };
        let n = ranked.len();
        live.iter_mut()
            .map(|&mut (id, _)| {
                let a = &self.agents[id];
                let weight = match ranked.iter().position(|&r| r == id) {
                    Some(r) if self.job.mode == "team" && n >= 3 && r < n / 3 => 2.0,
                    Some(r) if self.job.mode == "team" && n >= 3 && r >= n - n / 3 => 0.25,
                    _ => 1.0,
                };
                let room = a.max.saturating_sub(a.tokens.len()).max(1);
                (
                    id,
                    ((self.chunk as f32 * weight) as usize)
                        .clamp(32, room.max(32))
                        .min(room),
                )
            })
            .collect()
    }

    /// One round: every live agent decodes its share, batched by the engine into the same fires.
    /// Returns the agents that finished.
    async fn round(&mut self) -> Result<Vec<usize>> {
        let shares = self.shares();
        let (wave, seed, stops, chunk) =
            (self.wave, self.job.seed, self.job.stops.clone(), self.chunk);
        let stops = &stops;
        let results = join_all(
            self.agents
                .iter_mut()
                .filter(|a| a.state == State::Live && a.waiting == 0)
                .map(|a| {
                    let count = shares
                        .iter()
                        .find(|(id, _)| *id == a.id)
                        .map_or(chunk, |(_, c)| *c);
                    async move {
                        let s = seed
                            .wrapping_add((a.id as u32).wrapping_mul(104_729))
                            .wrapping_add(a.tokens.len() as u32 + wave as u32);
                        (a.id, a.mind.decode(a.next, count, s, stops).await)
                    }
                }),
        )
        .await;
        let mut finished = Vec::new();
        for (id, result) in results {
            let (tokens, hs) = result?;
            let a = &mut self.agents[id];
            self.written += tokens.len();
            a.next = tokens.last().map_or(a.next, |&t| t as i32);
            let stopped = tokens.last().is_some_and(|t| stops.contains(t));
            a.tokens.extend(tokens);
            a.entropy.extend(hs);
            // A worker is done the moment its finding line is complete.
            let reported = a.kind == Kind::Worker && {
                let text = a.text();
                text.rfind("RESULT:")
                    .is_some_and(|at| text[at..].contains('\n'))
            };
            if stopped || reported || a.tokens.len() >= a.max {
                finished.push(id);
            }
        }
        Ok(finished)
    }

    /// A worker that ran out of room without its `RESULT:` line is asked for just that line.
    async fn conclude(&mut self, id: usize) -> Result<()> {
        if self.agents[id].text().contains("RESULT:") {
            return Ok(());
        }
        let mut ask = self.job.seal.clone();
        ask.extend(self.turn(CONCLUDE));
        let (seed, stops) = (
            self.job.seed.wrapping_add(300 + id as u32),
            self.job.stops.clone(),
        );
        let a = &mut self.agents[id];
        a.tokens.pop();
        a.sent = a.sent.min(a.tokens.len());
        let first = a.mind.read(&ask, seed).await?;
        let (more, _) = a.mind.decode(first, 120, seed, &stops).await?;
        self.written += more.len() + 1;
        let a = &mut self.agents[id];
        a.tokens.extend(model::encode("\n"));
        a.tokens.push(first as u32);
        a.tokens.extend(more);
        Ok(())
    }

    /// Posts a finding where every later agent sees it: on the blackboard, which new agents fork
    /// from, and as a note to each running worker, read once there.
    async fn share(&mut self, blackboard: &mut Mind, from: usize, text: &str) -> Result<()> {
        let source = format!("{} ({})", self.agents[from].label, self.agents[from].task);
        let posted = self.turn(&format!("Team finding from {source}: {text}"));
        blackboard
            .read(&posted[..posted.len() - self.job.turn_close.len()], 3)
            .await?;
        self.findings += 1;
        let mut note = self.job.seal.clone();
        note.extend(self.turn(&format!(
            "Note from teammate {source}: {text}\nUse it if it helps your task, then continue."
        )));
        let listeners: Vec<usize> = self
            .agents
            .iter()
            .filter(|a| a.state == State::Live && a.kind == Kind::Worker && a.id != from)
            .map(|a| a.id)
            .collect();
        self.traces();
        // Every listener reads the note at once: one batched fire, not one round trip each.
        let seed = self.job.seed;
        let note = &note;
        let read = join_all(
            self.agents
                .iter_mut()
                .filter(|a| listeners.contains(&a.id))
                .map(|a| async move {
                    // Its last token was sampled but never fed; the note closes the turn before it.
                    a.tokens.pop();
                    a.sent = a.sent.min(a.tokens.len());
                    (
                        a.id,
                        a.mind
                            .read(note, seed.wrapping_add(200 + a.id as u32))
                            .await,
                    )
                }),
        )
        .await;
        for (id, next) in read {
            send(
                serde_json::json!({ "event": "trace", "id": id + 1, "text": format!("\n\n[note from {source}: {text}]\n\n") }),
            );
            let a = &mut self.agents[id];
            a.next = next?;
            a.heard += 1;
            a.tokens.extend(model::encode("\n"));
            a.sent = a.tokens.len();
            a.tokens.push(a.next as u32);
        }
        Ok(())
    }

    /// Runs live agents until none is left, recording each solver's boxed answer; stops early
    /// when `quorum` votes agree.
    async fn solve(&mut self, quorum: Option<usize>) -> Result<()> {
        while self.agents.iter().any(|a| a.state == State::Live) && self.written < self.job.budget {
            for id in self.round().await? {
                let answer = boxed(&self.agents[id].text());
                self.agents[id].answer = answer.clone();
                self.agents[id].retire(
                    State::Done,
                    if answer.is_some() {
                        "answered"
                    } else {
                        "no answer"
                    },
                );
                if let Some(answer) = answer {
                    self.say(format!(
                        "{} (agent {}) answered {answer}",
                        self.agents[id].role(),
                        id + 1
                    ));
                }
            }
            if let Some(quorum) = quorum
                && let Some((answer, votes, runner_up, _)) = tally(&self.agents)
                && votes >= quorum
                && votes > runner_up
            {
                let stopped = self
                    .agents
                    .iter()
                    .filter(|a| a.state == State::Live)
                    .count();
                for a in self.agents.iter_mut().filter(|a| a.state == State::Live) {
                    a.retire(State::Killed, "stopped");
                }
                self.say(format!("{votes} votes agree on {answer}; {stopped} still running are killed, their memory freed."));
                break;
            }
            self.board(None);
        }
        Ok(())
    }
}

/// The last `\boxed{...}` in `text`, braces balanced.
fn boxed(text: &str) -> Option<String> {
    let at = text.rfind("\\boxed{")?;
    let mut depth = 0;
    let mut out = String::new();
    for c in text[at + 7..].chars() {
        match c {
            '{' => depth += 1,
            '}' if depth == 0 => return Some(out.trim().to_string()),
            '}' => depth -= 1,
            _ => {}
        }
        out.push(c);
    }
    None
}

fn words(text: &str) -> HashSet<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 3)
        .map(String::from)
        .collect()
}

/// A worker's finding: its last `RESULT:` line, or its last words.
fn finding(text: &str) -> String {
    if let Some(at) = text.rfind("RESULT:") {
        let line = text[at + 7..].lines().next().unwrap_or_default().trim();
        return line.trim_matches(['`', '*']).chars().take(400).collect();
    }
    let tail: String = text.chars().rev().take(300).collect();
    tail.chars().rev().collect::<String>().trim().to_string()
}

/// The main agent's tasks: `T<n>: ...` lines, near-duplicates of each other or of any task already
/// run dropped, so no two agents do the same work. `None` means it said READY.
fn tasks(plan: &str, width: usize, done: &[String]) -> Option<Vec<String>> {
    let mut kept: Vec<String> = Vec::new();
    for line in plan.lines() {
        let line = line.trim().trim_start_matches(['*', '-', ' ']);
        let Some(rest) = line.strip_prefix('T') else {
            continue;
        };
        let Some((n, body)) = rest.split_once(':') else {
            continue;
        };
        if n.trim().parse::<u32>().is_err() || body.trim().len() < 8 {
            continue;
        }
        let body = body.trim().to_string();
        let these = words(&body);
        let repeats = kept.iter().chain(done.iter()).any(|k| {
            let other = words(k);
            these.intersection(&other).count() * 10 >= these.len().min(other.len()).max(1) * 7
        });
        if !repeats {
            kept.push(body);
        }
        if kept.len() == width {
            break;
        }
    }
    if kept.is_empty() && plan.to_uppercase().contains("READY") {
        return None;
    }
    Some(kept)
}

/// The answer most finished solvers and referees gave, its votes, the runner-up's, and the
/// surest agent that gave it.
fn tally(agents: &[Agent]) -> Option<(String, usize, usize, usize)> {
    let mut votes: Vec<(String, usize, f32, usize)> = Vec::new();
    for a in agents
        .iter()
        .filter(|a| a.state == State::Done && matches!(a.kind, Kind::Solver | Kind::Referee))
    {
        if let Some(answer) = &a.answer {
            let weight = if a.kind == Kind::Referee { 2 } else { 1 };
            let sure = -a.unsure(usize::MAX);
            match votes.iter_mut().find(|v| &v.0 == answer) {
                Some(v) => {
                    v.1 += weight;
                    if sure > v.2 {
                        v.2 = sure;
                        v.3 = a.id;
                    }
                }
                None => votes.push((answer.clone(), weight, sure, a.id)),
            }
        }
    }
    votes.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.total_cmp(&a.2)));
    let top = votes.first()?;
    Some((top.0.clone(), top.1, votes.get(1).map_or(0, |v| v.1), top.3))
}

impl Run {
    /// Handles every agent that finished this round, whatever it was doing. Returns true when
    /// enough votes agree: everything still running is killed and its memory freed.
    async fn settle(&mut self, finished: Vec<usize>, blackboard: &mut Mind) -> Result<bool> {
        let mut calls: Vec<(usize, String)> = Vec::new();
        let mut spawns: Vec<(usize, Vec<String>)> = Vec::new();
        let mut delivered: Vec<usize> = Vec::new();
        for id in finished {
            if self.job.mode == "code" {
                // A turn can end in a spawn block, a Python block, or neither.
                if let Some(tasks) = self.spawn_call(id) {
                    spawns.push((id, tasks));
                    continue;
                }
                if let Some(code) = self.tool_call(id) {
                    calls.push((id, code));
                    continue;
                }
                let a = &self.agents[id];
                let turn = model::decode(&a.tokens[a.turn_from..]).unwrap_or_default();
                if a.kind == Kind::Solver
                    && boxed(&turn).is_none()
                    && a.tools < TOOL_TURNS
                    && a.tokens.len() < a.max
                {
                    self.agents[id].tools += 1;
                    self.resume(id, NUDGE).await?;
                    continue;
                }
                if self.agents[id].kind == Kind::Worker {
                    self.conclude(id).await?;
                    let found = finding(&self.agents[id].text());
                    self.agents[id].finding = Some(found.clone());
                    self.agents[id].retire(State::Done, "reported");
                    if let Some(parent) = self.agents[id].parent {
                        let label = self.agents[id].label.clone();
                        let task = self.agents[id].task.clone();
                        self.agents[parent]
                            .reports
                            .push(format!("{label} ({task}): {found}"));
                        self.agents[parent].waiting -= 1;
                        if self.agents[parent].waiting == 0 {
                            delivered.push(parent);
                        }
                    }
                    self.say(format!("{} reports: {found}", self.agents[id].label));
                    continue;
                }
            }
            match self.agents[id].kind {
                Kind::Main => self.agents[id].retire(State::Done, "planned"),
                Kind::Worker => {
                    self.conclude(id).await?;
                    let found = finding(&self.agents[id].text());
                    self.agents[id].finding = Some(found.clone());
                    self.agents[id].retire(State::Done, "shared");
                    self.say(format!("{} found: {found}", self.agents[id].label));
                    self.share(blackboard, id, &found).await?;
                }
                Kind::Solver | Kind::Referee => {
                    let answer = boxed(&self.agents[id].text());
                    self.agents[id].answer = answer.clone();
                    self.agents[id].retire(
                        State::Done,
                        if answer.is_some() {
                            "answered"
                        } else {
                            "no answer"
                        },
                    );
                    if let Some(answer) = answer {
                        self.say(format!(
                            "{} (agent {}) answered {answer}",
                            self.agents[id].role(),
                            id + 1
                        ));
                    }
                }
            }
        }
        if !calls.is_empty() {
            self.run_tools(calls).await?;
        }
        for (id, tasks) in spawns {
            self.spawn_helpers(id, tasks).await?;
        }
        for id in delivered {
            let reports = std::mem::take(&mut self.agents[id].reports);
            self.resume(
                id,
                &format!("Your helpers' results:\n{}", reports.join("\n")),
            )
            .await?;
        }
        if let Some((answer, mut votes, runner_up, _)) = tally(&self.agents)
            && {
                if self.job.mode == "tree" {
                    votes = independent(&self.agents, &answer);
                }
                votes >= QUORUM && votes > runner_up.min(votes.saturating_sub(1))
            }
        {
            let stopped = self
                .agents
                .iter()
                .filter(|a| a.state == State::Live)
                .count();
            for a in self.agents.iter_mut().filter(|a| a.state == State::Live) {
                a.retire(State::Killed, "stopped");
            }
            self.say(format!("{votes} independent votes agree on {answer}; {stopped} agents still running are killed, their memory freed."));
            return Ok(true);
        }
        self.board(None);
        Ok(false)
    }

    /// The Python block an agent's finished turn ends on, if it asked to run one and may.
    fn tool_call(&self, id: usize) -> Option<String> {
        let a = &self.agents[id];
        if a.tools >= TOOL_TURNS || a.tokens.len() >= a.max {
            return None;
        }
        let turn = model::decode(&a.tokens[a.turn_from..]).unwrap_or_default();
        if boxed(&turn).is_some() {
            return None;
        }
        // Any fence is a request: ```python, ```py, or a bare one, the way the model writes it.
        let open = turn
            .rfind("```")
            .and_then(|close| turn[..close].rfind("```"))?;
        let body = &turn[open + 3..];
        let body = body
            .strip_prefix("python3")
            .or(body.strip_prefix("python"))
            .or(body.strip_prefix("py"))
            .unwrap_or(body);
        if body.trim_start().starts_with("spawn") {
            return None;
        }
        let end = body.find("```")?;
        Some(body[..end].trim().to_string())
    }

    /// The tasks of a `spawn` block an agent's finished turn ends on.
    fn spawn_call(&self, id: usize) -> Option<Vec<String>> {
        let a = &self.agents[id];
        if a.kind == Kind::Worker || a.tokens.len() >= a.max {
            return None;
        }
        let turn = model::decode(&a.tokens[a.turn_from..]).unwrap_or_default();
        if boxed(&turn).is_some() {
            return None;
        }
        let start = turn.rfind("```spawn")? + "```spawn".len();
        let end = turn[start..].find("```").map_or(turn.len(), |e| e + start);
        let tasks: Vec<String> = turn[start..end]
            .lines()
            .map(|l| {
                l.trim()
                    .trim_start_matches(|c: char| {
                        c.is_ascii_digit() || matches!(c, '-' | '*' | '.' | ')' | ' ')
                    })
                    .trim()
                    .to_string()
            })
            .filter(|l| l.len() >= 8)
            .take(SPAWN_MAX)
            .collect();
        (!tasks.is_empty()).then_some(tasks)
    }

    /// Starts a helper per task, each a fork of the agent as it is now: it knows everything the
    /// agent knows and pays only for its own task. The agent pauses until they all report.
    async fn spawn_helpers(&mut self, id: usize, tasks: Vec<String>) -> Result<()> {
        self.say(format!(
            "agent {} spawns {} helpers that run at once",
            id + 1,
            tasks.len()
        ));
        for task in &tasks {
            let mut ask = self.job.seal.clone();
            ask.extend(self.turn(&HELPER.replace("{task}", task)));
            let mut mind = self.agents[id].mind.fork()?;
            let helper = self.agents.len();
            let next = mind
                .read(&ask, self.job.seed.wrapping_add(700 + helper as u32))
                .await?;
            self.agents.push(Agent {
                id: helper,
                kind: Kind::Worker,
                label: format!("H{}", helper + 1),
                task: task.clone(),
                mind,
                next,
                tokens: vec![next as u32],
                entropy: Vec::new(),
                state: State::Live,
                max: WORKER_MAX,
                answer: None,
                finding: None,
                heard: 0,
                parent: Some(id),
                fate: "",
                sent: 0,
                turn_from: 0,
                tools: 0,
                waiting: 0,
                reports: Vec::new(),
            });
            self.say(format!("  H{}: {task}", helper + 1));
        }
        let a = &mut self.agents[id];
        a.waiting = tasks.len();
        a.fate = "delegated";
        self.board(None);
        Ok(())
    }

    /// Wakes a paused agent with `text` as its next turn.
    async fn resume(&mut self, id: usize, text: &str) -> Result<()> {
        let mut tokens = self.job.seal.clone();
        tokens.extend(self.turn(text));
        self.traces();
        send(
            serde_json::json!({ "event": "trace", "id": id + 1, "text": format!("\n\n[{text}]\n\n") }),
        );
        let seed = self.job.seed.wrapping_add(900 + id as u32);
        let a = &mut self.agents[id];
        a.next = a.mind.read(&tokens, seed).await?;
        a.sent = a.tokens.len();
        a.turn_from = a.tokens.len();
        a.tokens.push(a.next as u32);
        a.fate = "";
        Ok(())
    }

    /// Hands each call to the host, which runs it sandboxed, and writes every output into its
    /// agent's context as the next turn, all in one batched read: no agent re-reads anything.
    async fn run_tools(&mut self, calls: Vec<(usize, String)>) -> Result<()> {
        for (id, code) in &calls {
            send(serde_json::json!({ "event": "tool", "id": id + 1, "code": code }));
        }
        let mut outputs: Vec<(usize, String)> = Vec::new();
        while outputs.len() < calls.len() {
            let Some(reply) = session::receive().await else {
                return Err("the host went away while code ran".into());
            };
            let reply: serde_json::Value =
                serde_json::from_str(&reply).map_err(|e| e.to_string())?;
            let id = reply["id"].as_u64().unwrap_or(0) as usize;
            if id == 0 || id > self.agents.len() {
                continue;
            }
            outputs.push((
                id - 1,
                reply["output"].as_str().unwrap_or_default().to_string(),
            ));
        }
        self.traces();
        let turns: Vec<(usize, Vec<u32>)> = outputs
            .iter()
            .map(|(id, out)| {
                let mut tokens = self.job.seal.clone();
                tokens.extend(self.turn(&format!("Output:\n```\n{out}\n```")));
                (*id, tokens)
            })
            .collect();
        let seed = self.job.seed;
        let turns = &turns;
        let read = join_all(
            self.agents
                .iter_mut()
                .filter(|a| turns.iter().any(|(id, _)| *id == a.id))
                .map(|a| async move {
                    let tokens = &turns
                        .iter()
                        .find(|(id, _)| *id == a.id)
                        .expect("a turn per caller")
                        .1;
                    (
                        a.id,
                        a.mind
                            .read(tokens, seed.wrapping_add(500 + a.id as u32))
                            .await,
                    )
                }),
        )
        .await;
        for (id, next) in read {
            let out = &outputs
                .iter()
                .find(|(o, _)| *o == id)
                .expect("an output per caller")
                .1;
            send(
                serde_json::json!({ "event": "trace", "id": id + 1, "text": format!("\n\n[output]\n{out}\n\n") }),
            );
            let a = &mut self.agents[id];
            a.next = next?;
            a.tools += 1;
            a.sent = a.tokens.len();
            a.turn_from = a.tokens.len();
            a.tokens.push(a.next as u32);
            self.say(format!(
                "agent {} ran code ({} so far)",
                id + 1,
                self.agents[id].tools
            ));
        }
        Ok(())
    }

    /// Rounds until `busy` holds no live agent; true when the run is decided on the way.
    async fn until_done(&mut self, busy: &[usize], blackboard: &mut Mind) -> Result<bool> {
        while busy.iter().any(|&id| self.agents[id].state == State::Live)
            && self.written < self.job.budget
        {
            let finished = self.round().await?;
            if self.settle(finished, blackboard).await? {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

/// The team. Scouts race straight at the answer while the main agent plans: an easy question is
/// settled by two scouts agreeing, and the team is killed before it costs anything. A hard one
/// gets waves of parallel workers on a shared blackboard, then solvers and a referee from it.
async fn team(run: &mut Run, root: &Mind) -> Result<()> {
    let width = run.job.width.clamp(1, 16);
    let mut blackboard = root.fork()?;
    for _ in 0..SCOUTS {
        run.spawn(
            root,
            Kind::Solver,
            String::new(),
            "scout: a direct attempt, alone".into(),
            SOLVE_ALONE,
            SOLVER_MAX,
            None,
        )
        .await?;
    }
    run.say(format!(
        "{SCOUTS} scouts race at the answer while the main agent plans the team."
    ));
    let mut done_tasks: Vec<String> = Vec::new();
    for wave in 1..=WAVES {
        run.wave = wave;
        if run.written >= run.job.budget / 2 {
            break;
        }
        let least = FIRST_WAVE_MIN.min(width);
        let mut planned = None;
        let mut main = 0;
        for attempt in 0..2 {
            let mut ask = if wave == 1 {
                LEAD_FIRST
                    .replace("{min}", &least.to_string())
                    .replace("{width}", &width.to_string())
            } else {
                format!(
                    "{} Already done: {}.",
                    LEAD_NEXT.replace("{width}", &width.to_string()),
                    done_tasks.join("; ")
                )
            };
            if attempt > 0 {
                ask.push_str(&format!(" Your last plan had too few tasks: split the work into at least {least} parallel tasks."));
            }
            main = run
                .spawn(
                    &blackboard,
                    Kind::Main,
                    String::new(),
                    format!("plan wave {wave}"),
                    &ask,
                    500,
                    None,
                )
                .await?;
            run.board(None);
            if run.until_done(&[main], &mut blackboard).await? {
                return Ok(());
            }
            planned = tasks(&run.agents[main].text(), width, &done_tasks);
            if wave > 1 || planned.as_ref().is_some_and(|t| t.len() >= 2) {
                break;
            }
            run.agents[main].fate = "replanned";
            run.say("The main agent's first plan was too thin; asked to split the work further.");
        }
        let Some(planned) = planned.filter(|t| !t.is_empty()) else {
            run.say(format!(
                "Main agent (wave {wave}): the team has what it needs."
            ));
            break;
        };
        run.say(format!(
            "Main agent (wave {wave}) delegates {} parallel tasks:",
            planned.len()
        ));
        let mut workers = Vec::new();
        for (i, task) in planned.iter().enumerate() {
            let label = format!("T{}", done_tasks.len() + i + 1);
            run.say(format!("  {label}: {task}"));
            let ask = WORK.replace("{task}", task);
            workers.push(
                run.spawn(
                    &blackboard,
                    Kind::Worker,
                    label,
                    task.clone(),
                    &ask,
                    WORKER_MAX,
                    Some(main),
                )
                .await?,
            );
        }
        done_tasks.extend(planned);
        run.board(None);
        if run.until_done(&workers, &mut blackboard).await? {
            return Ok(());
        }
    }

    run.wave += 1;
    run.say(format!(
        "{} findings on the blackboard; {SOLVERS} solvers start from all of them.",
        run.findings
    ));
    let mut solvers = Vec::new();
    for _ in 0..SOLVERS {
        solvers.push(
            run.spawn(
                &blackboard,
                Kind::Solver,
                String::new(),
                "the whole solution, from the findings".into(),
                SOLVE,
                SOLVER_MAX,
                None,
            )
            .await?,
        );
    }
    let everyone: Vec<usize> = (0..run.agents.len()).collect();
    if run.until_done(&everyone, &mut blackboard).await? {
        return Ok(());
    }
    if let Some((first, _, runner_up, _)) = tally(&run.agents)
        && runner_up > 0
        && run.written < run.job.budget
    {
        let others: Vec<String> = run
            .agents
            .iter()
            .filter_map(|a| a.answer.clone())
            .filter(|a| a != &first)
            .collect();
        run.say(format!(
            "Solvers disagree ({first} vs {}); a referee checks.",
            others.join(", ")
        ));
        let ask = REFEREE.replace("{answers}", &format!("{first} and {}", others.join(", ")));
        let referee = run
            .spawn(
                &blackboard,
                Kind::Referee,
                String::new(),
                "settle the disagreement".into(),
                &ask,
                SOLVER_MAX,
                None,
            )
            .await?;
        run.until_done(&[referee], &mut blackboard).await?;
    }
    Ok(())
}

/// A branch's state at the start of its current chunk: if the chunk turns out to hold a
/// hesitation, siblings start here and take that decision differently.
struct Checkpoint {
    mind: Mind,
    next: i32,
    tokens: usize,
    entropy: usize,
}

/// Fork on doubt. One agent solves; before each short chunk pie checkpoints it (a copy-on-write
/// fork), and if the model hesitates inside the chunk, siblings start from the checkpoint and
/// take that decision their own way. They share everything written before it, so the only extra
/// work is the hard decision itself; any branch can branch again at its own next doubt. Branches
/// that converge onto another's path are killed at once, the live count stays within `width`,
/// and two branches that agree end the run. A question the model never hesitates on never
/// branches and runs at one agent's speed.
async fn tree(run: &mut Run, root: &Mind) -> Result<()> {
    run.chunk = TREE_CHUNK;
    let width = run.job.width.clamp(1, 16);
    let mut blackboard = root.fork()?;
    run.spawn(
        root,
        Kind::Solver,
        "B1".into(),
        "the solution".into(),
        SOLVE_ALONE,
        SOLVER_MAX,
        None,
    )
    .await?;
    run.board(None);
    loop {
        let live: Vec<usize> = run
            .agents
            .iter()
            .filter(|a| a.state == State::Live)
            .map(|a| a.id)
            .collect();
        if live.is_empty() || run.written >= run.job.budget {
            return Ok(());
        }
        // Checkpoints only while there is room to branch: a full tree cannot use one.
        let mut checkpoints: Vec<(usize, Checkpoint)> = Vec::new();
        if live.len() < width {
            for &id in &live {
                let a = &run.agents[id];
                let mind = a.mind.fork()?;
                checkpoints.push((
                    id,
                    Checkpoint {
                        mind,
                        next: a.next,
                        tokens: a.tokens.len(),
                        entropy: a.entropy.len(),
                    },
                ));
            }
        }
        let finished = run.round().await?;
        // Where did each live branch hesitate most in this chunk, and how much?
        let mut doubts: Vec<(usize, f32, usize)> = checkpoints
            .iter()
            .filter(|(id, _)| !finished.contains(id))
            .filter_map(|(id, c)| {
                let chunk = &run.agents[*id].entropy[c.entropy..];
                let (at, peak) =
                    chunk
                        .iter()
                        .copied()
                        .enumerate()
                        .fold(
                            (0, 0.0f32),
                            |best, (i, h)| if h > best.1 { (i, h) } else { best },
                        );
                // A fork late in an agent's budget has no room left to finish a different road.
                let early = c.tokens * 4 < run.agents[*id].max * 3;
                (early && peak > run.job.doubt).then_some((*id, peak, at))
            })
            .collect();
        doubts.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (id, peak, at) in doubts {
            if run.agents.iter().filter(|a| a.state == State::Live).count() >= width {
                break;
            }
            let Some(slot) = checkpoints.iter().position(|(c, _)| *c == id) else {
                continue;
            };
            let (_, mut c) = checkpoints.swap_remove(slot);
            // Replay the parent up to the token it hesitated on, then take any token but its.
            let (replay, taken, tokens, entropy) = {
                let parent = &run.agents[id];
                let mut replay = vec![c.next as u32];
                replay.extend(&parent.tokens[c.tokens..c.tokens + at]);
                (
                    replay,
                    parent.tokens[c.tokens + at],
                    parent.tokens[..c.tokens + at].to_vec(),
                    parent.entropy[..c.entropy + at].to_vec(),
                )
            };
            let sibling = run.agents.len();
            let seed = run.job.seed.wrapping_add(31 * sibling as u32 + 1);
            let next = c.mind.read_banning(&replay, seed, Some(taken)).await?;
            let label = format!("B{}", sibling + 1);
            let mut tokens = tokens;
            tokens.push(next as u32);
            let mut entropy = entropy;
            entropy.push(peak);
            let fork_at = tokens.len() - 1;
            run.agents.push(Agent {
                id: sibling,
                kind: Kind::Solver,
                label: label.clone(),
                task: format!(
                    "branch of agent {} at token {fork_at} (doubt {peak:.1})",
                    id + 1
                ),
                mind: c.mind,
                next,
                tokens,
                entropy,
                state: State::Live,
                max: SOLVER_MAX,
                answer: None,
                finding: None,
                heard: 0,
                parent: Some(id),
                fate: "",
                sent: fork_at,
                turn_from: 0,
                tools: 0,
                waiting: 0,
                reports: Vec::new(),
            });
            run.say(format!(
                "{label} forks from agent {} at token {fork_at}, where it hesitated (entropy {peak:.1}), and takes the other road.",
                id + 1
            ));
        }
        drop(checkpoints);
        if run.settle(finished, &mut blackboard).await? {
            return Ok(());
        }
        // Converged branches do the same work twice: the surer one stays.
        let live: Vec<usize> = run
            .agents
            .iter()
            .filter(|a| a.state == State::Live)
            .map(|a| a.id)
            .collect();
        for (i, &x) in live.iter().enumerate() {
            for &y in &live[i + 1..] {
                if run.agents[x].state != State::Live || run.agents[y].state != State::Live {
                    continue;
                }
                // Only what each wrote after they parted: a fresh fork shares its parent's text.
                let (tx, ty) = (&run.agents[x].tokens, &run.agents[y].tokens);
                let parted = tx.iter().zip(ty.iter()).take_while(|(a, b)| a == b).count();
                if tx.len() < parted + 128 || ty.len() < parted + 128 {
                    continue;
                }
                let (gx, gy) = (grams(&tx[parted..]), grams(&ty[parted..]));
                if gx.intersection(&gy).count() * 10 > gx.len().min(gy.len()) * 8 {
                    let (keep, lose) = if run.agents[x].unsure(256) <= run.agents[y].unsure(256) {
                        (x, y)
                    } else {
                        (y, x)
                    };
                    run.agents[lose].retire(State::Killed, "merged");
                    run.say(format!(
                        "agent {} rejoined agent {}: same path, no double work",
                        lose + 1,
                        keep + 1
                    ));
                }
            }
        }
        // With no rival left running, a finished answer is the answer.
        let still = run.agents.iter().filter(|a| a.state == State::Live).count();
        if still == 0 && run.agents.iter().any(|a| a.answer.is_some()) {
            return Ok(());
        }
    }
}

/// Agents that gave `answer` and did not reach it on the same path: two that share more than
/// half their text before they part are one line of reasoning, and count once.
fn independent(agents: &[Agent], answer: &str) -> usize {
    let mut kept: Vec<&Agent> = Vec::new();
    for a in agents
        .iter()
        .filter(|a| a.state == State::Done && a.answer.as_deref() == Some(answer))
    {
        let alone = kept.iter().all(|k| {
            let shared = k
                .tokens
                .iter()
                .zip(&a.tokens)
                .take_while(|(x, y)| x == y)
                .count();
            shared * 2 < k.tokens.len().min(a.tokens.len())
        });
        if alone {
            kept.push(a);
        }
    }
    kept.len()
}

/// Token 4-grams of an agent's last 256 tokens.
fn grams(tokens: &[u32]) -> HashSet<[u32; 4]> {
    tokens[tokens.len().saturating_sub(256)..]
        .windows(4)
        .map(|w| [w[0], w[1], w[2], w[3]])
        .collect()
}

#[inferlet::main]
async fn main(job: Job) -> Result<String> {
    let mut run = Run {
        chunk: ROUND,
        job,
        agents: Vec::new(),
        written: 0,
        began: inferlet::monotonic_now_ns(),
        findings: 0,
        wave: 0,
    };
    let mut root = Mind::new();
    root.read(&run.job.head.clone(), run.job.seed).await?;
    match run.job.mode.as_str() {
        "team" => team(&mut run, &root).await?,
        "tree" => tree(&mut run, &root).await?,
        "code" => {
            // One agent, like OpenCode's, that can run code and decides itself when to spawn.
            let mut blackboard = root.fork()?;
            let ask = format!("{CODE_RULES}\n{SPAWN_RULES}");
            run.spawn(
                &root,
                Kind::Solver,
                "A1".into(),
                "the solution".into(),
                &ask,
                CODE_MAX,
                None,
            )
            .await?;
            run.chunk = TREE_CHUNK * 2;
            run.board(None);
            while run.agents.iter().any(|a| a.state == State::Live) && run.written < run.job.budget
            {
                let finished = run.round().await?;
                if run.settle(finished, &mut blackboard).await? {
                    break;
                }
            }
        }
        mode => {
            // The baselines read the question with the plain instruction closing its turn.
            let n = if mode == "vote" {
                run.job.width.max(1)
            } else {
                1
            };
            for _ in 0..n {
                run.spawn(
                    &root,
                    Kind::Solver,
                    String::new(),
                    "the whole solution, alone".into(),
                    SOLVE_ALONE,
                    SOLVER_MAX,
                    None,
                )
                .await?;
            }
            run.board(None);
            run.solve(None).await?;
        }
    }
    let (answer, votes, winner) = match tally(&run.agents) {
        Some((answer, votes, _, winner)) => (answer, votes, Some(winner)),
        None => (String::new(), 0, None),
    };
    run.board(winner);
    let workers = run.agents.iter().filter(|a| a.kind == Kind::Worker).count();
    send(serde_json::json!({
        "event": "done", "answer": answer, "votes": votes, "agents": run.agents.len(), "workers": workers,
        "findings": run.findings, "written": run.written, "seconds": (run.seconds() * 10.0).round() / 10.0,
        "solution": winner.map(|w| run.agents[w].text()),
    }));
    Ok(String::new())
}
