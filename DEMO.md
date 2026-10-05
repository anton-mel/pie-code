# pie agents: a team of agents that organizes itself, on one MacBook

Everything here runs locally on Qwen3.8-27B (4-bit). The numbers are in [RESULTS.md](RESULTS.md).

## Setup

```sh
cd pie-mpp && cargo build --release -p pie --features metal --bin pie && cd ..
(cd army && cargo build --release --target wasm32-wasip2)
python -m venv .venv && .venv/bin/pip install -e pie-mpp/python/client transformers datasets aiohttp
```

## 1. Watch an agent army solve a hard problem

```sh
.venv/bin/python army.py ask "Find the number of ordered pairs (x,y), where both x and y are integers \
between -100 and 100, inclusive, such that 12x^2-xy-6y^2=0."
```

The terminal redraws the team live, one line per agent: its role (main, T1…, solver), what it is
doing, its tokens, its doubt, and its finding or answer. What happens:

- **Two scouts race straight at the answer** while the **main agent** plans; if the scouts agree
  the question was easy, and everything else is killed at once.
- Otherwise the main agent **delegates** 3–8 sub-tasks that run **in parallel** (T1, T2, …). It
  never solves alone, and it never assigns the same work twice.
- Each finding is shared the moment it lands: appended once to a **blackboard** in pie's memory that
  every later agent forks from (no re-reading), and written into every **running** teammate's
  context as a note.
- Each round, **surer agents get more tokens** and doubtful ones fewer (pie measures every token's
  uncertainty on the GPU for free).
- Solvers start from the full blackboard; **two that agree end the run** and the rest are killed,
  their memory freed instantly; a referee settles disagreements.

## 2. The same team on llama.cpp

```sh
.venv/bin/python army.py ask "<same question>" --engine llama
```

Same prompts and plan over llama.cpp's API (8 parallel slots, prompt caching): where pie forks an
agent from shared memory, llama.cpp re-reads it; where pie writes a note into a running agent,
llama.cpp re-issues the request.

## 3. The benchmark

```sh
./run_army.sh 15        # AIME 2025: pie team / single / vote, llama.cpp team / single
.venv/bin/python results.py
```

## 4. LatentMAS: agents that share hidden states, not text

```sh
.venv/bin/python latentmas.py --method latent --task gsm8k --steps 10 --n 100
.venv/bin/python latent_parity.py --model Qwen/Qwen3-4B --config pie-latent-q3-small.toml --scale 1.0974
```

pie gained two engine features for this (`pie-mpp` branch `feat/latent-steps`): a `hidden` reading
that returns the last layer's state, and a `latent` reading that feeds a vector in place of a token.
Agents then think in latent steps on the GPU with no host round trip, and the next agent reads
the previous agent's thoughts straight from the shared KV.
