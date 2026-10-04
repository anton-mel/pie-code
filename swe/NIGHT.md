# Overnight SWE-bench run: plan and log

Goal: SWE-bench Verified (16 easy tasks, 11 repos), many agents at once, scored by the official harness.
Compare: mini-swe-agent on llama.cpp (A), mini-swe-agent on pie (B), and a pie-native co-designed agent (D).

## Plan
1. B: mini-swe-agent, 16 agents at once, on pie (pie-swe.toml: 4,864-page KV pool, 9.5 GiB, plus 36 state slots).
2. A: mini-swe-agent, 16 agents at once, on llama.cpp (a slot per agent, 32k tokens each, 8-bit KV, 16 GiB).
3. D: pie-native agent: per task, parallel attempts forked from a shared prompt; first submitted patch wins, the rest are killed.
4. Official scoring of every run; report (md + html).

## Log
- 00:34 Fixed: client timeout 600 s caused retries and duplicate agents (now 3600 s, no retries). pie's auto-sized KV pool was only 1,024 pages (2 GB) for 16 agents; now 4,864. Earlier pie-16 runs kept as runs/pie-16-starved and runs/pie-16-small-pool. Started B.
- 00:46 B failed at once: GPU out of memory in the first 16-agent batch (pool too close to the limit; batch buffers not counted at boot). Pool now 3,072 pages (6 GiB). Restarted B.
- 00:48 B failed again with GPU out of memory at 3,072 pages, though system memory was 88% free. Switched to pie's automatic pool sizing with 36 state slots (was 64). Restarted B.
- 00:52 Auto pool is fixed at 1,024 pages: starvation again (13 of 16 parked, p90 step 45 s). Bigger pools OOM because batch scratch grows with lanes x context. Now 3,072 pages with batches capped at 8 agents. Restarted B.
- 00:57 3,072 pages OOMs even at 8-wide batches; 2,048 survives a 16-agent load test. Restarted B at 2,048 pages, 16-wide.
- 01:02 B running at 2,048 pages without OOM; median step 19 s, p90 66 s (9 agents queued). pie has only a bf16-KV build of this model (64 KiB/token) vs llama.cpp q8 (32 KiB/token): memory per token is pie's handicap here.
- 01:02 B finished: {"engine": "pie", "agents": 16, "wall_s": 307}. Started A (llama.cpp, 16).
- Plan update: at a 40-step limit contexts stay under ~25k tokens, so 16 agents fit in llama.cpp's q8 KV and it never re-reads; A and B should be close. Added a long-horizon pair (E): 120-step limit, contexts past 50k. llama.cpp then fits only 8 slots of 64k (8-bit) for 16 agents, so agents share slots and re-read; pie parks on SSD instead. Scripts: run_long.sh.
- 02:44 Correction: the 'B finished (307 s)' line above was stale, from the run that failed on out-of-memory; A was stopped before it ran. B (2,048 pages) is still running.
- 04:26 Night plan reset: 16 agents plus 16 containers OOMs the GPU mid-run (B hung 3 h after an OOM at 01:03). Now 8 agents at once on both engines, equal memory (~24 GB each), stall guard on.
- 04:26 Night plan reset: 16 agents plus 16 containers OOMs the GPU mid-run (B hung 3 h after an OOM at 01:03). Now 8 agents at once on both engines, equal memory (~24 GB each), stall guard on.
- 04:58 Stall guard: no progress for 15 min in './run.sh pie 8'; stopped it.
- 04:58 B8 done: 
- 07:15 A8 done: {"engine": "llama", "agents": 8, "wall_s": 8189}
- 07:15 B8 stalled at 04:43: 6 tasks started after the first batch were never admitted. The proxy never closes finished conversations, and pie's planner holds new processes while older ones are preempted. Fix: the proxy keeps at most agents + 2 sessions, closing the oldest idle one. Rerunning B8.
- 07:15 A8 (llama.cpp) stopped at 2 h 16 min with 13 of 16 tasks finished; its RAM prompt cache kept evicting 5 GB agent states and re-reading. Per finished task: median 926 s, 19 steps. pie's stalled B8 before the stall: median 452 s per task, 29 steps.
- 07:49 B8 done: {"engine": "pie", "agents": 8, "wall_s": 2041}
- 07:49 Scored B8 and A8.
- 07:50 Scoring failed: swebench 5.0.2 needs the SWE-bench/SWE-bench_Verified dataset (with image/eval_script fields). Stopped E8 to score B8 and A8 first.
- 07:56 SCORED (official harness): B8 pie solved 12/16 in 34 min (2,041 s, all 16 finished). A8 llama.cpp solved 12/16 but needed 2 h 16 min and still had 3 unfinished. Same quality; pie at least 4x faster.
- 08:03 Started D: pie-code with branch and prune, 16 tasks, 8 at a time x 2 attempts (16 agents), pie-bp.toml.
- 09:10 Stall guard stopped D (no progress for 15 min).
- 09:10 D done: {"engine": "pie-code-bp", "agents": 16, "tasks_at_once": 8, "attempts": 2, "wall_s": 3993}; OOM errors: 0. Scoring D.
- 09:14 D scored: Instances submitted: 14 Instances resolved: 12 
- 09:14 D scored 12/16 in 67 min (incl. a 15-min stall). Report written: REPORT.md / REPORT.html.
