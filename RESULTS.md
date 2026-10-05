# Results

## Agent army: AIME 2025, Qwen3.8-27B, one MacBook

| Engine · method | Solved | Total time | Mean per problem | Tokens written | Agents per problem |
|---|---|---|---|---|---|
| pie · single | **7 / 15** | 41 min | 162 s | 67,003 | 1.0 |
| pie · vote | **8 / 15** | 95 min | 378 s | 564,116 | 8.0 |
| pie · team | **8 / 15** | 170 min | 680 s | 430,158 | 14.2 |
| llama · single | **6 / 7** | 50 min | 427 s | 26,949 | 1.0 |
| llama · team | **7 / 8** | 219 min | 1644 s | 163,783 | 15.5 |

### The same team on both engines (problems 1–8, both finished)

Same prompts, same plan, same model. **pie is 2.7x faster per problem (median; 1.5x–12.1x)**, 3.7x in total: 60 min vs 219 min. Across all agents pie writes 42 tokens/s, llama.cpp 12. Solved: pie 6/8, llama.cpp 7/8 (sampling noise at this size).

| # | answer | pie | llama.cpp | llama.cpp / pie |
|---|---|---|---|---|
| 1 | 70 | ✓ 102 s | ✓ 209 s | 2.0x |
| 2 | 588 | ✗ 923 s | ✓ 2992 s | 3.2x |
| 3 | 16 | ✓ 291 s | ✓ 1004 s | 3.4x |
| 4 | 117 | ✓ 253 s | ✓ 377 s | 1.5x |
| 5 | 279 | ✓ 398 s | ✓ 970 s | 2.4x |
| 6 | 504 | ✓ 131 s | ✓ 308 s | 2.4x |
| 7 | 821 | ✗ 1146 s | ✗ 3302 s | 2.9x |
| 8 | 77 | ✓ 329 s | ✓ 3991 s | 12.1x |

## LatentMAS on pie (agents that think in hidden states and share one KV working memory)

Engine check, Qwen3-4B bf16: pie's latent vectors vs Hugging Face, cosine by step: 0.9998, 0.9991, 0.9980, 0.8112 … (rounding drift after, as HF shows against itself).

| Model | Task | Method | pie accuracy | pie time | authors' code (HF) accuracy |
|---|---|---|---|---|---|
| Qwen3.5-4B | arc_challenge | single | 73% | 272 s |  |
| Qwen3.5-4B | gsm8k | latent10 | 46% | 397 s |  |
| Qwen3.5-4B | gsm8k | single | 67% | 415 s |  |
