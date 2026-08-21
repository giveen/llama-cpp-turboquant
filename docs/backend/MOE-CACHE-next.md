# MoE cache roadmap: next work after heat-protected eviction

Status: heat-protected eviction is the only shipped change. Phase A (per-layer
reservation) was implemented then dropped after measurement. Phase B
(predictive prefetch) is the next planned phase, design-only.
Branch context: feature/moe-cache-expert-heat.

## Shipped: heat-protected eviction

On by default via GGML_CUDA_MOE_CACHE_HOT_USES (default 4). A slot whose
resident hit count exceeds the threshold is "hot" and is only evicted when every
eligible slot is hot. Measured on the 5090 (hot0 vs hot4, same configs):

| metric                     | soft hot0 -> hot4 | auto hot0 -> hot4 |
| -------------------------- | ----------------- | ----------------- |
| TG (t/s)                   | 20.38 -> 22.86    | 32.60 -> 35.04    |
| hit rate                   | 38.6 -> 47.0%     | 78.7 -> 81.0%     |
| evictions                  | -15%              | -13%              |
| fills enqueued             | -15%              | -12%              |
| hot experts sacrificed     | 11 -> 0           | 70 -> 0           |

This is the change to keep and ship.

## Dropped: Phase A per-layer top-K reservation

A strict per-layer reservation (pin top-K experts per layer so they are never
evicted) was implemented behind a knob, then measured. Because heat-protection
already yields `heat=0` (hot experts never sacrificed), the reservation added
nothing: reserve=0 vs reserve=4 was within noise and slightly negative on soft
(e.g. soft hot4 22.86 -> 22.26; auto 35.04 -> 35.51). The gate said drop it, so
it was reverted. Do not revive it without a scenario where a layer really loses
its hot experts to another layer's crowding.

## Next: Phase B predictive / layer-ahead prefetch

### Goal
Reduce the cold-start admission cost, the real warmup bottleneck on large MoE
(e.g. GLM-5.2's ~209 GiB expert set needs a long warmup). Prefetch the predicted
next-layer experts into VRAM before they are routed, so cold misses become
hidden warm fills.

### Design
1. Observe routing.
   The plan path already sees the routed expert ids per layer. Accumulate
   per-layer frequency counts and cross-layer transition counts
   (expert[L] -> expert[L+1]).
2. Predict.
   After layer L is routed, predict layer L+1's top candidates from the
   transition model, plus per-layer popularity. Submit fill jobs for the
   predicted experts on the existing low-priority background fill stream,
   before the next node plans.
3. Bound the waste.
   Wrong predictions waste PCIe and can displace useful entries. Cap prefetch
   volume and rate (GGML_CUDA_MOE_CACHE_PREFETCH_MAX_JOBS / _MB), and skip
   prefetch when demand-fill traffic is already high so prefetch never starves
   real demand.
4. Admission interaction.
   Prefetched experts should not burn the admission/readmit counters. They fill
   as candidates; demand still drives admission as today.
5. Config knobs.
   GGML_CUDA_MOE_CACHE_PREFETCH (default 0) to enable; plus the depth/volume
   caps above.
6. Stats.
   Count prefetch hits (predicted and later used) and prefetch waste
   (predicted, filled, then evicted before use), reported in teardown.

### Measurement gate
Cold-ramp TG (first N tokens after an empty/pool-reset start) with prefetch on
vs off; cold-phase hit rate; and prefetch hit/waste counters. This is the
higher-value phase but higher risk, so benchmark cold-start behavior
specifically rather than steady-state TG.

### Sequencing
Phase B is a design-and-measure spike: prove the predictor's cold-start benefit
on one model and one config before generalizing. Keep it behind its own env knob
so it is reversible and measurable in isolation from the shipped heat behavior.

## Saturated-pool sweep (answers review Q1/Q2 on hot_uses)

Context: the shipped hot=4 default was measured in the healthy regime (pool never
saturates, heat=0). Review asked: what happens under sustained heat > 0 (smaller
pool), and does the threshold want to scale with pool size. Ran the exact base
server command (Laguna UD-Q5_K_M, fit on, 1x5090) in fixed mode
`--moe-cache N` so the pool could be forced small. Note: at default verbosity
INFO is hidden in this fork (LLAMA_LOG_LEVEL_INFO maps to verbosity 4), so the
[ moe-cache] stats lines require `-lv 4`; earlier green runs without it show no
cache lines even when the cache engaged. 600 generated tokens per arm, one or two
repeats; top-p 0.95 sampling makes hit rate and TG noisy (+-3pp / +-0.5 tps).

| budget MiB (slots) | hot_uses | hit rate | TG t/s | heat (hot sacrificed) | evictions |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 (198) | 0 | 3.1% | 13.15 +- 0.22 | 135 | 183879 |
| 512 (198) | 4 | 3.7% | 13.34 +- 0.03 | 4 | 182565 |
| 512 (198) | 8 | 2.5% | 13.32 +- 0.02 | 0 | 185095 |
| 2048 (895) | 0 | 18.3% | 15.66 +- 0.06 | 112 | 152934 |
| 2048 (895) | 4 | 25.6% | 16.82 +- 0.28 | 4 | 138176 |
| 2048 (895) | 8 | 28.0% | 17.98 +- 0.05 | 0 | 133483 |

Method: llama-bench (not server + curl), which is faster and better suited: one
process runs both budgets via `--moe-cache 512,2048`, `--n-gen-warmup 256`
populates the pool before timing, `-r 3` gives variance, teardown stats print per
arm, and the pool geometry matches the server runs exactly (198 and 895 slots;
q5_K 532 + q6_K 289 + q8_0 74 at 2048 MiB). Requires `-v` (llama-bench has no
`-lv`; without `-v` llama logs are nulled). Model load ~25 s, whole 3-arm sweep
~8 min. One caveat: `-o json` and the INFO logs both go to stdout, so the JSON
array has log lines glued inside it; filter lines to `{`/`}`/`"`/`,` starters and
strip trailing log text after the closing `}`.

First attempt used llama-server + curl with 600 generated tokens per arm; those
numbers (hot4 b512 3.1% / b2048 18.8%) agree with the bench run within sampling
noise. Server runs additionally showed the same monotonic ladder at b2048
(18.8 -> 23.7% hits, 16.1 -> 17.1 t/s for hot4 -> hot8).

Findings:

1. No inversion under saturation. Even at 2.5-3.7% hit rate with the pool
   100% occupied and thrashing (admission ~1.4M against ~180k evictions), hot
   protection still does its job: heat=135 at hot0 collapses to 4 at hot4 and 0
   at hot8, with TG flat within noise (13.15 vs 13.34 vs 13.32). The recency
   fallback covers the all-hot case; the scan cost is negligible.
2. The threshold does move with pool size, and at the bigger pool the effect
   is large and monotone. At 512 MiB (198 slots, the smaller pool) hot4 is
   preferred over hot8 (hit rate 3.7% vs 2.5%: stale pins displace fresh
   demand). At 2048 MiB (895 slots) hot8 is clearly best: 28.0% hits and
   17.98 t/s vs 18.3% and 15.66 at hot0, +15% TG, zero sacrifices, and fewer
   evictions and admissions (better retention). Stronger pinning pays off
   exactly when the pool is large enough to retain a real working set, and
   hurts when the pool is tiny and any stale pin wastes most of it.
3. Keep default 4 but the doorway is open.

Flat 4 is right for the healthy default (auto/soft on big pools never saturate,
so the knob is mostly inactive there), and its protecting action is lossless in
the saturated regime. But the 2048 MiB line is evidence that values above 4 want
to be reached when the pool is big; if GGML_CUDA_MOE_CACHE_HOT_USES should
follow pool size, a simple scale like hot_uses = clamp(4, slots/128, 8) would
track both measurements and costs nothing when unset. Before building that,
repeat both lines with more repeats and a fixed seed to separate signal from
sampling noise.

## Cross-model check: Qwen3-Coder-30B-A3B-UD-Q4_K_XL (2026-08-17)

Second architecture to rule out Laguna-only behavior. 17.6 GB model, 48 layers,
128 experts x 8 used, q4_K experts (0.86 MiB each). Model fully fits the 32 GB
GPU, so the moe-cache fit refuses by design ("kept stock placement because the
complete model already meets the fit targets") and the cache cannot engage with
experts on GPU. Forced the intended regime with llama-bench
`-ngl 99 -ot 'blk.*ffn_.*exps=CPU'` (no -fitt: the fit path wipes -ot overrides
and -ncmoe is a no-op, it is not consumed anywhere in src/). Same sweep:
hot 0/4/8 x budgets 512/2048, -n 300 -r 3 --n-gen-warmup 256, plus PP control.

TG t/s (mean +- sd over 3 reps):

| budget (slots) | hot0 | hot4 | hot8 |
| --- | ---: | ---: | ---: |
| 512 (580) | 41.14 +- 0.89 | 44.68 +- 0.50 | 44.12 +- 1.32 |
| 2048 (2365) | 61.84 +- 2.35 | 62.64 +- 0.40 | 67.70 +- 1.76 |

Cache stats (session-wide): hits, heat, evictions, admission:

| budget | hot0 | hot4 | hot8 |
| --- | ---: | ---: | ---: |
| 512 | 23.1%, heat=122, ev=121550, adm=900120 | 27.6%, heat=0, ev=113957, adm=847001 | 30.9%, heat=0, ev=108663, adm=809300 |
| 2048 | 66.5%, heat=110, ev=49587, adm=391286 | 69.7%, heat=0, ev=44388, adm=353884 | 70.5%, heat=0, ev=43069, adm=344946 |

PP control (cache bypasses prompt nodes): flat across hot values, e.g. pp512
578-593 t/s, pp2048 ~1960-2036 t/s for all arms, within noise.

Cross-model findings:

1. Same as Laguna: no inversion under saturation. Both budgets run 100% full
   (580/580, 2365/2365) and heat collapses 122/110 at hot0 to 0 at hot4/hot8.
2. Same direction, larger effect at the small pool. Qwen3 routing is more
   concentrated than Laguna's, so protection pays even at 512 MiB: hot4 beats
   hot0 by +8.6% TG (44.68 vs 41.14) and +4.5pp hits. hot8 equals hot4 at
   512 MiB, and wins at 2048 MiB (+8% over hot4, +9.5% over hot0).
3. The 2048 MiB hot8 advantage repeats on both models: bigger pool + stronger
   pinning = better working-set retention (fewer evictions and admissions at
   hot8 on both models).
