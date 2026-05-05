# R2 — CP sampling latency budget

Phase 0, experiment R2 from `../journal/extension_plan.md`.

Microbenchmark only — no training, no real drafter backbone. Measures:

- `t_target`: a single *full* forward pass of the target (Qwen3-8B by default)
  at several seq_lens, no KV cache. Generous (worst-case) denominator.
- `t_target_cached`: a *verification-style* forward — k draft tokens over
  a KV cache of length L. Closer to what spec-decode actually pays per
  cycle. New in v2; grid is `{prefix_lens} × {draft_ks}`.
- `t_CP`: shared-trunk rank-$r$ CP sampling of a block of size $n$ on a
  synthetic vocab-$V$ trunk (mirrors `factorization_gap/fgap/heads.py::SharedTrunkCPHead`
  plus Basharin 2024 §3.4 sequential-within-block sampling with logsumexp
  mixture updates).
- `t_FF`: factorized single-head sampling as a baseline — the deployed
  DFlash drafter's output path.

Reports the full grid `(V, n, r) ↦ t_CP` plus `t_CP / t_target` against
each prefill seq_len and against the reference cached-verification cell
`(L = cached_prefix_target, k = cached_k_target)`, and `t_CP / t_FF` for
CP overhead relative to the factorized baseline. Kill criterion (hard
gate): the prefill ratio at (r=8, n=16, V=152064, L=2048) must be ≤ 0.3.
Cached-verification ratios are reported alongside for context but do not
drive the decision.

## Layout

```
configs/
  latency.yaml           # grid + kill threshold + model choice
r2lat/
  cp.py                  # SharedTrunkCPSampler + FFSampler
  target.py              # load + time target forward
  utils.py               # timing + io helpers
run_latency.py           # entry point
```

## Setup

```bash
cd r2_latency
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .

# The only gated model is the target; log in if your HF account needs it.
huggingface-cli login
```

(If you already have the `factorization_gap` venv active, its dependency set
is a superset of this one — you can skip the fresh venv and just run the
script.)

## Run

```bash
# Smoke test — 1 min on 1×H200, hits both kernels but reduced grid.
python run_latency.py --config configs/latency.yaml --smoke

# Full run — ~10–30 min on 1×H200 depending on grid and model.
python run_latency.py --config configs/latency.yaml

# Reuse earlier target timings (if you're iterating on the CP kernel only).
python run_latency.py --config configs/latency.yaml --skip-target
```

The `--skip-target` path requires `hidden_size_override` in the config so
the CP trunk is sized correctly; the decision line will read `UNAVAILABLE`
because there is no `t_target` to ratio against in that mode.

## Outputs

Each run writes `runs/r2/<run_name>_<ts>/` with:

- `config.json` — resolved config
- `run.log` — loguru log (includes every grid cell)
- `summary.json` — `target`, `target_cached`, `ff`, `cp`, `ratio_grid`,
  `kill_criterion` (including the cached reference ratio)
- `cp_latency.png` — CP time vs rank, one subplot per vocab
- `ratio_grid.png` — `t_CP / t_target` (prefill) across the grid, with
  the kill threshold drawn in
- `ratio_cached.png` — `t_CP / t_target_cached` at the reference cell
  (top row: vs rank at fixed k), plus k-sweep for a few representative
  (n, r) cells (bottom row)

## Kill criterion

| check | threshold | metric |
|---|---|---|
| CP-over-target ratio at (r=8, n=16, V=152064, L=2048) | ≤ 0.3 | `t_CP / t_target` |

The script also reports `t_CP / t_FF` per row so we can see the CP-specific
overhead independent of target cost — useful when the target forward is
cheap enough that the ratio-to-target threshold is dominated by kernel
launch noise.

## Notes / assumptions

- CP sampling *precomputes* the full `(n, r, V)` log-probs once per block
  and then runs `n` small kernels for sequential-mixture updates. This
  trades HBM for fewer launches. A streaming variant (compute logits per
  position) is not benchmarked here — the precomputed variant is what the
  Phase 1 drafter would use.
- Weights are random. We are measuring kernel cost, not quality.
- Multinomial sampling is included in the timed path (real deployment
  overhead).
- Dtype is bf16 for weights + activations; softmax and logsumexp run in
  fp32 for stability at V=152K.
- The target is loaded with `use_cache=False` so the forward cost is a
  full prefill, which is the heaviest per-step target work and a
  defensible denominator.
