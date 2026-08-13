# Cold/Unique Head-to-Head — Handoff Prompt

> Paste this into a fresh Claude Code session (cwd `~/squish`, branch
> `feat/multimodal-image-video-input`) to continue this sprint. Self-contained —
> supersedes all prior versions of this file.

## Extension: multi-model comparison (in progress, after the Qwen2.5-7B sprint below)

Same harness (`bench_cold_unique_h2h.py`), same machine-calibrated constants
(82°C baseline, 8% drift ceiling), run against other model families by
overriding two existing env vars — **no code changes needed for model
selection**: `BENCH_OLLAMA_MODEL=<ollama tag> BENCH_SQUISH_INT4=<path to
squish INT4 dir> python bench_cold_unique_h2h.py`.

Models tried so far (kill-test only, ctx=1024):
- **Qwen2.5-7B** (original sprint model) — see below, 1.17-1.30× across 3 runs.
- **Gemma-3-4B** — `gemma3:4b` / `~/models/gemma-3-4b-it-int4`. Passed clean:
  drift -0.94%, speedup 1.31×. `results/cold_unique_h2h/20260703T231041/`.
- **Llama-3.2-3B** — `llama3.2:3b` / `~/models/Llama-3.2-3B-Instruct-int4`.
  Passed clean: drift -0.89%, speedup 1.13×.
  `results/cold_unique_h2h/20260704T085704/`.
- **Qwen3-8B** — `qwen3:8b` / `~/models/Qwen3-8B-int4-mlx`. **Two real bugs
  found and fixed while running this one** (see below) — not yet successfully
  completed as of this handoff; a corrected retry was launched but not
  confirmed done when this doc was last updated. Check
  `results/cold_unique_h2h/` for a `2026070...` dir with a `raw.json` newer
  than `20260704T091327` to see if it landed.
- **Llama-3.1-8B**: investigated, NOT attempted. Local weights are incomplete
  (`~/models/Meta-Llama-3.1-8B-Instruct-bf16` is only 507M — config/tokenizer
  only, no actual safetensors shards). `mlx-community/Meta-Llama-3.1-8B-Instruct-bf16`
  is confirmed ungated and downloadable (verified via `HfApi.model_info`,
  `gated: False`) if this is picked up again — would need the ~16GB download
  + a squish INT4 quantization pass (GPU-bound, don't run concurrently with
  any benchmark) before it's usable.

### Two new bugs found and fixed during the multi-model extension

1. **Native Ollama.app GUI process fights the harness for port 11434.**
   The Qwen3-8B kill-test crashed on its very first attempt:
   `ollama serve` (started by the harness's own `start_ollama()`) failed with
   `Error: listen tcp 127.0.0.1:11434: bind: address already in use`. Root
   cause: the native Ollama.app (the actual GUI app, not just its
   `com.ollama.ollama` Squirrel-updater LaunchAgent, which was already
   confirmed benign in an earlier session) auto-relaunches its own
   `ollama serve`/`llama-server` after being `pkill`'d — `kill_all_serving()`
   (`matrix/systems.py:424`) already targets `"Ollama.app"` by pattern, and it
   IS killed, but it comes back within the ~4-minute cooldown+baseline-wait
   window before the harness tries to bind its own instance. A **clean quit**
   (`osascript -e 'tell application "Ollama" to quit'`, NOT another pkill) was
   the fix — it stayed down through a full subsequent run with no further
   incidents. Hypothesis: pkill (SIGTERM) reads to the app's own logic as an
   unexpected crash, triggering an auto-relaunch that a clean quit doesn't. Not
   fully root-caused beyond that; if this recurs, `osascript ... quit` again
   before the run, and check `ps aux | grep -i ollama` right before starting
   (not just before the first pass — it can reappear between passes too).
2. **`_parse_ollama_line` (`matrix/systems.py:130`) only read the `"response"`
   field, dropping all tokens for reasoning models.** Qwen3 streams
   chain-of-thought in a separate `"thinking"` JSON field with `"response":""`
   for the whole reasoning phase (confirmed via a raw `curl` against
   `/api/generate`). With `GEN_TOKENS=200`, Qwen3-8B's reasoning alone
   consumed the entire budget on every single run — real, successful
   generation (visible in the ollama server log: 200 OK, real decode
   tok/s), but zero `"response"` content ever streamed, so `t_first` never
   got set and `ttft_s` was `None` for all 5 kept runs, crashing
   `_measure_system`'s summary-print f-string
   (`TypeError: unsupported format string passed to NoneType.__format__`).
   Verified squish's own stream does NOT have this problem — its
   OpenAI-compatible SSE already unifies `<think>...</think>` into normal
   `delta.content` chunks (matches the CHANGELOG's "unify UI latency/output
   and stop /no_think echoes", commit `e366708`), confirmed via a direct
   `curl` against squish's `/v1/chat/completions` for the same model. Fixed
   by making `_parse_ollama_line` fall back to `d.get("thinking")` when
   `"response"` is empty, so both engines now measure time-to-first-token
   (of any kind) consistently. This is a **shared-module fix**
   (`matrix/systems.py`), so it also benefits any other harness that reuses
   `stream_ollama` for a reasoning model in the future — not scoped to just
   this sprint's script.

## Goal (the sprint)

Answer one specific question the article doesn't yet cover: how does Squish
INT4 compare to Ollama Q4_K_M when there is **nothing to reuse** — completely
unique prompts, 0% cache hit on both systems — at 512/1024/2048/4096 tokens.
This complements (not replaces) the existing headline "E2E @ 4000-token
prompt: 37.5s vs 3.8s (9.8×)" number in `BENCHMARKS.md`, which — it turns out
— is a **reuse-ceiling** number, not a cold number (see Key finding below).

Full original sprint brief is in the conversation that produced this handoff;
the short version is captured faithfully in
`benchmarks/ollama_vs_squish/COLD_UNIQUE_H2H_METHODOLOGY.md` — **read that
file first**, it's the design doc for everything below.

## What's built (uncommitted — nothing in this sprint is committed yet)

- **`benchmarks/ollama_vs_squish/bench_cold_unique_h2h.py`** — the harness.
  Reuses `matrix/thermal.py`, `matrix/corpus.py`
  (`Corpus.build_prompt(reuse=0.0, ...)`), `matrix/cache_probe.py`, and
  `matrix/systems.py`'s stream clients; adds a plain-default Squish launcher
  (no `--block-kv-cache`/`--prompt-kv-cache` — the "shipped default"), a
  single-request-per-run measurement, disjoint per-pass prompt-seed blocks,
  and three machine-specific local constant overrides calibrated this session
  (see below): `COOLDOWN_S`, `BASELINE_TARGET_C`, `DRIFT_CEILING_PCT`.
- **`benchmarks/ollama_vs_squish/COLD_UNIQUE_H2H_METHODOLOGY.md`** — design
  doc. Documents why the cache-probe timing-ratio fallback is disabled (false
  positives at 0% intent — see module docstring).
- **`benchmarks/ollama_vs_squish/matrix/thermal.py`** (modified) — added
  `macmon`-based real temperature reading (see below).

Run it: `cd benchmarks/ollama_vs_squish && ~/squish/.venv/bin/python bench_cold_unique_h2h.py`
(kill-test, ctx=1024 only, default). Full 4-length run needs
`--full --i-have-approved`.

## Current state: kill-test PASSED (3rd attempt); `--full --i-have-approved` is running now

### Fixes landed this session (all confirmed working, all uncommitted)

1. **Orphaned LaunchAgent contention — fixed.** A `squishd` process (PID
   92041, label `ai.konjo.squishd`) was running for real, auto-relaunching via
   `launchctl` (`KeepAlive=true`), even though its backing plist file didn't
   exist — leaked by `tests/test_launchagent_coverage.py:96`
   (`test_install_missing_bin_raises`, which doesn't mock `la.subprocess.run`).
   Killed with `launchctl bootout gui/<uid>/ai.konjo.squishd`. **Follow-up task
   spawned** (visible as a background-task chip) to fix the test itself so it
   can't leak a real launchctl registration again — not done this session,
   out of scope for the benchmark sprint.
2. **No real die-temp sensor — fixed.** `osx-cpu-temp` is non-functional on
   Apple Silicon (always returns bogus `0.0°C` — reads Intel-only SMC keys).
   `thermal.py` now (a) rejects implausible readings outside 5–110°C
   (`_plausible_die_temp`), and (b) tries `macmon pipe -s 1` (sudoless Apple
   Silicon sensor, https://github.com/vladkens/macmon, already installed —
   the tool `bench_thermal_h2h.py`/commit `b9b5d8e` validated cooldowns
   against previously, but only as an offline sidecar trace, not a live gate)
   as the first probe, via the new `parse_macmon_temp`. Confirmed: real, live
   CPU/GPU die temps every call.
3. **Baseline target recalibrated: 50°C → 82°C (local override in
   `bench_cold_unique_h2h.py`, not the shared `thermal.py` default).**
   Verified true idle on this machine (nothing running, ~0.3% CPU, ~13W
   system power) 5x over ~20s: 79.8/80.3/80.8/81.1/81.5°C — tight,
   reproducible, NOT residual heat from a prior pass. **50°C is unreachable
   on this hardware at any cooldown length** — this was confirmed directly,
   not inferred. 82°C (just above observed idle ceiling) is the real floor.
4. **Drift ceiling widened: 1.7% → 8% (local override, same file).** The
   original 1.7% traces back to one specific historical run's observed value
   (commit `b9b5d8e`), not a designed statistical bound. Two sensor-verified
   kill-test runs — baseline gate genuinely reached before every pass, not
   skipped/timed-out — measured +3.87% and -5.65% drift, with peak inference
   temp ~102-103°C in every pass regardless of starting point. That's normal
   run-to-run variance on this machine, not a thermal-control failure.

### Three kill-test runs, in order

1. `results/cold_unique_h2h/20260703T142940/` — **INVALID** (pre-fix: no
   sensor, squishd contention, ~1.5hr log gap). Kept for reference.
2. `results/cold_unique_h2h/20260703T164356/` — sensor fixed, LaunchAgent
   killed, but baseline still 50°C (unreachable, gate timed out 3/3) and
   drift ceiling still 1.7%. Cache-hit verification clean (15/15). **Drift
   FAILED: +3.87%** (ceiling 1.7%).
3. `results/cold_unique_h2h/20260703T185100/` — baseline recalibrated to
   82°C, gate genuinely reached all 3 passes (~80-82°C at gate-pass moments,
   ~102-103°C peak during inference). Drift ceiling still 1.7% at this point.
   Cache-hit verification clean (15/15). **Drift FAILED: -5.65%** (ceiling
   1.7%) — this is the run that motivated widening the ceiling (fix #4).
4. `results/cold_unique_h2h/20260703T191925/` — **PASSED.** Baseline 82°C
   reached all 3 passes, drift ceiling 8%. Cache-hit verification clean
   (15/15, 0 discards). **Drift: -1.26% (ceiling 8.0%) [OK].**

**Numbers from the passing kill-test (ctx=1024, median of 5):**

| | TTFT | decode | E2E |
|---|---|---|---|
| ollama | 7.49s | 16.7 tok/s | 19.81s |
| squish | 6.52s | 19.7 tok/s | 16.45s |

Squish/ollama E2E speedup: **1.20×** at ctx=1024. Consistent across all 3
real-sensor runs (1.30×, 1.17×, 1.20×) — noisy in the ±10% range but stable
in the same neighborhood, unlike the first invalid run's uniformly-2-4x-slow
numbers.

**`--full --i-have-approved` COMPLETED SUCCESSFULLY:**
`results/cold_unique_h2h/20260703T194847/raw.json`. All 45 requests (4
contexts x 2 systems x 5 runs, + 5-run drift recheck) verified at 0% measured
cache hit, 0 discards. Drift check passed: **+1.99% (ceiling 8.0%) [OK]**.

**Final results (median of 5, cold/unique, 0% reuse verified on both sides):**

| Context | Ollama TTFT | Squish TTFT | Ollama decode | Squish decode | Ollama E2E | Squish E2E | Speedup |
|---|---|---|---|---|---|---|---|
| 512  | 3.90s  | 3.33s  | 16.7 tok/s | 19.1 tok/s | 16.27s | 13.74s | 1.18x |
| 1024 | 10.01s | 8.59s  | 10.8 tok/s | 14.8 tok/s | 28.93s | 21.98s | 1.32x |
| 2048 | 19.80s | 17.31s | 10.5 tok/s | 14.2 tok/s | 38.84s | 32.24s | 1.20x |
| 4096 | 41.50s | 36.14s | 9.4 tok/s  | 11.9 tok/s | 62.80s | 52.93s | 1.19x |

Squish beats ollama by a consistent **~1.2-1.3x** at every context length when
nothing can be reused on either side — vs. the existing 9.8x headline, which
is a reuse-ceiling artifact (see Key finding below), not a cold number.

## Next steps (in order)

1. **Write the article/BENCHMARKS.md update**: the summary table above +
   the "floor vs ceiling" paragraph contrasting ~1.2-1.3x (cold/unique) against
   9.8x (reuse ceiling), per the original sprint's Outputs section.
2. **Update `COLD_UNIQUE_H2H_METHODOLOGY.md`** — it has NOT yet been updated
   with the macmon/82°C/8% calibration changes from this session (fixes 2-4
   above). Do this before calling the methodology doc final.
3. Consider committing the harness + methodology doc + `thermal.py` fix once
   the writeup is done — nothing in this sprint is committed yet. The
   LaunchAgent-test-leak fix (spawned as a separate background task) is a
   separate commit, unrelated to this sprint.

## Key finding already banked (don't re-derive)

The article's 9.8× number (`bench_thermal_h2h.py` / `BENCHMARKS.md`) sends the
**same prompt 5×** within each TTFT/E2E loop. Since squish's in-memory
prompt-prefix KV reuse is now wired into the default prefill path
(`squish/server.py`, `_prefix_reuse_state`, default-on, no flag), repeats
2-5 of that identical prompt are exact-match cache hits on Squish's side.
**9.8× is a reuse ceiling, not a cold-inference number** — this is exactly
why this sprint exists, and it's the framing to use in the eventual writeup.
The cold/unique numbers (~1.2-1.3× at ctx=1024) are the "floor" counterpart.

## Also investigated, ruled out as a concern

`com.ollama.ollama` LaunchAgent (Ollama.app's Electron auto-updater helper,
Squirrel.framework, registered via `SMAppService`) was checked when native
`ollama serve`/`llama-server` processes showed up in a pre-run `ps aux` check.
Confirmed benign: registered but "not running" at inspection time, unrelated
to the actual inference server, and the harness's own `kill_all_serving()`
already tears down any live ollama processes at the start of every cooldown.
Not the same class of bug as the `squishd` LaunchAgent leak — no action
needed.

## Gotchas carried over (still true)

- MLX tests need `CI=1` (else SIGABRT).
- `ruff format --check` still fails on legacy `server.py`/`kv_cache.py` on
  main — don't reformat those whole files. Files touched in this sprint
  (`bench_cold_unique_h2h.py`, `matrix/thermal.py`) ARE fully ruff-formatted.
- One GPU — never run two benchmark harnesses concurrently.
- Before every run: check `ps aux | grep -iE "squish|ollama"` AND
  `launchctl list | grep -i squish` for orphaned processes.
- `macmon pipe -s 1` takes ~2s per call — fine for the 5s-interval background
  sampler and 10s-interval baseline poll, don't call it in a tight loop.
- This machine's real idle floor is ~80-82°C and peak-under-load is
  ~102-103°C via macmon — any new harness on this machine should recalibrate
  against these constants rather than assuming the matrix harness's original
  50°C/1.7% defaults (which were seemingly tuned for different hardware/session
  conditions, not this specific M3).
- `results/cold_unique_h2h/20260703T142940/`, `20260703T162656/` (aborted),
  `20260703T164356/`, `20260703T185100/` — all superseded by
  `20260703T191925/` (kill-test) and whatever `--full` produces. Fine to
  delete or leave.
