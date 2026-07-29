# squish

Local LLM inference server — MLX-accelerated on Apple Silicon, with speculative decoding, quantization (INT4/INT3/SQINT2), agent tool execution, Ollama/OpenAI-compatible API, and the macOS SquishBar.

**v9.34.15**

## Org rules

@~/.konjo/kiban/plugins/konjo/skills/konjo/SKILL.md

The org ethos applies here: ship over optimize, kill-test first, statistical rigor,
honest negative results, evidence first, token-efficient context.

Editorial rules: no em dashes, no AI-tell vocabulary. The prose lint enforces it; run
`konjo-prose` on docs before pushing.

Log durable decisions with `konjo-decision decide` at `repo:squish` scope. Search with
`konjo-decision search` before reopening a settled call.

When you catch a mistake worth not repeating, invoke `correct`: it records a learning
with `konjo-learn` and proposes the smallest durable fix. A learning must name where
its rule lives (a CLAUDE.md line, a prose-lint word, a lane, or a gate), or it is
refused.

Build the Konjo way: the `craft` skill carries the four behaviors (think before coding,
simplicity first, surgical changes, goal-driven execution) plus the verify-loop and the
pre-implementation trust-boundary contract. `verify_cmd` is declared in
`.konjo/profile.yml`.

## Stack
Python 3.10+ · MLX + mlx-lm (Apple Silicon) · FastAPI · transformers · HuggingFace Hub · Swift (macOS SquishBar)

## Commands
```bash
python -m pytest tests/ -x                   # full test suite
python -m pytest tests/ -x -k "test_name"    # run a single test
python -m squish serve                        # start inference server
squish pull hf:<repo>                         # download + pre-scan HF model
squish trace                                  # observability report
squish compat                                 # backend compatibility check
```

## Invariants
- No `unwrap()`/`expect()` in Python and no silent failures — `repo:pre-commit "silent error swallowing scan"` (blocks the commit on bare/`Exception`-wide `except`)
- Quantization accuracy gates are hard stops: INT4 AWQ g=32 ≥ 70.6% arc_easy (Qwen2.5-1.5B); INT2 naive is **NEVER SHIP** — `repo:model_pipeline.yml` "Compress and validate — accuracy gate"
- MLX imports must be gated behind platform check — never imported on Linux paths — ADVISORY
- `squish.squash` is an **optional** import — never hard-depend on `squash-ai` — ADVISORY
- Pre-scan HF models **before** loading weights — `HFFileSummary` scan runs at `squish pull hf:` time — ADVISORY
- Prompt injection: system prompt content must never be controllable by request payload — ADVISORY (only checked by Wall 3, which is disabled in CI)
- Never log raw user prompt content at INFO level or above — log a hash or truncated prefix — ADVISORY (same reason)
- Version bumps touch `pyproject.toml` + `squish/__init__.py` — ADVISORY

## Repo map
| Module | Role |
|--------|------|
| `squish/server.py` | FastAPI app entry point, startup profiler, backend routing |
| `squish/cli.py` | `squish` CLI — serve, pull, trace, compat, agent |
| `squish/catalog.py` | Model registry: URI parsing (`ollama:` / `hf:`) + HF batch upload |
| `squish/serving/` | Backend router, Ollama/LocalAI compat, blazing TTFT, tool calling |
| `squish/hardware/` | Platform detector, production profiler, Apple Silicon routing |
| `squish/api/` | OpenAI-compatible v1 router |
| `squish/agent/` | Agent loop, tool name map, tool execution |
| `squish/quant/` | AWQ/INT3/INT4/SQINT2 quantization pipeline |
| `squish/kv/` | KV cache management |
| `squish/context/` | Context window management |
| `squish/platform/` | Cross-platform router and detector |
| `apps/macos/SquishBar/` | Swift macOS menu bar app (model picker, progress, hotkey) |

## Repo-specific rules

### Planning Docs
- `MODULES.md` — per-wave module reference (Waves 1–99+)
- `CHANGELOG.md` — all notable changes

### Konjo Quality Framework

**Wall 1 — Pre-commit** (`bash .konjo/scripts/install-hooks.sh`):
ruff lint, ruff format, bare-except scan, DRY check, TODO scan. Blocks the commit.

**Wall 2 — CI gate** (`.github/workflows/konjo-gate.yml`, `.github/workflows/ci.yml`):
- `ruff check` blocks for real on every job that runs it (0 standing violations).
- `ruff format`, `vulture`, `bandit`, `radon` complexity, `dry_check.py`, `interrogate`
  docstrings, and `mypy` are **ratcheted**: each blocks only on regression above (or, for
  docstrings, below) its measured baseline, not the full pre-existing backlog. Baselines
  live in `.konjo/*-ceiling.txt` / `*-floor.txt`; see `.konjo/scripts/ratchet_check.py`.
- File size ≤ 500L blocks for **new** files; legacy oversized files are grandfathered in
  `.konjo/oversized-allowlist.txt` (split them to remove, don't grow the list).
- Coverage and mutation testing stay soft — both are duplicated by real enforcement
  elsewhere (`ci.yml`'s own macOS coverage job; mutation's own documented CI-timeout
  constraint) — see `LEDGER.md`'s `Squish-Gate-Triage-1` for the full per-step table.

**Wall 2b — kiban `konjo-gates`** (`.github/workflows/konjo-gates.yml`):
Runs kiban's pinned gate orchestrator (`.konjo/kiban.ref`) against `.konjo/profile.yml`
— `gate_polarity` and `gate_claude_contract` (both advisory during this repo's adoption
ramp), plus the same format/lint tools above, net-new-diff scoped. Blocks for real.

**Wall 3 — Adversarial review** (local only — disabled in CI):
`git diff HEAD~1 | python3 .konjo/scripts/konjo_review.py`

See the `konjo-quality` skill (`.claude/skills/konjo-quality/`) for the full specification.

### Skills
See `.claude/skills/` — auto-loaded when relevant.
Run `/konjo` to boot a full session (Brief + Discovery + Plan).

## Pinning

This repo pins a kiban ref in `.konjo/kiban.ref` (currently `v1.9.0`) and `KIBAN_REF` in
`.github/workflows/konjo-gates.yml` — bump both together; a kiban change should not
silently reach the gate.
