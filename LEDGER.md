# Ledger

A running log of load-bearing design decisions — the ones that would be
expensive to silently re-litigate in a later sprint. One entry per sprint,
newest first. Not a changelog (that's `CHANGELOG.md`) — this is *why*, not
*what*.

## Track A2 — squish's first kiban connection: decorative-lint-job fix, gate triage, honest performance range

**KT-A2.1 (required kill-test, run before any triage): `konjo-gates` genuinely runs
against squish's new profile and CAN fail.** Ran the real, pinned engine (installed
from the local kiban checkout, since no `v1.9.0` tag exists yet in this sandboxed
clone — the real CI job resolves the tag once kiban itself cuts and pushes it)
against `.konjo/profile.yml` with `--base origin/main`. First finding, before any
deliberate test: the run **hung** — `mutation: mutmut` in the profile (inherited
verbatim from kiban's own `profiles/squish.yml` starting point) causes
`konjo_gates_py.cli`'s generic net-new dispatcher to shell out to a bare `mutmut run`
with **no timeout wrapper at all** (unlike the `cargo-mutants` path, which gets
`--jobs`/`--timeout` flags). Killed after several minutes with RSS still climbing.
This is the exact trap `profiles/lopi.yml`'s own comment documents avoiding for
`cargo-mutants` — confirmed live here for `mutmut` too. Fixed: `mutation:
"none-with-reason: ..."` (the `"none"`-prefix konjo_gates_py's dispatcher checks for
to skip the tool entirely); real mutation testing stays in `konjo-gate.yml` G3, which
already has the correct 7-minute internal cap. **This is exactly the kind of defect a
kill-test exists to catch before it reaches CI, not after** — recorded here instead of
quietly fixed with no trace.

With that fixed, the deliberate half of KT-A2.1: a scratch file
(`squish/_kt_a21_scratch.py`, never committed) with a genuine, non-globally-ignored
violation (`except Exception:` — BLE001, enforced on `squish/` — plus a bare `1 / 0`
expression, B018) was scanned via `--changed squish/_kt_a21_scratch.py`. Result: **RED**
— `repo:ruff` FAILed on the exact BLE001/B018 text, `repo:vulture` FAILed (unused
function), `repo:bandit` FAILed (B110 try-except-pass). `polarity` correctly PASSed (no
permissive-value shape here) and `threat_model` correctly SKIPped (file not in
`security_globs`). Scratch file deleted immediately after. **The mechanism is live and
correctly discriminating real violations from clean code, not just returning green by
default.**

**Real count and shapes of "soft" CI, vs the brief's stated 11.** `konjo-gate.yml`
carries exactly **10** literal `continue-on-error: true` steps (16 named steps total,
matching the brief's "16"), not 11 — a small but real discrepancy from the brief's own
count, reported rather than silently reconciled to match. Separately, and *not* caught
by any `continue-on-error` grep, `ci.yml`'s `lint-only` job's two steps
(`ruff check ... --exit-zero`, `mypy ... --no-error-summary || true`) plus the same
`--exit-zero` in the `test` and `test-linux` jobs' own `Lint (ruff)` steps — 4 more
structurally-cannot-fail shapes. **Total: 14 decorative quality-gate steps found across
both workflow files, not 11 and not 10** — the brief's "11" undercounts because it was
counting only `konjo-gate.yml`'s literal string, and this sprint's own instruction to
check for all three shapes (`continue-on-error`, `--exit-zero`, `|| true`) is exactly
what surfaced the other 4.

**Every one of the 14 got a real disposition** (promote / keep-soft-with-owner-and-date
/ delete-with-reason; see `.github/workflows/konjo-gate.yml` and `ci.yml`'s own inline
comments for the full per-step reasoning, not duplicated here):

| Step | Baseline measured | Disposition |
|---|---:|---|
| `ci.yml` `lint-only` ruff check | 0 | **PROMOTED** — `--exit-zero` removed, real block |
| `ci.yml` `lint-only` mypy check | 215 errors | **RATCHETED** — `.konjo/mypy-ceiling.txt`, real block on regression |
| `ci.yml` `test` job's `Lint (ruff)` | 0 | **PROMOTED** — `--exit-zero` removed |
| `ci.yml` `test-linux` job's `Lint (ruff)` | 0 | **PROMOTED** — `--exit-zero` removed |
| G1 `ruff lint` | 0 (10 fixed this sprint) | **PROMOTED** — real block |
| G1 `ruff format` | 363 files | **RATCHETED** — `.konjo/ruff-format-ceiling.txt` |
| G1 `vulture` | 108 | **RATCHETED** — `.konjo/vulture-ceiling.txt` |
| G1 `bandit` | 67 (2 real High fixed) | **RATCHETED** — `.konjo/bandit-ceiling.txt`; also fixed a real `--exclude` path-prefix bug that was silently scanning `tests/` too (134 findings measured with the old flag, not the real 67) |
| G2 `Run tests with coverage` + `Coverage gate` | n/a | **KEPT SOFT** — owner: squish maintainers, revisit-by 2026-09-30. Duplicate of `ci.yml`'s real macOS coverage job, broken as configured on `ubuntu-latest` (no mlx install, missing the Metal-unguarded-import ignore list, `--cov=.` instead of `--cov=squish`) — promoting today would red every PR on a false signal, not a real one. |
| G3 `Run mutation testing` | n/a | **KEPT SOFT** — owner: squish maintainers, revisit-by 2026-09-30. Already load-bearing for a documented, sound reason (7-min internal cap dodges the 8-min GH Actions SIGKILL); promoting needs a real mutation-survival threshold this same timeout risk prevents measuring today. |
| G4 `Complexity gate` | 146 | **RATCHETED** — `.konjo/complexity-ceiling.txt` |
| G4 `DRY check` | 99 | **RATCHETED** — `.konjo/dry-ceiling.txt` |
| G4 `Documentation gate` | 31.4% | **RATCHETED** (floor) — `.konjo/docstring-floor.txt` |
| G4 `File size gate` | n/a | Already the one hard-blocking step; no change needed. |

**`gate_polarity` full-tree scan (first connection — see `konjo-retrofit`'s baseline-
before-gating protocol): 17 standing findings, all triaged, none waved through.**
2 real (1 fixed this sprint, 1 flagged and deferred), 10 false positives in `squish/`
production code, 5 in `tests/**`/`benchmarks/**` (added to `polarity.exempt_globs`).
Given the brief's explicit instruction to scrutinize fail-open shapes extra carefully
on a daemon with a network listener and a model-loading path, every finding was traced
to its actual callers before being dismissed, not pattern-matched and dropped:

| Finding | Disposition |
|---|---|
| `squish/catalog.py`'s `has_prebuilt`, network-unavailable → `True` | **FIXED (docs only, zero behavior change).** Traced every caller (`cli.py`'s model-listing table, `catalog.py`'s own `__str__`) — display-only, not on the request-handling or model-loading path. Docstring now states the intentional "trust the last-known catalog entry" reasoning instead of leaving the comment/code relationship ambiguous. |
| `scripts/compress_and_upload.py`'s `_smoke_test`, `mlx_lm` not installed → `True` (coherence check treated as passed) | **REAL DEFECT, FLAGGED, DEFERRED.** Gates whether a quantized model's coherence was actually verified before upload; "couldn't run the check" silently becoming "check passed" is backwards, though it does warn to stderr. Quant-adjacent — CLAUDE.md's hard constraint on quantization accuracy gates means changing this script's return-value semantics needs its own review, not a drive-by fix in a CI-connection sprint. Named for the next squish sprint. |
| `squish/cli.py` ×2 (ASTC-loader `ImportError` → int4 fallback; default compression format selection) | False positive — visible warning printed, legitimate UX default, not a trust-boundary bypass. |
| `squish/catalog.py`'s hash-verify, no expected hash recorded → `(True, "")` | False positive — "nothing recorded to compare against" is a data-completeness case, not a failed evaluation. |
| `squish/server.py` ×3, `repetition_penalty` default `1.0` | False positive — the semantically-neutral value for that parameter (no penalty), a normal API default. |
| `squish/serving/ollama_compat.py` ×2, `stream` default `True` | False positive — matches upstream Ollama's own default; a response-format choice, not a trust-boundary bypass. |
| `squish/serving/scheduler.py` ×2, `req.done = True` after emitting completion | False positive — idempotent completion bookkeeping after real work (emitting the finish signal), not an unconfigured-fallback shape. |
| `tests/test_wave82_autoload_eagle3.py` | Exempt — test fixture, added `tests/**` to `polarity.exempt_globs`. |
| `benchmarks/**` ×3 (two readiness-poll `wait_ready` functions treating any HTTP response including errors as "server up"; one thermal-sensor-absent fallback with its own inline justification comment already in the code) | Exempt — offline perf-harness code, not the daemon's request path; the thermal one was already self-documenting before this sprint touched it. Added `benchmarks/**` to `polarity.exempt_globs`. |

**A live kiban-side finding, not a squish defect, reported upstream rather than worked
around by gaming squish's own code:** `repo:ruff`'s net-new dispatch (`konjo_gates_py.
cli`) diffs raw tool **stdout text** between the HEAD and base worktree scans, not
per-finding identity. When this sprint's own `pyproject.toml` per-file-ignore addition
(the fix for the whole-repo `ruff lint` promotion above) made a file's ruff output go
from several real findings (at base, old config) to `All checks passed!` (at HEAD, new
config), the dispatcher counted `All checks passed!` itself as a "1 net-new finding" —
a config-driven *cleanup* registering as a regression. Confirmed by isolating the exact
file and reproducing twice. Left as a known, upstream-reportable limitation rather than
suppressed; this PR's own `konjo-gates` run will show it.

A second, related false trigger caught and fixed rather than worked around dishonestly:
`gate_one_way_door`'s `_REMOVED_DEF` regex flags *any* diff line starting with `-def`/
`-class`, with no semantic understanding of "same function, modernized type annotation"
vs "function actually removed." A `ruff --fix`-applied `Optional[str]` → `str | None`
rewrite on a single-line function signature (`scripts/check_release_sync.py`) tripped
it. Rather than fabricate a `Konjo-Acknowledged-Oneway` trailer for a change that isn't
genuinely one-way, the line was reverted to its original spelling with a whole-file
`# ruff: noqa: UP045` directive (not a per-line one, which would touch the same `def`
line and retrigger the detector) — confirmed clean on both `one_way_door` and `ruff
lint` afterward.

**`perf_globs` corrected from a bare `squish/**` (kiban's own starting-point profile,
carried forward unquestioned) to the genuine hot-path directories** —
`quant/serving/kv/context/loaders/io/speculative/hardware/backend.py`, not the whole
package. Caught live by KT-A2.1's own kill-test run: `squish/**` matched literally
every file including `cli.py`'s UX code and `catalog.py`'s docstring-only edit,
demanding a bench-hardware `konjo-prove` MERGE verdict for changes with no measurable
perf effect. This is exactly the kind of kiban-inherited field the brief asked to be
re-verified against the real tree rather than left as a placeholder.

**`threat_model` exercised for real, not stubbed.** This sprint's own diff touches
`squish/catalog.py` and `squish/server.py`, both correctly in `security_globs`
(`network_ingress` boundary); the full 25-file changed-file set also matched
`authn_authz` on a diff-content scan (the word "auth"/"api" appears in a nearby
docstring, not in any changed authentication logic). Ran `konjo-threat classify` then
`record` for real (not simulated) against the full, real changed-file set — not the
partial 5-file set an earlier mid-sprint check used before every file was staged,
which produced a different (now-superseded) fingerprint. Mitigation for both
boundaries states the diff is a docstring clarification plus a `usedforsecurity=False`
annotation on two non-authentication hashes, zero behavior change to the request-
handling or auth-check path; `hmac.compare_digest` (the real API-key check) is
untouched. Trailer: `Konjo-Threat-Model: b94be0e76761`.

**`one_way_door` also fired for real on the full changed-file set, on a fourth kiban
false-positive class**: `_DIFF_RULES`'s `destructive-shell` pattern matches any diff
line containing `rm -rf` with no scope awareness of "inside a `mktemp -d` test-fixture
cleanup trap" vs. an actual destructive repo action — the same `rm -rf "$TMP"` idiom
lopi's own `.konjo/scripts/test_coverage_floor_killtest.sh` already uses, here in this
sprint's new `.konjo/scripts/test_ratchet_killtest.sh`. Rather than rewrite a safe,
standard test-cleanup idiom to dodge a detector, ran `konjo-oneway confirm` for real
(not simulated) and recorded the acknowledgement: `Konjo-Acknowledged-Oneway:
b94be0e76761` (same fingerprint — both trailers key on the identical sorted file set).

**`claude_contract` applied for real** (`docs/pilots/squish-claude-md.proposed.md`,
copied from kiban read-only, applied here): squish's `CLAUDE.md` had 4 of 6 required
sections missing before this sprint (org rules, invariants, repo map, repo-specific
rules) and no org import line — same finding kiban's own read-only reconciliation
recorded for both squish and vectro. Applied verbatim plus one necessary update the
static proposal couldn't have known about: the Konjo Quality Framework section's Wall 2
description no longer claims "blocks the merge" for gates this same sprint just
converted from decorative to real-but-ratcheted — re-verified against
`lib.claude_contract.check_contract` (`ok=True`) after editing, not just assumed clean.
`profile.yml`'s `claude_contract.advisory` stays `true` this sprint regardless (one
measurement of a freshly-converted document isn't the same bar as a document that's
run clean across several real subsequent PRs) — the next sprint that confirms it holds
can flip it.

**Performance-claim correction.** `pyproject.toml`'s PyPI description stated "5.4×
faster end-to-end on 4K-token prompts vs Ollama" as a single figure — traced to
`CHANGELOG.md`'s v5.1.1-era entry (12.78s vs 69.63s, an old benchmark run), superseded
by the current thermally-controlled benchmark that measures the same claim (4K-token
prompt, end-to-end vs Ollama) at 9.8× on exact repetition and 1.19× on a completely
unique prompt — a materially different number for the identical claim, from newer,
more rigorous data already in the repo (`docs/paper.md`, `BENCHMARKS.md`, the linked
blog post `docs/blog/posts/local-llm-fast-enough.md`, which independently states "1.15
to 14.7× faster than Ollama depending on how much your prompts repeat" in its own
description and TL;DR). `BENCHMARKS.md` already states its own rule for exactly this
situation — "quote the range, not a single number" — which the stale PyPI description
violated. Corrected to: "1.15-14.7x faster than Ollama depending on prompt
repetition." `README.md` already stated the honest range and needed no change.
`Formula/*.rb`'s Homebrew `desc` carries no numeric claim; nothing to fix there.

**Non-goal boundary held**: this sprint did not touch quantization algorithms, the
Homebrew formula, or PyPI packaging mechanics (build-system, publish workflow,
classifiers) — the one `pyproject.toml` edit beyond the version bump and the
per-file-ignore addition is the `description` string, which step 7 of this sprint's own
brief explicitly named as one of three places to check for performance claims.

**Honest caveat on this session's own final re-verification, not glossed over.** This
sandboxed environment's root filesystem ran critically low on disk mid-sprint (a
pre-existing condition, not caused by this sprint's own file additions, which total a
few hundred KB) forcing deletion of the isolated venv this sprint used for its clean,
authoritative measurements (the 215-error mypy baseline, the 0-violation ruff baseline,
all ratchet seeds). The final post-commit `konjo-gates` re-run used bare system Python
without squish's own runtime dependencies installed, which inflated `repo:mypy`'s
apparent finding count with spurious "not defined"/"attr-defined" noise from
unresolved imports (numpy, mlx, fastapi) — an environment artifact, not a real
regression. The pre-commit run, with the properly-installed venv, is the authoritative
measurement and is what every ratchet file and the CHANGELOG/LEDGER numbers above
report. Recorded here rather than silently reported as if the degraded re-run's larger
numbers were real.

**A fifth kiban-side limitation, found re-verifying with a properly-installed venv
(ruling out the environment-degradation explanation above): `repo:mypy` and
`repo:vulture`'s net-new dispatch reports far more "net-new" findings against this
PR's real 25-file diff (110 and 59) than any direct, single-pass measurement of HEAD
ever found (215 total mypy errors repo-wide, 108 total vulture findings repo-wide —
this diff touches 5 Python files, so 110/59 "new" findings on a docstring-and-two-
one-liners diff cannot be real). The two-pass mechanism scans HEAD and a separate
`git worktree` checkout of the base commit; the leading hypothesis (not fully root-
caused — this is kiban's own internal worktree/import-resolution mechanics, out of
scope for a "connect squish to kiban" sprint to debug) is that mypy resolves the
editable-installed `squish` package from its real site-packages redirect rather than
from the base worktree's own checkout, so files in the base pass get analyzed against
the WRONG (HEAD's) module contents, manufacturing a large, spurious diff. Because
`repo:mypy`/`repo:vulture`/`repo:ruff`/`repo:ruff-format` have no advisory/soft
setting in `konjo-gates` (unlike `polarity`/`claude_contract`), this PR's own first
`konjo-gates.yml` CI run will show these as real FAILs — disclosed here and in the PR
description rather than hidden, worked around, or silently re-ordered to dodge them.
This, the `repo:ruff` stdout-diffing issue, and the `one_way_door` regex false
positive are three upstream `konjoai/kiban` findings this connection sprint surfaced,
all real, none squish-side defects.
