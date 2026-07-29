#!/usr/bin/env python3
"""Konjo generic ratchet gate — a cleared bar never moves backward.

Same shape as lopi's `.konjo/scripts/coverage_floor_check.py`, generalized to cover
both directions a metric can be locked:

  --mode ceiling   a count that must never GROW (dead code, security findings,
                   complexity violations, DRY violations, files needing reformat).
                   FAIL if measured > locked value.
  --mode floor     a percentage/score that must never SHRINK (docstring coverage,
                   test coverage). FAIL if measured < locked value.

Introduced this sprint (Track A2, squish's first kiban connection) to convert several
`continue-on-error: true` steps in konjo-gate.yml from "cannot ever fail" to "fails
only on regression against squish's own measured baseline" — see LEDGER.md's
Squish-Gate-Triage-1 for the per-step disposition table and the baseline counts each
`.konjo/*-ceiling.txt` / `.konjo/*-floor.txt` file was seeded with.

This script does not run the underlying tool (ruff/vulture/bandit/radon/interrogate).
The caller runs the tool, extracts a single numeric measurement, and passes it via
--measured — kept this way (rather than each gate shelling out through this script)
so the tool invocation stays visible and greppable in the workflow YAML itself, the
same transparency lopi's coverage_floor_check.py's own workflow step already has.

Exit codes:
  0 — measured value at or better than the locked value
  1 — regression (ceiling exceeded, or floor undershot)
  2 — locked-value file missing or malformed, or --measured unparseable
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def read_locked_value(path: Path) -> float:
    """Read the locked value: the first non-comment, non-blank line."""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        return float(line)
    raise ValueError(f"{path} has no value (only comments/blank lines)")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["ceiling", "floor"])
    parser.add_argument("--name", required=True, help="metric name, for messages")
    parser.add_argument("--measured", required=True, help="the value the caller measured")
    parser.add_argument("--file", required=True, type=Path, help="the locked-value file")
    args = parser.parse_args(argv)

    try:
        measured = float(args.measured)
    except ValueError:
        print(f"::error::--measured {args.measured!r} is not a number.")
        return 2

    try:
        locked = read_locked_value(args.file)
    except (OSError, ValueError) as exc:
        print(f"::error::Cannot read {args.name} ratchet value from {args.file}: {exc}")
        return 2

    print(f"Measured {args.name}: {measured:g}")
    print(f"Locked {args.mode}:   {locked:g}")

    if args.mode == "ceiling":
        if round(measured, 4) > round(locked, 4):
            print(
                f"::error::{args.name} rose to {measured:g}, above the locked ceiling "
                f"{locked:g} ({args.file}). Fix the regression, or if this is a genuine "
                "measurement-method change, say why in the commit message and update the "
                "ceiling — never raise it silently to make a real regression pass."
            )
            return 1
        if measured < locked:
            print(
                f"{args.name} improved to {measured:g}, below the {locked:g} ceiling. "
                f"Consider ratcheting {args.file} down to {measured:g} in this PR."
            )
    else:  # floor
        if round(measured, 4) < round(locked, 4):
            print(
                f"::error::{args.name} dropped to {measured:g}, below the locked floor "
                f"{locked:g} ({args.file}). Fix the regression, or if this is a genuine "
                "measurement-method change, say why in the commit message and update the "
                "floor — never lower it silently to make a real regression pass."
            )
            return 1
        if measured > locked:
            print(
                f"{args.name} rose to {measured:g}, above the {locked:g} floor. Consider "
                f"ratcheting {args.file} up to {measured:g} in this PR."
            )

    print(f"{args.name} ratchet gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
