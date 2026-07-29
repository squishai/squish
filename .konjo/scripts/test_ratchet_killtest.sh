#!/usr/bin/env bash
# Kill-test for ratchet_check.py — Track A2 verification.
#
# Proves, against synthetic locked-value files (never squish's real ceiling/floor
# files — this must never depend on the current tree's actual counts):
#   1. A ceiling metric that regresses UP (more findings) fails.
#   2. A ceiling metric that holds or improves passes.
#   3. A floor metric that regresses DOWN (less coverage) fails.
#   4. A floor metric that holds or improves passes.
#   5. A missing/malformed locked-value file fails with exit 2, not a silent pass.
#   6. A non-numeric --measured fails with exit 2, not a silent pass.
#
# Usage: bash .konjo/scripts/test_ratchet_killtest.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK="$SCRIPT_DIR/ratchet_check.py"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

PASS=0
FAIL=0

assert_exit() {
  local desc="$1" expected="$2"
  shift 2
  set +e
  python3 "$CHECK" "$@" >"$TMP/out.log" 2>&1
  local actual=$?
  set -e
  if [ "$actual" -eq "$expected" ]; then
    echo "PASS: $desc (exit $actual)"
    PASS=$((PASS + 1))
  else
    echo "FAIL: $desc (expected exit $expected, got $actual)"
    cat "$TMP/out.log"
    FAIL=$((FAIL + 1))
  fi
}

echo "108" > "$TMP/ceiling_108.txt"
echo "31.4" > "$TMP/floor_31_4.txt"

echo "── ratchet_check.py kill-test ──"
assert_exit "ceiling holds exactly (108 == 108) passes"        0 --mode ceiling --name vulture --measured 108 --file "$TMP/ceiling_108.txt"
assert_exit "ceiling improves (90 < 108) passes"                0 --mode ceiling --name vulture --measured 90  --file "$TMP/ceiling_108.txt"
assert_exit "ceiling regresses (120 > 108) fails"                1 --mode ceiling --name vulture --measured 120 --file "$TMP/ceiling_108.txt"
assert_exit "floor holds exactly (31.4 == 31.4) passes"         0 --mode floor --name docstrings --measured 31.4 --file "$TMP/floor_31_4.txt"
assert_exit "floor improves (40 > 31.4) passes"                 0 --mode floor --name docstrings --measured 40   --file "$TMP/floor_31_4.txt"
assert_exit "floor regresses (20 < 31.4) fails"                  1 --mode floor --name docstrings --measured 20   --file "$TMP/floor_31_4.txt"
assert_exit "missing locked-value file fails closed (exit 2)"    2 --mode ceiling --name vulture --measured 10 --file "$TMP/does_not_exist.txt"
assert_exit "non-numeric --measured fails closed (exit 2)"       2 --mode ceiling --name vulture --measured "not-a-number" --file "$TMP/ceiling_108.txt"

echo
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
