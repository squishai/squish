"""
tests/test_wave148_run_existence_check.py

Wave 148 Part 2 — fix `cmd_run`'s stale "is this model downloaded" gate.

Before this fix, the auto-pull gate in `cmd_run` (squish/cli.py) checked only
for the *raw* bf16 directory (via `_MODEL_SHORTHAND` / `CatalogEntry.dir_name`,
both raw-repo naming). `squish pull`'s fast prebuilt path never creates that
raw directory, and a user who deletes it post-compression is doing the right
thing — either way the gate's `_expected_dir.exists()` check was `False`, so
`squish run <model>` printed a false "not found locally — pulling now" and
re-invoked `cmd_pull` on every run of an already-downloaded, already-compressed
model. `_resolve_presquished_dir` already solves exactly this elsewhere in the
same file (`_resolve_model`) — the gate just never called it.

These tests drive `cmd_run` up through the fixed gate and stop execution
immediately after (via a sentinel raised from `_detect_ram_gb`, the next thing
`cmd_run` calls) so no real server spawn or RAM detection ever runs.
"""
from __future__ import annotations

import contextlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import squish.cli as cli


class _StoppedAfterGate(Exception):
    """Sentinel raised from the mocked `_detect_ram_gb` to halt `cmd_run`
    immediately after the auto-pull gate under test, before any real RAM
    detection or server spawn happens."""


def _make_quant_dir(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "config.json").write_text('{"quantization": {"bits": 4}}')
    return d


def _run_args(model: str):
    import argparse

    return argparse.Namespace(
        model=model,
        daemon=False,
        prompt="",
        int2=False, int3=False, int4=False, int8=False,
    )


class TestRunExistenceCheckReusesPresquishedResolver:
    @contextlib.contextmanager
    def _patched(self, tmp_path):
        """Shared mocks: no-op health/service checks, and a sentinel that
        fires the instant cmd_run reaches the RAM-aware quant block right
        after the gate under test."""
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(cli, "_MODELS_DIR", tmp_path))
            stack.enter_context(patch.object(cli, "_CATALOG_AVAILABLE", True))
            stack.enter_context(patch.object(cli, "_first_run_health_gate", MagicMock()))
            stack.enter_context(
                patch.object(cli, "_detect_local_ai_services", MagicMock(return_value=[]))
            )
            stack.enter_context(
                patch.object(cli, "_detect_ram_gb", MagicMock(side_effect=_StoppedAfterGate))
            )
            yield

    def _exercise_gate(self, tmp_path, capsys):
        """Run cmd_run through the auto-pull gate under test and return
        (cmd_pull_mock, captured_stdout)."""
        cmd_pull_mock = MagicMock()
        with (
            self._patched(tmp_path),
            patch.object(cli, "cmd_pull", cmd_pull_mock),
            pytest.raises(_StoppedAfterGate),
        ):
            cli.cmd_run(_run_args("qwen3:8b"))
        return cmd_pull_mock, capsys.readouterr().out

    def test_int4_only_skips_repull_and_prints_nothing(self, tmp_path, capsys):
        """Only Qwen3-8B-int4/ on disk (no bf16) — the normal post-compression
        state — must not trigger a re-pull or the false 'not found' message."""
        _make_quant_dir(tmp_path, "Qwen3-8B-int4")
        cmd_pull_mock, out = self._exercise_gate(tmp_path, capsys)
        cmd_pull_mock.assert_not_called()
        assert "not found locally" not in out

    def test_neither_dir_present_still_pulls(self, tmp_path, capsys):
        """Genuinely nothing on disk — the legitimate 'not downloaded yet'
        case must still trigger a pull."""
        cmd_pull_mock, out = self._exercise_gate(tmp_path, capsys)
        cmd_pull_mock.assert_called_once()
        assert "not found locally" in out

    def test_int3_only_falls_back_through_resolver_chain(self, tmp_path, capsys):
        """Only Qwen3-8B-int3/ present (no int4, no bf16) — exercises
        _resolve_presquished_dir's int4 -> int3 fallback chain through this
        call site — must still be found, no re-pull."""
        _make_quant_dir(tmp_path, "Qwen3-8B-int3")
        cmd_pull_mock, out = self._exercise_gate(tmp_path, capsys)
        cmd_pull_mock.assert_not_called()
        assert "not found locally" not in out
