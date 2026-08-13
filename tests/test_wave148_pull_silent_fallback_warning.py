"""
tests/test_wave148_pull_silent_fallback_warning.py

Wave 148 Part 1 — kill the silent-fallback ambiguity in `squish pull`.

Before this fix, `_hf_list_files` swallowed every exception from the
HuggingFace prebuilt-weights listing call and returned `[]`, making a real
network failure (bad proxy, expired token, rate limit) indistinguishable
from "this repo genuinely has no prebuilt weights" — and neither case
printed anything unless `--verbose` was passed. A user could silently take
the expensive raw-download-and-compress path with zero explanation.

These tests pin:
  - `_hf_list_files` / `_has_squish_weights` now raise `_PrebuiltCheckError`
    (chaining the original exception) when the listing call itself fails,
    instead of returning `[]`.
  - `pull()` always prints a distinct warning for that case — not gated on
    `verbose` — and still falls back to raw-download-and-compress rather
    than crashing.
  - The genuine "listed fine, no squish files yet" case prints a distinct,
    non-alarming message and does NOT raise.
  - The pre-existing SSL-error re-raise path through `pull()` is untouched.

`huggingface_hub` is not installed in this environment, so it is faked via
`sys.modules` injection (same pattern as tests/test_catalog_ssl.py and
tests/test_wave147a_streaming_pull.py) rather than skipped.
"""
from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from squish.catalog import (
    CatalogEntry,
    _has_squish_weights,
    _hf_list_files,
    _PrebuiltCheckError,
    _SSLError,
    pull,
)


def _make_entry(**kwargs) -> CatalogEntry:
    defaults = dict(
        id="test:8b",
        name="Test 8B",
        hf_mlx_repo="mlx-community/Test-8B-bf16",
        size_gb=16.0,
        params="8B",
        context=8192,
        squished_size_gb=4.5,
        squish_repo="squishai/Test-8B-squished",
        tags=[],
        notes="",
    )
    defaults.update(kwargs)
    return CatalogEntry(**defaults)


def _inject_fake_hf(monkeypatch, **fns):
    fake_hf = types.SimpleNamespace(**fns)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    return fake_hf


def _fake_snapshot_download_raw(local_dir_holder):
    """Create a minimal complete raw bf16 dir so _is_raw_model_dir_complete
    passes without a real network call or real weights."""

    def _snapshot_download(repo_id, local_dir, token=None, ignore_patterns=None):
        d = Path(local_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text("{}", encoding="utf-8")
        (d / "model.safetensors").write_bytes(b"fake-weights")
        local_dir_holder["dir"] = d
        return str(d)

    return _snapshot_download


def _connection_error_list_repo_files(repo_id, token=None):
    raise ConnectionError("network unreachable")


def _pull_with_fake_listing(monkeypatch, tmp_path, entry, list_repo_files_fn):
    """Run pull() against a fake HF hub whose list_repo_files is
    *list_repo_files_fn*, with the raw-download and compress steps faked out
    so only the prebuilt-check branch under test is exercised for real."""
    local_dir_holder = {}
    _inject_fake_hf(
        monkeypatch,
        list_repo_files=list_repo_files_fn,
        snapshot_download=_fake_snapshot_download_raw(local_dir_holder),
    )
    monkeypatch.setattr(subprocess, "run", lambda cmd: types.SimpleNamespace(returncode=0))
    with patch("squish.catalog.resolve", return_value=entry):
        return pull("test:8b", models_dir=tmp_path, verbose=False)


# ── _hf_list_files / _has_squish_weights raise on real failure ────────────────

class TestHfListFilesRaisesOnFailure:
    def test_raises_prebuilt_check_error_on_connection_error(self, monkeypatch):
        _inject_fake_hf(monkeypatch, list_repo_files=_connection_error_list_repo_files)

        with pytest.raises(_PrebuiltCheckError) as exc_info:
            _hf_list_files("squishai/Test-8B-squished")

        assert isinstance(exc_info.value.__cause__, ConnectionError)

    def test_has_squish_weights_propagates_the_error(self, monkeypatch):
        _inject_fake_hf(monkeypatch, list_repo_files=_connection_error_list_repo_files)

        with pytest.raises(_PrebuiltCheckError):
            _has_squish_weights("squishai/Test-8B-squished")

    def test_missing_huggingface_hub_also_raises_prebuilt_check_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        monkeypatch.delitem(sys.modules, "huggingface_hub", raising=False)
        # Simulate ImportError by making the import itself fail: pop any real
        # module and don't provide a fake one that exposes list_repo_files.
        import builtins

        real_import = builtins.__import__

        def _blocking_import(name, *args, **kwargs):
            if name == "huggingface_hub":
                raise ImportError("no module named huggingface_hub")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocking_import)

        with pytest.raises(_PrebuiltCheckError):
            _hf_list_files("squishai/Test-8B-squished")


# ── _has_squish_weights: genuine "no files yet" is NOT an error ───────────────

class TestHasSquishWeightsGenuineFalse:
    def test_returns_false_without_raising_when_repo_has_no_markers(self, monkeypatch):
        def _list_repo_files(repo_id, token=None):
            return ["config.json", "README.md"]

        _inject_fake_hf(monkeypatch, list_repo_files=_list_repo_files)

        assert _has_squish_weights("squishai/Test-8B-squished") is False

    def test_returns_true_for_npz_marker(self, monkeypatch):
        def _list_repo_files(repo_id, token=None):
            return ["config.json", "squish_weights.npz"]

        _inject_fake_hf(monkeypatch, list_repo_files=_list_repo_files)

        assert _has_squish_weights("squishai/Test-8B-squished") is True

    def test_returns_true_for_npy_dir_marker(self, monkeypatch):
        def _list_repo_files(repo_id, token=None):
            return ["config.json", "squish_npy/layer0.npy"]

        _inject_fake_hf(monkeypatch, list_repo_files=_list_repo_files)

        assert _has_squish_weights("squishai/Test-8B-squished") is True


# ── pull(): the check itself failing always prints, even without verbose ─────

class TestPullSurfacesPrebuiltCheckFailure:
    def test_warns_and_falls_back_without_verbose(self, tmp_path, monkeypatch, capsys):
        entry = _make_entry()

        result = _pull_with_fake_listing(
            monkeypatch, tmp_path, entry, _connection_error_list_repo_files
        )

        out = capsys.readouterr().out
        assert "Could not verify prebuilt weights" in out
        assert entry.squish_repo in out
        assert result == tmp_path / "Test-8B-int4"

    def test_prebuilt_check_error_does_not_crash_pull(self, tmp_path, monkeypatch):
        entry = _make_entry()

        # Must not raise _PrebuiltCheckError (or anything else) out of pull().
        _pull_with_fake_listing(monkeypatch, tmp_path, entry, _connection_error_list_repo_files)


# ── pull(): genuine "no prebuilt weights yet" prints a distinct message ──────

class TestPullNoPrebuiltWeightsYet:
    def test_prints_distinct_info_message_and_does_not_raise(self, tmp_path, monkeypatch, capsys):
        entry = _make_entry()

        def _list_repo_files(repo_id, token=None):
            return ["config.json"]

        result = _pull_with_fake_listing(monkeypatch, tmp_path, entry, _list_repo_files)

        out = capsys.readouterr().out
        assert "No prebuilt weights" in out
        assert "Could not verify" not in out
        assert result == tmp_path / "Test-8B-int4"


# ── pull(): SSL failure during the prebuilt *download* still re-raises ───────

class TestPullSslErrorStillReraises:
    def test_ssl_error_during_prebuilt_download_propagates(self, tmp_path, monkeypatch):
        entry = _make_entry()

        def _list_repo_files(repo_id, token=None):
            return ["squish_weights.npz"]

        _inject_fake_hf(monkeypatch, list_repo_files=_list_repo_files)

        def _raising_hf_download(repo, local_dir, token=None):
            raise _SSLError("SSL certificate verification failed")

        with (
            patch("squish.catalog.resolve", return_value=entry),
            patch("squish.catalog._hf_download", side_effect=_raising_hf_download),
        ):
            with pytest.raises(_SSLError):
                pull("test:8b", models_dir=tmp_path, verbose=False)
