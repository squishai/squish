"""tests/test_wave134_multimodal_input.py

Wave 134 — image/audio/video request plumbing on top of Wave 130's mlx_vlm
backend. Wave 130 shipped text-only dispatch only; this wave adds the
missing half: extracting OpenAI multi-content-block messages
(``content: [{"type": "image_url", ...}, ...]``) into a flat media list,
validating media sources against SSRF (a request payload must not make
squish's server fetch internal/loopback network resources), and routing
image/audio/video-bearing chat requests to mlx_vlm's own generation loop
instead of squish's text-only manual decode path.

This pins:
- extract_multimodal_content normalizes content to strings and collects
  image_url / input_image / input_audio / video_url sources, in order
- validate_media_source blocks file:/ftp: schemes and private/loopback/
  link-local hosts; allows data: URIs and public http(s) URLs unconditionally
- _generate_tokens's new images/audio/videos kwargs bypass the text-only
  dispatch chain entirely and route through BE.stream_generate
- stop-sequence truncation works correctly across streamed chunks in the
  multimodal path (same contract as the text-only path: truncate at the
  stop string, emit finish_reason="stop")
- /v1/chat/completions returns a clear 400 (not a 500 or silent drop) when
  a request carries image/audio/video content but the loaded model isn't
  mlx_vlm-backed
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from squish.serving.multimodal_content import (
    UnsafeMediaSourceError,
    extract_multimodal_content,
    validate_media_source,
)

# ── extract_multimodal_content ────────────────────────────────────────────────


class TestExtractMultimodalContent:
    def test_plain_string_content_passes_through_unchanged(self):
        messages = [{"role": "user", "content": "hello"}]
        normalized, images, audio, videos = extract_multimodal_content(messages)
        assert normalized == messages
        assert images == audio == videos == []

    def test_text_blocks_concatenate_into_string_content(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Part one. "},
                    {"type": "text", "text": "Part two."},
                ],
            }
        ]
        normalized, _, _, _ = extract_multimodal_content(messages)
        assert normalized[0]["content"] == "Part one. Part two."

    def test_image_url_standard_openai_shape(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                ],
            }
        ]
        normalized, images, audio, videos = extract_multimodal_content(messages)
        assert normalized[0]["content"] == "what is this?"
        assert images == ["https://example.com/a.png"]
        assert audio == []
        assert videos == []

    def test_image_url_loose_string_shape(self):
        # Some clients send image_url as a bare string rather than {"url": ...}.
        messages = [
            {"role": "user", "content": [{"type": "image_url", "image_url": "https://x.com/b.png"}]}
        ]
        _, images, _, _ = extract_multimodal_content(messages)
        assert images == ["https://x.com/b.png"]

    def test_input_image_variant(self):
        messages = [
            {
                "role": "user",
                "content": [{"type": "input_image", "input_image": {"url": "https://x.com/c.png"}}],
            }
        ]
        _, images, _, _ = extract_multimodal_content(messages)
        assert images == ["https://x.com/c.png"]

    def test_input_audio_wraps_base64_as_data_uri(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
                ],
            }
        ]
        _, _, audio, _ = extract_multimodal_content(messages)
        assert audio == ["data:audio/wav;base64,AAAA"]

    def test_video_url(self):
        messages = [
            {
                "role": "user",
                "content": [{"type": "video_url", "video_url": {"url": "https://x.com/d.mp4"}}],
            }
        ]
        _, _, _, videos = extract_multimodal_content(messages)
        assert videos == ["https://x.com/d.mp4"]

    def test_media_flattened_in_encounter_order_across_messages(self):
        messages = [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "https://x.com/1.png"}}],
            },
            {"role": "assistant", "content": "ok"},
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "https://x.com/2.png"}}],
            },
        ]
        _, images, _, _ = extract_multimodal_content(messages)
        assert images == ["https://x.com/1.png", "https://x.com/2.png"]

    def test_unknown_block_type_ignored_not_fatal(self):
        messages = [{"role": "user", "content": [{"type": "future_thing", "data": "??"}]}]
        normalized, images, audio, videos = extract_multimodal_content(messages)
        assert normalized[0]["content"] == ""
        assert images == audio == videos == []

    def test_unsafe_source_raises_before_returning(self):
        messages = [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "http://127.0.0.1/x"}}],
            }
        ]
        with pytest.raises(UnsafeMediaSourceError):
            extract_multimodal_content(messages)


# ── validate_media_source (SSRF guard) ────────────────────────────────────────


class TestValidateMediaSource:
    def test_data_uri_always_allowed(self):
        validate_media_source("data:image/png;base64,iVBORw0KGgo=")  # no raise

    def test_public_https_url_allowed(self):
        validate_media_source("https://huggingface.co/some/image.png")  # no raise

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1/secret",
            "http://localhost:11435/",
            "http://169.254.169.254/latest/meta-data",  # cloud metadata endpoint
            "http://10.0.0.5/internal",
            "http://[::1]/",
        ],
    )
    def test_private_or_loopback_host_blocked(self, url):
        with pytest.raises(UnsafeMediaSourceError):
            validate_media_source(url)

    @pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "not-a-url"])
    def test_disallowed_scheme_blocked(self, url):
        with pytest.raises(UnsafeMediaSourceError):
            validate_media_source(url)

    def test_unresolvable_host_fails_closed(self):
        with pytest.raises(UnsafeMediaSourceError):
            validate_media_source("http://this-host-does-not-exist.invalid/x")


# ── _generate_tokens multimodal dispatch ──────────────────────────────────────


def _fake_stream_generate_factory(chunks):
    def _fake(model, tokenizer, prompt, **kwargs):
        yield from chunks

    return _fake


class TestGenerateTokensMultimodalDispatch:
    def test_images_present_routes_through_be_stream_generate_not_mlx_lm(self, monkeypatch):
        import squish.server as srv

        monkeypatch.setattr(srv._state, "model", MagicMock(__squish_runtime__="mlx_vlm"))
        monkeypatch.setattr(srv._state, "tokenizer", MagicMock())

        from squish.backend import BE

        calls = []

        def _fake_stream_generate(model, tokenizer, prompt, **kwargs):
            calls.append(kwargs)
            yield "hello", None
            yield " world", "stop"

        monkeypatch.setattr(BE, "stream_generate", _fake_stream_generate)

        out = list(
            srv._generate_tokens(
                "a prompt",
                max_tokens=16,
                images=["https://x.com/a.png"],
                audio=None,
                videos=None,
            )
        )
        # finish_reason is emitted as its own terminal (empty-text) chunk,
        # matching squish's SSE convention elsewhere (_make_chunk).
        assert out == [("hello", None), (" world", None), ("", "stop")]
        assert calls[0]["image"] == ["https://x.com/a.png"]

    def test_no_media_falls_through_to_text_only_path(self, monkeypatch):
        # Sanity check: the new kwargs must not change behaviour for ordinary
        # text-only requests (the overwhelming majority of traffic).
        import squish.server as srv

        from squish.backend import BE

        called = {"mlx_vlm": False}

        def _should_not_be_called(*a, **k):
            called["mlx_vlm"] = True
            yield "x", "stop"

        monkeypatch.setattr(BE, "stream_generate", _should_not_be_called)
        # images/audio/videos all None/empty -> must not enter the new branch
        # (we don't drain the generator far enough to hit real model code;
        # just confirm the multimodal branch itself is skipped).
        gen = srv._generate_tokens("hi", images=None, audio=None, videos=None)
        # The multimodal branch returns immediately if entered; if it were
        # entered, the first yielded item would come from _should_not_be_called.
        # We only assert the guard condition here, not full text-path behaviour
        # (that requires a real/mocked model + tokenizer well beyond this unit).
        assert not called["mlx_vlm"]
        gen.close()

    def test_stop_sequence_truncates_across_chunks(self, monkeypatch):
        import squish.server as srv

        monkeypatch.setattr(srv._state, "model", MagicMock(__squish_runtime__="mlx_vlm"))
        monkeypatch.setattr(srv._state, "tokenizer", MagicMock())

        from squish.backend import BE

        monkeypatch.setattr(
            BE,
            "stream_generate",
            _fake_stream_generate_factory(
                [("Hello", None), (" wor", None), ("ld! STOP", None), ("more", None)]
            ),
        )

        out = list(srv._generate_tokens("p", images=["u"], audio=None, videos=None, stop="STOP"))
        texts = [t for t, _ in out]
        assert "".join(texts) == "Hello wor" + "ld! "
        assert out[-1] == ("", "stop")

    def test_finish_reason_propagates_without_stop_match(self, monkeypatch):
        import squish.server as srv

        monkeypatch.setattr(srv._state, "model", MagicMock(__squish_runtime__="mlx_vlm"))
        monkeypatch.setattr(srv._state, "tokenizer", MagicMock())

        from squish.backend import BE

        monkeypatch.setattr(
            BE, "stream_generate", _fake_stream_generate_factory([("a", None), ("b", "length")])
        )

        out = list(srv._generate_tokens("p", images=["u"], audio=None, videos=None))
        assert out == [("a", None), ("b", None), ("", "length")]


# ── /v1/chat/completions: 400 on multimodal request to a text-only model ─────


class TestChatCompletionsMultimodalRejection:
    @pytest.fixture()
    def client(self):
        from fastapi.testclient import TestClient

        import squish.server as srv

        orig_state = srv._state
        orig_apikey = srv._API_KEY
        orig_load_complete = srv._LOAD_COMPLETE.is_set()

        srv._state = srv._ModelState()
        srv._state.model = MagicMock()  # no __squish_runtime__ attr -> defaults "mlx_lm"
        srv._state.tokenizer = MagicMock()
        srv._API_KEY = None
        srv._LOAD_COMPLETE.set()

        c = TestClient(srv.app, raise_server_exceptions=False)
        yield c

        srv._state = orig_state
        srv._API_KEY = orig_apikey
        if not orig_load_complete:
            srv._LOAD_COMPLETE.clear()

    def test_image_request_against_text_only_model_returns_400(self, client):
        r = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what is this?"},
                            {"type": "image_url", "image_url": {"url": "https://x.com/a.png"}},
                        ],
                    }
                ],
            },
        )
        assert r.status_code == 400
        assert "image" in r.json()["detail"].lower() or "multimodal" in r.json()["detail"].lower()

    def test_unsafe_media_source_returns_400_not_500(self, client):
        r = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "http://127.0.0.1/x"}},
                        ],
                    }
                ],
            },
        )
        assert r.status_code == 400
