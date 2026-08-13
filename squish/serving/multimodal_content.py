"""squish/serving/multimodal_content.py — OpenAI multi-content-block parsing.

Wave 134 — the request-plumbing half of image/video input support. Before
Wave 130's mlx_vlm backend can be *used*, chat requests need a way to carry
image/audio/video alongside text: OpenAI's ``messages[].content`` can be
either a plain string or a list of content blocks
(``{"type": "text", "text": ...}``, ``{"type": "image_url", "image_url":
{"url": ...}}``, etc.). Squish's request handling (``_apply_chat_template``,
concision-prefix injection, trace logging) has always assumed a string.

:func:`extract_multimodal_content` normalizes every message's ``content`` to
a plain string (so all of that existing string-only code keeps working
unchanged) and separately collects the raw image/audio/video source strings
(URLs or ``data:`` URIs) in message order, for the caller to hand to
mlx_vlm's own ``image=``/``audio=``/``video=`` generation kwargs — squish
does not fetch or decode media itself; ``mlx_vlm.utils.load_image`` already
handles URLs, ``data:`` URIs, and local paths internally, so squish's job is
only to extract and *validate* the source string (SSRF guard: block schemes
and hosts that let a request make the server fetch internal/loopback
resources) before handing it off.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

_LOG = logging.getLogger("squish.serving.multimodal_content")

_ALLOWED_URL_SCHEMES = frozenset({"http", "https", "data"})


class UnsafeMediaSourceError(ValueError):
    """A message content block referenced a media source squish refuses to fetch."""


def _is_private_or_loopback(host: str) -> bool:
    """True if *host* resolves to a loopback/private/link-local/reserved address."""
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        try:
            resolved = socket.gethostbyname(host)
            addr = ipaddress.ip_address(resolved)
        except (OSError, ValueError) as exc:
            _LOG.debug("could not resolve host %r for SSRF check: %s", host, exc)
            return True  # unresolvable host: fail closed, refuse to fetch
    return (
        addr.is_loopback
        or addr.is_private
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
    )


def validate_media_source(source: str) -> None:
    """Raise :class:`UnsafeMediaSourceError` if *source* is unsafe to fetch.

    ``data:`` URIs never touch the network — always allowed. ``http``/
    ``https`` URLs are allowed only when the host does not resolve to a
    loopback/private/link-local address (SSRF guard: a request payload must
    not be able to make squish's server fetch internal network resources).
    Any other scheme (``file:``, ``ftp:``, bare paths, ...) is rejected.
    """
    if source.startswith("data:"):
        return
    parsed = urlparse(source)
    if parsed.scheme not in _ALLOWED_URL_SCHEMES or not parsed.hostname:
        raise UnsafeMediaSourceError(
            f"Unsupported media source scheme {parsed.scheme!r}; "
            "only http(s) URLs and data: URIs are accepted."
        )
    if _is_private_or_loopback(parsed.hostname):
        raise UnsafeMediaSourceError(
            f"Refusing to fetch media from a private/loopback host: {parsed.hostname!r}"
        )


def _extract_text_and_media(content: list) -> tuple[str, list[str], list[str], list[str]]:
    """Split one message's content-block list into (text, images, audio, videos)."""
    text_parts: list[str] = []
    images: list[str] = []
    audio: list[str] = []
    videos: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(str(block.get("text", "")))
        elif block_type in ("image_url", "input_image"):
            raw = block.get("image_url") or block.get("input_image") or block.get("url")
            url = raw.get("url") if isinstance(raw, dict) else raw
            if isinstance(url, str) and url:
                validate_media_source(url)
                images.append(url)
        elif block_type == "input_audio":
            audio_obj = block.get("input_audio", {})
            data = audio_obj.get("data")
            fmt = audio_obj.get("format", "wav")
            if data:
                # OpenAI's input_audio ships raw base64 (no data: prefix) — wrap it
                # into a data URI so it round-trips through the same "source
                # string" contract as image_url.
                source = f"data:audio/{fmt};base64,{data}"
                audio.append(source)
        elif block_type == "video_url":
            raw = block.get("video_url")
            url = raw.get("url") if isinstance(raw, dict) else raw
            if isinstance(url, str) and url:
                validate_media_source(url)
                videos.append(url)
    return "".join(text_parts), images, audio, videos


def extract_multimodal_content(
    messages: list[dict],
) -> tuple[list[dict], list[str], list[str], list[str]]:
    """Normalize ``messages[].content`` to plain strings; collect media sources.

    Returns ``(normalized_messages, images, audio, videos)``. Messages whose
    ``content`` is already a string pass through untouched. Media lists are
    flattened across all messages, in encounter order — callers that need
    per-message association should not use this helper (Phase 1 only
    supports one flat request-level media list, matching mlx_vlm's own
    ``image=``/``audio=``/``video=`` kwarg shape).
    """
    normalized: list[dict] = []
    images: list[str] = []
    audio: list[str] = []
    videos: list[str] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            text, msg_images, msg_audio, msg_videos = _extract_text_and_media(content)
            normalized.append({**msg, "content": text})
            images.extend(msg_images)
            audio.extend(msg_audio)
            videos.extend(msg_videos)
        else:
            normalized.append(msg)
    return normalized, images, audio, videos
