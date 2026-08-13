#!/usr/bin/env python3
"""Generate the squishai org card (squishai/README Space README.md).

Mirrors the substance of the squish GitHub README + adds an HF-native model
index built from the live grounded per-model data in hf_cards_gen.py.
GitHub-relative links/images are rewritten to absolute URLs so they resolve on HF.
"""
import os
from hf_cards_gen import MODELS, GH, DOCS, PYPI, ORG, BENCH, LICENSE_URL

RAW = "https://raw.githubusercontent.com/konjoai/squish/main"

# Space frontmatter: preserve the existing Space metadata, but fix the license
# tag (was `mit`; the repo is BUSL-1.1, which is not a standard HF license id,
# so use `other` and state BUSL-1.1 in the body).
FRONTMATTER = """title: README
emoji: 🔥
colorFrom: blue
colorTo: red
sdk: static
pinned: false
license: other"""

# Display order for the model index: family, then ascending size.
ORDER = [
    "Qwen2.5-1.5B-Instruct-bf16-squished",
    "Qwen2.5-7B-Instruct-bf16-squished",
    "Qwen3-0.6B-bf16-squished",
    "Qwen3-4B-bf16-squished",
    "Qwen3-8B-bf16-squished",
    "Llama-3.2-1B-Instruct-bf16-squished",
    "Llama-3.2-3B-Instruct-bf16-squished",
    "gemma-3-1b-it-bf16-squished",
    "gemma-3-4b-it-bf16-squished",
]


def model_index_rows() -> str:
    rows = []
    for repo in ORDER:
        d = MODELS[repo]
        ctx = f"{d['ctx']}" if d.get("ctx") else "—"
        rows.append(
            f"| [{d['display']}](https://huggingface.co/squishai/{repo}) "
            f"| `squish run {d['shorthand']}` "
            f"| {d['raw_gb']} GB | **{d['sq_gb']} GB** | {d['saved']} | {ctx} |"
        )
    return "\n".join(rows)


def render() -> str:
    body = f"""
<div align="center">

<img src="{RAW}/assets/squish-logo-1.png" width="320" alt="Squish" />

# Squish: pre-squished models for Apple Silicon

**Fast local LLMs on Apple Silicon.** Sub-second model loads. Beats Ollama on
throughput, tail latency, and full-response time. One OpenAI and Ollama-compatible
daemon. No cloud, no API keys, fully offline.

[![GitHub](https://img.shields.io/badge/GitHub-konjoai%2Fsquish-black?logo=github)]({GH})
[![PyPI](https://img.shields.io/badge/PyPI-squish--ai-3775A9?logo=pypi&logoColor=white)]({PYPI})
[![Docs](https://img.shields.io/badge/docs-squish.run-8b5cf6)]({DOCS})
[![License](https://img.shields.io/badge/license-BUSL--1.1-2563eb)]({LICENSE_URL})

</div>

---

## Run a model in one command

```bash
brew tap konjoai/squish
brew install squish
squish run qwen2.5:7b      # pulls a pre-squished model + starts a local server
```

The daemon serves an OpenAI (`/v1/*`) **and** Ollama (`/api/*`) API on port
11435. Point any existing client at it and go. Prefer pipx?
`pipx install squish-ai`.

## Models in this org

Every model below is INT4-quantized (4-bit, group size 64, affine) into MLX
format and ready for `squish run`. Sizes are the actual on-disk download from
this org; raw is the source bf16 checkpoint.

| Model | Run it | Raw (bf16) | Squished | Saved | Context |
|-------|--------|-----------:|---------:|:-----:|--------:|
{model_index_rows()}

`squish run <id>` downloads these exact weights and loads them in
**under a second**: no compression wait, no Python environment, no cloud.

## Why Squish

Squish separates how a model's weights are *stored* from how they *run*: stored
compressed and Metal-native, then `mmap`-ed straight into unified memory, with no
dtype-conversion pass. It runs as a **persistent daemon** whose two-tier KV
cache reuses prefill across requests instead of re-running it.

Measured on an Apple **M3 (16 GB)**, **Qwen2.5-7B** vs Ollama, thermally
controlled (Squish INT4/INT3 vs Ollama Q4_K_M):

| Metric | Ollama | Squish |
|---|---:|---:|
| Full response @ 4,000-token prompt | 37.5 s | **3.8 s** (up to 9.8× faster) |
| Cold start (load + first token, 1.5B) | 20–30 s | **≈ 0.5 s** (54× load) |
| Decode throughput @ 75 tokens | 20.3 tok/s | **24.0 tok/s** (INT3) |
| Repeat-prompt TTFT (KV cache hit) | ~160 ms | **4–11 ms** |
| Peak RAM during inference | 5.14 GB | **3.50 GB** |
| Disk (7B INT4 / INT3) | 4.36 GB | **4.00 / 3.56 GB** |
| Cold short-prompt TTFT | **167 ms** | 192 ms *(honest loss)* |

The one place Ollama wins is single-token latency on a cold, novel prompt, stated
plainly. Full methodology and ablations: [BENCHMARKS.md]({BENCH}).

## What it doesn't do

If any of these matter, Ollama or LM Studio is the right call:

- **No GPU outside Apple Silicon.** It's MLX-based; CUDA users want vLLM or llama.cpp.
- **No multi-user serving.** One developer, one machine, not a production API.
- **No multimodal.** Text only.
- **Slower first token on a cold, short prompt** than Ollama (192 ms vs 167 ms).

## Links

- **GitHub**: [github.com/konjoai/squish]({GH})
- **Docs**: [squish.run]({DOCS})
- **Install**: [`squish-ai` on PyPI]({PYPI})
- **Benchmarks**: [BENCHMARKS.md]({BENCH})
- **License**: BUSL-1.1 ([LICENSE]({LICENSE_URL}))

---

*All models here are pre-squished by [Squish]({GH}). Run any of them in one
command on Apple Silicon.*
"""
    return f"---\n{FRONTMATTER}\n---\n{body}"


if __name__ == "__main__":
    os.makedirs("hf_cards_preview/_ORG_README_space", exist_ok=True)
    out = render()
    open("hf_cards_preview/_ORG_README_space/README.md", "w", encoding="utf-8").write(out)
    print("wrote hf_cards_preview/_ORG_README_space/README.md", f"({len(out)} bytes)")
