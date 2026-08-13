---
title: README
emoji: 🔥
colorFrom: blue
colorTo: red
sdk: static
pinned: false
license: other
---

<div align="center">

<img src="https://raw.githubusercontent.com/konjoai/squish/main/assets/squish-logo-1.png" width="320" alt="Squish" />

# Squish — pre-squished models for Apple Silicon

**The fastest way to run local LLMs on Apple Silicon.** Sub-second model loads.
Beats Ollama on throughput, tail latency, and full-response time. One
OpenAI/Ollama-compatible daemon — no cloud, no API keys, fully offline.

[![GitHub](https://img.shields.io/badge/GitHub-konjoai%2Fsquish-black?logo=github)](https://github.com/konjoai/squish)
[![PyPI](https://img.shields.io/badge/PyPI-squish--ai-3775A9?logo=pypi&logoColor=white)](https://pypi.org/project/squish-ai/)
[![Docs](https://img.shields.io/badge/docs-squish.run-8b5cf6)](https://squish.run)
[![License](https://img.shields.io/badge/license-BUSL--1.1-2563eb)](https://github.com/konjoai/squish/blob/main/LICENSE)

</div>

---

## Run a model in one command

```bash
brew tap konjoai/squish
brew install squish
squish run qwen2.5:7b      # pulls a pre-squished model + starts a local server
```

The daemon serves an OpenAI (`/v1/*`) **and** Ollama (`/api/*`) API on port
11435 — point any existing client at it and go. Prefer pipx?
`pipx install squish-ai`.

## Models in this org

Every model below is INT4-quantized (4-bit, group size 64, affine) into MLX
format and ready for `squish run`. Sizes are the actual on-disk download from
this org; raw is the source bf16 checkpoint.

| Model | Run it | Raw (bf16) | Squished | Saved | Context |
|-------|--------|-----------:|---------:|:-----:|--------:|
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/squishai/Qwen2.5-1.5B-Instruct-bf16-squished) | `squish run qwen2.5:1.5b` | 3.1 GB | **0.9 GB** | 72% | 32,768 |
| [Qwen2.5-7B-Instruct](https://huggingface.co/squishai/Qwen2.5-7B-Instruct-bf16-squished) | `squish run qwen2.5:7b` | 15.2 GB | **4.3 GB** | 72% | 32,768 |
| [Qwen3-0.6B](https://huggingface.co/squishai/Qwen3-0.6B-bf16-squished) | `squish run qwen3:0.6b` | 1.2 GB | **0.35 GB** | 71% | 40,960 |
| [Qwen3-4B](https://huggingface.co/squishai/Qwen3-4B-bf16-squished) | `squish run qwen3:4b` | 8.0 GB | **2.3 GB** | 72% | 40,960 |
| [Qwen3-8B](https://huggingface.co/squishai/Qwen3-8B-bf16-squished) | `squish run qwen3:8b` | 16.4 GB | **4.6 GB** | 72% | 40,960 |
| [Llama-3.2-1B-Instruct](https://huggingface.co/squishai/Llama-3.2-1B-Instruct-bf16-squished) | `squish run llama3.2:1b` | 2.5 GB | **0.7 GB** | 72% | 131,072 |
| [Llama-3.2-3B-Instruct](https://huggingface.co/squishai/Llama-3.2-3B-Instruct-bf16-squished) | `squish run llama3.2:3b` | 6.4 GB | **1.8 GB** | 72% | 131,072 |
| [Gemma-3-1B-Instruct](https://huggingface.co/squishai/gemma-3-1b-it-bf16-squished) | `squish run gemma3:1b` | 2.6 GB | **0.8 GB** | 72% | 32,768 |
| [Gemma-3-4B-Instruct](https://huggingface.co/squishai/gemma-3-4b-it-bf16-squished) | `squish run gemma3:4b` | 9.9 GB | **2.6 GB** | 74% | 131,072 |

`squish run <id>` downloads these exact weights and loads them in
**under a second** — no compression wait, no Python environment, no cloud.

## Why Squish

Squish separates how a model's weights are *stored* from how they *run*: stored
compressed and Metal-native, then `mmap`-ed straight into unified memory — no
dtype-conversion pass. It runs as a **persistent daemon** whose two-tier KV
cache reuses prefill across requests instead of re-running it.

Measured on an Apple **M3 (16 GB)**, **Qwen2.5-7B** vs Ollama, thermally
controlled (Squish INT4/INT3 vs Ollama Q4_K_M):

| Metric | Ollama | Squish |
|---|---:|---:|
| Full response @ 4,000-token prompt | 37.5 s | **3.8 s** (9.8× faster) |
| Cold start — load + first token (1.5B) | 20–30 s | **≈ 0.5 s** (54× load) |
| Decode throughput @ 75 tokens | 20.3 tok/s | **24.0 tok/s** (INT3) |
| Repeat-prompt TTFT (KV cache hit) | ~160 ms | **4–11 ms** |
| Peak RAM during inference | 5.14 GB | **3.50 GB** |
| Disk — 7B INT4 / INT3 | 4.36 GB | **4.00 / 3.56 GB** |
| Cold short-prompt TTFT | **167 ms** | 192 ms *(honest loss)* |

The one place Ollama wins is single-token latency on a cold, novel prompt — we
say so plainly. Full methodology and ablations: [BENCHMARKS.md](https://github.com/konjoai/squish/blob/main/BENCHMARKS.md).

## What it doesn't do

Honesty is a feature. If any of these matter, Ollama or LM Studio is the right call:

- **No GPU outside Apple Silicon** — it's MLX-based; CUDA users want vLLM or llama.cpp.
- **No multi-user serving** — one developer, one machine, not a production API.
- **No multimodal** — text only.
- **Slower first token on a cold, short prompt** than Ollama (192 ms vs 167 ms).

## Links

- **GitHub** — [github.com/konjoai/squish](https://github.com/konjoai/squish)
- **Docs** — [squish.run](https://squish.run)
- **Install** — [`squish-ai` on PyPI](https://pypi.org/project/squish-ai/)
- **Benchmarks** — [BENCHMARKS.md](https://github.com/konjoai/squish/blob/main/BENCHMARKS.md)
- **License** — BUSL-1.1 ([LICENSE](https://github.com/konjoai/squish/blob/main/LICENSE))

---

*All models here are pre-squished by [Squish](https://github.com/konjoai/squish) · run any of them in one
command on Apple Silicon.*
