---
title: README
emoji: 🔥
colorFrom: blue
colorTo: red
sdk: static
pinned: false
license: mit
---

<div align="center">

<img src="https://raw.githubusercontent.com/konjoai/squish/main/assets/squish-logo-1.png" width="330" alt="Squish" />

<h2>Squeeze the Most Out of Your Models</h2
                                         
<h3>Pre-compressed models for Apple Silicon. Load in under a second. Run fully local.</h3>

[![GitHub](https://img.shields.io/badge/GitHub-konjoai%2Fsquish-black?logo=github)](https://github.com/konjoai/squish)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/konjoai/squish/blob/main/LICENSE)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon%20M1–M5-lightgrey?logo=apple)](https://github.com/konjoai/squish)
[![Website](https://img.shields.io/badge/site-squish.run-blue)](https://squish.run)

</div>

---

## What is this?

This organization hosts models pre-compressed by [Squish](https://github.com/konjoai/squish) — a local inference engine for Apple Silicon that gets models off disk and into memory in under a second.

Every model here was compressed with Squish's INT4 quantization pipeline and is ready to load directly with `squish pull`. No setup, no Python environment, no cloud.

```bash
brew tap konjoai/squish
brew trust konjoai/squish
brew install squish
squish pull qwen3:8b
squish run qwen3:8b
```

---

## Why pre-compressed?

Compression takes time. A Qwen3-8B model compresses in roughly 8 minutes on an M3. You shouldn't have to wait for that on first run. Models in this org are pre-compressed and validated — pull once, load instantly every time after.

| Format | What it means |
|--------|--------------|
| `*-bf16-squished` | INT4-compressed, ready for `squish run` |

---

## Available models

| Model | Squish ID | Raw size | Squished size | Context |
|-------|-----------|----------|---------------|---------|
| [Qwen3-8B](https://huggingface.co/squishai/Qwen3-8B-bf16-squished) | `qwen3:8b` | 16.4 GB | 4.4 GB | 128k |
| [Qwen3-4B](https://huggingface.co/squishai/Qwen3-4B-bf16-squished) | `qwen3:4b` | 8.2 GB | 2.2 GB | 32k |
| [Qwen3-0.6B](https://huggingface.co/squishai/Qwen3-0.6B-bf16-squished) | `qwen3:0.6b` | 1.3 GB | 0.9 GB | 32k |
| [Qwen2.5-7B-Instruct](https://huggingface.co/squishai/Qwen2.5-7B-Instruct-bf16-squished) | `qwen2.5:7b` | 14.4 GB | 3.9 GB | 128k |
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/squishai/Qwen2.5-1.5B-Instruct-bf16-squished) | `qwen2.5:1.5b` | 3.1 GB | 0.9 GB | 32k |
| [Llama-3.2-3B-Instruct](https://huggingface.co/squishai/Llama-3.2-3B-Instruct-bf16-squished) | `llama3.2:3b` | 6.4 GB | 1.7 GB | 128k |
| [Llama-3.2-1B-Instruct](https://huggingface.co/squishai/Llama-3.2-1B-Instruct-bf16-squished) | `llama3.2:1b` | 2.5 GB | 0.7 GB | 128k |
| [Gemma-3-4B-Instruct](https://huggingface.co/squishai/gemma-3-4b-it-bf16-squished) | `gemma3:4b` | 9.8 GB | 2.6 GB | 128k |
| [Gemma-3-1B-Instruct](https://huggingface.co/squishai/gemma-3-1b-it-bf16-squished) | `gemma3:1b` | 2.0 GB | 0.5 GB | 32k |

More models added as the catalog grows. Run `squish catalog` for the full list.

---

## Load time comparison (M3 16GB)

| Model | Squish (INT4) | Ollama | llama.cpp |
|-------|--------------|--------|-----------|
| Qwen3-8B | **0.43s** | 4.2s | 6.1s |
| Llama-3.2-3B | **0.33s** | 1.8s | 2.4s |

*Measured cold-start on Apple M3 16GB. Results will vary by chip and storage.*

---

## OpenAI-compatible API

Squish runs a local server on port 11435. Any OpenAI client works out of the box:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11435/v1", api_key="squish")
response = client.chat.completions.create(
    model="qwen3:8b",
    messages=[{"role": "user", "content": "Explain attention mechanisms briefly."}]
)
print(response.choices[0].message.content)
```

```bash
# Or point your existing tools at it
export OPENAI_BASE_URL=http://localhost:11435/v1
export OPENAI_API_KEY=squish
```

---

## How models are compressed

Squish uses a three-tier compression pipeline:

- **INT4 quantization** via a Rust extension (`squish_quant_rs`) with ARM NEON acceleration — 8–12 GB/s throughput on Apple Silicon
- **Compressed weight loader** — weights stay compressed on disk and decompress directly into Metal-mapped memory at load time
- **KV cache quantization** — attention cache stored at reduced precision during generation, not just weights

The result is a model that fits in memory on a base M3 16GB and loads faster than Ollama can parse its configuration.

---

## Using models directly with mlx_lm

```python
from mlx_lm import load, generate

model, tokenizer = load("squishai/Qwen3-8B-bf16-squished")
response = generate(model, tokenizer, prompt="Hello", max_tokens=100)
```

---

## Requirements

- macOS 13.0 or later
- Apple Silicon (M1, M2, M3, M4, M5)
- Sufficient unified memory for the model (see table above)

> Intel Macs and Linux are not supported. Windows is not planned.

---

## Links

- CLI and inference engine: [github.com/konjoai/squish](https://github.com/konjoai/squish)
- Install: `brew tap konjoai/squish && brew install squish`
- Issues and discussions: [GitHub Issues](https://github.com/konjoai/squish/issues)

---

*Squish it. Run it. Go.*

Built by [Konjo AI](https://github.com/konjoai) &nbsp;·&nbsp; MIT License
