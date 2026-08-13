---
language:
- en
library_name: mlx
tags:
- squish
- apple-silicon
- quantized
- int4
- local-llm
- mlx
license: apache-2.0
base_model: mlx-community/Qwen2.5-1.5B-Instruct-bf16
---

# Qwen2.5-1.5B-Instruct, squished for Apple Silicon

This is **Qwen2.5-1.5B-Instruct** (1.5B parameters), compressed with
[Squish](https://github.com/konjoai/squish) into MLX format and ready to run locally on Apple Silicon. The
weights are INT4-quantized. The original model is
[mlx-community/Qwen2.5-1.5B-Instruct-bf16](https://huggingface.co/mlx-community/Qwen2.5-1.5B-Instruct-bf16).

## Run it

```bash
brew tap konjoai/squish
brew install squish
squish run qwen2.5:1.5b
```

`squish run qwen2.5:1.5b` pulls these exact weights
(`squishai/Qwen2.5-1.5B-Instruct-bf16-squished`) and starts a local OpenAI/Ollama-compatible server on
port 11435. That's the whole setup: no cloud, no API keys, fully offline.

## Why run it with Squish

Squish runs MLX models from a **persistent daemon** with a two-tier KV cache that
reuses prefill across requests instead of re-running it, so an agent that resends
the same long prompt every turn pays for it once, not every turn.

These are Squish's published benchmarks (Apple **M3, 16 GB**, **Qwen2.5-7B** vs
Ollama, thermally controlled). They are Squish's headline numbers, not this
model's measured result:

| Metric | Ollama | Squish |
|---|---:|---:|
| Full response @ 4,000-token prompt | 37.5 s | **3.8 s** (up to 9.8× faster) |
| Cold start (load + first token, 1.5B) | 20–30 s | **≈ 0.5 s** |
| Peak RAM during inference | 5.14 GB | **3.50 GB** |
| Disk (7B INT4) | 4.36 GB | **4.00 GB** |

Ollama wins cold single-token TTFT (167 ms vs 192 ms). Full methodology and
ablations: [BENCHMARKS.md](https://github.com/konjoai/squish/blob/main/BENCHMARKS.md).

## This model

| Property | Value |
|----------|-------|
| Base model | [mlx-community/Qwen2.5-1.5B-Instruct-bf16](https://huggingface.co/mlx-community/Qwen2.5-1.5B-Instruct-bf16) |
| Developer | Alibaba Cloud |
| Parameters | 1.5B |
| Quantization | INT4 (4-bit, group size 64, affine) |
| Size on disk | **0.9 GB** squished, from 3.1 GB bf16 (~72% smaller) |
| Context window | 32,768 tokens |
| Format | MLX safetensors |
| Requires | Apple Silicon (M1–M5), macOS 13+ |

## Use it from any client

```bash
# OpenAI-compatible endpoint (port 11435)
curl http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5:1.5b","messages":[{"role":"user","content":"Hello"}]}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11435/v1", api_key="squish")
resp = client.chat.completions.create(
    model="qwen2.5:1.5b",
    messages=[{"role": "user", "content": "Hello"}],
)
print(resp.choices[0].message.content)
```

Or load the weights directly with `mlx_lm`:

```python
from mlx_lm import load, generate

model, tokenizer = load("squishai/Qwen2.5-1.5B-Instruct-bf16-squished")
print(generate(model, tokenizer, prompt="Hello", max_tokens=100))
```

## Links

- **Squish on GitHub**: [github.com/konjoai/squish](https://github.com/konjoai/squish)
- **Docs**: [squish.run](https://squish.run)
- **Install (PyPI)**: [`squish-ai`](https://pypi.org/project/squish-ai/)
- **All pre-squished models**: [huggingface.co/squishai](https://huggingface.co/squishai)

## License

The original model weights are released by Alibaba Cloud under the **Apache 2.0** license; this is a derivative redistribution under that license. The Squish tooling that produced these weights is licensed
**BUSL-1.1** ([LICENSE](https://github.com/konjoai/squish/blob/main/LICENSE)).

---

*Pre-squished by [Squish](https://github.com/konjoai/squish). Run it in one command on Apple Silicon.*
