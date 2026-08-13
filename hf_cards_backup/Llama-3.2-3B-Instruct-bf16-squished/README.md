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
base_model: mlx-community/Llama-3.2-3B-Instruct-bf16
---

# Llama-3.2-3B-Instruct — Squished for Apple Silicon

This is **Llama-3.2-3B-Instruct** (3B parameters) compressed with [Squish](https://github.com/konjoai/squish) — a local inference engine for Apple Silicon.

Weights are INT4-quantized using Squish's ARM NEON-accelerated pipeline and load in under a second on M-series hardware.

## Quick start

```bash
brew tap konjoai/squish
brew install squish
squish pull llama3.2:3b
squish run llama3.2:3b
```

## Model details

| Property | Value |
|----------|-------|
| Parameters | 3B |
| Family | Llama 3.2 |
| Developer | Meta |
| Raw size | 6.4 GB |
| Squished size | 4.3 GB |
| Context window | 128,000 tokens |
| Minimum RAM | 8 GB unified memory |
| Quantization | INT4 (Squish pipeline) |
| Format | MLX-compatible safetensors |

## Use case

Capable Llama model with 128k context. Strong at reasoning and instruction following.

## Requirements

- macOS 13.0 or later
- Apple Silicon (M1, M2, M3, M4, M5)
- 8 GB unified memory minimum

> Intel Macs, Linux, and Windows are not supported.

## How to use with Squish

```bash
# Pull and run
squish pull llama3.2:3b
squish run llama3.2:3b

# OpenAI-compatible API on port 11435
curl http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.2:3b","messages":[{"role":"user","content":"Hello"}]}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11435/v1", api_key="squish")
response = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.choices[0].message.content)
```

## Load with mlx_lm directly

```python
from mlx_lm import load, generate

model, tokenizer = load("squishai/Llama-3.2-3B-Instruct-bf16-squished")
response = generate(model, tokenizer, prompt="Hello", max_tokens=100)
print(response)
```

## Compression details

This model was compressed using Squish's three-tier pipeline:

- **INT4 quantization** via `squish_quant_rs` Rust extension with ARM NEON acceleration
- **Compressed weight loader** — weights decompress directly into Metal-mapped memory at load time
- **KV cache quantization** — attention cache stored at reduced precision during generation

Source weights: [mlx-community/Llama-3.2-3B-Instruct-bf16](https://huggingface.co/mlx-community/Llama-3.2-3B-Instruct-bf16)

## License

The original model weights are subject to the license of the source model (Meta). The compression and tooling are MIT licensed. See [Squish license](https://github.com/konjoai/squish/blob/main/LICENSE) for details.

---

*Pre-compressed by [Konjo AI](https://github.com/konjoai) · [squish.run](https://squish.run)*
