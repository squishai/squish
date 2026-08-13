
# Gemma-3-1B-Instruct — Squished for Apple Silicon

This is **Gemma-3-1B-Instruct** (1B parameters) compressed with [Squish](https://github.com/konjoai/squish) — a local inference engine for Apple Silicon.

Weights are INT4-quantized using Squish's ARM NEON-accelerated pipeline and load in under a second on M-series hardware.

## Quick start

```bash
brew tap konjoai/squish
brew install squish
squish pull gemma3:1b
squish run gemma3:1b
```

## Model details

| Property | Value |
|----------|-------|
| Parameters | 1B |
| Family | Gemma 3 |
| Developer | Google DeepMind |
| Raw size | 2.0 GB |
| Squished size | 1.3 GB |
| Context window | 32,768 tokens |
| Minimum RAM | 8 GB unified memory |
| Quantization | INT4 (Squish pipeline) |
| Format | MLX-compatible safetensors |

## Use case

Google's smallest instruction-tuned model. Fast, compact, and capable for its size.

## Requirements

- macOS 13.0 or later
- Apple Silicon (M1, M2, M3, M4, M5)
- 8 GB unified memory minimum

> Intel Macs, Linux, and Windows are not supported.

## How to use with Squish

```bash
# Pull and run
squish pull gemma3:1b
squish run gemma3:1b

# OpenAI-compatible API on port 11435
curl http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma3:1b","messages":[{"role":"user","content":"Hello"}]}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11435/v1", api_key="squish")
response = client.chat.completions.create(
    model="gemma3:1b",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.choices[0].message.content)
```

## Load with mlx_lm directly

```python
from mlx_lm import load, generate

model, tokenizer = load("squishai/gemma-3-1b-it-bf16-squished")
response = generate(model, tokenizer, prompt="Hello", max_tokens=100)
print(response)
```

## Compression details

This model was compressed using Squish's three-tier pipeline:

- **INT4 quantization** via `squish_quant_rs` Rust extension with ARM NEON acceleration
- **Compressed weight loader** — weights decompress directly into Metal-mapped memory at load time
- **KV cache quantization** — attention cache stored at reduced precision during generation

Source weights: [mlx-community/gemma-3-1b-it-bf16](https://huggingface.co/mlx-community/gemma-3-1b-it-bf16)

## License

The original model weights are subject to the license of the source model (Google DeepMind). The compression and tooling are MIT licensed. See [Squish license](https://github.com/konjoai/squish/blob/main/LICENSE) for details.

---

*Pre-compressed by [Konjo AI](https://github.com/konjoai) · [squish.run](https://squish.run)*
