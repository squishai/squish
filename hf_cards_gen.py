#!/usr/bin/env python3
"""Generate Squish-funnel model-card bodies for the squishai HF org.

Frontmatter is NEVER generated here — it is preserved byte-for-byte from the
live card (see split_frontmatter / regenerate). This module only produces the
human-readable body below the YAML, from grounded per-model data.

All benchmark numbers trace to BENCHMARKS.md / README.md of the squish repo.
All sizes/quant/context trace to the live HF repo config + file metadata.
"""

# ---- canonical URLs (verified against squish/README.md) ----
GH = "https://github.com/konjoai/squish"
DOCS = "https://squish.run"
PYPI = "https://pypi.org/project/squish-ai/"
ORG = "https://huggingface.co/squishai"
BENCH = "https://github.com/konjoai/squish/blob/main/BENCHMARKS.md"
LICENSE_URL = "https://github.com/konjoai/squish/blob/main/LICENSE"

# Shared "Why Squish" block. Framed as Squish's measured performance on
# Qwen2.5-7B (the one benchmarked model), NOT as this model's result.
WHY_SQUISH = f"""## Why run it with Squish

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
ablations: [BENCHMARKS.md]({BENCH}).
"""


def render_body(m: dict) -> str:
    """Render the funnel body for one model from its grounded data dict."""
    ctx_row = (
        f"| Context window | {m['ctx']} tokens |\n" if m.get("ctx") else ""
    )
    base_id = m["base"]
    return f"""
# {m['display']}, squished for Apple Silicon

This is **{m['display']}** ({m['params']} parameters), compressed with
[Squish]({GH}) into MLX format and ready to run locally on Apple Silicon. The
weights are INT4-quantized. The original model is
[{base_id}](https://huggingface.co/{base_id}).

## Run it

```bash
brew tap konjoai/squish
brew install squish
squish run {m['shorthand']}
```

`squish run {m['shorthand']}` pulls these exact weights
(`squishai/{m['repo']}`) and starts a local OpenAI/Ollama-compatible server on
port 11435. That's the whole setup: no cloud, no API keys, fully offline.

{WHY_SQUISH}
## This model

| Property | Value |
|----------|-------|
| Base model | [{base_id}](https://huggingface.co/{base_id}) |
| Developer | {m['developer']} |
| Parameters | {m['params']} |
| Quantization | INT4 (4-bit, group size 64, affine) |
| Size on disk | **{m['sq_gb']} GB** squished, from {m['raw_gb']} GB bf16 (~{m['saved']} smaller) |
{ctx_row}| Format | MLX safetensors |
| Requires | Apple Silicon (M1–M5), macOS 13+ |

## Use it from any client

```bash
# OpenAI-compatible endpoint (port 11435)
curl http://localhost:11435/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{"model":"{m['shorthand']}","messages":[{{"role":"user","content":"Hello"}}]}}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11435/v1", api_key="squish")
resp = client.chat.completions.create(
    model="{m['shorthand']}",
    messages=[{{"role": "user", "content": "Hello"}}],
)
print(resp.choices[0].message.content)
```

Or load the weights directly with `mlx_lm`:

```python
from mlx_lm import load, generate

model, tokenizer = load("squishai/{m['repo']}")
print(generate(model, tokenizer, prompt="Hello", max_tokens=100))
```

## Links

- **Squish on GitHub**: [{GH.split('//')[1]}]({GH})
- **Docs**: [squish.run]({DOCS})
- **Install (PyPI)**: [`squish-ai`]({PYPI})
- **All pre-squished models**: [huggingface.co/squishai]({ORG})

## License

{m['license_body']} The Squish tooling that produced these weights is licensed
**BUSL-1.1** ([LICENSE]({LICENSE_URL})).

---

*Pre-squished by [Squish]({GH}). Run it in one command on Apple Silicon.*
"""


# ---- grounded per-model data ----
APACHE = (
    "The original model weights are released by {dev} under the **Apache 2.0** "
    "license; this is a derivative redistribution under that license."
)
LLAMA = (
    "The original model weights are released by Meta under the **Llama 3.2 "
    "Community License**. Review Meta's terms before redistribution or "
    "commercial use."
)
GEMMA = (
    "The original model weights are released by Google under the **Gemma Terms "
    "of Use**. Review Google's terms before redistribution or commercial use."
)

MODELS = {
    "Qwen2.5-7B-Instruct-bf16-squished": dict(
        display="Qwen2.5-7B-Instruct", params="7B",
        base="mlx-community/Qwen2.5-7B-Instruct-bf16",
        developer="Alibaba Cloud", shorthand="qwen2.5:7b",
        raw_gb="15.2", sq_gb="4.3", saved="72%", ctx="32,768",
        license_body=APACHE.format(dev="Alibaba Cloud"),
    ),
    "Qwen2.5-1.5B-Instruct-bf16-squished": dict(
        display="Qwen2.5-1.5B-Instruct", params="1.5B",
        base="mlx-community/Qwen2.5-1.5B-Instruct-bf16",
        developer="Alibaba Cloud", shorthand="qwen2.5:1.5b",
        raw_gb="3.1", sq_gb="0.9", saved="72%", ctx="32,768",
        license_body=APACHE.format(dev="Alibaba Cloud"),
    ),
    "Qwen3-0.6B-bf16-squished": dict(
        display="Qwen3-0.6B", params="0.6B",
        base="mlx-community/Qwen3-0.6B-bf16",
        developer="Alibaba Cloud", shorthand="qwen3:0.6b",
        raw_gb="1.2", sq_gb="0.35", saved="71%", ctx="40,960",
        license_body=APACHE.format(dev="Alibaba Cloud"),
    ),
    "Qwen3-4B-bf16-squished": dict(
        display="Qwen3-4B", params="4B",
        base="mlx-community/Qwen3-4B-bf16",
        developer="Alibaba Cloud", shorthand="qwen3:4b",
        raw_gb="8.0", sq_gb="2.3", saved="72%", ctx="40,960",
        license_body=APACHE.format(dev="Alibaba Cloud"),
    ),
    "Qwen3-8B-bf16-squished": dict(
        display="Qwen3-8B", params="8B",
        base="mlx-community/Qwen3-8B-bf16",
        developer="Alibaba Cloud", shorthand="qwen3:8b",
        raw_gb="16.4", sq_gb="4.6", saved="72%", ctx="40,960",
        license_body=APACHE.format(dev="Alibaba Cloud"),
    ),
    "Llama-3.2-1B-Instruct-bf16-squished": dict(
        display="Llama-3.2-1B-Instruct", params="1B",
        base="mlx-community/Llama-3.2-1B-Instruct-bf16",
        developer="Meta", shorthand="llama3.2:1b",
        raw_gb="2.5", sq_gb="0.7", saved="72%", ctx="131,072",
        license_body=LLAMA,
    ),
    "Llama-3.2-3B-Instruct-bf16-squished": dict(
        display="Llama-3.2-3B-Instruct", params="3B",
        base="mlx-community/Llama-3.2-3B-Instruct-bf16",
        developer="Meta", shorthand="llama3.2:3b",
        raw_gb="6.4", sq_gb="1.8", saved="72%", ctx="131,072",
        license_body=LLAMA,
    ),
    "gemma-3-1b-it-bf16-squished": dict(
        display="Gemma-3-1B-Instruct", params="1B",
        base="mlx-community/gemma-3-1b-it-bf16",
        developer="Google DeepMind", shorthand="gemma3:1b",
        raw_gb="2.6", sq_gb="0.8", saved="72%", ctx="32,768",
        license_body=GEMMA,
    ),
    "gemma-3-4b-it-bf16-squished": dict(
        display="Gemma-3-4B-Instruct", params="4B",
        base="mlx-community/gemma-3-4b-it-bf16",
        developer="Google DeepMind", shorthand="gemma3:4b",
        # ctx not in squished/mlx config (max_position_embeddings stripped by MLX
        # conversion); 131,072 from upstream google/gemma-3-4b-it config
        # (via non-gated unsloth mirror) and squish catalog.py.
        raw_gb="9.9", sq_gb="2.6", saved="74%", ctx="131,072",
        license_body=GEMMA,
    ),
}
for _repo, _d in MODELS.items():
    _d["repo"] = _repo

# Frontmatter license correction: the original cards declared apache-2.0 for ALL
# repos, which is wrong for Meta/Google community-licensed weights. Override the
# `license:` field for these (Qwen repos stay apache-2.0). Applied to the
# preserved frontmatter so regeneration is idempotent.
LICENSE_OVERRIDE = {
    "Llama-3.2-1B-Instruct-bf16-squished": "llama3.2",
    "Llama-3.2-3B-Instruct-bf16-squished": "llama3.2",
    "gemma-3-1b-it-bf16-squished": "gemma",
    "gemma-3-4b-it-bf16-squished": "gemma",
}


def apply_license_override(repo: str, fm: str) -> str:
    """Replace only the `license:` value in the frontmatter; leave all else."""
    import re as _re
    new = LICENSE_OVERRIDE.get(repo)
    if not new:
        return fm
    return _re.sub(r"(?m)^license:\s*.*$", f"license: {new}", fm)


if __name__ == "__main__":
    import sys, os
    only = sys.argv[1] if len(sys.argv) > 1 else None
    os.makedirs("hf_cards_preview", exist_ok=True)
    for repo, d in MODELS.items():
        if only and only not in repo:
            continue
        body = render_body(d)
        # read preserved frontmatter from backup
        fm = open(f"hf_cards_backup/{repo}/frontmatter.yaml", encoding="utf-8").read().rstrip("\n")
        fm = apply_license_override(repo, fm)
        full = f"---\n{fm}\n---\n{body}"
        os.makedirs(f"hf_cards_preview/{repo}", exist_ok=True)
        open(f"hf_cards_preview/{repo}/README.md", "w", encoding="utf-8").write(full)
        print("wrote", f"hf_cards_preview/{repo}/README.md", f"({len(full)} bytes)")
