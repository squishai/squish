#!/usr/bin/env python3
"""Workstream B driver: dry-run the tone pass on live squishai cards.

Fetches live cards (authoritative "old"), regenerates tone-fixed bodies, verifies
frontmatter is byte-identical to live, and writes preview + backup for diffing.
No pushes here.
"""
import os, re
from huggingface_hub import HfApi, hf_hub_download
import hf_cards_gen as G
import hf_org_card_gen as O

api = HfApi()
os.makedirs("hf_tone_backup", exist_ok=True)
os.makedirs("hf_tone_preview", exist_ok=True)

def split(t):
    m = re.match(r"^---\n(.*?)\n---\n?(.*)$", t, re.DOTALL)
    return (m.group(1), m.group(2)) if m else (None, t)

results = {}
# ---- model cards ----
for repo, d in G.MODELS.items():
    rid = f"squishai/{repo}"
    live = open(hf_hub_download(rid, "README.md", repo_type="model", force_download=True), encoding="utf-8").read()
    os.makedirs(f"hf_tone_backup/{repo}", exist_ok=True)
    open(f"hf_tone_backup/{repo}/README.md", "w", encoding="utf-8").write(live)
    live_fm, live_body = split(live)
    # regenerate
    fm = open(f"hf_cards_backup/{repo}/frontmatter.yaml", encoding="utf-8").read().rstrip("\n")
    fm = G.apply_license_override(repo, fm)
    new_body = G.render_body(d)
    new_full = f"---\n{fm}\n---\n{new_body}"
    os.makedirs(f"hf_tone_preview/{repo}", exist_ok=True)
    open(f"hf_tone_preview/{repo}/README.md", "w", encoding="utf-8").write(new_full)
    results[repo] = dict(fm_ok=(fm == live_fm), old_body=live_body, new_body=new_body)

# ---- org card ----
live = open(hf_hub_download("squishai/README", "README.md", repo_type="space", force_download=True), encoding="utf-8").read()
os.makedirs("hf_tone_backup/_ORG_README_space", exist_ok=True)
open("hf_tone_backup/_ORG_README_space/README.md", "w", encoding="utf-8").write(live)
live_fm, live_body = split(live)
new_full = O.render()
new_fm, new_body = split(new_full)
os.makedirs("hf_tone_preview/_ORG_README_space", exist_ok=True)
open("hf_tone_preview/_ORG_README_space/README.md", "w", encoding="utf-8").write(new_full)
results["_ORG_README_space"] = dict(fm_ok=(new_fm == live_fm), old_body=live_body, new_body=new_body, live_fm=live_fm, new_fm=new_fm)

# ---- report ----
print(f"{'repo':44} {'frontmatter==live':18} {'em(new body)':12} {'we/eng(new)':12}")
for repo, r in results.items():
    nb = r["new_body"]
    prose = "\n".join(l for l in nb.splitlines())
    # crude prose scan (strip fences)
    body_nofence = re.sub(r"```.*?```", "", nb, flags=re.DOTALL)
    em = nb.count("—")
    we = len(re.findall(r"\b(we|our|us)\b", body_nofence, re.I))
    eng = len(re.findall(r"\b(engine|engines)\b", body_nofence, re.I))
    print(f"{repo:44} {str(r['fm_ok']):18} {em:<12} {f'{we}/{eng}':12}")

# Save org frontmatter mismatch detail if any
org = results["_ORG_README_space"]
if not org["fm_ok"]:
    print("\n!! ORG frontmatter differs from live:")
    print("--- live ---\n", org["live_fm"])
    print("--- new  ---\n", org["new_fm"])
