#!/usr/bin/env bash
set -e
cd /workspace/mvlm-hallucination
source /venv/main/bin/activate
export HF_HOME=/workspace/hf_cache
python -m src.run_eval --model qwen2.5-vl-3b --out results/raw/e1_qwen3b.csv
python -m src.run_eval --model qwen2.5-vl-7b --out results/raw/e1_qwen7b.csv
python -m src.attn_probe --model qwen2.5-vl-3b --out results/raw/e2_verify.csv --limit 2 --verify
python -m src.attn_probe --model qwen2.5-vl-3b --out results/raw/e2_qwen3b.csv --qtypes forall_yes forall_no exists_yes
echo ALL_DONE
