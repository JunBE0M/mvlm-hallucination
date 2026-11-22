#!/usr/bin/env bash
set -e
cd /workspace/mvlm-hallucination
source /venv/main/bin/activate
export HF_HOME=/workspace/hf_cache
python -m src.attn_probe --model qwen2.5-vl-7b --out results/raw/e2_qwen7b.csv --qtypes forall_yes forall_no exists_yes
python -m src.run_eval --model qwen2.5-vl-3b --image_labels --out results/raw/e1lab_qwen3b.csv
python -m src.run_eval --model qwen2.5-vl-7b --image_labels --out results/raw/e1lab_qwen7b.csv
echo ALL_DONE
