#!/usr/bin/env bash
# 실험 기록 3: 재현 확인 (Qwen2.5-VL-7B E1 --limit 200 vs 기존 results/raw/e1_qwen7b.csv)
cd /workspace/mvlm-hallucination; source /venv/main/bin/activate; export HF_HOME=/workspace/.hf_home
CUDA_VISIBLE_DEVICES=${GPU:-0} python -m src.run_eval --model qwen2.5-vl-7b --out results/repro/e1_qwen7b_lim200_rtx4090b.csv --limit 200
