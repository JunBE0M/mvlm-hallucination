#!/usr/bin/env bash
# 실험 A: probe 검증 (p·V vs sdpa 출력 상대오차), 모델별 N=2, 16 각 2문항.  bash logs/verify_A.sh <model> <tag>
cd /workspace/mvlm-hallucination; source /venv/main/bin/activate; export HF_HOME=/workspace/.hf_home
M=$1; T=$2; O=results/novelty/verify; mkdir -p $O
python -m src.attn_probe --model $M --questions data/questions.jsonl --Ns 2 --qtypes forall_no --limit 2 --verify \
  --out $O/verify_$T.csv --npz_dir /tmp/verify_npz_$T 2>&1 | grep -E "verify\]|text_attn|Error"
python -m src.attn_probe --model $M --questions data/questions_bigN.jsonl --Ns 16 --qtypes forall_no --limit 2 --verify \
  --out $O/verify_$T.csv --npz_dir /tmp/verify_npz_$T 2>&1 | grep -E "verify\]|text_attn|Error"
