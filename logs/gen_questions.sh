#!/usr/bin/env bash
# 실험 기록 3: 문항 재생성 (seed 0) + 학습용 seed 1
cd /workspace/mvlm-hallucination; source /venv/main/bin/activate
python -m src.data --config configs/default.yaml --out data/questions.jsonl
python -m src.make_neg
python -m src.make_variants
python -m src.data --config configs/bigN.yaml  --out data/questions_bigN.jsonl
python -m src.data --config configs/small.yaml --out data/questions_small.jsonl
python -m src.make_neg --questions data/questions_bigN.jsonl --out data/questions_neg_bigN.jsonl
python -m src.data --config configs/train_s1.yaml --out data/questions_s1.jsonl
python -m src.distill split
python -m src.make_variants --quant --out data/questions_quant.jsonl
