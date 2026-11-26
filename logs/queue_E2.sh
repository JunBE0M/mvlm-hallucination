#!/usr/bin/env bash
# 실험 E 후속 (2026-11-26): ∀ 문항 이미지 묶음에서 M2 격리 판독을 ∃ 문장으로 (N 2-8 먼저, 그다음 12/16).
#   ∀ M2 의 No 쏠림이 "N장 모두?" 문장을 이미지 1장에 묻는 데서 온다는 진단 -> ∃ 문장 판독 + min 이면 임계값이 맞는지.
cd /workspace/mvlm-hallucination; source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
M=$1; T=$2
python -m src.isolated --model $M --questions data/questions_forall_exq.jsonl --Ns 2 4 6 8 --mode post --out results/raw/m2exq_post_$T.csv 2>&1 | grep -vE "Loading weights|Warning|warn"
python -m src.isolated --model $M --questions data/questions_forall_exq.jsonl --Ns 12 16 --mode post --out results/raw/m2exq_post_$T.csv 2>&1 | grep -vE "Loading weights|Warning|warn"
echo QUEUE_DONE $T E2
