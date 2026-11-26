#!/usr/bin/env bash
# 실험 B (실험 기록 3): M2 -> 단일 forward 자기증류.  bash logs/queue_B.sh <model> <tag> <step...>
#   teacherA/teacherB: 학습 문항 M2 (N 2,4,6 / N 8),  train_<loss>: LoRA 학습,  eval_<loss>: held-out 평가
cd /workspace/mvlm-hallucination
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} MKL_NUM_THREADS=${OMP_NUM_THREADS:-6}
M=$1; T=$2; shift 2
R=results/raw
run(){ echo "=== $(date +%H:%M) START $*"; "$@" 2>&1 | grep --line-buffered -vE "Loading weights|Warning|warn"; echo "=== $(date +%H:%M) END rc=${PIPESTATUS[0]}"; }
for s in "$@"; do case $s in
  teacherA) run python -m src.isolated --model $M --questions data/s1_train.jsonl --Ns 2 4 6 --mode post --out $R/B_teacher_${T}_a.csv ;;
  teacherB) run python -m src.isolated --model $M --questions data/s1_train.jsonl --Ns 8 --mode post --out $R/B_teacher_${T}_b.csv ;;
  train_*)  L=${s#train_}; python - <<PY
import pandas as pd, glob
pd.concat([pd.read_csv(f) for f in sorted(glob.glob("$R/B_teacher_${T}_*.csv"))]).to_csv("$R/B_teacher_${T}.csv", index=False)
PY
            run python -m src.distill train --model $M --teacher $R/B_teacher_${T}.csv --loss $L --out_dir ckpt/B_${T}_$L
            mkdir -p results/novelty/B_train; cp ckpt/B_${T}_$L/train_log.csv results/novelty/B_train/${T}_$L.csv
            cp ckpt/B_${T}_$L/train_config.json results/novelty/B_train/${T}_$L.json ;;
  eval_*)   L=${s#eval_}; A=ckpt/B_${T}_$L/${EPOCH:-epoch1}; S=${T}_${L}_${EPOCH:-epoch1}
            run python -m src.run_eval --model $M --adapter $A --questions data/heldout_questions.jsonl --out $R/B_e1_$S.csv
            run python -m src.per_image --model $M --adapter $A --questions data/heldout_questions.jsonl --out $R/B_m1_$S.csv
            run python -m src.run_eval --model $M --adapter $A --questions data/heldout_bigN.jsonl --out $R/B_e1big_$S.csv
            run python -m src.per_image --model $M --adapter $A --questions data/heldout_bigN.jsonl --out $R/B_m1big_$S.csv
            run python -m src.run_eval --model $M --adapter $A --questions data/heldout_var.jsonl --out $R/B_e1var_$S.csv ;;
esac; done
echo QUEUE_DONE $T "$@"
