#!/usr/bin/env bash
# 일반화 실험 큐 (2026-11-25, 4x RTX 4090 48GB).  bash logs/queue_gen.sh <model> <tag> [steps...]
#   steps: e1 m1 m2 m0 var big m1big e3 lens m2big  (기본: e1 m1 m2 m0 var big m1big e3)
cd /workspace/mvlm-hallucination
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} MKL_NUM_THREADS=${OMP_NUM_THREADS:-6}
M=$1; T=$2; shift 2
STEPS=${*:-e1 m1 m2 m0 var big m1big e3}
R=results/raw
run(){ echo "=== $(date +%H:%M) START $*"; "$@" 2>&1 | grep --line-buffered -vE "Loading weights|Warning|warn"; echo "=== $(date +%H:%M) END rc=${PIPESTATUS[0]}"; }
for s in $STEPS; do case $s in
  e1)    run python -m src.run_eval --model $M --out $R/e1_$T.csv ;;
  m1agg) run python -m src.per_image --aggregate --single $R/m1_single_$T.csv --out $R/m1_agg_$T.csv ;;
  m1)    run python -m src.per_image --model $M --out $R/m1_single_$T.csv
         run python -m src.per_image --aggregate --single $R/m1_single_$T.csv --out $R/m1_agg_$T.csv ;;
  m2)    run python -m src.isolated --model $M --qtypes forall_yes forall_no --mode post --out $R/m2_post_$T.csv ;;
  m0)    run python -m src.run_eval --model $M --questions data/questions_neg.jsonl --out $R/m0neg_$T.csv ;;
  var)   run python -m src.run_eval --model $M --questions data/questions_var.jsonl --out $R/e1var_$T.csv ;;
  big)   run python -m src.run_eval --model $M --questions data/questions_bigN.jsonl --out $R/e1big_$T.csv ;;
  m1big) run python -m src.per_image --model $M --questions data/questions_bigN.jsonl --out $R/m1_single_big_$T.csv ;;
  e3)    run python -m src.intervene --model $M --qtypes forall_no --conditions none:1 absent:4 absent:0 present:4 --out_dir $R/e3_$T ;;
  lens)  run python -m src.lens --model $M --out $R/lens_$T.csv ;;
  small) run python -m src.run_eval --model $M --questions data/questions_small.jsonl --out $R/e1small_$T.csv ;;
  m1small) run python -m src.per_image --model $M --questions data/questions_small.jsonl --out $R/m1_single_small_$T.csv ;;
  m2big) run python -m src.isolated --model $M --questions data/questions_bigN.jsonl --qtypes forall_yes forall_no --mode post --out $R/m2_post_big_$T.csv ;;
esac; done
echo QUEUE_DONE $T
