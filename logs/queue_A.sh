#!/usr/bin/env bash
# 실험 A (실험 기록 3, 2026-11-26): E2 attention probe, 5 모델 × N∈{2..16} × {forall_yes, forall_no, exists_yes}
# + M0 부정 질문(forall_no) + (추가) exists_no.   bash logs/queue_A.sh <model> <tag> <part...>
#   part: small(N2-8) big(N12,16) neg negbig exno exnobig compact
cd /workspace/mvlm-hallucination
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} MKL_NUM_THREADS=${OMP_NUM_THREADS:-6}
M=$1; T=$2; shift 2
R=results/raw
QT="forall_yes forall_no exists_yes"
run(){ echo "=== $(date +%H:%M) START $*"; "$@" 2>&1 | grep --line-buffered -vE "Loading weights|Warning|warn"; echo "=== $(date +%H:%M) END rc=${PIPESTATUS[0]}"; }
P="python -m src.attn_probe --model $M"
for s in "$@"; do case $s in
  small)   run $P --questions data/questions.jsonl      --qtypes $QT --out $R/e2x_$T.csv ;;
  big)     run $P --questions data/questions_bigN.jsonl --qtypes $QT --out $R/e2x_${T}_big.csv ;;
  neg)     run $P --questions data/questions_neg.jsonl      --qtypes forall_no --out $R/e2xneg_$T.csv ;;
  negbig)  run $P --questions data/questions_neg_bigN.jsonl --qtypes forall_no --out $R/e2xneg_${T}_big.csv ;;
  exno)    run $P --questions data/questions.jsonl      --qtypes exists_no --out $R/e2x_${T}_exno.csv ;;
  exnobig) run $P --questions data/questions_bigN.jsonl --qtypes exists_no --out $R/e2x_${T}_exnobig.csv ;;
  compact) run python -m src.attn_theory compact --tag $T --e2 $(ls $R/e2x_${T}.csv $R/e2x_${T}_*.csv 2>/dev/null) \
               --neg $(ls $R/e2xneg_${T}*.csv 2>/dev/null) ;;
esac; done
echo QUEUE_DONE $T "$@"
