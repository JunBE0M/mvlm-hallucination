#!/usr/bin/env bash
# 실험 기록 3: C, D, E, FOCUS 기준선 큐.  bash logs/queue_novelty.sh <model> <tag> <step...>
#   m2ex : seed0 ∃ 문항 M2 (실험 E 보정 + 실험 B ∃ 비교)
#   D    : 그룹 테스팅 N=8,12,16          C : 일반 양화사 E1 + M1 + M2(any/all 문장)
#   F03 / F10 : FOCUS 노이즈 가리기 λ=0.3 / 1.0 (N=2-8 ∀)
cd /workspace/mvlm-hallucination
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} MKL_NUM_THREADS=${OMP_NUM_THREADS:-6}
M=$1; T=$2; shift 2
R=results/raw
run(){ echo "=== $(date +%H:%M) START $*"; "$@" 2>&1 | grep --line-buffered -vE "Loading weights|Warning|warn"; echo "=== $(date +%H:%M) END rc=${PIPESTATUS[0]}"; }
for s in "$@"; do case $s in
  m2ex) run python -m src.isolated --model $M --qtypes exists_yes exists_no --mode post --out $R/m2ex_post_$T.csv ;;
  D)    run python -m src.group_test --model $M --Ns 8 --out $R/D_group_$T.csv
        run python -m src.group_test --model $M --questions data/questions_bigN.jsonl --out $R/D_group_$T.csv ;;
  C)    run python -m src.run_eval --model $M --questions data/questions_quant.jsonl --out $R/C_e1_$T.csv
        run python -m src.per_image --model $M --questions data/questions_quant.jsonl --out $R/C_m1_$T.csv
        run python -m src.isolated --model $M --questions data/questions_quant.jsonl --qtypes quant_any --mode post --out $R/C_m2_$T.csv ;;
  Call) run python -m src.isolated --model $M --questions data/questions_quant.jsonl --qtypes quant_all --mode post --out $R/C_m2_$T.csv ;;
  F03)  run python -m src.focus_noise --model $M --lam 0.3 --Ns 2 4 6 8 --out $R/F_focus_l0.3_$T.csv ;;
  F10)  run python -m src.focus_noise --model $M --lam 1.0 --Ns 2 4 6 8 --out $R/F_focus_l1.0_$T.csv ;;
esac; done
echo QUEUE_DONE $T "$@"
