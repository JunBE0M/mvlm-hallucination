#!/usr/bin/env bash
# 일반화 실험 큐 (2026-11-25, 4x RTX 4090 48GB).  bash logs/queue_gen.sh <model> <tag> [steps...]
#   steps: e1 m1 m2 m0 var big m1big e3 lens m2big  (기본: e1 m1 m2 m0 var big m1big e3)
#   COCO 2017 분할 steps: coco_e1 coco_m1 coco_m2 coco_dab coco_dabtune coco_teacher coco_train_<loss> coco_eval_<loss>  (아래 coco 블록)
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
  # ---- COCO 2017 분할 coco2017_seed1·2·3 (docs/split_coco.md, 2026-11-28). 문항 data/splits/coco2017_seed$S/{train,val,test,test_extN}.jsonl
  #   결과 results/raw/coco2017_s${S}_<method>_<split>_$T.csv
  #   DSEEDS: 데이터 seed (기본 1 2 3). seed 마다 train/val/test 가 따로 있고, B 학습 seed 도 같은 값을 쓴다
  #   SPLITS: 평가할 split. 학습 없는 방법(coco_e1/m1/m2)의 기본값은 val test.
  #           선택이 있는 방법(coco_dab, coco_eval_*)의 기본값은 val 뿐이다. seed 마다 val 로 설정을 동결한 뒤 SPLITS=test 로 1회
  coco_e1)  for S in ${DSEEDS:-1 2 3}; do for sp in ${SPLITS:-val test}; do Q=data/splits/coco2017_seed$S/$sp.jsonl
              run python -m src.run_eval --model $M --questions $Q --out $R/coco2017_s${S}_e1_${sp}_$T.csv; done; done ;;
  coco_m1)  for S in ${DSEEDS:-1 2 3}; do for sp in ${SPLITS:-val test}; do Q=data/splits/coco2017_seed$S/$sp.jsonl
              run python -m src.per_image --model $M --questions $Q --out $R/coco2017_s${S}_m1_single_${sp}_$T.csv
              run python -m src.per_image --aggregate --questions $Q --single $R/coco2017_s${S}_m1_single_${sp}_$T.csv --out $R/coco2017_s${S}_m1_agg_${sp}_$T.csv; done; done ;;
  coco_m2)  for S in ${DSEEDS:-1 2 3}; do for sp in ${SPLITS:-val test}; do Q=data/splits/coco2017_seed$S/$sp.jsonl
              run python -m src.isolated --model $M --questions $Q --mode post --out $R/coco2017_s${S}_m2_post_${sp}_$T.csv; done; done ;;
  #   DAB (src/dab.py): ALPHAS 별로 한 파일. 기본은 논문값 α=0.5, base_ratio(BR)=0.2, val 만.
  #   coco_dabtune: seed 마다 val 에서 α 그리드 -> python -m src.dab_eval --splits val 로 seed 별 α 를 고르고 docs/ 에 동결
  #   동결 후 seed 마다: DSEEDS=K SPLITS=test ALPHAS="0.5 <그 seed 에서 고른 값>" bash ... coco_dab
  coco_dab) for S in ${DSEEDS:-1 2 3}; do for a in ${ALPHAS:-0.5}; do for sp in ${SPLITS:-val}; do
              run python -m src.dab --model $M --questions data/splits/coco2017_seed$S/$sp.jsonl --alpha $a --base_ratio ${BR:-0.2} --out $R/coco2017_s${S}_dab_a${a}_${sp}_$T.csv; done; done; done ;;
  coco_dabtune) for S in ${DSEEDS:-1 2 3}; do for a in 0.1 0.25 0.5 0.75 1.0; do
              run python -m src.dab --model $M --questions data/splits/coco2017_seed$S/val.jsonl --alpha $a --base_ratio ${BR:-0.2} --out $R/coco2017_s${S}_dab_a${a}_val_$T.csv; done; done ;;
  #   B: seed 마다 teacher(M2) -> LoRA 학습 1회(학습 seed = S). GPU 별로 seed 를 나눠 띄우면 기다리는 시간은 1회분
  coco_teacher) for S in ${DSEEDS:-1 2 3}; do
              run python -m src.isolated --model $M --questions data/splits/coco2017_seed$S/train.jsonl --mode post --out $R/coco2017_s${S}_teacher_$T.csv; done ;;
  coco_train_*) L=${s#coco_train_}; for S in ${DSEEDS:-1 2 3}; do C=ckpt/coco2017_s${S}_B_${T}_$L
              run python -m src.distill train --model $M --teacher $R/coco2017_s${S}_teacher_$T.csv --train_q data/splits/coco2017_seed$S/train.jsonl --loss $L --seed $S --out_dir $C
              mkdir -p results/coco2017/B_train; cp $C/train_log.csv results/coco2017/B_train/s${S}_${T}_$L.csv; cp $C/train_config.json results/coco2017/B_train/s${S}_${T}_$L.json; done ;;
  coco_eval_*) L=${s#coco_eval_}; for S in ${DSEEDS:-1 2 3}; do A=ckpt/coco2017_s${S}_B_${T}_$L/${EPOCH:-epoch1}
              for sp in ${SPLITS:-val}; do
                run python -m src.run_eval --model $M --adapter $A --questions data/splits/coco2017_seed$S/$sp.jsonl --out $R/coco2017_s${S}_B_e1_${sp}_${T}_$L.csv; done; done ;;
esac; done
echo QUEUE_DONE $T
