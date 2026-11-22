#!/usr/bin/env bash
cd /workspace/mvlm-hallucination
source /venv/main/bin/activate
export HF_HOME=/workspace/hf_cache
run(){ echo "=== $(date +%H:%M) START $*"; "$@" 2>&1 | grep --line-buffered -vE "Loading weights|Warning"; echo "=== $(date +%H:%M) END rc=${PIPESTATUS[0]} $3 $4 $5"; }
M=qwen2.5-vl-7b; T=qwen7b
run python -m src.per_image --model $M --out results/raw/m1_single_$T.csv
run python -m src.per_image --aggregate --single results/raw/m1_single_$T.csv --out results/raw/m1_agg_$T.csv
run python -m src.run_eval --model $M --questions data/questions_neg.jsonl --out results/raw/m0neg_$T.csv
run python -m src.attn_probe --model qwen2.5-vl-7b --out results/raw/e2_qwen7b.csv --qtypes forall_yes forall_no exists_yes
run python -m src.intervene --model qwen2.5-vl-7b --qtypes forall_no --conditions none:1 absent:2 absent:4 absent:0 present:2 present:4 --out_dir results/raw/e3_qwen7b
run python -m src.isolated --model qwen2.5-vl-7b --qtypes forall_yes forall_no --mode post --out results/raw/m2_post_qwen7b.csv
run python -m src.focus --model qwen2.5-vl-7b --m2 results/raw/m2_post_qwen7b.csv --alphas 4 16 --query post_image after_target --out_dir results/raw/m3_qwen7b
run python -m src.intervene --model qwen2.5-vl-7b --qtypes forall_no --conditions absent:0 absent:4 present:4 --query after_target --out_dir results/raw/e3_qwen7b
run python -m src.isolated --model qwen2.5-vl-7b --qtypes forall_yes forall_no --mode strict --out results/raw/m2_strict_qwen7b.csv
echo QUEUE_DONE
