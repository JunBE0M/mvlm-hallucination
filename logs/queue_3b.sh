#!/usr/bin/env bash
cd /workspace/mvlm-hallucination
source /venv/main/bin/activate
export HF_HOME=/workspace/hf_cache
run(){ echo "=== $(date +%H:%M) START $*"; "$@" 2>&1 | grep --line-buffered -vE "Loading weights|Warning"; echo "=== $(date +%H:%M) END rc=${PIPESTATUS[0]} $3 $4 $5"; }
run python -m src.attn_probe --model qwen2.5-vl-3b --out results/raw/e2_qwen3b.csv --qtypes forall_yes forall_no exists_yes
run python -m src.intervene --model qwen2.5-vl-3b --qtypes forall_no --conditions none:1 absent:2 absent:4 absent:0 present:2 present:4 --out_dir results/raw/e3_qwen3b
run python -m src.isolated --model qwen2.5-vl-3b --qtypes forall_yes forall_no --mode post --out results/raw/m2_post_qwen3b.csv
run python -m src.focus --model qwen2.5-vl-3b --m2 results/raw/m2_post_qwen3b.csv --alphas 4 16 --query post_image after_target --out_dir results/raw/m3_qwen3b
run python -m src.intervene --model qwen2.5-vl-3b --qtypes forall_no --conditions absent:0 absent:4 present:4 --query after_target --out_dir results/raw/e3_qwen3b
run python -m src.isolated --model qwen2.5-vl-3b --qtypes forall_yes forall_no --mode strict --out results/raw/m2_strict_qwen3b.csv
echo QUEUE_DONE
