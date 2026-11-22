cd /workspace/mvlm-hallucination
source /venv/main/bin/activate
export HF_HOME=/workspace/hf_cache
run(){ echo "=== $(date +%H:%M) START $*"; "$@" 2>&1 | grep --line-buffered -vE "Loading weights|Warning"; echo "=== $(date +%H:%M) END rc=${PIPESTATUS[0]} $3 $4 $5"; }
M=qwen2.5-vl-7b; T=qwen7b
run python -m src.focus --model $M --m2 results/raw/m2_post_$T.csv --alphas 4 16 --query post_image after_target --out_dir results/raw/m3_$T
run python -m src.intervene --model $M --qtypes forall_no --conditions absent:0 absent:4 present:4 --query after_target --out_dir results/raw/e3_$T
run python -m src.isolated --model $M --qtypes forall_yes forall_no --mode strict --out results/raw/m2_strict_$T.csv
echo QUEUE_DONE
