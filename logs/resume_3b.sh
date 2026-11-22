cd /workspace/mvlm-hallucination
source /venv/main/bin/activate
export HF_HOME=/workspace/hf_cache
run(){ echo "=== $(date +%H:%M) START $*"; "$@" 2>&1 | grep --line-buffered -vE "Loading weights|Warning"; echo "=== $(date +%H:%M) END rc=${PIPESTATUS[0]} $3 $4 $5"; }
run python -m src.isolated --model qwen2.5-vl-3b --qtypes forall_yes forall_no --mode strict --out results/raw/m2_strict_qwen3b.csv
echo QUEUE_DONE
