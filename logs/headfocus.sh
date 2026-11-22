cd /workspace/mvlm-hallucination; source /venv/main/bin/activate; export HF_HOME=/workspace/hf_cache
T=$1; M=$2
for k in 5 1; do
  echo "=== $(date +%H:%M) k=$k"
  python -m src.focus --model $M --m2 results/raw/hf/heads_k${k}_$T.csv --alphas 4 16 --query post_image \
     --out_dir results/raw/m5_k${k}_$T 2>&1 | grep -vE "Loading weights|Warning"
done
echo QUEUE_DONE
