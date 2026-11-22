#!/usr/bin/env bash
# 두 큐가 모두 끝나면(QUEUE_DONE) 분석을 자동 실행
cd /workspace/mvlm-hallucination
until grep -q QUEUE_DONE logs/queue_3b.log && grep -q QUEUE_DONE logs/queue_7b.log; do sleep 60; done
source /venv/main/bin/activate
echo "=== $(date +%H:%M) analyze"
python -m src.analyze --e1 results/raw/e1_qwen3b.csv results/raw/e1_qwen7b.csv \
    --e2 results/raw/e2_qwen3b.csv results/raw/e2_qwen7b.csv --out_dir results 2>&1 | grep -vE "Warning|warn"
echo "=== $(date +%H:%M) compare_methods"
python -m src.compare_methods --out results/summary_methods.csv
echo AUTO_ANALYZE_DONE
