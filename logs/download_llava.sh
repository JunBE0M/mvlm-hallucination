#!/usr/bin/env bash
# LLaVA-OV 다운로드 재시도 (첫 시도가 651MB 에서 멈춤). 끝날 때까지 반복.
source /venv/main/bin/activate; export HF_HOME=/workspace/.hf_home HF_XET_HIGH_PERFORMANCE=1
for i in 1 2 3 4 5 6; do
  timeout 1800 python -c "from huggingface_hub import snapshot_download as s; print('DONE', s('llava-hf/llava-onevision-qwen2-7b-ov-hf'))" && break
  echo "retry $i"
done
