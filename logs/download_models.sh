#!/usr/bin/env bash
# 실험 기록 3 (2026-11-26): 5개 모델 가중치 다운로드 (HF_TOKEN 불필요, 공개 모델)
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home HF_HUB_ENABLE_HF_TRANSFER=1
for r in Qwen/Qwen2.5-VL-7B-Instruct Qwen/Qwen2.5-VL-3B-Instruct OpenGVLab/InternVL3-8B-hf Qwen/Qwen3-VL-8B-Instruct llava-hf/llava-onevision-qwen2-7b-ov-hf; do
  echo "== $(date +%H:%M) $r"
  python -c "from huggingface_hub import snapshot_download as s; print(s('$r'))" || echo "FAILED $r"
done
echo DOWNLOAD_DONE
