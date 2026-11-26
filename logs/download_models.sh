#!/usr/bin/env bash
# 실험 기록 3 (2026-11-26): 모델 가중치 병렬 다운로드 (공개 모델, HF_TOKEN 불필요). 끊기면 다시 실행(이어받기).
# Qwen2.5-VL-7B 는 먼저 받음.
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home HF_XET_HIGH_PERFORMANCE=1
for r in Qwen/Qwen2.5-VL-3B-Instruct OpenGVLab/InternVL3-8B-hf Qwen/Qwen3-VL-8B-Instruct llava-hf/llava-onevision-qwen2-7b-ov-hf; do
  ( python -c "from huggingface_hub import snapshot_download as s; print('DONE', s('$r'))" 2>&1 | grep -E "DONE|Error" || echo "FAILED $r" ) &
done
wait; echo DOWNLOAD_ALL_DONE
