#!/usr/bin/env bash
# GPU0: Qwen2.5-VL 3B/7B 추가 실험. 프로세스 2개까지만 (4개면 OOM)
cd /workspace/mvlm-hallucination
CUDA_VISIBLE_DEVICES=0 nohup bash -c "bash logs/queue_gen.sh qwen2.5-vl-3b qwen3b var; bash logs/queue_gen.sh qwen2.5-vl-7b qwen7b var" > logs/gen_qwen25_a.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup bash -c "bash logs/queue_gen.sh qwen2.5-vl-3b qwen3b big m1big; bash logs/queue_gen.sh qwen2.5-vl-7b qwen7b big m1big" > logs/gen_qwen25_b.log 2>&1 &
