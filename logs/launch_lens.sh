#!/usr/bin/env bash
# 각 GPU 의 b 큐가 끝나면 lens 실행
cd /workspace/mvlm-hallucination
W(){ gpu=$1 log=$2; shift 2  # $1=QUEUE_DONE 개수, $2=명령
     nohup bash -c "until grep -q QUEUE_DONE $log 2>/dev/null && [ \$(grep -c QUEUE_DONE $log) -ge $1 ]; do sleep 60; done; $2" > logs/lens_wait_$gpu.log 2>&1 & }
W 1 logs/gen_qwen3vl8b_b.log 1 "CUDA_VISIBLE_DEVICES=1 bash logs/queue_gen.sh qwen3-vl-8b qwen3vl8b lens"
W 2 logs/gen_internvl8b_b.log 1 "CUDA_VISIBLE_DEVICES=2 bash logs/queue_gen.sh internvl3-8b internvl8b lens"
W 3 logs/gen_llavaov7b_b.log 1 "CUDA_VISIBLE_DEVICES=3 bash logs/queue_gen.sh llava-onevision-7b llavaov7b lens"
W 0 logs/gen_qwen25_b.log 2 "CUDA_VISIBLE_DEVICES=0 bash logs/queue_gen.sh qwen2.5-vl-3b qwen3b lens; CUDA_VISIBLE_DEVICES=0 bash logs/queue_gen.sh qwen2.5-vl-7b qwen7b lens"
