#!/usr/bin/env bash
# 2026-11-25: GPU당 큐 2개 (같은 GPU, 다른 출력 파일) 로 CPU 전처리 병목 완화
cd /workspace/mvlm-hallucination
L(){ CUDA_VISIBLE_DEVICES=$1 nohup bash logs/queue_gen.sh $2 $3 ${@:5} > logs/gen_$3_$4.log 2>&1 & }
L 1 qwen3-vl-8b qwen3vl8b a e1 m1 m2 e3
L 1 qwen3-vl-8b qwen3vl8b b var m0 big m1big
L 2 internvl3-8b internvl8b a e1 m1 m2 e3
L 2 internvl3-8b internvl8b b var m0 big m1big
L 3 llava-onevision-7b llavaov7b a e1 m1 m2 e3
L 3 llava-onevision-7b llavaov7b b var m0 big m1big
L 0 qwen2.5-vl-3b qwen3b a var
L 0 qwen2.5-vl-3b qwen3b b big m1big
L 0 qwen2.5-vl-7b qwen7b a var
L 0 qwen2.5-vl-7b qwen7b b big m1big
