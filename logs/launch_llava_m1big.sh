#!/usr/bin/env bash
# 2026-11-25 17:35: LLaVA-OV m1big(단일 이미지 anyres, 0.65s/img x ~13k) 를 5개 shard 로 나눠 비는 GPU 에 배치
cd /workspace/mvlm-hallucination
M=llava-onevision-7b; T=llavaov7b
waitpid(){ while kill -0 $1 2>/dev/null; do sleep 30; done; }
waitdone(){ until [ "$(grep -c QUEUE_DONE $1 2>/dev/null)" -ge $2 ] 2>/dev/null; do sleep 60; done; }
sh(){ echo "CUDA_VISIBLE_DEVICES=$1 bash -c 'source /venv/main/bin/activate; export HF_HOME=/workspace/.hf_home OMP_NUM_THREADS=6; python -m src.per_image --model $M --questions data/questions_bigN.jsonl --shard $2/5 --out results/raw/m1_single_big_${T}_s$2.csv'"; }
F="$(declare -f waitpid waitdone)"
# GPU3: per_image(m1) 끝나면 m1agg, m0, 그다음 shard 0
nohup bash -c "$F; waitpid 19262; CUDA_VISIBLE_DEVICES=3 bash logs/queue_gen.sh $M $T m1agg m0; $(sh 3 0)" > logs/gen_${T}_c.log 2>&1 &
# GPU1: Qwen3-VL lens 끝나면 shard 1, 이어서 shard 4
nohup bash -c "$F; waitdone logs/lens_wait_1.log 1; $(sh 1 1); $(sh 1 4)" > logs/gen_${T}_s14.log 2>&1 &
# GPU2: InternVL lens 끝나면 shard 2
nohup bash -c "$F; waitdone logs/lens_wait_2.log 1; $(sh 2 2)" > logs/gen_${T}_s2.log 2>&1 &
# GPU0: LLaVA E3 끝나면 shard 3
nohup bash -c "$F; waitdone logs/gen_${T}_e3.log 1; $(sh 0 3)" > logs/gen_${T}_s3.log 2>&1 &
