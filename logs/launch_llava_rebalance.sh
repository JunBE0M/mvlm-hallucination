#!/usr/bin/env bash
# 2026-11-25 17:20: LLaVA-OV 남은 단계를 비는 GPU 로 분산.
#   진행 중이던 per_image(pid 19262), e1var(pid 12553) 는 그대로 끝나게 두고 큐 bash 만 종료함.
cd /workspace/mvlm-hallucination
M=llava-onevision-7b; T=llavaov7b
waitpid(){ while kill -0 $1 2>/dev/null; do sleep 30; done; }
waitdone(){ until [ $(grep -c QUEUE_DONE $1 2>/dev/null || echo 0) -ge $2 ]; do sleep 60; done; }
Q="bash logs/queue_gen.sh $M $T"
# GPU1: Qwen3-VL a 큐 끝나면 M2 (N=2,4,6)
nohup bash -c "$(declare -f waitdone); waitdone logs/gen_qwen3vl8b_a.log 1; CUDA_VISIBLE_DEVICES=1 bash -c 'source /venv/main/bin/activate; export HF_HOME=/workspace/.hf_home OMP_NUM_THREADS=6; python -m src.isolated --model $M --qtypes forall_yes forall_no --mode post --Ns 2 4 6 --out results/raw/m2_post_${T}_N246.csv'" > logs/gen_${T}_m2a.log 2>&1 &
# GPU2: InternVL a 큐 끝나면 M2 (N=8)
nohup bash -c "$(declare -f waitdone); waitdone logs/gen_internvl8b_a.log 1; CUDA_VISIBLE_DEVICES=2 bash -c 'source /venv/main/bin/activate; export HF_HOME=/workspace/.hf_home OMP_NUM_THREADS=6; python -m src.isolated --model $M --qtypes forall_yes forall_no --mode post --Ns 8 --out results/raw/m2_post_${T}_N8.csv'" > logs/gen_${T}_m2b.log 2>&1 &
# GPU0: Qwen2.5 a 큐(3B,7B var) 끝나면 E3
nohup bash -c "$(declare -f waitdone); waitdone logs/gen_qwen25_a.log 2; CUDA_VISIBLE_DEVICES=0 $Q e3" > logs/gen_${T}_e3.log 2>&1 &
# GPU3 slot A: per_image 끝나면 aggregate, m0, m1big
nohup bash -c "$(declare -f waitpid); waitpid 19262; CUDA_VISIBLE_DEVICES=3 $Q m1agg m0 m1big" > logs/gen_${T}_c.log 2>&1 &
# GPU3 slot B: e1var 끝나면 big, lens
nohup bash -c "$(declare -f waitpid); waitpid 12553; CUDA_VISIBLE_DEVICES=3 $Q big lens" > logs/gen_${T}_d.log 2>&1 &
