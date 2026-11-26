#!/usr/bin/env bash
# 7B teacher 가 끝나면 B 학습+평가 3 변형을 작업 목록 맨 앞에 넣는다 (스케줄러 우선순위).
cd /workspace/mvlm-hallucination
until grep -q QUEUE_DONE logs/B_qwen7b_tA.log && grep -q QUEUE_DONE logs/B_qwen7b_tB.log; do sleep 60; done
{ head -1 logs/jobs_novelty.txt
  for L in both mse rank; do echo "B_qwen7b_$L|bash logs/queue_B.sh qwen2.5-vl-7b qwen7b train_$L eval_$L"; done
  tail -n +2 logs/jobs_novelty.txt; } > logs/jobs_novelty.tmp && mv logs/jobs_novelty.tmp logs/jobs_novelty.txt
echo "$(date +%H:%M) enqueued B qwen7b"
