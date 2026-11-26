#!/usr/bin/env bash
# B 확장: teacher 작업(Bt_<tag>)이 끝나면 학습+평가(EXCL, loss=both)를 작업 목록 맨 앞에 넣는다.  bash logs/enqueue_B_ext.sh <model> <tag>
cd /workspace/mvlm-hallucination; M=$1; T=$2
until grep -q QUEUE_DONE logs/run/Bt_$T.log 2>/dev/null; do sleep 60; done
{ head -1 logs/jobs_novelty.txt; echo "B_${T}_both|EXCL;bash logs/queue_B.sh $M $T train_both eval_both"
  tail -n +2 logs/jobs_novelty.txt; } > logs/jobs_novelty.tmp && mv logs/jobs_novelty.tmp logs/jobs_novelty.txt
echo "$(date +%H:%M) enqueued B $T"
