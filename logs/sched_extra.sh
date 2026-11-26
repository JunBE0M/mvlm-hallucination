#!/usr/bin/env bash
# 2026-11-26 03:40 KST: 추가 실험 스케줄러. GPU 여유 메모리 >= 22GB 이면 다음 작업을 그 GPU 에 띄운다.
#   1) M2 큰 N (N=12,16): 해결책이 큰 N 에서도 통하는지   2) 작은 물체(bbox 면적 <= 5%, N=4,8): E1 + M1
cd /workspace/mvlm-hallucination
JOBS=(
 "internvl3-8b internvl8b m2big"
 "qwen3-vl-8b qwen3vl8b m2big"
 "qwen2.5-vl-3b qwen3b m2big"
 "qwen2.5-vl-7b qwen7b m2big"
 "qwen3-vl-8b qwen3vl8b small m1small"
 "internvl3-8b internvl8b small m1small"
 "qwen2.5-vl-3b qwen3b small m1small"
 "qwen2.5-vl-7b qwen7b small m1small"
)
i=0
while [ $i -lt ${#JOBS[@]} ]; do
  for g in 0 1 2 3; do
    [ $i -ge ${#JOBS[@]} ] && break
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    if [ "$used" -lt 26000 ]; then
      set -- ${JOBS[$i]}
      echo "$(date +%H:%M) launch job $i on GPU $g: ${JOBS[$i]}"
      CUDA_VISIBLE_DEVICES=$g setsid nohup bash logs/queue_gen.sh "$@" > logs/extra_${2}_$i.log 2>&1 < /dev/null &
      i=$((i+1)); sleep 180   # 모델 로딩으로 메모리가 잡힐 때까지 대기
    fi
  done
  sleep 60
done
echo "$(date +%H:%M) all extra jobs launched"
