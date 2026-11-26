#!/usr/bin/env bash
# 실험 기록 3 스케줄러: logs/jobs_novelty.txt 의 작업("이름|명령")을 순서대로, GPU 마다 동시에 최대 2개
# (8B 모델 3개 이상이면 OOM) + 여유 메모리 >= 22GB 일 때 띄운다. 작업 파일은 매 루프마다 다시 읽으므로 뒤에 추가 가능.
# 명령 앞에 "WAIT:<hf_id>;" 가 있으면 그 모델 다운로드가 끝날 때까지 미룬다.  실행 기록: logs/jobs_launched.txt
cd /workspace/mvlm-hallucination
J=logs/jobs_novelty.txt; L=logs/jobs_launched.txt; touch $J $L; mkdir -p logs/run
declare -A PIDS
ready(){ HF_HOME=/workspace/.hf_home /venv/main/bin/python -c "from huggingface_hub import snapshot_download as s; s('$1', local_files_only=True)" >/dev/null 2>&1; }
while true; do
  for g in 0 1 2 3; do
    n=0; for p in ${PIDS[$g]}; do kill -0 $p 2>/dev/null && n=$((n+1)); done
    # 다른 경로로 띄운 프로세스도 센다 (CUDA_VISIBLE_DEVICES=g 인 queue_*.sh)
    ext=$(for p in $(pgrep -f "bash logs/queue_"); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$g" && echo $p; done | wc -l)
    [ $ext -gt $n ] && n=$ext
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g)
    [ $n -ge 2 ] || [ $free -lt 22000 ] && continue
    while IFS='|' read -r name cmd; do
      [ -z "$name" ] && continue; [[ $name == \#* ]] && continue
      grep -qx "$name" $L && continue
      if [[ $cmd == WAIT:* ]]; then hf=${cmd#WAIT:}; hf=${hf%%;*}; cmd=${cmd#*;}; ready $hf || continue; fi
      echo "$name" >> $L
      echo "$(date +%H:%M) GPU$g launch $name: $cmd"
      CUDA_VISIBLE_DEVICES=$g setsid nohup bash -c "$cmd" > logs/run/$name.log 2>&1 < /dev/null &
      PIDS[$g]="${PIDS[$g]} $!"; sleep 120; break
    done < $J
  done
  sleep 30
done
