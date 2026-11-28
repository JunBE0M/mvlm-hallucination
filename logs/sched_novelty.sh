#!/usr/bin/env bash
# 실험 기록 3 스케줄러: logs/jobs_novelty.txt 의 작업("이름|명령")을 순서대로, GPU 마다 동시에 최대 2개
# (8B 모델 3개 이상이면 OOM) + 여유 메모리 >= 22GB 일 때 띄운다. 작업 파일은 매 루프마다 다시 읽으므로 뒤에 추가 가능.
# 명령 앞 접두사:
#   WAIT:<hf_id>;  그 모델 다운로드가 끝날 때까지 미룬다
#   EXCL;          GPU 단독 실행 (실험 B 학습: checkpointing 없이 7B 1문항 활성화 ~25GB). 대기 중인 첫 작업이 EXCL 이면
#                  실행 중 작업이 가장 적은 GPU 를 예약해 새 작업을 넣지 않고, 비면 띄운다.
# 실행 기록: logs/jobs_launched.txt. (2026-11-26 13:10 EXCL 지원 추가 후 재시작)
# 작업 파일과 실행 기록 파일은 환경변수로 바꿀 수 있다 (2026-11-28, coco2017_custom 바닐라: logs/jobs_coco_e1.txt)
#   JOBS=logs/jobs_coco_e1.txt LAUNCHED=logs/run/jobs_coco_e1_launched.txt nohup bash logs/sched_novelty.sh > logs/run/sched_coco_e1.log 2>&1 &
cd /workspace/mvlm-hallucination
J=${JOBS:-logs/jobs_novelty.txt}; L=${LAUNCHED:-logs/jobs_launched.txt}; mkdir -p logs/run; touch $J $L
ready(){ HF_HOME=/workspace/.hf_home /venv/main/bin/python -c "from huggingface_hub import snapshot_download as s; s('$1', local_files_only=True)" >/dev/null 2>&1; }
count(){  # GPU $1 에서 돌고 있는 우리 작업 수 (CUDA_VISIBLE_DEVICES=$1 인 queue_*.sh / verify_A.sh). EXCL 작업은 2 로 센다
  local n=0 p
  for p in $(pgrep -f "bash logs/(queue_|verify_A)"); do
    tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" || continue
    if tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "SCHED_EXCL=1"; then n=$((n+2)); else n=$((n+1)); fi
  done
  echo $n
}
pending(){ while IFS='|' read -r name cmd; do
    [ -z "$name" ] && continue; [[ $name == \#* ]] && continue; grep -qx "$name" $L && continue
    echo "$name|$cmd"; done < $J; }
launch(){ # $1=gpu $2=name $3=cmd $4=excl
  echo "$2" >> $L
  echo "$(date +%H:%M) GPU$1 launch $2${4:+ (EXCL)}: $3"
  CUDA_VISIBLE_DEVICES=$1 SCHED_EXCL=${4:-0} setsid nohup bash -c "$3" > logs/run/$2.log 2>&1 < /dev/null &
}
while true; do
  head=$(pending | head -1)
  reserved=-1
  if [[ ${head#*|} == EXCL\;* ]]; then
    best=9; for g in 0 1 2 3; do c=$(count $g); [ $c -lt $best ] && best=$c && reserved=$g; done
    if [ $best -eq 0 ]; then launch $reserved "${head%%|*}" "${head#*|EXCL;}" 1; sleep 120; continue; fi
  fi
  for g in 0 1 2 3; do
    [ $g -eq $reserved ] && continue
    n=$(count $g); free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g)
    { [ $n -ge 2 ] || [ $free -lt 22000 ]; } && continue
    while IFS='|' read -r name cmd; do
      [[ $cmd == EXCL\;* ]] && continue
      if [[ $cmd == WAIT:* ]]; then hf=${cmd#WAIT:}; hf=${hf%%;*}; cmd=${cmd#*;}; ready $hf || continue; fi
      launch $g "$name" "$cmd"; sleep 120; break
    done < <(pending)
  done
  sleep 30
done
