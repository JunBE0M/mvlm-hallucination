#!/usr/bin/env bash
# COCO 2017 분할 (docs/split_coco.md): coco2017_seed1·2·3 의 train/val 문항에 쓰는 COCO train2017 이미지만 받는다.
#   bash logs/fetch_train2017.sh      # 파이썬 환경을 켠 상태로 실행. 레포 루트의 train2017/ 에 저장
# 1순위: images.cocodataset.org 에서 파일별 다운로드 (이미 받은 파일은 건너뜀, 8개 병렬, 재시도 3회)
# 2순위: 남은 파일이 있으면 HF detection-datasets/coco train parquet 에서 image_id 로 꺼낸다 (shard 하나씩 받고 지움).
#        이 미러에는 annotation 이 없는 이미지가 빠져 있다(logs/coco_from_hf.sh 참고). 그래도 남으면
#        COCO 서버에 접속되는 곳(예: 로컬 Mac)에서 이 스크립트를 돌려 train2017/ 을 올린다.
set -euo pipefail
cd "$(dirname "$0")/.."
LIST=$(mktemp)
python -m src.splits images --splits train val > "$LIST"
echo "필요한 이미지: $(wc -l < "$LIST" | tr -d ' ')장"
mkdir -p train2017
missing() { while read -r p; do [ -s "$p" ] || echo "$p"; done < "$LIST"; }

missing | sed 's#^train2017/##' | xargs -P 8 -I{} sh -c \
  'wget -q -T 30 -t 3 -O "train2017/{}.part" "http://images.cocodataset.org/train2017/{}" && mv "train2017/{}.part" "train2017/{}" || rm -f "train2017/{}.part"'
N=$(missing | wc -l | tr -d ' ')
echo "COCO 서버 다운로드 후 없는 파일: ${N}장"

if [ "$N" -gt 0 ]; then
  missing > "$LIST.missing"
  python - "$LIST.missing" <<'PY'
import os, sys
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
need = {int(os.path.basename(l.strip()).split(".")[0]): os.path.basename(l.strip()) for l in open(sys.argv[1]) if l.strip()}
files = sorted(f for f in HfApi().list_repo_files("detection-datasets/coco", repo_type="dataset") if f.startswith("data/train"))
for f in files:
    if not need:
        break
    p = hf_hub_download("detection-datasets/coco", f, repo_type="dataset", local_dir="train2017/_hf_tmp")
    for batch in pq.ParquetFile(p).iter_batches(columns=["image_id", "image"], batch_size=256):
        for r in batch.to_pylist():
            name = need.pop(r["image_id"], None)
            if name:
                with open(f"train2017/{name}", "wb") as g:
                    g.write(r["image"]["bytes"])
    os.remove(p)
    print(f"{f}: 남은 파일 {len(need)}장", flush=True)
PY
  rm -rf train2017/_hf_tmp
fi

N=$(missing | wc -l | tr -d ' ')
echo "최종 없는 파일: ${N}장"
rm -f "$LIST" "$LIST.missing"
[ "$N" -eq 0 ]
