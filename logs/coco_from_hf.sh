#!/usr/bin/env bash
# 2026-11-26: images.cocodataset.org 접속 불가(타임아웃) -> HF Hub 미러에서 COCO val2017 복구.
#  - 이미지 4952장: detection-datasets/coco (val parquet, 원본 JPEG bytes). annotation 없는 48장은 빠져 있음
#  - 나머지 48장: lmms-lab/COCO-Caption2017 (val parquet, 5000장). 두 미러의 bytes 가 300/300 동일함을 확인
#  - annotation: sashankSaya/coco2017 annotations/instances_val2017.json (images 5000, annotations 36781)
#  검증: 재생성한 문항의 image_ids 가 기존 results/raw csv 와 일치, E1 재현 (EXPERIMENTS.md 실험 기록 3)
set -e
R=/workspace/mvlm-hallucination; D=/workspace/coco_dl
source /venv/main/bin/activate; export HF_HOME=/workspace/.hf_home
mkdir -p $D $R/val2017 $R/annotations
python - <<'PY'
from huggingface_hub import hf_hub_download as d, HfApi
D="/workspace/coco_dl"
for f in [f for f in HfApi().list_repo_files("detection-datasets/coco", repo_type="dataset") if f.startswith("data/val")]:
    d("detection-datasets/coco", f, repo_type="dataset", local_dir=D)
for f in ["data/val-00000-of-00002.parquet", "data/val-00001-of-00002.parquet"]:
    d("lmms-lab/COCO-Caption2017", f, repo_type="dataset", local_dir=f"{D}/cap")
d("sashankSaya/coco2017", "annotations/instances_val2017.json", repo_type="dataset", local_dir=D)
PY
cp $D/annotations/instances_val2017.json $R/annotations/
python -I - <<'PY'
import glob, json, os, pyarrow.parquet as pq
R, D = "/workspace/mvlm-hallucination", "/workspace/coco_dl"
names = {im["id"]: im["file_name"] for im in json.load(open(f"{R}/annotations/instances_val2017.json"))["images"]}
for p in sorted(glob.glob(f"{D}/data/val-*.parquet")):
    for r in pq.read_table(p, columns=["image_id", "image"]).to_pylist():
        open(f"{R}/val2017/{names[r['image_id']]}", "wb").write(r["image"]["bytes"])
for p in sorted(glob.glob(f"{D}/cap/data/val*.parquet")):
    for r in pq.read_table(p, columns=["file_name", "image"]).to_pylist():
        if not os.path.exists(f"{R}/val2017/{r['file_name']}"):
            open(f"{R}/val2017/{r['file_name']}", "wb").write(r["image"]["bytes"])
print("val2017:", len(os.listdir(f"{R}/val2017")))
PY
