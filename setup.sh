#!/usr/bin/env bash
# Vast.ai 인스턴스에서 실행:  bash setup.sh
# 비밀값은 코드에 넣지 않는다. 필요하면 실행 전에 환경변수로:
#   export HF_TOKEN=...      (gated 모델일 때만 필요)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COCO_ROOT="${COCO_ROOT:-$REPO_DIR}"          # configs/default.yaml 의 coco.root 와 맞출 것
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
mkdir -p "$HF_HOME" "$COCO_ROOT"

# 다음 셸에서도 HF_HOME 유지
if ! grep -q "export HF_HOME=" ~/.bashrc 2>/dev/null; then
  echo "export HF_HOME=$HF_HOME" >> ~/.bashrc
fi

echo "== pip install =="
python -m pip install --upgrade pip
python -m pip install -r "$REPO_DIR/requirements.txt"

download_and_unzip () {   # $1=url  $2=확인할 디렉터리
  local url="$1" target="$2" zip
  zip="$COCO_ROOT/$(basename "$url")"
  if [ -d "$target" ] && [ -n "$(ls -A "$target" 2>/dev/null)" ]; then
    echo "skip: $target already exists"
    return
  fi
  if [ ! -f "$zip" ]; then
    echo "download: $url"
    wget -q --show-progress -c -O "$zip.part" "$url" && mv "$zip.part" "$zip"
  fi
  echo "unzip: $zip"
  python -m zipfile -e "$zip" "$COCO_ROOT"   # unzip 미설치 환경 대비
  rm -f "$zip"
}

echo "== COCO val2017 =="
download_and_unzip http://images.cocodataset.org/zips/val2017.zip "$COCO_ROOT/val2017"
download_and_unzip http://images.cocodataset.org/annotations/annotations_trainval2017.zip "$COCO_ROOT/annotations"
echo "val2017 images: $(ls "$COCO_ROOT/val2017" | wc -l)  (expected 5000)"

if [ -n "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN detected (value not printed)"
fi

echo "== GPU =="
nvidia-smi || echo "nvidia-smi not available"
python - <<'EOF'
try:
    import torch
    print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
except ImportError:
    print("torch not installed yet")
EOF
