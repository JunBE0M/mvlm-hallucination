"""COCO val2017 annotation으로 다중 이미지 객체 존재 문항(∀/∃)을 생성한다.

사용:
    python -m src.data --config configs/default.yaml --out data/questions.jsonl

문항 유형 (N = 이미지 수):
    forall_yes : N장 모두 present                          -> Yes
    forall_no  : forall_yes 묶음에서 1장만 absent로 교체     -> No   (같은 pair_id)
    exists_yes : 1장 present + N-1장 absent                 -> Yes
    exists_no  : N장 모두 absent                            -> No

면적 비율: 이미지별로 해당 클래스 bbox 중 최대 (w*h) / (W*H). segmentation area가 아니라 bbox 기준.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

QTYPES = ("forall_yes", "forall_no", "exists_yes", "exists_no")


@dataclass
class CocoIndex:
    cat_name_to_id: dict[str, int]
    images: dict[int, dict]                    # img_id -> {"file_name", "width", "height"}
    area_ratio: dict[int, dict[int, float]]    # cat_id -> {img_id: max bbox area ratio}


def load_coco_index(ann_path: str | os.PathLike) -> CocoIndex:
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    images = {
        im["id"]: {"file_name": im["file_name"], "width": im["width"], "height": im["height"]}
        for im in coco["images"]
    }
    cat_name_to_id = {c["name"]: c["id"] for c in coco["categories"]}
    area_ratio: dict[int, dict[int, float]] = defaultdict(dict)
    for ann in coco["annotations"]:
        im = images[ann["image_id"]]
        _, _, w, h = ann["bbox"]
        r = (w * h) / float(im["width"] * im["height"])
        d = area_ratio[ann["category_id"]]
        d[ann["image_id"]] = max(d.get(ann["image_id"], 0.0), r)
    return CocoIndex(cat_name_to_id, images, dict(area_ratio))


def build_pools(index: CocoIndex, cls: str, area_range=None) -> tuple[list[int], list[int]]:
    """(present_ids, absent_ids). present는 area_range=[lo, hi]로 필터 가능 (양끝 포함).
    absent는 해당 클래스 annotation이 전혀 없는 이미지 (area 필터와 무관)."""
    if cls not in index.cat_name_to_id:
        raise KeyError(f"class '{cls}' not in COCO categories")
    ratios = index.area_ratio.get(index.cat_name_to_id[cls], {})
    present = sorted(ratios)
    if area_range is not None:
        lo, hi = area_range
        present = [i for i in present if lo <= ratios[i] <= hi]
    absent = sorted(set(index.images) - set(ratios))
    return present, absent


def parse_position_mode(mode: str) -> int | None:
    """'uniform' -> None, 'fixed:k' -> k."""
    if mode == "uniform":
        return None
    if mode.startswith("fixed:"):
        return int(mode.split(":", 1)[1])
    raise ValueError(f"unknown odd_position mode: {mode}")


def odd_positions(n_items: int, N: int, mode: str, rng: random.Random) -> list[int]:
    """n_items개 문항에 쓸 '홀로 다른 이미지' 위치 목록.
    uniform : 0..N-1 을 차례로 n_items 개 채운 뒤 섞는다. n_items 가 N 의 배수일 때만 정확히 균등하고,
              배수가 아니면 앞쪽 위치가 더 많으며, n_items < N 이면 뒤쪽 위치가 아예 빠진다
              (예: n_items=5, N=8 -> 0..4 만). 칸마다 같은 패턴이라 클래스를 합쳐도 치우침이 남는다.
              기존 진단 세트(configs/*.yaml) 재현을 위해 동작은 그대로 둔다.
    random  : 문항마다 0..N-1 에서 같은 확률로 독립 추출 (COCO 2017 분할 src/splits.py 에서 사용).
    fixed:k : 항상 k (k<0 이면 뒤에서부터)."""
    if mode == "random":
        return [rng.randrange(N) for _ in range(n_items)]
    fixed = parse_position_mode(mode)
    if fixed is None:
        pos = [i % N for i in range(n_items)]   # 셀 안에서 정확히 균등
        rng.shuffle(pos)
        return pos
    if not -N <= fixed < N:
        raise ValueError(f"fixed position {fixed} out of range for N={N}")
    return [fixed % N] * n_items


def _record(index, cls, N, qtype, question, img_ids, cid, seed, idx, pair_id, coco_image_dir):
    ratios = index.area_ratio.get(cid, {})
    present_flags = [i in ratios for i in img_ids]
    return {
        "qid": f"{cls.replace(' ', '_')}_N{N}_{qtype}_{idx:04d}",
        "pair_id": pair_id,
        "cls": cls,
        "N": N,
        "qtype": qtype,
        "question": question,
        "answer": "Yes" if qtype.endswith("_yes") else "No",
        "image_ids": img_ids,
        "image_paths": [f"{coco_image_dir}/{index.images[i]['file_name']}" for i in img_ids],
        "present_positions": [p for p, f in enumerate(present_flags) if f],
        "absent_positions": [p for p, f in enumerate(present_flags) if not f],
        "area_ratios": [round(ratios.get(i, 0.0), 6) for i in img_ids],
        "seed": seed,
    }


def generate_questions(index: CocoIndex, cfg: dict) -> list[dict]:
    seed = cfg["seed"]
    d = cfg["data"]
    image_dir = cfg.get("coco", {}).get("image_dir", "val2017")
    tmpl = d["templates"]
    out: list[dict] = []
    for cls in d["classes"]:
        cid = index.cat_name_to_id[cls]
        present, absent = build_pools(index, cls, d.get("area_range"))
        for N in d["Ns"]:
            # (seed, cls, N)별 독립 RNG: 클래스/N 목록 순서가 바뀌어도 같은 셀은 같은 문항
            rng = random.Random(f"{seed}|{cls}|{N}")
            if len(present) < N or len(absent) < N:
                print(f"[warn] skip cls={cls} N={N}: present={len(present)} absent={len(absent)}")
                continue
            q_all = tmpl["forall"].format(obj=cls, N=N)
            q_any = tmpl["exists"].format(obj=cls, N=N)

            # forall 쌍
            n_pairs = d["n_forall_pairs"]
            for k, p in enumerate(odd_positions(n_pairs, N, d["odd_position"], rng)):
                yes_ids = rng.sample(present, N)
                no_ids = list(yes_ids)
                no_ids[p] = rng.choice(absent)
                pair_id = f"{cls.replace(' ', '_')}_N{N}_pair{k:04d}"
                out.append(_record(index, cls, N, "forall_yes", q_all, yes_ids, cid, seed, k, pair_id, image_dir))
                out.append(_record(index, cls, N, "forall_no", q_all, no_ids, cid, seed, k, pair_id, image_dir))

            # exists_yes: 홀로 다른(present) 이미지 위치도 같은 모드로 제어
            for k, p in enumerate(odd_positions(d["n_exists_yes"], N, d["odd_position"], rng)):
                ids = rng.sample(absent, N - 1)
                ids.insert(p, rng.choice(present))
                out.append(_record(index, cls, N, "exists_yes", q_any, ids, cid, seed, k, None, image_dir))

            # exists_no
            for k in range(d["n_exists_no"]):
                ids = rng.sample(absent, N)
                out.append(_record(index, cls, N, "exists_no", q_any, ids, cid, seed, k, None, image_dir))
    return out


def write_jsonl(records: list[dict], path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_config(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="data/questions.jsonl")
    ap.add_argument("--ann", default=None, help="annotation json 경로 override")
    ap.add_argument("--area_range", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    ap.add_argument("--odd_position", default=None, help="uniform | random | fixed:k")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.area_range is not None:
        cfg["data"]["area_range"] = list(args.area_range)
    if args.odd_position is not None:
        cfg["data"]["odd_position"] = args.odd_position
    if args.seed is not None:
        cfg["seed"] = args.seed
    ann = args.ann or os.path.join(cfg["coco"]["root"], cfg["coco"]["ann_file"])

    index = load_coco_index(ann)
    records = generate_questions(index, cfg)
    write_jsonl(records, args.out)
    counts = defaultdict(int)
    for r in records:
        counts[r["qtype"]] += 1
    print(f"wrote {len(records)} questions to {args.out}: {dict(counts)}")


if __name__ == "__main__":
    main()
