"""COCO 2017 데이터 분할 (docs/split_coco.md): seed 마다 train / val / test 를 통째로 따로 뽑은 3벌 (coco2017_seed1·2·3).

    python -m src.splits build  [--seeds 1 2 3]                         # 이미지 풀 선택 + 문항 생성 + manifest
    python -m src.splits check  [--seeds 1 2 3]                         # sha256, 문항 수, 칸 균등, 중복, 이미지 분리 검증
    python -m src.splits images [--seeds 1 2 3] [--splits train val]    # 필요한 이미지 경로 목록 (다운로드용)

출력: data/splits/coco2017_seed{K}/ 에 manifest.json, train.jsonl, val.jsonl, test.jsonl, test_extN.jsonl
이미지 출처는 COCO 2017 (train2017 / val2017) 뿐이다. COCO 2014 는 출처로 쓰지 않는다
(2017 은 2014 와 같은 사진을 다시 나눈 것이라 이미지 id 가 같다. 아래 MIHBench 제외에서만 2014 파일명을 id 로 바꿔 쓴다).
seed K 하나 안에서 (문항 생성 규칙은 src/data.py 의 generate_questions 를 그대로 쓴다):
    train      COCO train2017 풀 8,000장 (클래스당 present >= 240), N 2·4·6·8, 칸당 유형별 10 -> 3,200
    val        COCO train2017 풀 3,000장 (같은 seed 의 train 과 분리, 클래스당 present >= 120), N 2·4·6·8, 칸당 5 -> 1,600
    test       COCO val2017 전체, N 2·4·6·8, 칸당 5 -> 1,600
    test_extN  COCO val2017 전체, N 10·12·14, 칸당 5 -> 1,200 (지금 생성·고정, 실행은 나중)
  - 칸 = (클래스, N). 유형 = forall_yes / forall_no / exists_yes / exists_no. 클래스는 configs/default.yaml 의 20개.
  - 이미지 풀 선택과 문항 생성의 난수는 모두 seed K 에서 나온다 (문항 seed 문자열: "coco2017-s{K}-{split}").
    seed 가 다르면 풀과 문항이 모두 새로 뽑힌다 (독립 반복). B(LoRA) 학습 seed 도 K 로 기록한다.
  - generate_questions 는 문항마다 풀에서 무작위로 뽑으므로 split 안에서 같은 이미지가 다른 조합으로 다시 쓰일 수 있다.
    split 사이 이미지 중복과 bundle 중복은 없다.
  - MIHBench(github.com/pgtrece/DAB) 문항이 쓰는 COCO val2014 이미지(1,908장: train2017 에 1,663장, val2017 에 245장)는
    모든 split 에서 뺀다. train/val 은 누수 방지, test 는 우리 test 와 MIHBench 를 서로 독립된 평가로 두기 위해서.
  - bundle(같은 클래스·유형의 같은 이미지 조합)은 seed 하나 안에서 한 번만 나온다. 중복이 생긴 (클래스, N) 칸만
    seed 문자열에 접미사("r1" 등)를 붙여 다시 뽑고 manifest 에 기록한다.
  - qid / pair_id 앞에 "coco2017-s{K}-{split}-" 을 붙인다.
  - test-unseen-class 는 선택 규칙과 그 규칙으로 고른 클래스만 manifest 에 기록한다 (seed 와 무관하게 같은 5개, 문항은 나중에).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from .data import QTYPES, CocoIndex, generate_questions, load_coco_index, load_config, read_jsonl, write_jsonl

NAME = "coco2017"
SEEDS = [1, 2, 3]
OUT_ROOT = Path("data/splits")
CLASS_CONFIG = "configs/default.yaml"          # 메인 20개 클래스와 질문 문장을 여기서 가져온다
MIH_DIR = "bench/mihbench/DAB/Questions"
SPLITS = ("train", "val", "test", "test_extN")

POOLS = {   # train2017 에서 이 순서로 뽑는다 (val 은 train 과 겹치지 않게)
    "train": {"size": 8000, "min_present": 240},
    "val": {"size": 3000, "min_present": 120},
}
NS_MAIN = [2, 4, 6, 8]
NS_EXT = [10, 12, 14]
UNSEEN = {"n": 5, "seed": f"{NAME}-unseen"}

MIH_RE = re.compile(r"COCO_val2014_(\d{12})\.jpg")


def seed_dir(seed: int, root=OUT_ROOT) -> Path:
    return Path(root) / f"{NAME}_seed{seed}"


def split_specs(seed: int) -> dict[str, dict]:
    return {
        "train": dict(source="train2017", pool="train", Ns=NS_MAIN, n_per_type=10,
                      role="method development / training"),
        "val": dict(source="train2017", pool="val", Ns=NS_MAIN, n_per_type=5, role="parameter selection only"),
        "test": dict(source="val2017", pool=None, Ns=NS_MAIN, n_per_type=5,
                     role="final evaluation (frozen settings, run once)"),
        "test_extN": dict(source="val2017", pool=None, Ns=NS_EXT, n_per_type=5,
                          role="N extension (generated now, run later)"),
    }


# ------------------------------------------------------------------ 이미지 풀
def mihbench_coco_ids(qdir: str | os.PathLike) -> set[int]:
    """MIHBench 질문 파일(image_list)에 나오는 COCO 이미지 id. 파일명은 COCO_val2014_<12자리 id>.jpg."""
    files = sorted(glob.glob(os.path.join(str(qdir), "**", "*.jsonl"), recursive=True))
    if not files:
        raise FileNotFoundError(f"MIHBench 질문 파일이 없음: {qdir} (git clone https://github.com/pgtrece/DAB)")
    ids: set[int] = set()
    for f in files:
        for q in read_jsonl(f):
            for name in q.get("image_list", []):
                m = MIH_RE.search(name)
                if m:
                    ids.add(int(m.group(1)))
    return ids


def restrict_index(index: CocoIndex, keep) -> CocoIndex:
    """풀 이미지만 남긴 인덱스. generate_questions 의 present/absent 풀이 자동으로 풀 안으로 제한된다."""
    keep = set(keep)
    return CocoIndex(
        index.cat_name_to_id,
        {i: v for i, v in index.images.items() if i in keep},
        {c: {i: r for i, r in d.items() if i in keep} for c, d in index.area_ratio.items()},
    )


def select_pools(index: CocoIndex, classes, exclude, pools=POOLS, rng_seed: str = "pools") -> dict[str, list[int]]:
    """pools 순서대로 서로 겹치지 않는 이미지 풀을 고른다.
    클래스마다 present 를 min_present 장 확보한 뒤(드문 클래스부터) 나머지를 무작위 이미지로 size 장까지 채운다."""
    rng = random.Random(rng_seed)
    order = sorted(set(index.images) - set(exclude))
    rng.shuffle(order)
    present = {c: set(index.area_ratio.get(index.cat_name_to_id[c], {})) for c in classes}
    taken: set[int] = set()
    out: dict[str, list[int]] = {}
    for name, p in pools.items():
        chosen: list[int] = []
        chosen_set: set[int] = set()
        for c in sorted(classes, key=lambda c: (len(present[c]), c)):
            need = p["min_present"] - len(present[c] & chosen_set)
            if need <= 0:
                continue
            cand = [i for i in order if i in present[c] and i not in taken and i not in chosen_set]
            if len(cand) < need:
                raise ValueError(f"pool {name}: class {c!r} needs {need} more present images, only {len(cand)} left")
            chosen += cand[:need]
            chosen_set.update(cand[:need])
        if len(chosen) > p["size"]:
            raise ValueError(f"pool {name}: present quotas need {len(chosen)} images > size {p['size']}")
        for i in order:
            if len(chosen) >= p["size"]:
                break
            if i not in taken and i not in chosen_set:
                chosen.append(i)
                chosen_set.add(i)
        if len(chosen) < p["size"]:
            raise ValueError(f"pool {name}: only {len(chosen)} images available, need {p['size']}")
        out[name] = sorted(chosen)
        taken |= chosen_set
    return out


# ------------------------------------------------------------------ 문항
def make_cfg(classes, templates, Ns, n_per_type, seed, image_dir) -> dict:
    """generate_questions 에 넘길 설정 (configs/*.yaml 과 같은 구조). seed 는 정수나 문자열."""
    return {
        "seed": seed,
        "coco": {"image_dir": image_dir},
        "data": {"classes": list(classes), "Ns": list(Ns), "n_forall_pairs": n_per_type,
                 "n_exists_yes": n_per_type, "n_exists_no": n_per_type, "odd_position": "uniform",
                 "area_range": None, "templates": templates},
    }


def bundle_key(q: dict) -> tuple:
    return q["cls"], q["qtype"], tuple(sorted(q["image_ids"]))


def generate_split(index: CocoIndex, cfg: dict, seen: set, max_retry: int = 20) -> tuple[list[dict], list[dict]]:
    """generate_questions + bundle 중복 제거.
    seen: 앞선 split 에서 이미 쓴 bundle. 중복이 있는 (cls, N) 칸만 seed 에 접미사를 붙여 그 칸만 다시 뽑는다.
    반환: (문항, 다시 뽑은 칸 목록)."""
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for q in generate_questions(index, cfg):
        by_cell[(q["cls"], q["N"])].append(q)
    out, regenerated, local = [], [], set()
    for (cls, N), qs in by_cell.items():
        k = 0
        while True:
            keys = [bundle_key(q) for q in qs]
            if len(set(keys)) == len(keys) and not (set(keys) & (seen | local)):
                break
            k += 1
            if k > max_retry:
                raise RuntimeError(f"bundle 중복을 없애지 못함: cls={cls} N={N} (풀이 너무 작음)")
            sub = dict(cfg, seed=f"{cfg['seed']}r{k}", data=dict(cfg["data"], classes=[cls], Ns=[N]))
            qs = generate_questions(index, sub)
        if k:
            regenerated.append({"cls": cls, "N": N, "seed": f"{cfg['seed']}r{k}"})
        local.update(bundle_key(q) for q in qs)
        out.extend(qs)
    return out, regenerated


def add_prefix(recs: list[dict], seed: int, split: str) -> list[dict]:
    pre = f"{NAME}-s{seed}-{split}-"
    for q in recs:
        q["qid"] = pre + q["qid"]
        if q.get("pair_id"):
            q["pair_id"] = pre + q["pair_id"]
        q["dataset"] = f"{NAME}_seed{seed}"
        q["split"] = split
    return recs


# ------------------------------------------------------------------ test-unseen-class 선택 규칙
def unseen_classes(val_index: CocoIndex, main_classes, n=UNSEEN["n"], seed=UNSEEN["seed"]) -> dict:
    """자격: 메인 클래스가 아닌 COCO 클래스 중
      (1) val2017 present 이미지 수 >= 메인 클래스들의 최솟값
      (2) 이미지별 최대 bbox 면적 비율의 중앙값이 메인 클래스들 중앙값의 [최소, 최대] 범위 안
    자격을 갖춘 클래스에서 고정 seed 로 n 개를 고른다."""
    def stats(cid):
        r = val_index.area_ratio.get(cid, {})
        return len(r), (statistics.median(r.values()) if r else 0.0)
    main = {c: stats(val_index.cat_name_to_id[c]) for c in main_classes}
    min_count = min(v[0] for v in main.values())
    lo, hi = min(v[1] for v in main.values()), max(v[1] for v in main.values())
    eligible = []
    for name, cid in sorted(val_index.cat_name_to_id.items()):
        if name in main_classes:
            continue
        cnt, med = stats(cid)
        if cnt >= min_count and lo <= med <= hi:
            eligible.append(name)
    if len(eligible) < n:
        raise ValueError(f"unseen-class 후보 {len(eligible)}개 < {n}")
    chosen = sorted(random.Random(seed).sample(eligible, n))
    return {"status": "deferred (classes fixed now, questions generated later)",
            "rule": "COCO classes not in main 20 with val2017 present count >= min over main classes and median "
                    "max-bbox-area ratio within [min, max] of main-class medians; random.Random(seed).sample(eligible, n)",
            "min_count": min_count, "median_area_range": [round(lo, 6), round(hi, 6)], "seed": seed, "n": n,
            "eligible": eligible, "chosen": chosen}


# ------------------------------------------------------------------ build / check
def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def build(seeds=SEEDS, ann_dir="annotations", out_root=OUT_ROOT, mih_dir=MIH_DIR, no_mihbench=False,
          classes=None, templates=None, specs=split_specs, pools=POOLS, unseen=UNSEEN) -> dict[int, dict]:
    """seed 마다 data/splits/coco2017_seed{K}/ 를 만든다. specs: seed -> {split: spec} 함수. 반환 {seed: manifest}."""
    if classes is None or templates is None:
        c = load_config(CLASS_CONFIG)["data"]
        classes = classes or c["classes"]
        templates = templates or c["templates"]
    ann = {"train2017": Path(ann_dir) / "instances_train2017.json", "val2017": Path(ann_dir) / "instances_val2017.json"}
    index = {src: load_coco_index(p) for src, p in ann.items()}
    ann_info = {src: {"path": str(p), "sha256": sha256_file(p)} for src, p in ann.items()}
    mih = set() if no_mihbench else mihbench_coco_ids(mih_dir)
    mih_in = {src: len(mih & set(index[src].images)) for src in index}
    # test 쪽(val2017)도 MIHBench 이미지를 뺀 인덱스에서 뽑는다. train/val 은 select_pools 에서 뺀다
    val_index = restrict_index(index["val2017"], set(index["val2017"].images) - mih)
    unseen_info = unseen_classes(val_index, classes, unseen["n"], unseen["seed"])
    head = _git_head()
    out = {}
    for seed in seeds:
        d = seed_dir(seed, out_root)
        d.mkdir(parents=True, exist_ok=True)
        pool_seed = f"{NAME}-s{seed}-pools"
        pool_ids = select_pools(index["train2017"], classes, mih, pools, pool_seed)
        manifest = {
            "dataset": f"{NAME}_seed{seed}",
            "seed": seed,
            "doc": "docs/split_coco.md",
            "created_by": "python -m src.splits build",
            "git_head": head,
            "classes": list(classes),
            "templates": templates,
            "generator": "src/data.py generate_questions (minimal pairs, uniform odd position, per-(seed, class, N) RNG)",
            "source": "COCO 2017 (train/val splits from train2017, test splits from val2017)",
            "coco2014": "not used as an image source. COCO 2017 re-splits the 2014 images with the same image ids; "
                        "2014 file names appear only in the MIHBench exclusion below (converted to ids)",
            "annotations": ann_info,
            "mihbench_excluded": {"source": "github.com/pgtrece/DAB Questions/**/*.jsonl (COCO_val2014_<id>.jpg)",
                                  "skipped": no_mihbench, "scope": "all splits (train, val, test, test_extN)",
                                  "n": len(mih), "n_in": mih_in, "image_ids": sorted(mih)},
            "pools": {name: {"source": "train2017", "rng_seed": pool_seed, **pools[name], "image_ids": ids}
                      for name, ids in pool_ids.items()},
            "b_train_seed": seed,
            "splits": {},
            "unseen_class": unseen_info,
        }
        seen: set = set()
        for split, sp in specs(seed).items():
            idx = val_index if sp["source"] == "val2017" else index[sp["source"]]
            if sp["pool"]:
                idx = restrict_index(idx, pool_ids[sp["pool"]])
            qseed = f"{NAME}-s{seed}-{split}"
            cfg = make_cfg(classes, templates, sp["Ns"], sp["n_per_type"], qseed, sp["source"])
            recs, regen = generate_split(idx, cfg, seen)
            seen |= {bundle_key(q) for q in recs}
            add_prefix(recs, seed, split)
            path = d / f"{split}.jsonl"
            write_jsonl(recs, path)
            manifest["splits"][split] = {
                **sp, "question_seed": qseed, "file": str(path), "n_questions": len(recs),
                "n_unique_images": len({i for q in recs for i in q["image_ids"]}),
                "regenerated_cells": regen, "sha256": sha256_file(path),
            }
            print(f"{NAME}_seed{seed} {split:10s} {len(recs):5d} questions  "
                  f"{manifest['splits'][split]['n_unique_images']:5d} images  regenerated cells {len(regen)}  -> {path}")
        with open(d / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=1)
        out[seed] = manifest
    print(f"MIHBench excluded {len(mih)} (train2017 {mih_in['train2017']}, val2017 {mih_in['val2017']}), "
          f"unseen-class {unseen_info['chosen']}")
    return out


def _answer_ok(q: dict) -> bool:
    n_abs, n_pre = len(q["absent_positions"]), len(q["present_positions"])
    return {"forall_yes": n_abs == 0, "forall_no": n_abs == 1, "exists_yes": n_pre == 1,
            "exists_no": n_pre == 0}[q["qtype"]] and n_abs + n_pre == q["N"] == len(q["image_ids"])


def check(manifest_path) -> list[str]:
    """seed 하나의 manifest 와 jsonl 이 설계대로인지 검증. 문제 목록을 반환 (빈 목록이면 통과)."""
    m = json.load(open(manifest_path, encoding="utf-8"))
    errs: list[str] = []
    seed = m["seed"]
    pools = {k: set(v["image_ids"]) for k, v in m["pools"].items()}
    names = list(pools)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if pools[a] & pools[b]:
                errs.append(f"pool {a} and {b} share {len(pools[a] & pools[b])} images")
    mih = set(m["mihbench_excluded"]["image_ids"])
    for k, ids in pools.items():
        if ids & mih:
            errs.append(f"pool {k} contains {len(ids & mih)} MIHBench images")
        if len(ids) != m["pools"][k]["size"]:
            errs.append(f"pool {k} has {len(ids)} images, expected {m['pools'][k]['size']}")
    qids, keys = set(), set()
    by_source: dict[str, set] = defaultdict(set)
    for split, s in m["splits"].items():
        path = Path(s["file"])
        if sha256_file(path) != s["sha256"]:
            errs.append(f"{split}: sha256 mismatch ({path})")
        recs = read_jsonl(path)
        if len(recs) != s["n_questions"]:
            errs.append(f"{split}: {len(recs)} questions, manifest says {s['n_questions']}")
        cells = Counter((q["cls"], q["N"], q["qtype"]) for q in recs)
        expect = Counter({(c, N, t): s["n_per_type"] for c in m["classes"] for N in s["Ns"] for t in QTYPES})
        if cells != expect:
            errs.append(f"{split}: per-cell counts differ from {s['n_per_type']} per (class, N, type)")
        for q in recs:
            if q["qid"] in qids:
                errs.append(f"{split}: duplicate qid {q['qid']}")
            qids.add(q["qid"])
            if not q["qid"].startswith(f"{NAME}-s{seed}-{split}-"):
                errs.append(f"{split}: qid without prefix {q['qid']}")
            if not all(p.startswith(f"{s['source']}/") for p in q["image_paths"]):
                errs.append(f"{split}: image path not under {s['source']}/ ({q['qid']})")
            if s["pool"] and not set(q["image_ids"]) <= pools[s["pool"]]:
                errs.append(f"{split}: image outside pool {s['pool']} ({q['qid']})")
            if len(set(q["image_ids"])) != len(q["image_ids"]):
                errs.append(f"{split}: repeated image inside one question ({q['qid']})")
            if set(q["image_ids"]) & mih:
                errs.append(f"{split}: MIHBench image in question ({q['qid']})")
            if not _answer_ok(q):
                errs.append(f"{split}: present/absent positions inconsistent with {q['qtype']} ({q['qid']})")
            k = bundle_key(q)
            if k in keys:
                errs.append(f"{split}: duplicate bundle {k[:2]} ({q['qid']})")
            keys.add(k)
            by_source[s["source"]].update(q["image_ids"])
    srcs = list(by_source)
    for i, a in enumerate(srcs):
        for b in srcs[i + 1:]:
            if by_source[a] & by_source[b]:
                errs.append(f"{a} and {b} questions share {len(by_source[a] & by_source[b])} image ids")
    return errs


def list_images(seeds, splits, root=OUT_ROOT) -> list[str]:
    paths = set()
    for seed in seeds:
        for split in splits:
            for q in read_jsonl(seed_dir(seed, root) / f"{split}.jsonl"):
                paths.update(q["image_paths"])
    return sorted(paths)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    b.add_argument("--ann_dir", default="annotations")
    b.add_argument("--out_root", default=str(OUT_ROOT))
    b.add_argument("--mih_dir", default=MIH_DIR)
    b.add_argument("--no_mihbench", action="store_true", help="MIHBench 제외를 건너뜀 (테스트용, 기록에 남음)")
    c = sub.add_parser("check")
    c.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    c.add_argument("--out_root", default=str(OUT_ROOT))
    i = sub.add_parser("images")
    i.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    i.add_argument("--splits", nargs="+", default=["train", "val"])
    i.add_argument("--out_root", default=str(OUT_ROOT))
    args = ap.parse_args(argv)
    if args.cmd == "build":
        build(args.seeds, args.ann_dir, args.out_root, args.mih_dir, args.no_mihbench)
    if args.cmd in ("build", "check"):
        bad = 0
        for seed in args.seeds:
            errs = check(seed_dir(seed, args.out_root) / "manifest.json")
            print(f"check {NAME}_seed{seed}:", "OK" if not errs else f"{len(errs)} problems")
            for e in errs[:20]:
                print("  ", e)
            bad += len(errs)
        raise SystemExit(1 if bad else 0)
    if args.cmd == "images":
        for p in list_images(args.seeds, args.splits, args.out_root):
            print(p)


if __name__ == "__main__":
    main()
