"""src/splits.py (COCO 2017 분할 coco2017_seed1·2·3) 검증. 작은 가짜 COCO annotation 으로 CPU 에서 돈다."""
import json
from collections import Counter

import pytest

from src import splits as S
from src.data import QTYPES, load_coco_index, read_jsonl

CLASSES = ["cat", "dog", "car"]
EXTRA = ["cup", "fork", "bench"]          # unseen-class 후보
TEMPLATES = {"forall": "Is there a {obj} in all {N} images?", "exists": "Is there a {obj} in any of the {N} images?"}
TRAIN_IDS = range(1, 1501)                # 가짜 train2017
VAL_IDS = range(100001, 100601)           # 가짜 val2017
MIH_IDS = {7, 11, 100005}                 # MIHBench 가 쓰는 COCO id (train 쪽 2개, val 쪽 1개)
POOLS = {"train": {"size": 300, "min_present": 40}, "val": {"size": 120, "min_present": 20}}
SEEDS = [1, 2]


def specs(seed):
    return {
        "train": dict(source="train2017", pool="train", Ns=[2, 4], n_per_type=5, role="t"),
        "val": dict(source="train2017", pool="val", Ns=[2, 4], n_per_type=2, role="v"),
        "test": dict(source="val2017", pool=None, Ns=[2, 4], n_per_type=2, role="x"),
        "test_extN": dict(source="val2017", pool=None, Ns=[6], n_per_type=2, role="e"),
    }


def write_fake_coco(path, ids):
    """클래스 k 는 (id + k) % 3 == 0 인 이미지에 있음 (모든 클래스 출현 수와 면적이 같게)."""
    cats = [{"id": k + 1, "name": n} for k, n in enumerate(CLASSES + EXTRA)]
    images, anns = [], []
    for i in ids:
        images.append({"id": i, "file_name": f"{i:012d}.jpg", "width": 100, "height": 100})
        for k, c in enumerate(cats):
            if (i + k) % 3 == 0:
                anns.append({"image_id": i, "category_id": c["id"], "bbox": [0, 0, 30, 30]})
    path.write_text(json.dumps({"images": images, "categories": cats, "annotations": anns}))


@pytest.fixture(scope="module")
def coco_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("coco")
    write_fake_coco(d / "instances_train2017.json", TRAIN_IDS)
    write_fake_coco(d / "instances_val2017.json", VAL_IDS)
    q = d / "mih" / "Questions" / "Existence"
    q.mkdir(parents=True)
    rows = [{"image_list": [f"COCO_val2014_{i:012d}.jpg" for i in sorted(MIH_IDS)] + ["wineglass/1/frame000053.jpg"]}]
    (q / "x.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return d


def do_build(coco_dir, out_root, seeds=SEEDS):
    return S.build(seeds=seeds, ann_dir=coco_dir, out_root=out_root, mih_dir=coco_dir / "mih" / "Questions",
                   classes=CLASSES, templates=TEMPLATES, specs=specs, pools=POOLS, unseen={"n": 2, "seed": "u"})


@pytest.fixture(scope="module")
def built(coco_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("splits")
    return do_build(coco_dir, out), out


def test_layout_and_check_passes(built):
    ms, out = built
    for seed in SEEDS:
        d = out / f"coco2017_seed{seed}"
        assert sorted(p.name for p in d.iterdir()) == ["manifest.json", "test.jsonl", "test_extN.jsonl",
                                                       "train.jsonl", "val.jsonl"]
        assert S.check(d / "manifest.json") == []
        assert ms[seed]["dataset"] == f"coco2017_seed{seed}" and ms[seed]["b_train_seed"] == seed


def test_pools_disjoint_sized_and_mihbench_excluded(built, coco_dir):
    ms, _ = built
    idx = load_coco_index(coco_dir / "instances_train2017.json")
    for seed in SEEDS:
        m = ms[seed]
        tr, va = set(m["pools"]["train"]["image_ids"]), set(m["pools"]["val"]["image_ids"])
        assert len(tr) == 300 and len(va) == 120 and not (tr & va)
        assert not ((tr | va) & MIH_IDS)
        assert set(m["mihbench_excluded"]["image_ids"]) == MIH_IDS
        for c in CLASSES:
            present = set(idx.area_ratio[idx.cat_name_to_id[c]])
            assert len(present & tr) >= 40 and len(present & va) >= 20


def test_seeds_are_independent_draws(built):
    ms, _ = built
    a, b = ms[1], ms[2]
    assert a["pools"]["train"]["image_ids"] != b["pools"]["train"]["image_ids"]
    for split in S.SPLITS:
        assert a["splits"][split]["sha256"] != b["splits"][split]["sha256"]
        assert a["splits"][split]["question_seed"] == f"coco2017-s1-{split}"
    assert a["unseen_class"]["chosen"] == b["unseen_class"]["chosen"]          # 클래스 선택은 seed 와 무관


def test_counts_prefix_and_paths(built):
    ms, _ = built
    for seed in SEEDS:
        for split, sp in specs(seed).items():
            recs = read_jsonl(ms[seed]["splits"][split]["file"])
            assert len(recs) == len(CLASSES) * len(sp["Ns"]) * len(QTYPES) * sp["n_per_type"]
            cells = Counter((q["cls"], q["N"], q["qtype"]) for q in recs)
            assert set(cells.values()) == {sp["n_per_type"]}
            pre = f"coco2017-s{seed}-{split}-"
            assert all(q["qid"].startswith(pre) and q["split"] == split and q["dataset"] == f"coco2017_seed{seed}"
                       for q in recs)
            assert all(p.startswith(sp["source"] + "/") for q in recs for p in q["image_paths"])
            assert all(q["pair_id"].startswith(pre) for q in recs if q["qtype"] == "forall_no")


def test_train_val_images_stay_in_pools_and_test_uses_val2017(built):
    ms, _ = built
    m = ms[1]
    tr = set(m["pools"]["train"]["image_ids"])
    train_imgs = {i for q in read_jsonl(m["splits"]["train"]["file"]) for i in q["image_ids"]}
    test_imgs = {i for q in read_jsonl(m["splits"]["test"]["file"]) for i in q["image_ids"]}
    assert train_imgs <= tr
    assert all(i in VAL_IDS for i in test_imgs)


def test_mihbench_excluded_from_all_splits(built):
    ms, _ = built
    for seed in SEEDS:
        for split in S.SPLITS:
            imgs = {i for q in read_jsonl(ms[seed]["splits"][split]["file"]) for i in q["image_ids"]}
            assert not (imgs & MIH_IDS), (seed, split)
        mx = ms[seed]["mihbench_excluded"]
        assert mx["n_in"] == {"train2017": 2, "val2017": 1} and "all splits" in mx["scope"]
        assert ms[seed]["source"].startswith("COCO 2017")


def test_check_flags_mihbench_image_in_test(coco_dir, tmp_path):
    do_build(coco_dir, tmp_path, seeds=[1])
    f = tmp_path / "coco2017_seed1" / "test.jsonl"
    lines = f.read_text().splitlines()
    q = json.loads(lines[0])
    q["image_ids"][0] = 100005                                   # MIHBench 가 쓰는 val2017 이미지
    lines[0] = json.dumps(q)
    f.write_text("\n".join(lines) + "\n")
    errs = S.check(tmp_path / "coco2017_seed1" / "manifest.json")
    assert any("MIHBench image in question" in e for e in errs)


def test_random_positions_used_and_recorded(built):
    ms, _ = built
    for seed in SEEDS:
        m = ms[seed]
        assert m["odd_position"] == "random"
        for split, sp in specs(seed).items():
            counts = m["splits"][split]["odd_position_counts"]
            assert set(counts) == {"forall_no", "exists_yes"}
            for qtype in counts:
                assert set(counts[qtype]) == {str(N) for N in sp["Ns"]}
                for n, c in counts[qtype].items():
                    assert len(c) == int(n) and sum(c) == len(CLASSES) * sp["n_per_type"]
    assert S.make_cfg(CLASSES, TEMPLATES, [2], 1, 0, "val2017")["data"]["odd_position"] == "random"


def test_missing_positions_rule():
    counts = {"forall_no": {"8": [13, 12, 12, 13, 12, 13, 25, 0]},    # 기대 12.5 >= 5 인데 마지막 자리 비어 있음
              "exists_yes": {"4": [2, 2, 2, 0]}}                    # 기대 1.5 < 5 -> 검사 안 함
    assert S.missing_positions(counts) == [("forall_no", 8, [7])]
    assert S.missing_positions({"forall_no": {"8": [20, 20, 20, 20, 20, 0, 0, 0]}}) == [("forall_no", 8, [5, 6, 7])]


def test_check_flags_missing_positions(coco_dir, tmp_path):
    """예전 uniform 방식처럼 뒤쪽 자리가 비어 있는 split 은 check 에서 실패해야 한다."""
    big = {"test": dict(source="val2017", pool=None, Ns=[8], n_per_type=20, role="x")}
    S.build(seeds=[1], ann_dir=coco_dir, out_root=tmp_path, mih_dir=coco_dir / "mih" / "Questions",
            classes=CLASSES, templates=TEMPLATES, specs=lambda s: big, pools=POOLS, unseen={"n": 2, "seed": "u"})
    mpath = tmp_path / "coco2017_seed1" / "manifest.json"
    assert S.check(mpath) == []                                 # 무작위: 60문항 / 8자리, 빈 자리 없음
    f = tmp_path / "coco2017_seed1" / "test.jsonl"
    recs = read_jsonl(f)
    for q in recs:                                              # 객체 없는 사진을 앞 5자리로만 몰기
        if q["qtype"] == "forall_no" and q["absent_positions"][0] >= 5:
            q["absent_positions"] = [q["absent_positions"][0] % 5]
    f.write_text("\n".join(json.dumps(q) for q in recs) + "\n")
    errs = S.check(mpath)
    assert any("odd image never at positions [5, 6, 7]" in e for e in errs)


def test_build_is_deterministic(coco_dir, tmp_path):
    a = do_build(coco_dir, tmp_path / "a", seeds=[1])
    b = do_build(coco_dir, tmp_path / "b", seeds=[1])
    assert a[1]["pools"] == b[1]["pools"]
    assert {k: v["sha256"] for k, v in a[1]["splits"].items()} == {k: v["sha256"] for k, v in b[1]["splits"].items()}


def test_check_detects_tampering(coco_dir, tmp_path):
    do_build(coco_dir, tmp_path, seeds=[1])
    f = tmp_path / "coco2017_seed1" / "val.jsonl"
    lines = f.read_text().splitlines()
    q = json.loads(lines[0])
    q["answer"] = "No" if q["answer"] == "Yes" else "Yes"
    lines[0] = json.dumps(q)
    f.write_text("\n".join(lines) + "\n")
    errs = S.check(tmp_path / "coco2017_seed1" / "manifest.json")
    assert any("sha256 mismatch" in e for e in errs)


def test_list_images(built):
    _, out = built
    paths = S.list_images([1, 2], ["train", "val"], out)
    assert paths and all(p.startswith("train2017/") for p in paths) and paths == sorted(set(paths))


def test_generate_split_resolves_bundle_collisions(coco_dir):
    idx = load_coco_index(coco_dir / "instances_val2017.json")
    cfg = S.make_cfg(CLASSES, TEMPLATES, [2, 4], 3, "x", "val2017")
    first, regen0 = S.generate_split(idx, cfg, set())
    assert regen0 == []
    seen = {S.bundle_key(q) for q in first}
    second, regen = S.generate_split(idx, cfg, seen)       # 같은 seed -> 모든 칸이 충돌 -> 칸마다 다시 뽑음
    assert len(regen) == len(CLASSES) * 2
    assert not ({S.bundle_key(q) for q in second} & seen)
    assert all(str(q["seed"]).startswith("xr") for q in second)


def test_generate_split_raises_when_pool_too_small(coco_dir):
    idx = load_coco_index(coco_dir / "instances_val2017.json")
    cat = idx.cat_name_to_id["cat"]
    cat_imgs = sorted(idx.area_ratio[cat])[:2]                   # cat present 2장뿐
    others = [i for i in sorted(idx.images) if i not in idx.area_ratio[cat]][:50]
    small = S.restrict_index(idx, cat_imgs + others)
    cfg = S.make_cfg(["cat"], TEMPLATES, [2], 3, 0, "val2017")   # forall_yes 3개인데 가능한 2장 조합은 1개
    with pytest.raises(RuntimeError):
        S.generate_split(small, cfg, set())


def test_unseen_rule(coco_dir):
    idx = load_coco_index(coco_dir / "instances_val2017.json")
    a = S.unseen_classes(idx, CLASSES, n=2, seed="u")
    b = S.unseen_classes(idx, CLASSES, n=2, seed="u")
    assert set(a["eligible"]) == set(EXTRA)
    assert set(a["chosen"]) <= set(EXTRA) and len(a["chosen"]) == 2
    assert a["chosen"] == b["chosen"]


def test_mihbench_ids(coco_dir):
    assert S.mihbench_coco_ids(coco_dir / "mih" / "Questions") == MIH_IDS
    with pytest.raises(FileNotFoundError):
        S.mihbench_coco_ids(coco_dir / "nope")
