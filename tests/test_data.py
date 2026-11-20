"""가짜 COCO annotation으로 data.py를 CPU에서 검증."""
import copy
import json
import random
from collections import Counter

import pytest

from src import data as D

CLASSES = ["cat", "dog", "teddy bear"]


@pytest.fixture
def fake_ann(tmp_path):
    rng = random.Random(123)
    images, anns = [], []
    aid = 0
    for img_id in range(1, 61):
        W, H = 640, 480
        images.append({"id": img_id, "file_name": f"{img_id:012d}.jpg", "width": W, "height": H})
        for cid in (1, 2, 3):
            if rng.random() < 0.4:
                for _ in range(rng.randint(1, 3)):
                    w, h = rng.uniform(10, 600), rng.uniform(10, 450)
                    aid += 1
                    anns.append({"id": aid, "image_id": img_id, "category_id": cid,
                                 "bbox": [0, 0, w, h], "area": w * h, "iscrowd": 0})
    coco = {"images": images, "annotations": anns,
            "categories": [{"id": i + 1, "name": n} for i, n in enumerate(CLASSES)]}
    p = tmp_path / "instances.json"
    p.write_text(json.dumps(coco))
    return p, coco


@pytest.fixture
def cfg():
    return {
        "seed": 0,
        "coco": {"image_dir": "val2017"},
        "data": {
            "classes": CLASSES, "Ns": [2, 4, 6, 8],
            "n_forall_pairs": 8, "n_exists_yes": 8, "n_exists_no": 4,
            "odd_position": "uniform", "area_range": None,
            "templates": {
                "forall": "Is there a {obj} in all {N} images? Please answer with Yes or No.",
                "exists": "Is there a {obj} in any of the {N} images? Please answer with Yes or No.",
            },
        },
    }


def _truth(coco):
    """(cat_id -> set(img_id)), (cat_id, img_id) -> max bbox ratio."""
    has, ratio = {}, {}
    sizes = {im["id"]: im["width"] * im["height"] for im in coco["images"]}
    for a in coco["annotations"]:
        has.setdefault(a["category_id"], set()).add(a["image_id"])
        r = a["bbox"][2] * a["bbox"][3] / sizes[a["image_id"]]
        k = (a["category_id"], a["image_id"])
        ratio[k] = max(ratio.get(k, 0), r)
    return has, ratio


def test_index_area_ratio(fake_ann):
    path, coco = fake_ann
    idx = D.load_coco_index(path)
    _, ratio = _truth(coco)
    for (cid, iid), r in ratio.items():
        assert idx.area_ratio[cid][iid] == pytest.approx(r)


def test_labels_and_presence(fake_ann, cfg):
    path, coco = fake_ann
    idx = D.load_coco_index(path)
    has, _ = _truth(coco)
    recs = D.generate_questions(idx, cfg)
    assert recs
    for r in recs:
        cid = idx.cat_name_to_id[r["cls"]]
        flags = [i in has.get(cid, set()) for i in r["image_ids"]]
        assert len(r["image_ids"]) == r["N"] == len(r["image_paths"]) == len(r["area_ratios"])
        assert len(set(r["image_ids"])) == r["N"], "duplicate image within a question"
        n_present = sum(flags)
        expected = {"forall_yes": r["N"], "forall_no": r["N"] - 1, "exists_yes": 1, "exists_no": 0}
        assert n_present == expected[r["qtype"]]
        assert r["answer"] == ("Yes" if r["qtype"] in ("forall_yes", "exists_yes") else "No")
        assert r["absent_positions"] == [p for p, f in enumerate(flags) if not f]
        assert r["present_positions"] == [p for p, f in enumerate(flags) if f]
        for p, f in enumerate(flags):
            assert (r["area_ratios"][p] > 0) == f
        word = "all" if r["qtype"].startswith("forall") else "any of the"
        assert f"Is there a {r['cls']} in {word} {r['N']} images?" in r["question"]


def test_pairs_differ_in_one_position(fake_ann, cfg):
    idx = D.load_coco_index(fake_ann[0])
    recs = D.generate_questions(idx, cfg)
    pairs = {}
    for r in recs:
        if r["pair_id"] is not None:
            pairs.setdefault(r["pair_id"], {})[r["qtype"]] = r
    assert pairs
    for pid, pr in pairs.items():
        y, n = pr["forall_yes"], pr["forall_no"]
        diff = [i for i in range(y["N"]) if y["image_ids"][i] != n["image_ids"][i]]
        assert diff == n["absent_positions"] and len(diff) == 1
    # exists 문항엔 pair_id 없음
    assert all(r["pair_id"] is None for r in recs if r["qtype"].startswith("exists"))
    assert len({r["qid"] for r in recs}) == len(recs)


def test_uniform_positions_balanced(fake_ann, cfg):
    idx = D.load_coco_index(fake_ann[0])
    recs = D.generate_questions(idx, cfg)
    for cls in CLASSES:
        for N in (2, 4, 8):   # n_forall_pairs=8 은 N의 배수 → 정확히 균등
            c = Counter(r["absent_positions"][0] for r in recs
                        if r["cls"] == cls and r["N"] == N and r["qtype"] == "forall_no")
            if c:
                assert set(c) == set(range(N)) and len(set(c.values())) == 1


def test_fixed_position(fake_ann, cfg):
    idx = D.load_coco_index(fake_ann[0])
    c = copy.deepcopy(cfg)
    c["data"]["odd_position"] = "fixed:-1"
    recs = D.generate_questions(idx, c)
    for r in recs:
        if r["qtype"] == "forall_no":
            assert r["absent_positions"] == [r["N"] - 1]
        if r["qtype"] == "exists_yes":
            assert r["present_positions"] == [r["N"] - 1]


def test_area_filter(fake_ann, cfg):
    idx = D.load_coco_index(fake_ann[0])
    c = copy.deepcopy(cfg)
    c["data"]["area_range"] = [0.2, 0.6]
    c["data"]["Ns"] = [2]
    recs = D.generate_questions(idx, c)
    assert recs
    for r in recs:
        for p in r["present_positions"]:
            assert 0.2 <= r["area_ratios"][p] <= 0.6


def test_deterministic_and_order_independent(fake_ann, cfg):
    idx = D.load_coco_index(fake_ann[0])
    a = D.generate_questions(idx, cfg)
    b = D.generate_questions(idx, copy.deepcopy(cfg))
    assert a == b
    c = copy.deepcopy(cfg)
    c["data"]["classes"] = list(reversed(CLASSES))
    assert sorted(a, key=lambda r: r["qid"]) == sorted(D.generate_questions(idx, c), key=lambda r: r["qid"])
    c2 = copy.deepcopy(cfg)
    c2["seed"] = 1
    assert D.generate_questions(idx, c2) != a


def test_unknown_class_raises(fake_ann, cfg):
    idx = D.load_coco_index(fake_ann[0])
    with pytest.raises(KeyError):
        D.build_pools(idx, "unicorn")


def test_cli_roundtrip(fake_ann, cfg, tmp_path):
    import yaml
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "q.jsonl"
    D.main(["--config", str(cfg_path), "--ann", str(fake_ann[0]), "--out", str(out),
            "--odd_position", "fixed:0"])
    recs = D.read_jsonl(out)
    assert recs and all(r["absent_positions"] == [0] for r in recs if r["qtype"] == "forall_no")
