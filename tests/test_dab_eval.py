"""src/dab_eval.py 검증: 정답을 아는 합성 결과로 AUC·bootstrap·α 선택·β·출력 파일을 확인한다 (CPU)."""
import json

import numpy as np
import pandas as pd
import pytest

from src import agg_analysis as A
from src import dab_eval as E

CLASSES = ["cat", "dog"]
SEEDS = [1, 2]


def make_questions(rng, pre, n_per=12, Ns=(2, 4)):
    """∀ 는 pair(Yes/No, 이미지 하나만 교체), ∃ 는 독립 문항. 이미지별 증거 ev: present +2, absent −2 (+잡음)."""
    qs, ev, iid = [], {}, 0

    def img(cls, present):
        nonlocal iid
        iid += 1
        ev[f"{cls}|{iid}"] = (2.0 if present else -2.0) + rng.normal(0, 1)
        return iid

    for c in CLASSES:
        for N in Ns:
            for k in range(n_per):
                yes = [img(c, True) for _ in range(N)]
                no = list(yes)
                no[k % N] = img(c, False)
                pid = f"{pre}{c}_N{N}_pair{k:04d}"
                qs.append(dict(qid=f"{pid}_y", pair_id=pid, cls=c, N=N, qtype="forall_yes", answer="Yes", image_ids=yes))
                qs.append(dict(qid=f"{pid}_n", pair_id=pid, cls=c, N=N, qtype="forall_no", answer="No", image_ids=no))
                ex_yes = [img(c, False) for _ in range(N - 1)] + [img(c, True)]
                ex_no = [img(c, False) for _ in range(N)]
                qs.append(dict(qid=f"{pre}{c}_N{N}_ey{k}", pair_id=None, cls=c, N=N, qtype="exists_yes",
                               answer="Yes", image_ids=ex_yes))
                qs.append(dict(qid=f"{pre}{c}_N{N}_en{k}", pair_id=None, cls=c, N=N, qtype="exists_no",
                               answer="No", image_ids=ex_no))
    return qs, ev


@pytest.fixture()
def synth(tmp_path):
    rng = np.random.default_rng(0)
    qroot, raw = tmp_path / "q", tmp_path / "raw"
    raw.mkdir()
    for seed in SEEDS:
        (qroot / f"coco2017_seed{seed}").mkdir(parents=True)
        for split in ("val", "test"):
            qs, ev = make_questions(rng, f"coco2017-s{seed}-{split}-")
            (qroot / f"coco2017_seed{seed}" / f"{split}.jsonl").write_text("\n".join(json.dumps(q) for q in qs) + "\n")
            pre = raw / f"coco2017_s{seed}_"
            pd.DataFrame({"key": list(ev), "m": list(ev.values())}).to_csv(f"{pre}m1_single_{split}_tg.csv", index=False)
            E_ = [np.array([ev[f"{q['cls']}|{i}"] for i in q["image_ids"]]) for q in qs]
            fam = [q["qtype"].split("_")[0] for q in qs]
            qid = [q["qid"] for q in qs]
            mean = np.array([e.mean() for e in E_])                                          # 평균 집계 (base 처럼)
            agg = np.array([e.min() if f == "forall" else e.max() for e, f in zip(E_, fam)])  # 규칙 집계 (M2 처럼)
            pd.DataFrame({"qid": qid, "m": mean}).to_csv(f"{pre}e1_{split}_tg.csv", index=False)
            pd.DataFrame({"qid": qid, "m": agg}).to_csv(f"{pre}m2_post_{split}_tg.csv", index=False)
            for a, noise in ((0.1, 2.0), (0.5, 0.5)):
                m = mean + rng.normal(0, noise, len(mean))
                pd.DataFrame({"qid": qid, "m": m}).to_csv(f"{pre}dab_a{a}_{split}_tg.csv", index=False)
            pd.DataFrame({"qid": qid[:-5], "m": agg[:-5]}).to_csv(f"{pre}B_e1_{split}_tg_both.csv", index=False)
    return qroot, raw, tmp_path / "out"


def test_auc_matches_agg_analysis():
    rng = np.random.default_rng(1)
    s, y = rng.normal(size=200).round(1), rng.random(200) < 0.4                 # 동점 포함
    assert E.auc(s, y) == pytest.approx(A.auc(s[y], s[~y]))


def test_forall_bootstrap_keeps_pairs_together(synth):
    qroot, _, _ = synth
    T = E.question_table(1, "val", str(qroot)).reset_index()
    F = T[T.family == "forall"].reset_index(drop=True)
    for ix in E._resamples("forall", F, 20, np.random.default_rng(0)):
        assert F.yes.to_numpy()[ix].sum() * 2 == len(ix)                          # pair 마다 Yes 1 + No 1


def test_evaluate_finds_rule_aggregation_better(synth):
    qroot, raw, _ = synth
    auc_df, diff_df = E.evaluate(["tg"], SEEDS, ["val", "test"], str(raw), str(qroot), nb=200)
    assert {"base", "M2", "DAB(a=0.1)", "DAB(a=0.5)", "B[both]"} <= set(auc_df.method)
    assert set(auc_df.N) == {"all", "2", "4"} and set(auc_df.seed) == set(SEEDS)
    a = auc_df[(auc_df.seed == 1) & (auc_df.split == "test") & (auc_df.N == "all")
               & (auc_df.family == "forall")].set_index("method")
    assert a.loc["M2", "auc"] > a.loc["base", "auc"]
    assert ((auc_df.lo <= auc_df.auc + 1e-12) & (auc_df.auc <= auc_df.hi + 1e-12)).all()
    d = diff_df[(diff_df.seed == 1) & (diff_df.split == "test") & (diff_df.ref == "base")
                & (diff_df.family == "forall") & (diff_df.N == "all")].set_index("method")
    assert d.loc["M2", "diff"] > 0 and d.loc["M2", "lo"] > 0
    assert "B[both]" in d.index                                                  # 빠진 문항이 있어도 따로 비교
    assert (diff_df.ref == "M2").any()


def test_dab_alpha_selected_per_seed_on_val(synth):
    qroot, raw, _ = synth
    auc_df, _ = E.evaluate(["tg"], SEEDS, ["val"], str(raw), str(qroot), nb=50)
    t = E.dab_alpha_table(auc_df)
    assert set(t.alpha) == {0.1, 0.5}
    sel = t[t.selected].set_index("seed")["alpha"]
    assert sel.to_dict() == {1: 0.5, 2: 0.5}                                      # seed 마다 잡음이 작은 쪽


def test_beta_signs(synth):
    qroot, raw, _ = synth
    b = E.beta_table(["tg"], [1], ["test"], str(raw), str(qroot), nboot=10)
    f = b[b.family == "forall"].set_index("method")
    assert f.loc["M2", "beta_norm"] > 0                                          # min 집계 -> min 쪽
    assert abs(f.loc["base", "beta_norm"]) < f.loc["M2", "beta_norm"]           # 평균 집계 -> 0 근처


def test_main_writes_outputs(synth):
    qroot, raw, out = synth
    E.main(["--tags", "tg", "--seeds", "1", "2", "--splits", "val", "test", "--raw", str(raw),
            "--qroot", str(qroot), "--out_dir", str(out), "--nb", "50"])
    for f in ("auc.csv", "diff.csv", "dab_alpha.csv", "beta.csv", "summary.txt"):
        assert (out / f).exists()
    text = (out / "summary.txt").read_text()
    assert "P1" in text and "(seed 2)" in text
