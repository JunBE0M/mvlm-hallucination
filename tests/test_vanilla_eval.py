"""src/vanilla_eval.py 검증: 정답을 아는 합성 원자료(run_eval 과 같은 열)로 각 지표를 확인한다 (CPU)."""
import json

import numpy as np
import pandas as pd
import pytest

from src import vanilla_eval as V

PLANTED = 999999          # 모든 모델이 강하게 Yes 라고 하는 '객체 없는' 사진 (라벨 누락 의심 후보)


def make_raw(path, tag, seed, split, rng, odd_top1=False):
    """cat: ∀ 에서 Δ = 4/N (1/N 희석, γ = 1), ∃ 도 구분됨. dog: 점수에 정보 없음 (AUC ≈ 0.5)."""
    rows, iid = [], [seed * 1_000_000 + (0 if split == "val" else 500_000)]

    def new():
        iid[0] += 1
        return iid[0]

    def add(qid, pid, cls, N, qtype, ids, pres, m):
        pred = "Yes" if m > 0 else "No"
        ans = "Yes" if qtype.endswith("yes") else "No"
        top1 = "The" if odd_top1 and len(rows) < 12 else pred
        rows.append(dict(qid=qid, pair_id=pid, cls=cls, N=N, qtype=qtype, answer=ans, image_ids=json.dumps(ids),
                         present_positions=json.dumps(pres),
                         absent_positions=json.dumps([p for p in range(N) if p not in pres]),
                         area_ratios=json.dumps([0.1] * N), logit_yes=m, logit_no=0.0, m=m, pred=pred,
                         correct=int(pred == ans), p_yes=0.6, p_no=0.39, top1_token=top1,
                         n_image_tokens=json.dumps([64] * N), model=tag))

    for cls in ("cat", "dog"):
        for N in (2, 4, 8):
            for k in range(20):
                pre = f"coco2017-s{seed}-{split}-{cls}_N{N}"
                pid = f"{pre}_pair{k:04d}"
                yes = [new() for _ in range(N)]
                p = k % N
                no = list(yes)
                no[p] = PLANTED if (cls == "cat" and N == 4 and k == 0) else new()
                if cls == "cat":
                    my, mn = 2 + rng.normal(0, .3), 2 - 4 / N + rng.normal(0, .3)
                    mey, men = 1 + rng.normal(0, .5), -1 + rng.normal(0, .5)
                else:
                    my, mn = rng.normal(0, .3), rng.normal(0, .3)
                    mey, men = rng.normal(0, 1), rng.normal(0, 1)
                if no[p] == PLANTED:
                    mn = 10.0
                add(f"{pid}_y", pid, cls, N, "forall_yes", yes, list(range(N)), my)
                add(f"{pid}_n", pid, cls, N, "forall_no", no, [i for i in range(N) if i != p], mn)
                ey = [new() for _ in range(N)]
                add(f"{pre}_ey{k}", "", cls, N, "exists_yes", ey, [p], mey)
                add(f"{pre}_en{k}", "", cls, N, "exists_no", [new() for _ in range(N)], [], men)
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    rng = np.random.default_rng(0)
    for tag in ("ta", "tb"):
        for seed in (1, 2):
            for split in ("val", "test"):
                make_raw(d / f"coco2017_s{seed}_e1_{split}_{tag}.csv", tag, seed, split, rng, odd_top1=(tag == "tb"))
    return d


@pytest.fixture(scope="module")
def res(raw):
    data = V.load_all(str(raw), ["ta", "tb", "missing"], [1, 2], ["val", "test"])
    assert len(data) == 8                                          # 없는 태그는 건너뜀
    return V.evaluate(data, nb=100, top_k=20)


def test_gamma_fit_recovers_known_exponent():
    rng = np.random.default_rng(1)
    pairs = pd.DataFrame([dict(N=n, delta=3 * n ** -1.5 + rng.normal(0, 1e-3)) for n in (2, 4, 6, 8) for _ in range(50)])
    g = V.gamma_fit(pairs, nb=50, rng=rng)
    assert g["gamma"] == pytest.approx(1.5, abs=0.05) and g["gamma_lo"] <= g["gamma"] <= g["gamma_hi"]


def test_gamma_near_one_for_1_over_N(res):
    g = res["gamma"]
    pooled = g[(g.seed == "all") & (g.split == "test")]
    assert set(pooled.tag) == {"ta", "tb"}
    assert pooled.gamma.between(0.6, 1.4).all()


def test_accuracy_tables(res):
    acc, am = res["accuracy"], res["accuracy_mean"]
    assert set(acc.seed) == {1, 2} and set(am.n_seeds) == {2}
    no = am[am.qtype.isin(["forall_no", "exists_no"])]
    yes = am[am.qtype.isin(["forall_yes", "exists_yes"])]
    assert no.false_yes_rate_mean.notna().all() and yes.false_yes_rate_mean.isna().all()


def test_class_auc_separates_informative_class(res):
    c = res["class_auc"]
    p = c[(c.seed == "all") & (c.split == "test") & (c.family == "forall")].set_index(["tag", "cls"]).auc
    for tag in ("ta", "tb"):
        assert p[(tag, "cat")] > p[(tag, "dog")] + 0.2
        assert 0.3 < p[(tag, "dog")] < 0.7


def test_answer_format_flags_non_yes_no_tokens(res):
    f = res["answer_format"].set_index(["tag", "seed", "split"])
    assert (f.loc["ta"].top1_not_yesno == 0).all()
    assert (f.loc["tb"].top1_not_yesno > 0).all() and "The" in f.loc[("tb", 1, "test"), "top1_other_examples"]
    assert "tb" in V.summarize(res)


def test_label_audit_puts_planted_image_first(res):
    la = res["label_audit_candidates"]
    top = la.iloc[0]
    assert top.absent_image_id == PLANTED and top.n_models == 2 and top.cls == "cat"
    assert top.image_path == f"val2017/{PLANTED:012d}.jpg" and top.frac_pred_yes == 1.0
    assert len(la) == 20


def test_position_and_sdt(res):
    pos = res["position"]
    n8 = pos[(pos.tag == "ta") & (pos.split == "test") & (pos.qtype == "forall_no") & (pos.N == 8)]
    assert sorted(n8.odd_pos) == list(range(8)) and set(pos.n_seeds) == {2}
    assert {"d_prime", "criterion"} <= set(res["sdt"].columns)


def test_main_writes_outputs(raw, tmp_path):
    out = tmp_path / "vanilla"
    V.main(["--tags", "ta", "tb", "--seeds", "1", "2", "--raw", str(raw), "--out_dir", str(out), "--nb", "50"])
    for name in ("accuracy", "accuracy_mean", "delta", "gamma", "sdt", "position", "class_auc", "answer_format",
                 "label_audit_candidates"):
        assert (out / f"{name}.csv").exists(), name
    text = (out / "summary.txt").read_text()
    assert "γ =" in text and "라벨 검수 후보" in text
