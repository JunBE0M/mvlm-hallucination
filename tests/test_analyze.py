"""정답을 아는 가짜 E1/E2 결과로 analyze.py 지표와 그래프 생성을 검증 (torch 불필요)."""
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from src import analyze as A
from src.io_utils import ResumableCsvWriter, atomic_save_npz
from src.run_eval import COLUMNS

NS = (2, 4, 8)
N_PAIRS = 8
L, H = 5, 3


def _row(qid, pair_id, N, qtype, m, absent, present):
    pred = "Yes" if m > 0 else "No"
    answer = "Yes" if qtype.endswith("_yes") else "No"
    return dict(qid=qid, pair_id=pair_id, cls="cat", N=N, qtype=qtype, answer=answer,
                image_ids=list(range(N)), present_positions=present, absent_positions=absent,
                area_ratios=[0.1] * N, logit_yes=m, logit_no=0.0, m=m, pred=pred, correct=int(pred == answer),
                n_image_tokens=[10] * N, model="fake", max_pixels=200704)


def make_rows():
    """m(forall_yes) = 3, m(forall_no) = 3 − 4/N  → Δ = 4/N, Δ×N = 4 (평균 집계 가설의 모양).
    forall_no 는 absent 가 위치 0 일 때만 맞힘(m<0이 되도록)."""
    rows = []
    for N in NS:
        for k in range(N_PAIRS):
            p = k % N
            m_yes = 3.0
            m_no = 3.0 - 4.0 / N if p != 0 else -1.0
            pid = f"N{N}_p{k}"
            allpos = list(range(N))
            rows.append(_row(f"{pid}_y", pid, N, "forall_yes", m_yes, [], allpos))
            rows.append(_row(f"{pid}_n", pid, N, "forall_no", m_no, [p], [i for i in allpos if i != p]))
            rows.append(_row(f"N{N}_ey{k}", None, N, "exists_yes", 1.0, [i for i in allpos if i != p], [p]))
            rows.append(_row(f"N{N}_en{k}", None, N, "exists_no", 1.0 if k < 2 else -1.0, allpos, []))
    return rows


@pytest.fixture
def e1_csv(tmp_path):
    p = tmp_path / "e1.csv"
    with ResumableCsvWriter(p, COLUMNS) as w:
        for r in make_rows():
            w.write(r)
    return p


def test_accuracy_and_false_yes(e1_csv):
    acc = A.accuracy_table(A.load_results([e1_csv])).set_index(["N", "qtype"])
    for N in NS:
        assert acc.loc[(N, "forall_yes"), "accuracy"] == 1.0
        assert np.isnan(acc.loc[(N, "forall_yes"), "false_yes_rate"])
        n_pos0 = sum(1 for k in range(N_PAIRS) if k % N == 0)
        assert acc.loc[(N, "forall_no"), "accuracy"] == pytest.approx(n_pos0 / N_PAIRS)
        assert acc.loc[(N, "forall_no"), "false_yes_rate"] == pytest.approx(1 - n_pos0 / N_PAIRS)
        assert acc.loc[(N, "exists_no"), "false_yes_rate"] == pytest.approx(2 / N_PAIRS)
        assert acc.loc[(N, "exists_yes"), "n"] == N_PAIRS


def test_pair_delta(e1_csv):
    df = A.load_results([e1_csv])
    pairs = A.pair_deltas(df)
    assert len(pairs) == len(NS) * N_PAIRS
    d = pairs[pairs["absent_pos"] != 0]
    np.testing.assert_allclose(d["delta"] * d["N"], 4.0)
    summ = A.delta_summary(d, n_boot=200).set_index("N")
    for N in NS:
        assert summ.loc[N, "delta_mean"] == pytest.approx(4.0 / N)
        assert summ.loc[N, "delta_x_N"] == pytest.approx(4.0)
        assert summ.loc[N, "delta_ci_lo"] <= summ.loc[N, "delta_mean"] <= summ.loc[N, "delta_ci_hi"]


def test_sdt_formula():
    r = A.sdt(hits=8, n_signal=10, fas=2, n_noise=10)
    z = NormalDist().inv_cdf
    H, F = 8.5 / 11, 2.5 / 11
    assert r["d_prime"] == pytest.approx(z(H) - z(F))
    assert r["criterion"] == pytest.approx(-(z(H) + z(F)) / 2)
    assert A.sdt(10, 10, 10, 10)["d_prime"] == pytest.approx(0.0)          # 다 Yes → d′=0
    assert A.sdt(10, 10, 10, 10)["criterion"] < 0                           # liberal(Yes 편향)
    assert np.isfinite(A.sdt(0, 10, 10, 10)["d_prime"])                     # 0/1 비율도 유한


def test_sdt_table_and_position(e1_csv):
    df = A.load_results([e1_csv])
    s = A.sdt_table(df).set_index("N")
    assert set(s.index) == set(NS)
    pos = A.position_table(df)
    fn = pos[pos["qtype"] == "forall_no"].set_index(["N", "odd_pos"])
    for N in NS:
        assert fn.loc[(N, 0), "accuracy"] == 1.0
        if N_PAIRS > N or N == 2:
            assert fn.loc[(N, 1), "accuracy"] == 0.0


def _e2(tmp_path):
    """absent 이미지가 present 보다 attention 을 덜 받는 가짜 E2."""
    rows = [r for r in make_rows() if r["qtype"] in ("forall_no", "forall_yes")]
    p = tmp_path / "e2.csv"
    rng = np.random.default_rng(0)
    with ResumableCsvWriter(p, COLUMNS + ["npz", "verify_max_abs_diff"]) as w:
        for r in rows:
            N = r["N"]
            img = np.full((L, H, N), 0.5 / N) + rng.normal(0, 1e-4, (L, H, N))
            for a in r["absent_positions"]:
                img[:, :, a] *= 0.5
            n_tok = np.full(N, 10)
            npz = tmp_path / "npz" / f"{r['qid']}.npz"
            atomic_save_npz(npz, img_sum=img, img_mean=img / n_tok, text_sum=np.full((L, H), 0.3),
                            sys_sum=np.full((L, H), 0.2), marker_sum=np.zeros((L, H)), n_tokens=n_tok)
            w.write(dict(r, npz=str(npz)))
    return p


def test_e2_summaries(tmp_path):
    meta = A.load_results([_e2(tmp_path)])
    arrays = A.load_e2(meta)
    summ = A.e2_layer_summary(A.e2_long(meta, arrays))
    fn = summ[summ["qtype"] == "forall_no"].pivot_table(index=["N", "layer"], columns="role", values="attn_sum")
    assert (fn["absent"] < fn["present"]).all()
    np.testing.assert_allclose(fn["absent"] / fn["present"], 0.5, rtol=1e-2)
    assert set(summ[summ["qtype"] == "forall_yes"]["role"]) == {"present"}
    diffs = A.e2_head_diff(meta, arrays, "img_sum")
    assert set(diffs) == {("fake", N) for N in NS}
    for (_, N), x in diffs.items():
        assert x.shape == (L, H) and (x < 0).all()


def test_main_writes_outputs(tmp_path, e1_csv):
    e2 = _e2(tmp_path)
    out = tmp_path / "results"
    written = A.main(["--e1", str(e1_csv), "--e2", str(e2), "--out_dir", str(out), "--n_boot", "100"])
    names = {p.name for p in written}
    for n in ("summary_accuracy.csv", "summary_delta.csv", "summary_sdt.csv", "summary_position.csv",
              "summary_e2_layers.csv", "accuracy_fake.png", "delta_sdt_fake.png", "position_fake.png",
              "e2_layers_attn_sum_forall_no_fake.png", "e2_headdiff_img_mean_fake.png"):
        assert n in names, n
    for p in written:
        assert p.exists() and p.stat().st_size > 0
    assert len(pd.read_csv(out / "summary_accuracy.csv")) == len(NS) * 4
