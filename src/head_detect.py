"""M5 1단계: E2 npz 에서 '탐지 헤드'를 골라 이미지별 의심 점수를 만든다 (focus.py 가 읽는 m_list 형식).

    python -m src.head_detect --e2 results/raw/e2_qwen3b.csv --k 1 5 --out_prefix results/raw/hf/heads --tag qwen3b

헤드 선택 (leave-one-class-out): 평가 클래스 c 를 뺀 나머지 클래스의 forall_no 문항에서
    s[l,h] = (absent 이미지의 토큰당 attention / 이미지 평균) − 1
의 평균/표준편차(z)가 가장 작은 k개 헤드(= 객체 없는 이미지를 일관되게 덜 보는 헤드)를 고른다.
점수: 그 헤드들에서 이미지별 (토큰당 attention / 이미지 평균) 의 평균. 낮을수록 객체가 없을 가능성이 큼
-> focus.py 가 forall 문항에서 argmin 이미지를 집중 대상으로 쓴다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e2", required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--out_prefix", default="results/raw/hf/heads")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args(argv)

    d = pd.read_csv(args.e2)
    d = d[d.qtype.str.startswith("forall")].reset_index(drop=True)
    Z = []
    for p in d.npz:
        z = np.load(p)["img_mean"]                        # [L, H, N]
        Z.append(z / z.mean(-1, keepdims=True))
    no = (d.qtype == "forall_no").values
    for k in args.k:
        ms = [None] * len(d)
        for c in d.cls.unique():
            tr = ((d.cls != c) & no).values
            S = np.stack([Z[i][:, :, json.loads(d.absent_positions[i])[0]] - 1 for i in np.where(tr)[0]])
            t = S.mean(0) / (S.std(0) + 1e-6)
            idx = np.argsort(t.ravel())[:k]
            for i in np.where((d.cls == c).values)[0]:
                sc = Z[i].reshape(-1, Z[i].shape[-1])[idx].mean(0)
                ms[i] = json.dumps([round(float(x), 5) for x in sc])
        out = Path(f"{args.out_prefix}_k{k}_{args.tag}.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        d.assign(m_list=ms)[["qid", "qtype", "N", "cls", "m_list"]].to_csv(out, index=False)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
