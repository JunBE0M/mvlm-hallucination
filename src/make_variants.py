"""질문 문장/양화사 변형: 같은 이미지 묶음에 다른 질문 템플릿을 붙인다 (A2 비대칭 확장 + 문장 다양화).

    python -m src.make_variants --questions data/questions.jsonl --out data/questions_var.jsonl
    python -m src.make_variants --quant --out data/questions_quant.jsonl      # 실험 C: 일반 양화사

--quant (실험 C, 2026-11-26): present 이미지 수 c 를 통제한 새 이미지 묶음을 만들고 일반 양화사 질문을 붙인다.
  (class, N) 마다 c = 0..N 각각 reps 개 묶음 (N=4: 4, N=8: 3) -> 모든 k 에 대해 경계(k-1, k) 문항이 균등하게 들어간다.
  present 위치는 무작위. RNG = Random(f"quant|{seed}|{cls}|{N}").  정답은 c 로 결정:
    atleast2      "Is there a {obj} in at least 2 of the {N} images?"   Yes iff c >= 2
    atleast_half  "Is there a {obj} in at least {k} of the {N} images?" k = N/2 (N=4 이면 atleast2 와 같아서 생략)
    most          "Is there a {obj} in most of the {N} images?"         Yes iff c > N/2
    none          "Do none of the {N} images contain a {obj}?"          Yes iff c == 0  (기존 none 문장)
    all / any     기존 ∀ / ∃ 문장 (같은 묶음에서 비교용)
  필드: set_id, n_present, template, quant, k(해당 시), qid = f"{set_id}__{template}".

각 레코드에 template(이름), quant(forall | exists | none) 필드를 추가. qid / pair_id 에 "__{template}" 접미사.
none(= ¬∃, "하나도 없나?")은 exists 묶음에 붙이고 정답을 뒤집는다 (exists_no 묶음 -> Yes).
"""
from __future__ import annotations

import argparse
import json

SUFFIX = " Please answer with Yes or No."
TEMPLATES = {
    # name: (원래 qtype 접두사, quant, 문장, 정답 뒤집기)
    "all_contain": ("forall", "forall", "Do all {N} images contain a {obj}?", False),
    "every_show": ("forall", "forall", "Does every one of the {N} images show a {obj}?", False),
    "atleast1": ("exists", "exists", "Does at least one of the {N} images contain a {obj}?", False),
    "any_show": ("exists", "exists", "Do any of the {N} images show a {obj}?", False),
    "none": ("exists", "none", "Do none of the {N} images contain a {obj}?", True),
}


QUANT_TEMPLATES = {
    # name: (문장, k 함수(N) 또는 None, 정답 함수(c, N))
    "all": ("Is there a {obj} in all {N} images?", None, lambda c, N: c == N),
    "any": ("Is there a {obj} in any of the {N} images?", None, lambda c, N: c >= 1),
    "atleast2": ("Is there a {obj} in at least 2 of the {N} images?", lambda N: 2, lambda c, N: c >= 2),
    "atleast_half": ("Is there a {obj} in at least {k} of the {N} images?", lambda N: N // 2, lambda c, N: c >= N // 2),
    "most": ("Is there a {obj} in most of the {N} images?", None, lambda c, N: c > N / 2),
    "none": ("Do none of the {N} images contain a {obj}?", None, lambda c, N: c == 0),
}


def make_quant(args):
    import random

    from .data import build_pools, load_coco_index, load_config
    cfg = load_config(args.config)
    index = load_coco_index(f"{cfg['coco']['root']}/{cfg['coco']['ann_file']}")
    image_dir = cfg["coco"]["image_dir"]
    reps = {4: 4, 8: 3}
    n = 0
    with open(args.out, "w") as g:
        for cls in cfg["data"]["classes"]:
            cid = index.cat_name_to_id[cls]
            present, absent = build_pools(index, cls)
            ratios = index.area_ratio.get(cid, {})
            for N in (4, 8):
                rng = random.Random(f"quant|{args.seed}|{cls}|{N}")
                for c in range(N + 1):
                    for r in range(reps[N]):
                        pos = sorted(rng.sample(range(N), c))
                        pres = rng.sample(present, c)
                        ab = rng.sample(absent, N - c)
                        ids = [pres.pop() if p in pos else ab.pop() for p in range(N)]
                        set_id = f"{cls.replace(' ', '_')}_N{N}_c{c}_r{r}"
                        for name, (tmpl, kf, ans) in QUANT_TEMPLATES.items():
                            if name == "atleast_half" and N == 4:
                                continue
                            k = kf(N) if kf else None
                            rec = dict(qid=f"{set_id}__{name}", pair_id=set_id, set_id=set_id, cls=cls, N=N,
                                       qtype=f"quant_{name}", template=name, quant=name, k=k, n_present=c,
                                       question=tmpl.format(obj=cls, N=N, k=k) + SUFFIX,
                                       answer="Yes" if ans(c, N) else "No", image_ids=ids,
                                       image_paths=[f"{image_dir}/{index.images[i]['file_name']}" for i in ids],
                                       present_positions=pos, absent_positions=[p for p in range(N) if p not in pos],
                                       area_ratios=[round(ratios.get(i, 0.0), 6) for i in ids], seed=args.seed)
                            g.write(json.dumps(rec) + "\n")
                            n += 1
    print(f"wrote {n} questions to {args.out}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/questions.jsonl")
    ap.add_argument("--out", default="data/questions_var.jsonl")
    ap.add_argument("--templates", nargs="*", default=list(TEMPLATES))
    ap.add_argument("--quant", action="store_true", help="실험 C: present 수를 통제한 일반 양화사 문항")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.quant:
        return make_quant(args)
    qs = [json.loads(l) for l in open(args.questions)]
    n = 0
    with open(args.out, "w") as g:
        for name in args.templates:
            fam, quant, tmpl, flip = TEMPLATES[name]
            for q in qs:
                if not q["qtype"].startswith(fam):
                    continue
                r = dict(q)
                r["qid"] = f"{q['qid']}__{name}"
                r["pair_id"] = f"{q['pair_id']}__{name}"
                r["question"] = tmpl.format(N=q["N"], obj=q["cls"]) + SUFFIX
                if flip:
                    r["answer"] = "No" if q["answer"] == "Yes" else "Yes"
                r["template"], r["quant"] = name, quant
                g.write(json.dumps(r) + "\n")
                n += 1
    print(f"wrote {n} questions to {args.out}")


if __name__ == "__main__":
    main()
