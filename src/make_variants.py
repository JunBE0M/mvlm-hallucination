"""질문 문장/양화사 변형: 같은 이미지 묶음에 다른 질문 템플릿을 붙인다 (A2 비대칭 확장 + 문장 다양화).

    python -m src.make_variants --questions data/questions.jsonl --out data/questions_var.jsonl

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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/questions.jsonl")
    ap.add_argument("--out", default="data/questions_var.jsonl")
    ap.add_argument("--templates", nargs="*", default=list(TEMPLATES))
    args = ap.parse_args(argv)
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
