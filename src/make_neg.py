"""M0 용 부정 질문 생성: forall 질문을 "없는 이미지가 하나라도 있나?" 로 바꾸고 정답을 뒤집는다.

    python -m src.make_neg --questions data/questions.jsonl --out data/questions_neg.jsonl
"""
from __future__ import annotations

import argparse
import json

TEMPLATE = ("Is there any image among the {N} images that does not contain a {cls}? "
            "Please answer with Yes or No.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/questions.jsonl")
    ap.add_argument("--out", default="data/questions_neg.jsonl")
    args = ap.parse_args(argv)
    n = 0
    with open(args.questions) as f, open(args.out, "w") as g:
        for line in f:
            q = json.loads(line)
            if not q["qtype"].startswith("forall"):
                continue
            q["question"] = TEMPLATE.format(N=q["N"], cls=q["cls"])
            q["answer"] = "No" if q["answer"] == "Yes" else "Yes"
            g.write(json.dumps(q) + "\n")
            n += 1
    print(f"wrote {n} questions to {args.out}")


if __name__ == "__main__":
    main()
