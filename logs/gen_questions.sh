#!/usr/bin/env bash
# 실험 기록 3: 문항 재생성 (seed 0) + 학습용 seed 1
cd /workspace/mvlm-hallucination; source /venv/main/bin/activate
python -m src.data --config configs/default.yaml --out data/questions.jsonl
python -m src.make_neg
python -m src.make_variants
python -m src.data --config configs/bigN.yaml  --out data/questions_bigN.jsonl
python -m src.data --config configs/small.yaml --out data/questions_small.jsonl
python -m src.make_neg --questions data/questions_bigN.jsonl --out data/questions_neg_bigN.jsonl
python -m src.data --config configs/train_s1.yaml --out data/questions_s1.jsonl
python -m src.distill split
python -m src.make_variants --quant --out data/questions_quant.jsonl
# 실험 E 후속: ∀ 이미지 묶음(N 2-16)에 ∃ 문장을 붙인 파일 (M2 이미지별 판독을 ∃ 문장으로 -> min 집계). qid 접미사 __exq
python - <<'PY'
import json
T = "Is there a {obj} in any of the {N} images? Please answer with Yes or No."
with open("data/questions_forall_exq.jsonl", "w") as g:
    for f in ["data/questions.jsonl", "data/questions_bigN.jsonl"]:
        for l in open(f):
            q = json.loads(l)
            if q["qtype"].startswith("forall"):
                q["question"] = T.format(obj=q["cls"], N=q["N"]); q["qid"] += "__exq"; g.write(json.dumps(q) + "\n")
PY
