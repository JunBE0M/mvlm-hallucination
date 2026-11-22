# mvlm-hallucination

다중 이미지 VLM의 객체 존재 환각 연구용 실험 코드.
**실험 결과, 결론, 다음 실험은 [EXPERIMENTS.md](EXPERIMENTS.md)에 정리했다.**
"Is there a {obj} in all {N} images?"에 대해 한 장에 객체가 없어도 N≥4부터 거의 항상 Yes라고 답하는 현상
(Qwen2.5-VL 3B/7B, COCO)을 다룬다. 가설: 모델이 이미지 간 정보를 min(∀)이 아니라 max/mean(∃)으로 집계하고,
마지막 토큰의 attention이 객체가 없는 이미지를 오히려 덜 본다.

## 실행 (Vast.ai)

```bash
git clone <repo> && cd mvlm-hallucination
export HF_TOKEN=...            # 필요할 때만, 코드/파일에 저장하지 말 것
bash setup.sh                  # HF_HOME=/workspace/hf_cache, pip, COCO val2017+annotations(있으면 skip), nvidia-smi
python -m src.data --config configs/default.yaml --out data/questions.jsonl
# 옵션: --area_range 0.05 0.3   --odd_position fixed:-1   --seed 1
pytest -m gpu -s                                  # 실제 모델로 Yes/No 토큰 확인
python -m src.run_eval --model qwen2.5-vl-3b --out results/raw/e1_qwen3b.csv --limit 5   # 디버그
python -m src.run_eval --model qwen2.5-vl-3b --out results/raw/e1_qwen3b.csv             # 전체 (같은 --out이면 이어서)
# 옵션: --Ns 4 8  --qtypes forall_yes forall_no  --max_pixels 200704  --attn_impl sdpa
python -m src.attn_probe --model qwen2.5-vl-3b --out results/raw/e2_qwen3b.csv --limit 2 --verify   # 검증
python -m src.attn_probe --model qwen2.5-vl-3b --out results/raw/e2_qwen3b.csv --qtypes forall_yes forall_no exists_yes
python -m src.analyze --e1 results/raw/e1_qwen3b.csv results/raw/e1_qwen7b.csv --e2 results/raw/e2_qwen3b.csv
```

로컬(CPU) 테스트: `python -m pytest -m "not gpu"`
- `test_tokens.py`, `test_tiny_model.py`는 Qwen2.5-VL의 tokenizer/processor/config 파일(가중치 제외, 수 MB)이 HF 캐시에 있어야 실행되고 없으면 skip:
  `python -c "from huggingface_hub import snapshot_download as s; [s(r, allow_patterns=['*.json','*.txt']) for r in ['Qwen/Qwen2.5-VL-3B-Instruct','Qwen/Qwen2.5-VL-7B-Instruct']]"`
- `test_tiny_model.py`는 실제 config를 축소한 **랜덤 가중치 소형 모델**(3층, 4 head / 2 KV head)로 실제 transformers 코드 경로를 CPU에서 통과시킨다.

## 레포 구조

| 경로 | 역할 |
|---|---|
| `setup.sh` | Vast 환경 세팅 |
| `configs/default.yaml` | 데이터·모델 설정 |
| `src/data.py` | COCO에서 문항 jsonl 생성 |
| `src/models.py` | 모델 로딩/입력 구성 (모델별 차이 흡수) |
| `src/run_eval.py` | E1: yes/no logit |
| `src/attn_probe.py` | E2: 마지막 토큰→이미지별 attention |
| `src/analyze.py` | 지표, 그래프 |
| `src/intervene.py` | E3: 특정 이미지 attention ×α 개입 |
| `src/make_neg.py` | M0: 부정 질문 jsonl 생성 |
| `src/per_image.py` | M1: 이미지별 단일 질문 + 집계 |
| `src/isolated.py` | M2: 이미지별 격리 판독 (attention knockout) |
| `src/focus.py` | M3/M5: 선택한 이미지에 attention 집중 |
| `src/head_detect.py` | M5: E2 npz에서 탐지 헤드 선택, 이미지별 점수 |
| `src/compare_methods.py` | 방법 비교 표 |
| `logs/*.sh` | 실제로 돌린 실험 큐 |
| `results/` | 요약 csv, 그림(`figs/`), 문항별 원자료 csv(`raw/`). npz와 eager 시절 결과(`raw/_eager_old/`)는 gitignore |

## 설계 결정

### 데이터 (`src/data.py`)
- **present**: 해당 클래스 annotation이 1개 이상 있는 이미지. **absent**: 해당 클래스 annotation이 전혀 없는 이미지.
  (COCO 라벨 누락 가능성은 남아 있음.)
- **면적 비율**: 이미지별로 해당 클래스 bbox 중 최대 `w*h / (W*H)`. segmentation `area`가 아니라 bbox 기준.
  `area_range: [lo, hi]`(양끝 포함)는 present 이미지에만 적용.
- **유형**: `forall_yes`(N장 present), `forall_no`(forall_yes 묶음에서 1장만 absent로 교체, 같은 `pair_id`),
  `exists_yes`(1 present + N−1 absent), `exists_no`(N absent). Δ = m(forall_yes) − m(forall_no)는 `pair_id`로 짝지어 계산.
- **홀로 다른 이미지 위치**(forall_no의 absent, exists_yes의 present): `odd_position: uniform`이면 (class, N) 셀 안에서
  0..N−1을 정확히 균등하게 배분한 뒤 섞고, `fixed:k`면 고정(`k<0`은 뒤에서부터).
- **재현성**: (seed, class, N)마다 독립 RNG(`random.Random(f"{seed}|{cls}|{N}")`). 클래스/N 목록 순서를 바꿔도 같은 셀은 같은 문항.
- jsonl 필드: `qid, pair_id, cls, N, qtype, question, answer, image_ids, image_paths, present_positions,
  absent_positions, area_ratios, seed`. `image_paths`는 `coco.root` 기준 상대경로.

### 모델 (`src/models.py`)
- 공통 인터페이스: `load_model(name, attn_impl, ...)` → `vlm.build_inputs(images, question)` →
  `vlm.get_yes_no_logits(inputs) -> (logit_yes, logit_no)`, `vlm.segments(inputs)`(E2용 이미지/text/system 구간).
- **이미지 해상도**: `model.max_pixels`(기본 448*448=200704). processor *호출 시* `size={"longest_edge", "shortest_edge"}`로 전달.
  종횡비는 유지되므로 이미지당 토큰 수는 같은 max_pixels에서도 조금씩 다르다 (640×480 → 234 토큰, 기본값이면 391).
  실제 토큰 수는 E1 csv의 `n_image_tokens`에 기록.
- **Yes/No logit**: chat template(`add_generation_prompt=True`) 적용 후 마지막 위치(= assistant 첫 응답 토큰을 예측하는 위치)의
  logit. `generate` 없이 forward 1회, `logits_to_keep=1`로 마지막 위치만 계산. `m = logit("Yes") − logit("No")`, `pred = Yes if m > 0`.
  참고용으로 `" Yes"`/`"yes"` 변형 logit, `p_yes`, `p_no`(전체 vocab softmax), top-1 토큰도 저장 → Yes/No 외 토큰으로 답하려는지 확인 가능.
- `run_eval.py` 재시작: 같은 `--out`이면 기존 csv의 qid를 건너뜀. 행 단위 append + 주기적 fsync,
  끊겨서 잘린 마지막 줄은 다음 실행 때 자동 제거.

### E2: attention probe (`src/attn_probe.py`)
**목표**: 마지막 query 토큰(= 답 토큰을 예측하는 위치)이 각 이미지·텍스트·시스템 구간에 주는 attention 총량을 [layer, head, image] 단위로 저장.

**구현 방식** (`output_attentions=True` 미사용 — 전 층 [H, L, L] 를 쌓으면 수십 GB):
1. transformers 5.22.2의 `Qwen2_5_VLAttention.forward`는 q/k에 mrope를 적용한 뒤
   `ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation, eager_attention_forward)`로 함수를 골라
   `fn(module, query[B,H,L,d], key[B,H_kv,L,d], value, attention_mask, scaling=..., ...)`로 호출한다.
   `"eager"`는 레지스트리에 기본 등록되어 있지 않아 모델 파일의 `eager_attention_forward`가 쓰인다.
2. probe는 `ALL_ATTENTION_FUNCTIONS["eager"]`에 래퍼를 **임시 등록**(인스턴스 local mapping, `with` 종료 시 원복)한다.
   모델은 `attn_implementation="eager"`로 로드.
3. 래퍼는 원래 `eager_attention_forward`로 출력을 계산해 **그대로 반환**(모델 동작 불변)하고, 언어모델 attention 모듈일 때만
   (`isinstance(module, Qwen2_5_VLAttention)` — 비전 인코더 attention도 같은 함수를 호출하므로 제외)
   `softmax(q_last · repeat_kv(K)ᵀ · scaling + mask[-1])`를 fp32로 직접 계산한다. GQA는 `repeat_kv`
   (KV head마다 연속 복제, transformers와 같은 순서)로 처리.
4. 결과 [H, L]을 즉시 구간 one-hot [S, L]에 곱해 [H, S]로 축약 후 CPU로 옮긴다 → 층당 저장 H×S.

**구간** (`src/segments.py`, 토큰 id는 tokenizer에서 조회): 이미지 i = i번째 `<|vision_start|>`와 `<|vision_end|>` *사이*의 토큰,
system = user 턴을 여는 `<|im_start|>` 이전(기본 system prompt와 첫 토큰 포함), marker = vision_start/end 토큰 자체,
text = 나머지(질문, chat 포맷 토큰, 마지막 query 토큰 자신 포함). 모든 위치가 정확히 한 구간에 속하므로 구간 합의 총합 = 1.

**저장** (문항별 `<out stem>_npz/{qid}.npz`, float32, 압축; 3B N=8 기준 수십 KB):
`img_sum[L,H,N]`, `img_mean[L,H,N]`(= sum / 이미지 토큰 수), `text_sum[L,H]`, `sys_sum[L,H]`, `marker_sum[L,H]`,
`n_tokens[N]`, `n_text_tokens`, `n_sys_tokens`, `logit_yes`, `logit_no`. 메타 csv는 E1과 같은 열 + `npz`, `verify_max_abs_diff`.
npz는 임시파일 → `os.replace`로 원자적 저장, csv와 npz가 모두 있는 qid만 완료로 간주(재시작 지원). batch size 1 전용.

**검증**:
- `tests/test_probe_math.py`: (a) 랜덤 텐서에서 `last_row_attention` == transformers `eager_attention_forward` 전체 행렬의 마지막 행
  (GQA 1:1/4:1/7:1, float/bool/없는 마스크), (b) 소형 랜덤 Qwen2.5-VL에서 probe가 logits를 바꾸지 않음, eager attn_weights와 오차 < 1e-5,
  **`output_attentions=True` 결과의 구간 합과 층별로 일치**, 구간 합 = 1, 예외 시 레지스트리 원복, CLI 재시작.
- Vast에서 `--verify`: eager가 함께 반환하는 attn_weights 마지막 행과 직접 계산한 값의 최대 절대오차를 출력/저장.
  bf16에서는 eager가 matmul을 bf16으로 하고 probe는 fp32로 하므로 ~1e-3 수준 차이는 정상.

### 분석 (`src/analyze.py`)
- 정확도·Yes 비율·False-Yes rate(정답 No 유형에서 P(pred=Yes)) — model × N × qtype.
- forall 쌍 Δ = m(forall_yes) − m(forall_no) (`pair_id`로 매칭): 평균, 95% bootstrap CI, **Δ×N**.
  평균(mean) 집계라면 absent 1장의 영향이 1/N로 희석되어 Δ×N이 대략 일정, max 집계라면 Δ ≈ 0.
- SDT: forall_yes = signal, forall_no = noise. log-linear 보정 `(k+0.5)/(n+1)` 후 d′ = z(H) − z(F), c = −(z(H)+z(F))/2 (c<0 = Yes 편향).
- 홀로 다른 이미지 위치별 정확도 (forall_no의 absent 위치, exists_yes의 present 위치).
- E2: 층별 absent vs present 이미지 attention(헤드 평균; sum과 토큰당 mean 두 가지), forall_no의 (absent − present) layer×head 히트맵.
- E1 csv 없이 E2 csv만 줘도 행동 지표를 계산(E2도 logit 저장).

### 확인된 사항 (transformers 5.22.2 소스와 Qwen2.5-VL tokenizer/config 파일을 직접 확인)
| 항목 | 값 |
|---|---|
| 모델 클래스 / processor | `Qwen2_5_VLForConditionalGeneration` / `Qwen2_5_VLProcessor` (`AutoProcessor`), image processor `Qwen2VLImageProcessor` |
| `from_pretrained` dtype 인자 | `dtype=` (`torch_dtype` 아님) |
| 해상도 지정 | 호출 시 `size={"longest_edge": max_pixels, "shortest_edge": min_pixels}`. `from_pretrained(size=...)`는 **반영 안 됨**, `min_pixels/max_pixels` 호출 인자는 deprecated |
| Yes/No 토큰 | `"Yes"`=9454, `"No"`=2753, `" Yes"`=7414, `" No"`=2308 — 서로 다름. 프롬프트가 `<\|im_start\|>assistant\n`으로 끝나므로 공백 없는 쪽 사용 |
| 특수 토큰 | `<\|vision_start\|>`=151652, `<\|vision_end\|>`=151653, `<\|image_pad\|>`=151655, `<\|im_start\|>`=151644 (코드는 tokenizer에서 조회) |
| 기본 system prompt | chat template이 `You are a helpful assistant.`를 자동 삽입 |
| 3B 구조 | 36층, 16 head, 2 KV head (GQA 8:1) |
| 7B 구조 | 28층, 28 head, 4 KV head (GQA 7:1) |
| 기타 | processor import 시 torchvision 필요. `qwen-vl-utils` 불필요 |

## 확인 필요 (추측 금지)
- 확장 모델 Qwen3-VL-8B, InternVL3-8B, LLaVA-OneVision-7B: HF repo id, transformers 클래스명, 필요 transformers 버전,
  이미지 구간 경계 토큰은 **확인 필요**. 현재는 `MODEL_REGISTRY`에 이름만 있고 로드 시 `NotImplementedError`.
- Vast: PyPI의 `torch==2.14.1` Linux 휠이 요구하는 CUDA 드라이버 버전은 확인 필요 → `setup.sh` 출력의 `cuda: True` 확인.
  Vast 템플릿에 다른 torch가 깔려 있으면 pip이 교체한다.
