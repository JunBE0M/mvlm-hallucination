# 실험 기록 2: 5개 모델 확장 (2026-11-26, Vast.ai RTX 4090 48GB ×4)

모델 5개: Qwen2.5-VL-3B/7B, **Qwen3-VL-8B, InternVL3-8B, LLaVA-OneVision-7B**(새로 `MODEL_REGISTRY`에 구현, `src/models.py` `GenericVLM`).
같은 COCO 문항(셀당 200문항), 같은 지표 m = logit("Yes") − logit("No"). 집계 분석은 `python -m src.agg_multi` → `results/multi/*.csv`, 요약 출력 `results/multi_summary.txt`,
그림은 `python -m src.plot_multi` → `results/figs/multi_*.png`. 실행 큐: `logs/queue_gen.sh`(단계별), `logs/launch_*.sh`, `logs/sched_extra.sh`.

**모델별 입력 설정 (공정 비교용)**
- Qwen2.5-VL: max_pixels 448², Qwen3-VL: 같은 픽셀 상한(이미지당 약 190토큰).
- InternVL3: `crop_to_patches=False` → 이미지당 448px 타일 1개(256토큰).
- LLaVA-OV: 다중 이미지는 anyres 없이 base 384px(730토큰, 원 논문 multi-image 설정). 단일 이미지(M1)는 모델 기본 anyres.
- 재현 확인: Qwen2.5-VL E1을 RTX 5090 → 4090에서 다시 돌려 답 일치 98.7%/99.3%, m 상관 0.999.

## 0. 한눈에 보는 결론 (논문 주장 후보)

1. **현상은 5개 모델 모두에서 나온다.** ∀ 질문 AUC가 N과 함께 떨어진다 (N=2 → 8 → 16).
   Qwen2.5-3B .84→.61→.53, Qwen2.5-7B .88→.65→.56, Qwen3-VL .95→.78→.70, InternVL3 .97→.69→.58, LLaVA-OV .94→.66→.55.
   ∃ 질문은 같은 N에서도 AUC .94–.99를 유지한다.
2. **Δ ∝ N^−γ (1/N 희석).** γ(N=2–16) = 1.47 / 1.31 / **0.52** / 1.50 / 1.41. 문항 단위로도 "평균 모형"이 가장 잘 맞는다
   (R² .46 / .58 / .36 / .75 / .70; 희석 없는 모형 .04–.21, min 모형 .09–.44).
   예외는 Qwen3-VL: 희석이 약하고(γ≈0.5) min 모형 R²(.44)가 평균 모형(.36)보다 약간 높다. 그래도 ∀ AUC는 N=16에서 .70까지 떨어진다.
3. **양화사 비대칭 (가장 새로운 결과): ∃는 max로, ∀는 평균으로 합친다. 5개 모델 × 모든 문장 변형에서 재현.**
   정규화 집계 온도 β·sd(e) (<0 max 쪽, 0 평균, >0 min 쪽):
   ∃ −2.7 / −2.9 / −5.7 / −3.5 / −9.2 (R² .91–.92), ∀ −0.51 / −0.11 / +0.35 / −0.07 / −0.03.
   ¬∃("none")도 max 쪽(−1.5 / −0.9 / −5.7 / −1.1 / −2.9). 문장을 바꿔도(all_contain, every_show, atleast1, any_show) 같다.
   모델은 "하나라도"에는 올바른 연산자(max)로 전환하지만 "모두"에는 min으로 전환하지 못한다.
4. **비대칭은 판독이 생기는 중·후반 층에서 동시에 형성된다 (logit lens).** 층별로 읽은 m이 판별력을 갖기 시작하는 지점(상대 깊이 약 0.6–0.7)부터
   ∃는 β<0(max)로 가고 ∀는 끝까지 β≈0(평균)에 머문다 (`multi_lens_beta.png`).
5. **인과 검증 (E3, 5개 모델).** absent 이미지 attention ×4 → 거짓 Yes 감소 (전체 N: .45→.40, .26→.09, .14→.02, .47→.27, .17→.05),
   ×0(가림) → 증가 (.58, .44, .39, .64, .32). 방향이 모든 모델에서 예측과 같다. 대조(present ×4)는 오히려 Yes 쪽.
6. **해결책 M2 (같은 입력·같은 프롬프트, 이미지별 격리 판독 후 min).** ∀ AUC(N=2–8):
   base .66/.74/.84/.79/.76 → **M2 .91/.93/.83/.95/.93** (4개 모델에서 p<.001, paired bootstrap).
   **Qwen3-VL에서는 개선 없음**(.83 vs .84, p=.10). M0(부정 질문)은 +.01–.10.
   M1_min(이미지마다 따로 묻기, 상한선) .97–.99.
7. **견고성**: 작은 물체(bbox 면적 ≤5%)에서도 같은 패턴(∀ AUC N=8: .54/.58/.66/.61, ∃ .94–.95; ∀ β≈0, ∃ β<0).

## 1. 결과 표

### 행동 (forall, N=2 / 8 / 16)
| 모델 | 거짓 Yes (forall_no) | d′ | ∀ AUC | ∃ AUC |
|---|---|---|---|---|
| Qwen2.5-VL-3B | .14 / .82 / .97 | 1.45 / .28 / .07 | .84 / .61 / .53 | .99 / .97 / .94 |
| Qwen2.5-VL-7B | .27 / .35 / .25 | 1.50 / .53 / .25 | .88 / .65 / .56 | .99 / .98 / .95 |
| Qwen3-VL-8B | .04 / .14 / .25 | 2.64 / 1.31 / .74 | .95 / .78 / .70 | .99 / .99 / .97 |
| InternVL3-8B | .02 / .77 / .94 | 2.91 / .59 / .19 | .97 / .69 / .58 | 1.00 / .98 / .97 |
| LLaVA-OV-7B | .01 / .34 / .54 | 2.47 / .59 / .15 | .94 / .66 / .55 | .99 / .97 / .96 |

Yes 편향 방향은 모델마다 다르다(3B·InternVL은 Yes, Qwen3-VL·7B는 No 쪽). 공통점은 **판별력(d′, AUC) 붕괴**다.

### 질문 문장 변형 (N=8 AUC)
| 모델 | ∀ all_contain | ∀ every_show | ∃ atleast1 | ∃ any_show | ¬∃ none |
|---|---|---|---|---|---|
| Qwen2.5-VL-3B | .61 | .61 | .97 | .98 | .91 |
| Qwen2.5-VL-7B | .65 | .64 | .98 | .98 | .98 |
| Qwen3-VL-8B | .77 | .75 | .99 | .99 | .98 |
| InternVL3-8B | .69 | .69 | .98 | .98 | .97 |
| LLaVA-OV-7B | .67 | .66 | .97 | .97 | .97 |

### 방법 비교 ∀ AUC (N=2–8 전체, 95% CI는 `results/multi/methods.csv`)
| 모델 | base | M0 부정 질문 | M1_mean | **M2 격리 판독** | M1_min (상한) |
|---|---|---|---|---|---|
| Qwen2.5-VL-3B | .663 | .713 | .864 | **.913** | .970 |
| Qwen2.5-VL-7B | .739 | .836 | .848 | **.928** | .982 |
| Qwen3-VL-8B | .844 | .849 | .867 | .831 | .987 |
| InternVL3-8B | .791 | .840 | .896 | **.953** | .989 |
| LLaVA-OV-7B | .763 | .817 | .881 | **.927** | .989 |

N=12/16에서 M2: Qwen2.5-3B .78/.76, 7B .86/.84 (base .54/.53, .60/.56). Qwen3-VL·InternVL의 N=12/16 M2는 재실행 중이라 표에서 제외.

## 2. 해석과 한계
- 주장 구조: (현상) 다중 이미지 ∀ 환각 → (법칙) Δ∝1/N → (메커니즘) 양화사 비대칭: ∃=max, ∀=mean, 중·후반 층에서 형성 → (인과) attention 개입 → (해결) M2.
- **Qwen3-VL이 예외**다: 희석이 약하고(γ≈0.5) M2도 효과가 없다. 대신 ∀ β가 약간 min 쪽(+0.35)이다.
  "최신 모델은 부분적으로 min 쪽으로 이동했지만 ∃ 수준의 연산자 전환에는 못 미친다"로 쓸 수 있다. M2가 안 통하는 이유는 추가 분석 필요(이미지별 m의 스케일·임계값 문제 가능).
- M2는 임계값 0에서 No 쪽으로 쏠리는 문제가 그대로다(라벨 없는 보정 미해결).
- 데이터는 COCO 한 종류, 클래스 20개. 공개 벤치마크(MIOH, MIHBench)와 기존 방법(VCD, PAI) 비교는 아직이다.
- lens의 β는 판독이 우연 수준인 층(AUC<0.6)에서는 의미가 없어서 그림에서 가렸다.

## 3. 재현
```bash
python -m src.make_neg; python -m src.make_variants
python -m src.data --config configs/bigN.yaml  --out data/questions_bigN.jsonl
python -m src.data --config configs/small.yaml --out data/questions_small.jsonl
CUDA_VISIBLE_DEVICES=0 bash logs/queue_gen.sh qwen3-vl-8b qwen3vl8b e1 m1 m2 m0 var big m1big e3 lens small m1small m2big
python -m src.agg_multi --out_dir results/multi > results/multi_summary.txt
python -m src.plot_multi
```
주의: GPU당 8B 모델 프로세스 2개까지(3개 이상은 OOM). LLaVA-OV 단일 이미지(M1)는 anyres라 0.65s/장으로 느리다(`per_image --shard i/k`로 분할).

---

# 실험 기록 1: Qwen2.5-VL (2026-11-22, Vast.ai RTX 5090)

Qwen2.5-VL 3B/7B, COCO val2017. 문항은 20개 클래스 × N∈{2,4,6,8} × 유형별 10개이고, 한 셀(model × N × qtype)당 200문항이다.
지표 m = logit("Yes") − logit("No") (마지막 위치), pred = Yes if m > 0.
재현 스크립트는 `logs/*.sh`, 문항별 원자료는 `results/raw/`, 요약 표는 `results/summary_*.csv`, 그림은 `results/figs/`에 있다.

## 0. 한눈에 보는 결론

1. **현상**: N이 커질수록 "Is there a {obj} in all N images?"에서 판별력이 무너진다.
   3B는 N=8 forall_no의 82%를 Yes라고 답하고, d′는 1.45에서 0.28로 떨어진다. 7B는 d′가 1.50에서 0.53으로 떨어진다.
2. **원인은 인식이나 attention이 아니라 집계 방식이다.** 이미지별 증거를 min이 아니라 **평균으로** 합친다.
   - 객체가 없는 이미지 1장의 효과 Δ는 1/N로 희석된다. Δ×N은 N과 무관하게 거의 일정하다.
   - 이미지별 단일 질문 결과를 평균으로 합치면 원래 모델과 정확도가 같고(3B 0.616 = 0.616), min으로 합치면 0.854가 된다.
   - 마지막 토큰은 객체 없는 이미지를 실제로 덜 보고(0.7~0.8배), 객체 있는 이미지를 더 본다(최대 4.9배).
     모델은 어느 이미지에 객체가 있는지 구분하고 있다.
3. **인과 검증**: 객체 없는 이미지로 가는 attention을 ×4로 키우면 거짓 Yes가 줄고(7B N=8 35%→15%), 가리면 늘어난다.
   그 이미지의 "No" 증거는 답에 쓰이지만, 평균에 희석된다.
4. **해결 (같은 입력·같은 프롬프트 조건)**:
   - **M2 isolated readout**(질문 토큰이 이미지 하나씩만 보게 attention 차단, 이미지별 m의 min)이 가장 강하다.
     AUC 0.91/0.93이지만, 답이 No 쪽으로 쏠려서 임계값 보정이 필요하다.
   - **M5 탐지 헤드 → attention 집중**(forward 2회)은 중간 정도의 개선이다(AUC 3B 0.66→0.71, 7B 0.74→0.82).
   - M1(이미지마다 따로 묻기)은 입력과 프롬프트를 바꾸므로 **방법이 아니라 상한선/진단용**이다.

## 1. 구현상 주의점 (결과 해석 전에 반드시)

- **eager attention은 쓰지 말 것.** transformers의 `eager_attention_forward`는 QKᵀ·softmax·PV를 bf16으로 계산한다.
  3B에서는 오차가 작지만, 7B에서는 답이 sdpa와 크게 달랐다(N=2 평균 m: sdpa −2.26 vs eager +0.40, 답 일치율 73%).
  그래서 probe와 개입 모두 래퍼는 `"eager"` 키에 등록하되, 실제 계산은 sdpa(fp32 누적)로 한다.
  sdpa 래퍼 경로는 E1(sdpa)과 비교해 7B 상관 0.9994, 답 일치 100%를 확인했다.
  eager로 돌린 옛 결과(`results/raw/_eager_old/`)는 무효이고 커밋하지 않았다.
- **lm_head는 fp32로 계산한다.** bf16이면 m이 0.125 간격으로 양자화되어 동점이 생긴다. argmax는 변하지 않음을 확인했다.
- E2 `--verify`는 eager 가중치 반환이 필요하므로 그때만 base를 eager로 쓴다(검증 전용).

## 2. 실험 목록

| ID | 무엇을 | 스크립트 | 원자료 |
|---|---|---|---|
| E1 | 기본 yes/no logit, 4개 유형 전부 | `src.run_eval` | `e1_qwen{3b,7b}.csv` |
| E1-lab | 이미지 앞에 "Image i:" 번호 | `src.run_eval --image_labels` | `e1lab_*.csv` |
| E2 | 마지막 토큰 → 이미지별 attention (층×헤드) | `src.attn_probe` | `e2_*.csv` (+npz, 미커밋) |
| E3 | 특정 이미지 attention ×α (인과) | `src.intervene` | `e3_*/` |
| M0 | 부정 질문 "없는 이미지가 하나라도 있나?" → 답 뒤집기 | `src.make_neg` + `src.run_eval` | `m0neg_*.csv` |
| M1 | 이미지마다 단일 질문 → min (상한선) | `src.per_image` | `m1_single_*`, `m1_agg_*` |
| M2 | 다중 이미지 입력 그대로, 질문 토큰이 이미지 i만 보게 차단 → min | `src.isolated --mode post/strict` | `m2_{post,strict}_*.csv` |
| M3 | M2로 찾은 최저 이미지에 attention ×α | `src.focus --m2 m2_post_*` | `m3_*/` |
| M4 | m_full + w·m2_min (w는 leave-one-class-out) | `src.compare_methods` | — |
| M5 | 탐지 헤드로 의심 이미지 선택 → attention ×α | `src.head_detect` → `src.focus --m2 hf/heads_k*` | `m5_k{1,5}_*/` |

## 3. 결과

### E1: 현상

정확도 (셀당 200문항)

| 모델 | 유형 | N=2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| 3B | forall_yes | .640 | .460 | .755 | .885 |
| 3B | forall_no | .865 | .760 | .385 | **.180** |
| 3B | exists_yes | .820 | .805 | .755 | .680 |
| 3B | exists_no | .995 | .990 | .990 | .990 |
| 7B | forall_yes | .815 | .545 | .445 | .555 |
| 7B | forall_no | .730 | .845 | .740 | .655 |
| 7B | exists_yes | .835 | .835 | .780 | .685 |
| 7B | exists_no | .990 | .995 | .990 | .995 |

SDT (forall_yes = signal, forall_no = noise)

| 모델 | N=2 | 4 | 6 | 8 |
|---|---|---|---|---|
| 3B d′ / c | 1.45 / .37 | 0.60 / .40 | 0.40 / −.49 | **0.28 / −1.05** |
| 7B d′ / c | 1.50 / −.14 | 1.12 / .45 | 0.50 / .39 | **0.53 / .13** |

- 3B는 판별력 하락에 Yes 편향(c<0)까지 겹친다. 7B는 편향은 작고 판별력만 떨어진다.
- **Δ×N이 거의 일정하다**(3B 3.7/2.8/2.7/3.0, 7B 10.4/9.2/9.6/9.0, 95% CI는 `summary_delta.csv`).
  absent 1장의 영향이 1/N로 희석된다는 뜻이고, 평균 집계의 지문이다. max 집계였다면 Δ≈0이어야 한다.
- **최근 이미지 편향**: N=8 exists_yes에서 객체 있는 이미지가 맨 앞이면 정확도 .55(3B)/.68(7B), 맨 뒤면 .95/.95.

### E1-lab: "Image i:" 번호

- forall_no는 좋아지고(N=8: 3B .18→.34, 7B .655→.775), forall_yes는 나빠진다(3B .885→.75, 7B .555→.41).
- d′는 그대로라서(AUC 3B .663→.677, 7B .739→.739) 기준만 옮겨 가는 효과이고, 해결책이 아니다. 최근 편향도 남는다.

### E2: attention

뒤쪽 절반 층에서, 토큰당 attention을 '홀로 다른 이미지 / 나머지 이미지' 비율로 본 값

| 모델 | 유형 | N=2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| 3B | forall_no (absent / present) | 0.69 | 0.74 | 0.71 | 0.80 |
| 3B | exists_yes (present / absent) | 1.64 | 2.74 | 3.89 | 4.91 |
| 7B | forall_no (absent / present) | 0.72 | 0.76 | 0.75 | 0.82 |
| 7B | exists_yes (present / absent) | 1.50 | 2.21 | 3.01 | 3.68 |

- attention은 내용을 따라간다. 객체 있는 이미지를 더 보고, 없는 이미지는 덜 본다.
- 그래서 "없는 이미지를 덜 봐서 환각한다"는 단순한 설명은 맞지 않는다. 덜 보는 것은 결과이지, 위치를 통제하면 m과의 상관도 없다.
- **헤드 평균**으로 argmin을 취해 absent 이미지를 맞히는 비율은 N=8에서 .15/.22(우연 .125)로 거의 못 찾는다.
  **선택한 헤드**(다른 클래스 forall_no로 고른 z-score 최저 헤드, leave-one-class-out)를 쓰면
  3B는 헤드 1개로 .98/.84/.72/.66(N=2/4/6/8)까지 찾는다. 7B는 헤드 5개로 .92/.55/.44/.34로 약하다.

### E3: attention 개입 (forall_no, 전 층, query = 마지막 이미지 뒤 토큰)

거짓 Yes 비율

| 조건 | 3B N=2 | 4 | 6 | 8 | 7B N=2 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|---|---|
| 개입 없음 | .13 | .23 | .61 | .83 | .27 | .16 | .26 | .35 |
| absent ×4 | .14 | .14 | .55 | .74 | .11 | .01 | .08 | **.15** |
| absent ×0 (knockout) | .45 | .36 | .68 | .82 | .58 | .36 | .37 | .46 |
| present ×4 (대조) | .23 | .28 | .68 | .86 | .44 | .26 | .30 | .38 |

평균 Δm (개입 − 기준)

| 조건 | 3B N=2 | 4 | 6 | 8 | 7B N=2 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|---|---|
| absent ×4 | −.01 | −.27 | −.22 | −.20 | −2.07 | −2.23 | −1.97 | −1.60 |
| absent ×4, after_target | −.41 | −.39 | −.21 | −.26 | −1.41 | −1.70 | −1.73 | −1.46 |
| absent ×0 | +1.02 | +.33 | +.14 | +.13 | +2.73 | +1.25 | +.81 | +.62 |
| present ×4 | +.51 | +.23 | +.24 | +.16 | +1.60 | +.49 | +.28 | +.14 |

- 방향은 두 모델 모두 예측대로다. absent를 키우면 No 쪽으로, 가리면 Yes 쪽으로 움직인다.
- **효과 크기는 7B가 3B보다 훨씬 크다.** 3B는 정답 이미지를 정확히 키워도 N=8 거짓 Yes가 83%에서 74%로만 줄어든다.
- after_target은 뒤쪽 이미지 토큰을 거쳐 들어오는 우회 경로까지 막는 변형인데, post_image와 결과가 비슷하다.
  우회 경로가 주된 경로는 아니다.

### 방법 비교 (forall_yes vs forall_no, 전체 N)

| 방법 | 입력/프롬프트 그대로? | forward 수 | AUC 3B | AUC 7B | 정확도 3B / 7B (임계값 0) | 정확도 3B / 7B (임계값 보정*) |
|---|---|---|---|---|---|---|
| base | ✓ | 1 | .663 | .739 | .616 / .666 | .612 / .670 |
| E1-lab (번호) | 프롬프트 변경 | 1 | .677 | .739 | .615 / .671 | .607 / .660 |
| M0 부정 질문 | 프롬프트 변경 | 1 | .713 | .836 | .612 / .729 | .641 / .756 |
| M1_min (상한선) | ✗ (단일 이미지) | N | **.970** | **.982** | .854 / .818 | **.911 / .914** |
| M1_mean | ✗ | N | .864 | .848 | .616 / .632 | .787 / .740 |
| **M2_post** | ✓ | N+1 | **.913** | **.928** | .549 / .595 | **.831 / .846** |
| M2_strict | ✓ | N+1 | .951 | .803 | .594 / .519 | .883 / .721 |
| M3 α16 (M2로 선택) | ✓ | N+2 | .677 | .858 | .631 / .696 | .631 / .752 |
| M4 fuse | ✓ | N+1 | .793 | .803 | .688 / .707 | .708 / .735 |
| **M5 k5 α16** (탐지 헤드) | ✓ | 2 | .708 | .820 | .651 / .724 | .639 / .737 |

\* 임계값 보정: 그 클래스를 제외한 나머지 19개 클래스에서 balanced accuracy가 최대인 임계값을 골라 적용했다(leave-one-class-out).
N별 표는 `results/summary_methods.csv`(임계값 0)와 `results/summary_m5.csv`(M5와 주요 방법, AUC·보정 정확도)에 있다.

- **M2**는 판별력이 가장 좋다(N=8 AUC .864/.931). 그런데 m이 매우 음수로 쏠려서, 임계값 0에서는 거의 다 No로 답한다
  (forall_yes 적중 .10/.19). 임계값 보정이 꼭 필요하다.
  M2_strict는 7B에서 N≥4이면 전부 No로 무너진다(d′ 0).
- **M5**는 임계값 보정 없이 바로 쓸 수 있다. 7B에서 N=8 거짓 Yes가 35%→16%로, 정답을 아는 E3 상한(15%)과 거의 같다.
  다만 forall_yes 적중도 56%→50%로 같이 떨어진다. 3B는 attention 개입 자체의 효과가 작아서 개선이 작다.
  헤드 선택에는 다른 클래스의 forall_no 정답 위치(라벨)를 썼다.
  이번 실험에서는 1번째 forward(탐지)를 E2에서 저장한 npz로 대신했다.
- **M3**는 M2보다 약하다. 정확한 이미지를 골라도 attention 증폭만으로는 평균 희석을 이기지 못한다(특히 3B).

## 4. 한계

- 프롬프트 템플릿 1개, 모델 1종류(Qwen2.5-VL), 셀당 200문항. 방법 간 차이에 대한 신뢰구간과 검정은 아직 하지 않았다.
- COCO 라벨 누락이 있을 수 있어 absent 이미지가 실제로는 객체를 포함할 수 있다.
- M5 헤드와 보정 임계값은 다른 클래스의 라벨로 골랐다. 클래스 간 일반화만 확인했고, 다른 데이터셋이나 프롬프트는 확인하지 않았다.
- exists 과제에 대한 방법(M2 max 등)은 계산만 해 두고 비교 표에 넣지 않았다. 지금 비교는 forall 과제만이다.

## 5. 다음 실험 (우선순위 순)

1. **M2-post를 forward 약 1회 비용으로 만들기.**
   post 모드는 이미지 토큰끼리의 attention을 건드리지 않으므로, 프리픽스(시스템+이미지) KV는 모든 이미지 i에서 같다.
   프리픽스 KV를 한 번 계산한 뒤, 질문 꼬리(약 30토큰)만 N개 마스크로 배치해서 한 번에 돌리면 결과는 같고 비용은 forward 1회 남짓이 된다.
   기존 `m2_post_*` 결과와 수치가 같은지 검증하고, 실행 시간을 잰다. 성공하면 주장은 "같은 입력·프롬프트, 약 1 forward, AUC .91–.93"이 된다.
2. **M2 임계값을 라벨 없이 보정하기.** 후보는 m_full과의 차이로 정규화, 이미지별 m_i의 분포 기반 offset, 소수 dev 분할로 고정 offset.
   보고는 클래스 단위 held-out으로 한다.
3. **forward 1회 안에서 탐지와 집중을 같이 하기 (M5 개선).** 앞쪽 층의 탐지 헤드로 의심 이미지를 고르고,
   같은 forward의 뒤쪽 층에서 그 이미지에 attention ×α를 준다. 7B 탐지가 약한 문제도 함께 본다(헤드 조합이나 층 선택).
4. **일반화.** 다른 모델(Qwen3-VL, InternVL3, LLaVA-OneVision — `MODEL_REGISTRY` 구현 필요),
   다른 질문 문장, exists 과제(M2 max로 최근 이미지 편향이 줄어드는지), N>8, 작은 객체(`area_range`).
5. **통계 보강.** 셀당 문항 수를 늘리고, 방법 간 차이를 paired bootstrap으로 신뢰구간을 내서 검정한다.
6. **메커니즘 심화.** 평균 집계가 어느 층과 헤드에서 일어나는지를 본다.
   이미지별 hidden state를 logit lens로 읽어 보고, 헤드 단위 activation patching을 한다.

## 6. 재현

```bash
python -m src.make_neg                                   # data/questions_neg.jsonl (M0)
bash logs/queue_3b.sh; bash logs/queue_7b.sh             # E2, E3, M0, M1, M2, M3 (E1은 logs/run_all.sh)
python -m src.head_detect --e2 results/raw/e2_qwen3b.csv --tag qwen3b   # M5 1단계: 탐지 헤드 점수 (E2 npz 필요)
bash logs/headfocus.sh qwen3b qwen2.5-vl-3b              # M5 2단계: attention 집중
python -m src.analyze --e1 results/raw/e1_qwen3b.csv results/raw/e1_qwen7b.csv \
    --e2 results/raw/e2_qwen3b.csv results/raw/e2_qwen7b.csv --out_dir results
python -m src.compare_methods
```
E2 npz(약 200MB)는 커밋하지 않았으므로, M5를 다시 하려면 E2부터 다시 돌려야 한다.
