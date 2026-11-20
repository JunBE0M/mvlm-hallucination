"""입력 토큰 시퀀스를 [이미지 0..N-1 | text | system | marker] 구간으로 나눈다 (torch 불필요).

정의 (Qwen chat template 기준, 토큰 id는 tokenizer에서 받아서 넘긴다):
    image i : i번째 vision_start 와 vision_end *사이*의 토큰 (image_pad 들)
    marker  : vision_start / vision_end 토큰 자체
    system  : user 턴을 여는 im_start 이전의 모든 토큰
              (= 첫 vision_start 이전 마지막 im_start 의 앞쪽. system prompt 와 첫 토큰(sink) 포함)
    text    : 나머지 전부 (user 턴 헤더, 질문, <|im_end|>, assistant 헤더, 마지막 query 토큰 자신 포함)
모든 위치는 정확히 하나의 구간에 속하므로 마지막 query 행의 attention 합은 구간 합의 총합 = 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Segments:
    image_spans: list[tuple[int, int]]   # half-open [start, end) — 이미지 토큰만
    labels: np.ndarray                   # [L] int: 0..N-1 = 이미지, N = text, N+1 = system, N+2 = marker
    n_images: int

    @property
    def text_label(self) -> int:
        return self.n_images

    @property
    def system_label(self) -> int:
        return self.n_images + 1

    @property
    def marker_label(self) -> int:
        return self.n_images + 2

    @property
    def n_labels(self) -> int:
        return self.n_images + 3

    def n_tokens(self) -> np.ndarray:
        return np.array([e - s for s, e in self.image_spans], dtype=np.int64)


def image_spans(ids, vision_start_id: int, vision_end_id: int) -> list[tuple[int, int]]:
    ids = list(map(int, ids))
    starts = [i for i, t in enumerate(ids) if t == vision_start_id]
    ends = [i for i, t in enumerate(ids) if t == vision_end_id]
    if len(starts) != len(ends):
        raise ValueError(f"unbalanced vision markers: {len(starts)} starts vs {len(ends)} ends")
    spans = []
    for s, e in zip(starts, ends):
        if not s < e:
            raise ValueError(f"vision_end at {e} precedes vision_start at {s}")
        spans.append((s + 1, e))
    for (s0, e0), (s1, _) in zip(spans, spans[1:]):
        if s1 <= e0:
            raise ValueError("nested/overlapping vision spans")
    return spans


def split_segments(ids, vision_start_id: int, vision_end_id: int, im_start_id: int,
                   expected_images: int | None = None) -> Segments:
    ids = list(map(int, ids))
    spans = image_spans(ids, vision_start_id, vision_end_id)
    n = len(spans)
    if expected_images is not None and n != expected_images:
        raise ValueError(f"found {n} image spans, expected {expected_images}")
    labels = np.full(len(ids), n, dtype=np.int64)        # default: text
    if n:
        first_vs = spans[0][0] - 1
        im_starts = [i for i in range(first_vs) if ids[i] == im_start_id]
        sys_end = im_starts[-1] if im_starts else 0
    else:
        im_starts = [i for i, t in enumerate(ids) if t == im_start_id]
        sys_end = im_starts[1] if len(im_starts) > 1 else 0
    labels[:sys_end] = n + 1
    for i, (s, e) in enumerate(spans):
        labels[s:e] = i
        labels[s - 1] = n + 2
        labels[e] = n + 2
    return Segments(spans, labels, n)
