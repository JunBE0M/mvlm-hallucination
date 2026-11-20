import numpy as np
import pytest

from src.segments import image_spans, split_segments

VS, VE, PAD, IMS, IME = 900, 901, 902, 800, 801


def make_ids(n_tokens_per_image):
    # <ims> system ... <ime> \n <ims> user \n [<vs> pad*k <ve>]* question <ime> \n <ims> assistant \n
    ids = [IMS, 1, 2, 3, IME, 10, IMS, 4, 10]
    for k in n_tokens_per_image:
        ids += [VS] + [PAD] * k + [VE]
    ids += [20, 21, 22, IME, 10, IMS, 5, 10]
    return ids


def test_spans_and_labels():
    ids = make_ids([3, 5, 2])
    seg = split_segments(ids, VS, VE, IMS, expected_images=3)
    assert seg.n_images == 3
    assert list(seg.n_tokens()) == [3, 5, 2]
    for i, (s, e) in enumerate(seg.image_spans):
        assert all(t == PAD for t in ids[s:e])
        assert ids[s - 1] == VS and ids[e] == VE
        assert (seg.labels[s:e] == i).all()
    # system: user 턴 im_start 이전 6개
    assert (seg.labels[:6] == seg.system_label).all()
    assert seg.labels[6] == seg.text_label
    assert seg.labels[-1] == seg.text_label              # 마지막 query 토큰은 text
    assert (seg.labels == seg.marker_label).sum() == 6
    # 모든 위치가 정확히 하나의 구간에 → 구간별 합의 총합 = 전체 합
    p = np.random.default_rng(0).random(len(ids))
    total = sum(p[seg.labels == l].sum() for l in range(seg.n_labels))
    assert total == pytest.approx(p.sum())


def test_expected_count_mismatch():
    with pytest.raises(ValueError):
        split_segments(make_ids([2, 2]), VS, VE, IMS, expected_images=3)


def test_unbalanced_markers():
    with pytest.raises(ValueError):
        image_spans([VS, PAD, PAD], VS, VE)
