"""소형 랜덤 Qwen2.5-VL (실제 모델 코드 경로)로 CPU에서 E1 입력/forward 검증."""
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from tiny_qwen import make_tiny_vlm  # noqa: E402


@pytest.fixture(scope="module")
def vlm():
    return make_tiny_vlm("eager")


def _images(n):
    return [Image.new("RGB", (640, 480) if i % 2 else (480, 640), (40 * i, 80, 120)) for i in range(n)]


def test_last_logits_match_full_forward(vlm):
    inp = vlm.build_inputs(_images(3), "Is there a cat in all 3 images? Please answer with Yes or No.")
    lg = vlm.forward_last_logits(inp)
    with torch.inference_mode():
        full = vlm.model(**inp, use_cache=False).logits[0, -1].float()
    assert torch.allclose(full, lg, atol=1e-5)
    ly, ln = vlm.get_yes_no_logits(inp)
    assert ly == pytest.approx(float(lg[9454])) and ln == pytest.approx(float(lg[2753]))


def test_segments_match_grid(vlm):
    inp = vlm.build_inputs(_images(4), "q")
    seg = vlm.segments(inp)
    assert seg.n_images == 4
    assert seg.n_tokens().tolist() == (inp["image_grid_thw"].prod(-1) // 4).tolist()
