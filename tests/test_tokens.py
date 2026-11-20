"""Yes/No 토큰 id 와 이미지 구간을 실제 Qwen2.5-VL tokenizer/processor로 확인.

- CPU 테스트: 가중치 없이 tokenizer/processor 파일만 필요. HF 캐시에 없으면 skip.
  (받기: huggingface_hub.snapshot_download(repo, allow_patterns=["*.json", "*.txt"]))
- GPU 테스트(@gpu): 실제 모델 forward로 "Yes"/"No" 가 상위 토큰인지 확인. Vast에서 `pytest -m gpu`.
"""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from PIL import Image  # noqa: E402

from src.models import MODEL_REGISTRY, QwenVL25  # noqa: E402

REPOS = [MODEL_REGISTRY["qwen2.5-vl-3b"].hf_id, MODEL_REGISTRY["qwen2.5-vl-7b"].hf_id]


def _processor(repo):
    try:
        return transformers.AutoProcessor.from_pretrained(repo, local_files_only=True)
    except Exception as e:  # 캐시에 없음
        pytest.skip(f"processor files for {repo} not cached: {type(e).__name__}")


@pytest.mark.parametrize("repo", REPOS)
def test_yes_no_token_ids(repo):
    tok = _processor(repo).tokenizer
    enc = {s: tok.encode(s, add_special_tokens=False) for s in ["Yes", "No", " Yes", " No"]}
    print(repo, enc)
    assert all(len(v) == 1 for v in enc.values())
    assert enc["Yes"] != enc[" Yes"] and enc["No"] != enc[" No"]
    # generation prompt 가 개행으로 끝나므로 첫 응답 토큰은 공백 없는 "Yes"/"No"
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    text = _processor(repo).apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    assert text.endswith("<|im_start|>assistant\n")


@pytest.mark.parametrize("N", [2, 4])
def test_inputs_and_segments(N):
    repo = REPOS[0]
    proc = _processor(repo)
    vlm = QwenVL25.__new__(QwenVL25)        # 가중치 없이 입력 구성 부분만 검사
    vlm.processor = proc
    tok = proc.tokenizer
    vlm.vision_start_id = tok.convert_tokens_to_ids("<|vision_start|>")
    vlm.vision_end_id = tok.convert_tokens_to_ids("<|vision_end|>")
    vlm.im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
    vlm.size = {"longest_edge": 448 * 448, "shortest_edge": 56 * 56}
    images = [Image.new("RGB", (640, 480) if i % 2 else (480, 640)) for i in range(N)]
    text = vlm.chat_text(N, f"Is there a cat in all {N} images? Please answer with Yes or No.")
    inputs = proc(text=[text], images=images, size=vlm.size, return_tensors="pt")
    seg = vlm.segments(inputs)
    pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
    ids = inputs["input_ids"][0]
    merge = proc.image_processor.merge_size ** 2
    expected = (inputs["image_grid_thw"].prod(-1) // merge).tolist()
    assert seg.n_tokens().tolist() == expected
    for s, e in seg.image_spans:
        assert (ids[s:e] == pad_id).all()
    # max_pixels 가 실제로 적용됐는지: 이미지당 패치 수 <= 448^2 / 14^2
    assert all(t * h * w * 14 * 14 <= 448 * 448 for t, h, w in inputs["image_grid_thw"].tolist())
    # system 구간은 "You are a helpful assistant" 를 포함
    sys_ids = ids[torch.as_tensor(seg.labels == seg.system_label)]
    assert "helpful assistant" in tok.decode(sys_ids)


@pytest.mark.gpu
def test_gpu_forward_prefers_yes_no():
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    from src.models import load_model
    vlm = load_model("qwen2.5-vl-3b", attn_impl="sdpa")
    images = [Image.new("RGB", (448, 448), c) for c in ("red", "blue")]
    inputs = vlm.build_inputs(images, "Is there a cat in all 2 images? Please answer with Yes or No.")
    logits = vlm.forward_last_logits(inputs)
    top5 = torch.topk(logits, 5).indices.tolist()
    print("top5:", [vlm.processor.tokenizer.decode([t]) for t in top5])
    assert vlm.answer_tokens.yes in top5 or vlm.answer_tokens.no in top5
