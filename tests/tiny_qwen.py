"""CPU 테스트용: 실제 Qwen2.5-VL config/processor 로 *아주 작은 랜덤 가중치* 모델을 만든다.

가중치를 받지 않고 실제 transformers 모델 코드(입력 처리, mrope, attention 경로)를 그대로 통과시키기 위함.
processor/config 파일이 HF 캐시에 없으면 pytest.skip.
"""
import pytest

REPO = "Qwen/Qwen2.5-VL-3B-Instruct"


def make_tiny_vlm(attn_impl="eager", max_pixels=112 * 112, seed=0):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from src.models import QwenVL25

    try:
        cfg = transformers.AutoConfig.from_pretrained(REPO, local_files_only=True)
        proc = transformers.AutoProcessor.from_pretrained(REPO, local_files_only=True)
    except Exception as e:
        pytest.skip(f"{REPO} config/processor not cached: {type(e).__name__}")

    v, t = cfg.vision_config, cfg.text_config
    v.depth, v.hidden_size, v.intermediate_size, v.num_heads = 2, 64, 128, 2
    v.out_hidden_size, v.fullatt_block_indexes = 64, [1]
    t.hidden_size, t.intermediate_size, t.num_hidden_layers = 64, 128, 3
    t.num_attention_heads, t.num_key_value_heads = 4, 2           # GQA: 2 query heads per KV head
    t.rope_parameters = dict(t.rope_parameters, mrope_section=[2, 3, 3])   # head_dim 16 -> 8 = 2+3+3
    if hasattr(t, "layer_types"):
        t.layer_types = t.layer_types[:3]
    torch.manual_seed(seed)
    model = transformers.Qwen2_5_VLForConditionalGeneration._from_config(
        cfg, attn_implementation=attn_impl, dtype=torch.float32)
    return QwenVL25("tiny-qwen2.5-vl", REPO, attn_impl=attn_impl, max_pixels=max_pixels,
                    min_pixels=56 * 56, device="cpu", model=model, processor=proc)
