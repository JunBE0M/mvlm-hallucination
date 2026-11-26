"""모델 로딩 / 입력 구성 / yes-no logit 추출. 모델별 차이를 여기서 흡수한다.

공통 인터페이스
    vlm = load_model(name, attn_impl="sdpa", max_pixels=448*448)
    inputs = vlm.build_inputs(images, question)       # PIL 이미지 리스트, 질문 문자열
    logit_yes, logit_no = vlm.get_yes_no_logits(inputs)
    seg = vlm.segments(inputs)                         # 이미지/text/system 구간 (E2)

확인된 사항 (transformers==5.22.2 소스 및 Qwen2.5-VL tokenizer 파일 직접 확인):
    - 클래스: Qwen2_5_VLForConditionalGeneration, 프로세서: Qwen2_5_VLProcessor (AutoProcessor)
    - from_pretrained 인자는 `dtype` (torch_dtype 아님)
    - 해상도: processor *호출 시* size={"longest_edge": max_pixels, "shortest_edge": min_pixels}.
      from_pretrained(size=...)는 무시됨. min_pixels/max_pixels 호출 인자는 deprecated.
    - "Yes"=9454, " Yes"=7414, "No"=2753, " No"=2308 (서로 다른 토큰).
      generation prompt가 "<|im_start|>assistant\n"으로 끝나므로 첫 응답 토큰은 공백 없는 "Yes"/"No".
    - <|vision_start|>=151652, <|vision_end|>=151653, <|image_pad|>=151655, <|im_start|>=151644
      (코드에서는 하드코딩하지 않고 tokenizer에서 가져온다)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .segments import Segments, split_segments


@dataclass(frozen=True)
class ModelSpec:
    family: str
    hf_id: str | None


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "qwen2.5-vl-3b": ModelSpec("qwen2_5_vl", "Qwen/Qwen2.5-VL-3B-Instruct"),
    "qwen2.5-vl-7b": ModelSpec("qwen2_5_vl", "Qwen/Qwen2.5-VL-7B-Instruct"),
    # transformers 5.22.2 에서 확인 (2026-11-25): 클래스는 AutoModelForImageTextToText 로 로드.
    "qwen3-vl-8b": ModelSpec("qwen3_vl", "Qwen/Qwen3-VL-8B-Instruct"),          # Qwen3VLForConditionalGeneration
    "internvl3-8b": ModelSpec("internvl3", "OpenGVLab/InternVL3-8B-hf"),         # InternVLForConditionalGeneration
    "llava-onevision-7b": ModelSpec("llava_onevision", "llava-hf/llava-onevision-qwen2-7b-ov-hf"),
}


@dataclass
class AnswerTokens:
    yes: int
    no: int
    # 기록용 변형 (pred 계산에는 사용하지 않음)
    variants: dict[str, int] = field(default_factory=dict)


def _single_token_id(tokenizer, s: str) -> int:
    ids = tokenizer.encode(s, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{s!r} is not a single token: {ids}")
    return ids[0]


class BaseVLM:
    name: str
    model: object
    processor: object
    answer_tokens: AnswerTokens

    def build_inputs(self, images, question: str) -> dict:
        raise NotImplementedError

    def forward_last_logits(self, inputs):
        """forward 1회, 마지막 위치 logits [V] (float32, cpu)."""
        raise NotImplementedError

    def get_yes_no_logits(self, inputs) -> tuple[float, float]:
        logits = self.forward_last_logits(inputs)
        return float(logits[self.answer_tokens.yes]), float(logits[self.answer_tokens.no])

    def segments(self, inputs) -> Segments:
        raise NotImplementedError

    def is_text_attention(self, module) -> bool:
        """E2 probe가 기록할 언어모델 self-attention 모듈인지 (비전 인코더 attention 제외)."""
        raise NotImplementedError

    @property
    def num_layers(self) -> int:
        raise NotImplementedError

    @property
    def num_heads(self) -> int:
        raise NotImplementedError


class QwenVL25(BaseVLM):
    def __init__(self, name: str, hf_id: str, attn_impl: str = "sdpa", dtype: str = "bfloat16",
                 max_pixels: int | None = 448 * 448, min_pixels: int | None = None, device: str = "cuda",
                 model=None, processor=None, image_labels: bool = False):
        """model/processor 를 직접 넘기면 from_pretrained 를 건너뜀 (테스트용 소형 랜덤 모델).
        image_labels=True: 각 이미지 앞에 "Image i: " 텍스트를 넣는다 (위치/구분 효과 통제 조건)."""
        self.image_labels = image_labels
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention

        self._torch = torch
        self._attn_cls = Qwen2_5_VLAttention
        self.name = name
        self.hf_id = hf_id
        self.device = device
        if model is None:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                hf_id, dtype=getattr(torch, dtype), attn_implementation=attn_impl, device_map=device,
            )
        self.model = model.eval()
        self.processor = processor if processor is not None else AutoProcessor.from_pretrained(hf_id)
        tok = self.processor.tokenizer

        default = self.processor.image_processor.size
        self.size = {
            "longest_edge": max_pixels if max_pixels is not None else default.longest_edge,
            "shortest_edge": min_pixels if min_pixels is not None else default.shortest_edge,
        }

        self.answer_tokens = AnswerTokens(
            yes=_single_token_id(tok, "Yes"), no=_single_token_id(tok, "No"),
            variants={s: _single_token_id(tok, s) for s in (" Yes", " No", "yes", "no")},
        )
        self.vision_start_id = tok.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end_id = tok.convert_tokens_to_ids("<|vision_end|>")
        self.im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
        for k in ("vision_start_id", "vision_end_id", "im_start_id"):
            if getattr(self, k) in (None, tok.unk_token_id):
                raise ValueError(f"special token for {k} not found in tokenizer")
        print(f"[{name}] answer tokens: Yes={self.answer_tokens.yes} No={self.answer_tokens.no} "
              f"variants={self.answer_tokens.variants} | vision_start={self.vision_start_id} "
              f"vision_end={self.vision_end_id} | image size={self.size} | attn={attn_impl}")

    def chat_text(self, n_images: int, question: str) -> str:
        if getattr(self, "image_labels", False):
            content = []
            for i in range(n_images):
                content += [{"type": "text", "text": f"Image {i + 1}: "}, {"type": "image"}]
            content.append({"type": "text", "text": "\n" + question})
        else:
            content = [{"type": "image"} for _ in range(n_images)] + [{"type": "text", "text": question}]
        msgs = [{"role": "user", "content": content}]
        return self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def build_inputs(self, images, question: str) -> dict:
        text = self.chat_text(len(images), question)
        inputs = self.processor(text=[text], images=list(images), size=self.size, return_tensors="pt")
        return inputs.to(self.model.device)

    def forward_last_logits(self, inputs):
        """본체는 모델 dtype(bf16)으로, 마지막 위치의 lm_head 만 fp32 로 계산.
        bf16 lm_head 출력은 logit ~20 부근에서 0.125 단위로 양자화되어 m=logit(Yes)-logit(No) 에 동점/잡음이 생긴다.
        transformers 5.22.2 Qwen2_5_VLForConditionalGeneration.forward 와 동일 경로: self.model(...)[0] -> lm_head."""
        torch = self._torch
        with torch.inference_mode():
            hidden = self.model.model(**inputs, use_cache=False)[0][0, -1]          # [D]
            if getattr(self, "_lm_head_fp32", None) is None:
                self._lm_head_fp32 = self.model.lm_head.weight.detach().float()     # [V, D]
            logits = torch.nn.functional.linear(hidden.float(), self._lm_head_fp32)
        return logits.cpu()

    def segments(self, inputs) -> Segments:
        ids = inputs["input_ids"][0].tolist()
        n_img = int(inputs["image_grid_thw"].shape[0]) if "image_grid_thw" in inputs else None
        return split_segments(ids, self.vision_start_id, self.vision_end_id, self.im_start_id, n_img)

    def is_text_attention(self, module) -> bool:
        return isinstance(module, self._attn_cls)

    @property
    def num_layers(self) -> int:
        return self.model.config.text_config.num_hidden_layers

    @property
    def num_heads(self) -> int:
        return self.model.config.text_config.num_attention_heads


class GenericVLM(QwenVL25):
    """Qwen3-VL / InternVL3(-hf) / LLaVA-OneVision(-hf). 셋 다 Qwen 계열 tokenizer 라
    "Yes"=9454 / "No"=2753, chat template 이 "<|im_start|>assistant\n" 으로 끝난다 (직접 확인).

    이미지 구간:
      qwen3_vl        : <|vision_start|> ... <|vision_end|>  (Qwen2.5-VL 과 동일)
      internvl3       : <img> <IMG_CONTEXT>*256 </img>.  processor 호출 시 crop_to_patches=False -> 이미지당 448px 타일 1개
      llava_onevision : <image> 토큰 연속 구간 (경계 마커 없음). images 를 [[...]] 로 넘기면 다중 이미지 =
                        anyres 없이 base 384px (원 논문 multi-image 설정), 단일 이미지는 anyres (모델 기본 동작).
                        이미지들이 붙어 있으므로 연속 구간을 N 등분 (다중 이미지 base 모드는 이미지당 토큰 수 동일).
    """

    def __init__(self, name: str, hf_id: str, family: str, attn_impl: str = "sdpa", dtype: str = "bfloat16",
                 max_pixels: int | None = 448 * 448, min_pixels: int | None = None, device: str = "cuda",
                 model=None, processor=None, image_labels: bool = False):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self._torch = torch
        self.family = family
        self.image_labels = image_labels
        self.name, self.hf_id, self.device = name, hf_id, device
        if model is None:
            model = AutoModelForImageTextToText.from_pretrained(
                hf_id, dtype=getattr(torch, dtype), attn_implementation=attn_impl, device_map=device)
        self.model = model.eval()
        self.processor = processor if processor is not None else AutoProcessor.from_pretrained(hf_id)
        tok = self.processor.tokenizer
        lm = self.model.model.language_model
        self._text_attn_ids = {id(m) for m in lm.modules() if hasattr(m, "layer_idx") and hasattr(m, "q_proj")}
        if family == "qwen3_vl":
            default = self.processor.image_processor.size
            self.size = {"longest_edge": max_pixels if max_pixels is not None else default.longest_edge,
                         "shortest_edge": min_pixels if min_pixels is not None else default.shortest_edge}
            self._call_kw = {"size": self.size}
            self.vision_start_id = tok.convert_tokens_to_ids("<|vision_start|>")
            self.vision_end_id = tok.convert_tokens_to_ids("<|vision_end|>")
        elif family == "internvl3":
            self.size = {"longest_edge": 448 * 448, "shortest_edge": None}
            self._call_kw = {"crop_to_patches": False}
            self.vision_start_id = tok.convert_tokens_to_ids("<img>")
            self.vision_end_id = tok.convert_tokens_to_ids("</img>")
        elif family == "llava_onevision":
            self.size = {"longest_edge": 384 * 384, "shortest_edge": None}
            self._call_kw = {}
            self.vision_start_id = self.vision_end_id = None
            self.image_token_id = self.processor.image_token_id
        else:
            raise ValueError(family)
        self.im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
        self.answer_tokens = AnswerTokens(
            yes=_single_token_id(tok, "Yes"), no=_single_token_id(tok, "No"),
            variants={s: _single_token_id(tok, s) for s in (" Yes", " No", "yes", "no")},
        )
        print(f"[{name}] family={family} Yes={self.answer_tokens.yes} No={self.answer_tokens.no} "
              f"call_kw={self._call_kw} attn={attn_impl} text_attn_modules={len(self._text_attn_ids)}")

    def build_inputs(self, images, question: str) -> dict:
        text = self.chat_text(len(images), question)
        imgs = [list(images)] if self.family == "llava_onevision" else list(images)
        inputs = self.processor(text=[text], images=imgs, return_tensors="pt", **self._call_kw)
        return inputs.to(self.model.device)

    def forward_last_logits(self, inputs):
        torch = self._torch
        with torch.inference_mode():
            hidden = self.model.model(**inputs, use_cache=False)[0][0, -1]
            if getattr(self, "_lm_head_fp32", None) is None:
                self._lm_head_fp32 = self.model.lm_head.weight.detach().float()
            logits = torch.nn.functional.linear(hidden.float(), self._lm_head_fp32)
        return logits.cpu()

    def segments(self, inputs) -> Segments:
        ids = inputs["input_ids"][0].tolist()
        if self.family != "llava_onevision":
            return split_segments(ids, self.vision_start_id, self.vision_end_id, self.im_start_id)
        import numpy as np
        pos = [i for i, t in enumerate(ids) if t == self.image_token_id]
        n_img = int(inputs["batch_num_images"][0]) if "batch_num_images" in inputs else 1
        runs = []
        for i in pos:
            if runs and runs[-1][1] == i:
                runs[-1][1] = i + 1
            else:
                runs.append([i, i + 1])
        if len(runs) != 1:
            raise ValueError(f"expected one contiguous <image> run, got {len(runs)}")
        s0, e0 = runs[0]
        if (e0 - s0) % n_img:
            raise ValueError(f"image run {e0 - s0} not divisible by {n_img}")
        k = (e0 - s0) // n_img
        spans = [(s0 + j * k, s0 + (j + 1) * k) for j in range(n_img)]
        n = n_img
        labels = np.full(len(ids), n, dtype=np.int64)
        ims = [i for i in range(s0) if ids[i] == self.im_start_id]
        labels[:ims[-1] if ims else 0] = n + 1
        for j, (a, b) in enumerate(spans):
            labels[a:b] = j
        return Segments(spans, labels, n)

    def is_text_attention(self, module) -> bool:
        return id(module) in self._text_attn_ids


class _TodoVLM(BaseVLM):
    def __init__(self, name, *a, **k):
        raise NotImplementedError(
            f"{name}: 아직 미구현. HF repo id, transformers 클래스명, 필요한 transformers 버전, "
            f"chat template의 이미지 구간 토큰을 소스에서 확인한 뒤 구현할 것 (README 'TODO' 참고).")


_FAMILY = {"qwen2_5_vl": QwenVL25, "qwen3_vl": GenericVLM, "internvl3": GenericVLM, "llava_onevision": GenericVLM}


def load_model(name: str, attn_impl: str = "sdpa", **kwargs) -> BaseVLM:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {name}; choose from {sorted(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[name]
    cls = _FAMILY[spec.family]
    if cls is _TodoVLM:
        return cls(name)
    if cls is GenericVLM:
        return cls(name, spec.hf_id, spec.family, attn_impl=attn_impl, **kwargs)
    return cls(name, spec.hf_id, attn_impl=attn_impl, **kwargs)


def load_images(paths, root):
    from pathlib import Path

    from PIL import Image
    out = []
    for p in paths:
        with Image.open(Path(root) / p) as im:
            out.append(im.convert("RGB"))
    return out
