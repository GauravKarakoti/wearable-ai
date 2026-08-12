"""
jepa_bridge_model.py -- drop-in VideoQAModel subclass for the test-phase
harness. Paste this class into model.py (or `from jepa_bridge_model import
JepaBridgeModel` at the top of model.py if you keep it as a separate file
in the same directory), then wire it in as shown at the bottom of this file.
"""

import os
import re
import numpy as np
import torch
import torch.nn as nn

NUM_JEPA_FRAMES = 64
NUM_VIDEO_TOKENS = 16

_PROMPT_RE = re.compile(
    r"Question:\s*(?P<question>.*?)\n\nOptions:\s*\n?(?P<options>.*?)\n\nAnswer with ONLY",
    re.DOTALL,
)


def _parse_question_and_options(messages: list[dict[str, str]]) -> tuple[str, dict[str, str]]:
    """Reverse-engineer question + {letter: option_text} from the harness's
    combined prompt string. Falls back to raising a clear error rather than
    silently guessing wrong, since a wrong parse here means every prediction
    is wrong."""
    combined_text = ""
    for msg in messages:
        if msg.get("role") == "user":
            combined_text = msg["content"]
            break

    match = _PROMPT_RE.search(combined_text)
    if not match:
        raise ValueError(
            "Could not parse question/options out of the harness prompt. "
            "The LONGQA_PROMPT_TEMPLATE format may have changed -- check "
            "run_generate_longqa.py against this regex before submitting."
        )

    question = match.group("question").strip()
    options_block = match.group("options").strip()

    option_matches = re.findall(r'([A-D])\.\s*(.*?)(?=\s[A-D]\.\s|$)', options_block)
    options = {letter: text.strip() for letter, text in option_matches}
    if len(options) < 2:
        raise ValueError(f"Parsed fewer than 2 options from: {options_block!r}")

    return question, options


def _resample_frames_to_array(frames: list, target_count: int) -> np.ndarray:
    """PIL images (any count) -> uint8 numpy array of exactly target_count frames,
    via nearest-index resampling (upsamples with repeats if fewer frames were
    given than the model expects, downsamples by dropping otherwise)."""
    n = len(frames)
    if n == 0:
        raise ValueError("Received zero frames -- cannot run inference on an empty clip.")
    indices = np.linspace(0, n - 1, target_count).round().astype(int)
    resampled = [np.array(frames[i].convert("RGB")) for i in indices]
    return np.stack(resampled, axis=0)


class JepaBridgeModel:
    """VideoQAModel implementation: frozen V-JEPA2 encoder -> pooling ->
    trained bridge -> frozen Qwen2.5-1.5B-Instruct, scored via per-option
    teacher-forced loss."""

    def __init__(self, model_id: str) -> None:
        """model_id is treated as a LOCAL directory baked into the image at
        build time (see Containerfile notes), NOT a Hugging Face hub id --
        nothing can reach the network at eval time. Expected layout under
        model_id:
            jepa/            <- V-JEPA2 encoder + processor, via save_pretrained
            qwen/            <- Qwen2.5-1.5B-Instruct + tokenizer, via save_pretrained
            bridge.pt        <- your trained bridge state_dict
        """
        from transformers import AutoModel, AutoVideoProcessor, AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        jepa_dir = os.path.join(model_id, "jepa")
        qwen_dir = os.path.join(model_id, "qwen")
        bridge_path = os.path.join(model_id, "bridge.pt")

        self.jepa_processor = AutoVideoProcessor.from_pretrained(jepa_dir)
        self.jepa = AutoModel.from_pretrained(jepa_dir, device_map=self.device, attn_implementation="sdpa")
        self.jepa.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(qwen_dir)
        self.qwen = AutoModelForCausalLM.from_pretrained(qwen_dir, torch_dtype=torch.float32, device_map=self.device)
        self.qwen.eval()

        qwen_hidden = self.qwen.get_input_embeddings().weight.shape[1]
        self.bridge = nn.Sequential(
            nn.Linear(1024, qwen_hidden),
            nn.GELU(),
            nn.Linear(qwen_hidden, qwen_hidden),
        ).to(device=self.device, dtype=self.qwen.dtype)
        self.bridge.load_state_dict(torch.load(bridge_path, map_location=self.device))
        self.bridge.eval()

    def _encode_video(self, frames: list) -> torch.Tensor:
        frame_array = _resample_frames_to_array(frames, NUM_JEPA_FRAMES)
        inputs = self.jepa_processor(frame_array, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.jepa(**inputs).last_hidden_state
        pooled = torch.nn.functional.adaptive_avg_pool1d(
            out.transpose(1, 2), NUM_VIDEO_TOKENS
        ).transpose(1, 2)
        with torch.no_grad():
            projected = self.bridge(pooled.to(self.qwen.dtype))
        return projected

    def _option_loss(self, projected_video, question: str, option_text: str) -> float:
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        target = option_text + self.tokenizer.eos_token

        q_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        a_ids = self.tokenizer(target, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

        q_embeds = self.qwen.get_input_embeddings()(q_ids)
        a_embeds = self.qwen.get_input_embeddings()(a_ids)
        inputs_embeds = torch.cat([projected_video, q_embeds, a_embeds], dim=1)

        video_len, q_len = projected_video.shape[1], q_ids.shape[1]
        labels = torch.cat([
            torch.full((1, video_len + q_len), -100, dtype=torch.long, device=self.device),
            a_ids,
        ], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)

        with torch.no_grad():
            loss = self.qwen(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels).loss
        return loss.item()

    def generate(self, frames: list, messages: list[dict[str, str]], max_new_tokens: int = 256) -> str:
        question, options = _parse_question_and_options(messages)
        projected_video = self._encode_video(frames)

        losses = {
            letter: self._option_loss(projected_video, question, text)
            for letter, text in options.items()
        }
        return min(losses, key=losses.get)

    def generate_batch(self, batch_frames, batch_messages, max_new_tokens: int = 256) -> list[str]:
        return [
            self.generate(frames, messages, max_new_tokens)
            for frames, messages in zip(batch_frames, batch_messages)
        ]