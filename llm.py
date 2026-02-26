from __future__ import annotations

import ast
import re
import time
from dataclasses import dataclass
from typing import Any

from modal_voice.prompts import MODAL_SYSTEM_PROMPT


@dataclass
class LLMResult:
    text: str
    latency_s: float


class ModalVLLM:
    """vLLM wrapper for container-scoped generation."""

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        max_model_len: int = 2048,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = True,
    ) -> None:
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enforce_eager = enforce_eager

        self.tokenizer: Any | None = None
        self.llm: Any | None = None

    def load(self) -> None:
        if self.llm is not None and self.tokenizer is not None:
            return

        from transformers import AutoTokenizer
        from vllm import LLM

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.llm = LLM(
            model=self.model_name,
            trust_remote_code=True,
            dtype="float16",
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enforce_eager=self.enforce_eager,
        )

    def _clean_output(self, text: str) -> str:
        out = text.strip()
        if not out:
            return out

        # Convert stringified Python list output into a clean sentence.
        if out.startswith("[") and out.endswith("]"):
            try:
                parsed = ast.literal_eval(out)
                if isinstance(parsed, list):
                    items = [str(x).strip() for x in parsed if str(x).strip()]
                    out = " ".join(items)
            except Exception:
                pass

        out = re.sub(r"\\s+", " ", out).strip()
        return out

    def _finalize_sentences(self, text: str, max_sentences: int = 4, max_words: int = 110) -> str:
        cleaned = self._clean_output(text)
        if not cleaned:
            return cleaned

        words = cleaned.split()
        if len(words) > max_words:
            cleaned = " ".join(words[:max_words]).strip()

        # Keep only first N sentences to bound TTS time while staying complete.
        parts = re.split(r"(?<=[.!?])\\s+", cleaned)
        pruned = " ".join(parts[:max_sentences]).strip()
        if not pruned:
            pruned = cleaned

        # Avoid trailing partial clauses when punctuation is missing.
        if pruned[-1] not in ".!?":
            last_punct = max(pruned.rfind("."), pruned.rfind("!"), pruned.rfind("?"))
            if last_punct > 20:
                pruned = pruned[: last_punct + 1].strip()
            else:
                pruned = pruned.rstrip(",;:") + "."

        return pruned

    def generate(self, user_text: str, max_tokens: int = 64, temperature: float = 0.1) -> LLMResult:
        if self.llm is None or self.tokenizer is None:
            raise RuntimeError("vLLM model is not loaded")
        from vllm import SamplingParams

        messages = [
            {"role": "system", "content": MODAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_text.strip()},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        sampling = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.05,
        )

        start = time.perf_counter()
        outputs = self.llm.generate([prompt], sampling, use_tqdm=False)
        latency_s = time.perf_counter() - start

        text = ""
        if outputs and outputs[0].outputs:
            text = self._finalize_sentences(outputs[0].outputs[0].text)

        return LLMResult(text=text, latency_s=latency_s)

    def generate_with_context(
        self,
        user_text: str,
        context_snippets: list[dict[str, str]] | None = None,
        max_tokens: int = 80,
        temperature: float = 0.1,
    ) -> LLMResult:
        context_snippets = context_snippets or []
        context_lines: list[str] = []
        for i, snip in enumerate(context_snippets, start=1):
            title = snip.get("title", "Modal docs")
            url = snip.get("url", "")
            text = snip.get("text", "")
            context_lines.append(f"[{i}] {title} | {url}\\n{text}")

        normalized_user = user_text.strip()
        lower_user = normalized_user.lower()
        if lower_user in {"what does modal do", "what is modal", "tell me about modal"}:
            normalized_user = (
                "What does Modal do? Give a direct one-sentence definition first, then one practical use case."
            )

        if not context_lines:
            return LLMResult(
                text="I do not have enough verified Modal docs context for that question. Please rephrase with Modal-specific terms.",
                latency_s=0.0,
            )

        contextual_user = normalized_user + (
            "\\n\\nUse ONLY factual information consistent with these Modal docs snippets when answering. "
            "Prioritize concrete Modal behavior and avoid generic ML platform descriptions.\\n\\nSources:\\n"
            + "\\n\\n".join(context_lines)
        )

        return self.generate(contextual_user, max_tokens=max_tokens, temperature=temperature)
