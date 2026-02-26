from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class STTResult:
    text: str
    latency_s: float


class WhisperSTT:
    """Whisper wrapper for container-scoped STT inference."""

    def __init__(
        self,
        model_name: str = "small",
        device: str = "cuda",
        compute_type: str = "float16",
        beam_size: int = 1,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.model: Any | None = None

    def load(self) -> None:
        if self.model is None:
            from faster_whisper import WhisperModel

            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )

    def transcribe(self, audio_bytes: bytes, language: str | None = None) -> STTResult:
        if self.model is None:
            raise RuntimeError("Whisper model is not loaded")

        start = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=".tmp", delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()

            segments_iter, _ = self.model.transcribe(
                tmp.name,
                language=language,
                beam_size=self.beam_size,
                vad_filter=True,
            )
            segments = list(segments_iter)

            if not segments:
                segments_iter, _ = self.model.transcribe(
                    tmp.name,
                    language=language,
                    beam_size=self.beam_size,
                    vad_filter=False,
                )
                segments = list(segments_iter)

        text = " ".join(s.text.strip() for s in segments if s.text).strip()
        return STTResult(text=text, latency_s=time.perf_counter() - start)
