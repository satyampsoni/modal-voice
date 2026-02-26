from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class TTSResult:
    audio_bytes: bytes
    latency_s: float
    mime_type: str


class CoquiTTS:
    """Coqui TTS wrapper for container-scoped speech synthesis."""

    def __init__(self, model_name: str = "tts_models/en/ljspeech/tacotron2-DDC", use_gpu: bool = True) -> None:
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.model: Any | None = None

    def load(self) -> None:
        if self.model is None:
            from TTS.api import TTS

            self.model = TTS(model_name=self.model_name, progress_bar=False, gpu=self.use_gpu)

    def synthesize(self, text: str) -> TTSResult:
        if self.model is None:
            raise RuntimeError("TTS model is not loaded")

        clean_text = text.strip()
        if not clean_text:
            clean_text = "I could not hear a clear question. Please try again."

        start = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            self.model.tts_to_file(text=clean_text, file_path=tmp.name)
            tmp.flush()
            tmp.seek(0)
            audio = tmp.read()

        return TTSResult(audio_bytes=audio, latency_s=time.perf_counter() - start, mime_type="audio/wav")
