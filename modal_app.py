from __future__ import annotations

import os
import time
from pathlib import Path

import modal
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

try:
    from modal_voice.llm import ModalVLLM
    from modal_voice.stt import WhisperSTT
    from modal_voice.tts import CoquiTTS
except ModuleNotFoundError:
    # Support flat repo layout where files live at repository root.
    from llm import ModalVLLM
    from stt import WhisperSTT
    from tts import CoquiTTS

app = modal.App("modal-voice")
hf_cache = modal.Volume.from_name("modalvoice-hf-cache", create_if_missing=True)
tts_cache = modal.Volume.from_name("modalvoice-tts-cache", create_if_missing=True)
rag_cache = modal.Volume.from_name("modalvoice-rag-cache", create_if_missing=True)

base_image = modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg")
gpu_base_image = modal.Image.from_registry(
    "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04",
    add_python="3.11",
).apt_install("ffmpeg", "git", "espeak-ng")

stt_image = (
    gpu_base_image
    .pip_install("faster-whisper==1.1.1", "requests", "fastapi==0.115.8", "python-multipart==0.0.20")
    .add_local_python_source("modal_voice")
)

llm_image = (
    gpu_base_image
    .pip_install(
        "vllm==0.6.6.post1",
        "transformers==4.45.2",
        "requests",
        "beautifulsoup4",
        "chromadb==0.5.11",
        "sentence-transformers==3.2.0",
        "fastapi==0.115.8",
        "python-multipart==0.0.20",
    )
    .add_local_python_source("modal_voice")
)

tts_image = (
    gpu_base_image
    .pip_install("TTS==0.22.0", "fastapi==0.115.8", "python-multipart==0.0.20")
    .add_local_python_source("modal_voice")
)

web_image = (
    base_image
    .pip_install("fastapi==0.115.8", "python-multipart==0.0.20")
    .add_local_python_source("modal_voice")
    .add_local_dir("modal_voice/static", remote_path="/root/static")
)


@app.cls(
    image=stt_image,
    gpu="A10G",
    scaledown_window=1800,
    timeout=600,
    min_containers=1,
    max_containers=1,
    volumes={"/root/.cache/huggingface": hf_cache},
)
class STTService:
    @modal.enter()
    def load_once(self) -> None:
        model_name = os.environ.get("WHISPER_MODEL", "tiny")
        compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")
        beam_size = int(os.environ.get("WHISPER_BEAM_SIZE", "1"))
        self.stt = WhisperSTT(model_name=model_name, compute_type=compute_type, beam_size=beam_size)
        self.stt.load()
        hf_cache.commit()

    @modal.method()
    def transcribe(self, audio_bytes: bytes, language: str | None = None) -> dict:
        result = self.stt.transcribe(audio_bytes, language)
        return {"text": result.text, "latency_s": result.latency_s}


@app.cls(
    image=llm_image,
    gpu="A10G",
    scaledown_window=1800,
    timeout=900,
    min_containers=1,
    max_containers=1,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/rag": rag_cache,
    },
)
class LLMService:
    @modal.enter()
    def load_once(self) -> None:
        model_name = os.environ.get("VLLM_MODEL", "microsoft/Phi-3-mini-4k-instruct")
        max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "2048"))
        gpu_mem_util = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.9"))
        enforce_eager = os.environ.get("VLLM_ENFORCE_EAGER", "true").lower() in {"1", "true", "yes"}
        rag_pages = int(os.environ.get("RAG_MAX_PAGES", "50"))
        self.llm = ModalVLLM(
            model_name=model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_mem_util,
            enforce_eager=enforce_eager,
        )
        self.llm.load()
        try:
            from modal_voice.rag import ModalExamplesRAG
        except ModuleNotFoundError:
            from rag import ModalExamplesRAG

        self.rag = ModalExamplesRAG(max_pages=rag_pages)
        self.rag.load(max_cache_age_s=int(os.environ.get("RAG_CACHE_MAX_AGE_S", "86400")))
        self.answer_cache: dict[str, dict] = {}
        hf_cache.commit()
        rag_cache.commit()
        rag_docs = self.rag.collection.count() if getattr(self.rag, "collection", None) else 0
        print({"event": "llm_ready", "model": model_name, "enforce_eager": enforce_eager})
        print({"event": "rag_ready", "documents": rag_docs, "max_pages": rag_pages})

    @modal.method()
    def generate(self, prompt: str) -> dict:
        max_tokens = int(os.environ.get("VLLM_MAX_OUTPUT_TOKENS", "120"))
        rag_k = int(os.environ.get("RAG_TOP_K", "2"))
        cache_key = " ".join(prompt.lower().split())
        cached = self.answer_cache.get(cache_key)
        if cached:
            return cached
        t0 = time.perf_counter()
        contexts = self.rag.retrieve(prompt, k=rag_k) if hasattr(self, "rag") else []
        rag_latency_s = time.perf_counter() - t0
        result = self.llm.generate_with_context(
            prompt,
            context_snippets=contexts,
            max_tokens=max_tokens,
            temperature=0.1,
        )
        out = {
            "text": result.text,
            "latency_s": result.latency_s,
            "rag_latency_s": rag_latency_s,
            "rag_hits": len(contexts),
        }
        if len(self.answer_cache) >= 256:
            self.answer_cache.pop(next(iter(self.answer_cache)))
        self.answer_cache[cache_key] = out
        return out


@app.cls(
    image=tts_image,
    gpu="A10G",
    scaledown_window=1800,
    timeout=600,
    min_containers=1,
    max_containers=1,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.local/share/tts": tts_cache,
    },
)
class TTSService:
    @modal.enter()
    def load_once(self) -> None:
        model_name = os.environ.get("TTS_MODEL", "tts_models/en/ljspeech/vits")
        self.tts = CoquiTTS(model_name=model_name, use_gpu=True)
        self.tts.load()
        self.cache: dict[str, dict] = {}
        tts_cache.commit()
        print({"event": "tts_ready", "model": model_name})

    @modal.method()
    def synthesize(self, text: str) -> dict:
        cached = self.cache.get(text)
        if cached:
            return cached
        result = self.tts.synthesize(text)
        out = {
            "audio_bytes": result.audio_bytes,
            "latency_s": result.latency_s,
            "mime_type": result.mime_type,
        }
        # Small in-memory cache for repeated prompts.
        if len(self.cache) >= 128:
            self.cache.pop(next(iter(self.cache)))
        self.cache[text] = out
        return out


stt_service = STTService()
llm_service = LLMService()
tts_service = TTSService()


@app.function(image=web_image)
@modal.asgi_app()
def web_app() -> FastAPI:
    api = FastAPI(title="ModalVoice")
    static_dir = Path("/root/static")

    api.mount("/static", StaticFiles(directory=static_dir), name="static")

    @api.get("/")
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @api.post("/api/voice")
    async def voice(audio: UploadFile = File(...), language: str | None = Form(default=None)) -> Response:
        if audio.content_type is None or not audio.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="Expected an audio file")

        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Audio payload is empty")

        total_start = time.perf_counter()

        stt = await stt_service.transcribe.remote.aio(audio_bytes=audio_bytes, language=language)
        transcript = (stt.get("text") or "").strip()
        if not transcript:
            transcript = "Please ask a question about Modal deployment, GPUs, autoscaling, or functions."

        llm = await llm_service.generate.remote.aio(prompt=transcript)
        answer = (llm.get("text") or "").strip()
        thinking_time_s = float(stt.get("latency_s", 0.0)) + float(llm.get("latency_s", 0.0))

        tts = await tts_service.synthesize.remote.aio(text=answer)
        audio_out = tts.get("audio_bytes") or b""
        if not audio_out:
            raise HTTPException(status_code=500, detail="TTS produced empty audio")
        total_latency = time.perf_counter() - total_start

        # Server-side latency logging for observability.
        print(
            {
                "event": "modalvoice_request",
                "stt_latency_s": stt.get("latency_s"),
                "rag_latency_s": llm.get("rag_latency_s"),
                "rag_hits": llm.get("rag_hits"),
                "llm_latency_s": llm.get("latency_s"),
                "thinking_time_s": thinking_time_s,
                "tts_latency_s": tts.get("latency_s"),
                "total_latency_s": total_latency,
            }
        )
        print(f"[ModalVoice] thinking_time_s={thinking_time_s:.3f} total_latency_s={total_latency:.3f}")

        headers = {
            "x-stt-latency-s": f"{stt.get('latency_s', 0.0):.3f}",
            "x-llm-latency-s": f"{llm.get('latency_s', 0.0):.3f}",
            "x-rag-latency-s": f"{llm.get('rag_latency_s', 0.0):.3f}",
            "x-rag-hits": str(llm.get("rag_hits", 0)),
            "x-thinking-time-s": f"{thinking_time_s:.3f}",
            "x-tts-latency-s": f"{tts.get('latency_s', 0.0):.3f}",
            "x-total-latency-s": f"{total_latency:.3f}",
        }
        return Response(content=audio_out, media_type=tts.get("mime_type", "audio/wav"), headers=headers)

    @api.post("/api/greet")
    async def greet() -> Response:
        greeting = "Hey, I am ModalVoice. How can I help you?"
        tts = await tts_service.synthesize.remote.aio(text=greeting)
        audio_out = tts.get("audio_bytes") or b""
        if not audio_out:
            raise HTTPException(status_code=500, detail="Greeting TTS produced empty audio")
        return Response(content=audio_out, media_type=tts.get("mime_type", "audio/wav"))

    return api


@app.local_entrypoint()
def main() -> None:
    print("Serve with: modal serve modal_app.py")
