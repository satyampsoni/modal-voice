# ModalVoice

ModalVoice is a fully self-hosted voice assistant on Modal:

1. Browser audio input
2. Whisper STT (GPU)
3. RAG retrieval from Modal docs (`/docs/examples`, `/docs/guide`, `/docs`)
4. vLLM response generation with open-weight model (GPU)
5. Coqui TTS synthesis (open-source)
6. Audio-only response playback in browser

No paid APIs or API keys are required.

## Project Structure

- `modal_voice/modal_app.py`: Modal app + web API + service orchestration
- `modal_voice/stt.py`: Whisper transcription wrapper
- `modal_voice/llm.py`: vLLM generation wrapper
- `modal_voice/tts.py`: Coqui TTS wrapper
- `modal_voice/prompts.py`: Modal-specialized system prompt
- `modal_voice/rag.py`: lightweight RAG indexer/retriever for Modal examples docs
- `modal_voice/static/index.html`: minimal audio-only UI
- `modal_voice/static/styles.css`: clean minimal styling
- `modal_voice/static/app.js`: record/send/autoplay behavior

## Architecture

- `STTService` (`@app.cls`): loads Whisper once per container and transcribes audio.
- `LLMService` (`@app.cls`): loads vLLM once, builds/loads cached RAG index from Modal examples docs, and generates grounded answers.
- `TTSService` (`@app.cls`): loads Coqui model once per container and synthesizes WAV audio.
- `web_app` (`@modal.asgi_app`): serves UI and `/api/voice`.

## Request Flow

1. Frontend records microphone audio.
2. `POST /api/voice` uploads the recording.
3. Backend calls STT -> LLM -> TTS in sequence.
4. LLM stage retrieves top matching examples snippets from local RAG cache first.
5. Backend returns audio bytes (`audio/wav`) only.
6. Frontend auto-plays returned audio.

## Latency Logging

For each request, backend logs:

- `stt_latency_s`
- `rag_latency_s`
- `rag_hits`
- `llm_latency_s`
- `tts_latency_s`
- `total_latency_s`

These are also returned as HTTP headers:

- `x-stt-latency-s`
- `x-rag-latency-s`
- `x-rag-hits`
- `x-llm-latency-s`
- `x-tts-latency-s`
- `x-total-latency-s`

## Deployment

From project root (recommended):

```bash
modal serve -m modal_voice.modal_app
```

Alternative (matches requested style):

```bash
cd modal_voice
modal serve modal_app.py
```

## Optional Model Settings

```bash
export WHISPER_MODEL=small
export WHISPER_COMPUTE_TYPE=float16

export VLLM_MODEL=microsoft/Phi-3-mini-4k-instruct
export VLLM_MAX_MODEL_LEN=2048
export VLLM_GPU_MEMORY_UTILIZATION=0.9
export VLLM_ENFORCE_EAGER=true
export VLLM_MAX_OUTPUT_TOKENS=64

export TTS_MODEL=tts_models/en/ljspeech/tacotron2-DDC

export RAG_MAX_PAGES=50
export RAG_TOP_K=3
export RAG_CACHE_MAX_AGE_S=86400
```

## UI Behavior

- White background
- Single centered circular record button
- Subtle pulse animation while recording
- No transcript text shown
- Auto-play audio response
- Mobile responsive

## Notes

- First request may be slower due to model cold start.
- Service containers are configured warm (`min_containers=1`) and use Modal Volumes for HF/TTS caches to reduce repeated downloads.
- RAG index is cached in a Modal Volume (`modalvoice-rag-cache`) so retrieval is local and fast after initial build.
- If vLLM OOM occurs, switch to a smaller open model.
- For lower latency, you can reduce model sizes or token limits.
