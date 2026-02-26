const recordBtn = document.getElementById("recordBtn");
const logoBtn = document.getElementById("logoBtn");
const player = document.getElementById("player");
const statusText = document.getElementById("statusText");

let stream = null;
let recorder = null;
let chunks = [];
let recording = false;
let busy = false;
let currentAudioUrl = null;

function setBusy(value) {
  busy = value;
  recordBtn.disabled = value;
  recordBtn.style.opacity = value ? "0.55" : "1";
}

function setStatus(message) {
  statusText.textContent = message;
}

async function playAudioBlob(blob) {
  if (!blob || blob.size === 0) throw new Error("Received empty audio response");
  stopSpeaking();
  const url = URL.createObjectURL(blob);
  currentAudioUrl = url;
  player.src = url;
  try {
    player.muted = false;
    player.volume = 1.0;
    await player.play();
    setStatus("Playing response...");
  } catch (error) {
    throw new Error(`Audio playback failed: ${error.message}`);
  }
}

function stopSpeaking() {
  player.pause();
  player.currentTime = 0;
  player.removeAttribute("src");
  player.load();
  if (currentAudioUrl) {
    URL.revokeObjectURL(currentAudioUrl);
    currentAudioUrl = null;
  }
}

async function startRecording() {
  stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  recorder = new MediaRecorder(stream);
  chunks = [];

  recorder.ondataavailable = (event) => {
    if (event.data.size > 0) chunks.push(event.data);
  };

  recorder.onstop = async () => {
    const audioBlob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
    stream.getTracks().forEach((track) => track.stop());
    stream = null;

    setBusy(true);
    setStatus("Thinking...");
    try {
      const form = new FormData();
      form.append("audio", audioBlob, "input.webm");
      form.append("language", "en");

      const response = await fetch("/api/voice", {
        method: "POST",
        body: form,
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "Voice request failed");
      }

      const audioReply = await response.blob();
      await playAudioBlob(audioReply);
    } catch (error) {
      console.error(error);
      setStatus(`Error: ${error.message}`);
    } finally {
      setBusy(false);
    }
  };

  recorder.start();
  recording = true;
  recordBtn.classList.add("recording");
  setStatus("Listening...");
}

function stopRecording() {
  if (recorder && recorder.state !== "inactive") recorder.stop();
  recording = false;
  recordBtn.classList.remove("recording");
  setStatus("Processing...");
}

recordBtn.addEventListener("click", async () => {
  if (busy) return;

  // If assistant audio is playing, use the main button as an interrupt.
  if (!player.paused && player.currentTime > 0) {
    stopSpeaking();
    setStatus("Playback stopped.");
    return;
  }

  try {
    if (!recording) {
      await startRecording();
    } else {
      stopRecording();
    }
  } catch (error) {
    console.error(error);
    recording = false;
    recordBtn.classList.remove("recording");
    setStatus(`Error: ${error.message}`);
  }
});

logoBtn.addEventListener("click", async () => {
  if (busy) return;
  setBusy(true);
  setStatus("Preparing greeting...");
  try {
    const response = await fetch("/api/greet", { method: "POST" });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || "Greeting request failed");
    }
    const audioReply = await response.blob();
    await playAudioBlob(audioReply);
  } catch (error) {
    console.error(error);
    setStatus(`Error: ${error.message}`);
  } finally {
    setBusy(false);
  }
});

player.addEventListener("ended", () => {
  setStatus("Tap to ask about Modal");
});
