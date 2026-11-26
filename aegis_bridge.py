from flask import Flask, request, jsonify
import requests
import json
import threading
import queue
import time
import os
import socket
import logging
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


class SuppressRegisterFilter(logging.Filter):
    """Hides repetitive 30s /register heartbeats while keeping startup URLs visible."""
    def filter(self, record):
        return "POST /register" not in record.getMessage()


logging.getLogger("werkzeug").addFilter(SuppressRegisterFilter())

app = Flask(__name__)

# --- CONFIGURATION ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "aegis"
WHISPER_SIZE = "base"                                   # ~250-400ms on CPU
CPU_THREADS = max(4, (os.cpu_count() or 8) // 2)        # Physical cores avoid SMT contention

# Audio / VAD tuning (16kHz mono, 30ms frames)
SAMPLE_RATE = 16000
BLOCK_SIZE = 480                # 30ms per chunk
PRE_ROLL_SEC = 0.25             # Capture 250ms before trigger so first syllable isn't cut
SILENCE_HANGOVER_SEC = 0.30     # Trigger inference 300ms after speech stops
MIN_SPEECH_SEC = 0.35           # Ignore brief clicks/coughs under 350ms
MAX_UTTERANCE_SEC = 4.5         # Commands are short; prevents long noise captures

# --- SHARED THREAD-SAFE STATE ---
state_lock = threading.Lock()
action_history = []
telemetry_state = {"main_light": False, "left_light": False}
esp32_ip = None

# Persistent HTTP sessions (eliminates TCP handshake latency)
ollama_session = requests.Session()
esp_session = requests.Session()

print(f"[INIT] Loading Whisper '{WHISPER_SIZE}' STT model ({CPU_THREADS} threads)...")
stt_model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8", cpu_threads=CPU_THREADS)
print("[INIT] Whisper STT Ready.")
try:
    ollama_session.post(OLLAMA_URL, json={"model": MODEL_NAME, "keep_alive": -1}, timeout=15)
    print("[INIT] A.E.G.I.S. brain pinned in memory.")
except requests.RequestException as e:
    print(f"[WARN] Could not pre-warm Ollama: {e}")


def normalize_json_command(raw_text: str) -> str:
    """Ensures valid JSON even when Ollama stops at '}' due to Modelfile stop token."""
    cleaned = raw_text.strip()
    if cleaned.startswith("{") and not cleaned.endswith("}"):
        cleaned += "}"
    return cleaned


def query_aegis_brain(user_input: str) -> str:
    """Queries local Ollama model and updates rolling history & telemetry."""
    with state_lock:
        current_telemetry = json.dumps(telemetry_state)
        current_history = json.dumps(action_history)

    prompt = f"Telemetry: {current_telemetry}\nHistory: {current_history}\nInput: {user_input}"

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "keep_alive": -1  # Keep model pinned in RAM/VRAM permanently
    }

    response = ollama_session.post(OLLAMA_URL, json=payload, timeout=15)
    raw_output = response.json().get("response", "")
    ai_command = normalize_json_command(raw_output)

    try:
        parsed_cmd = json.loads(ai_command)
        device = parsed_cmd.get("device")
        state = parsed_cmd.get("state")

        if device in ("main_light", "left_light", "both") and state in ("on", "off"):
            turn_on = (state == "on")
            with state_lock:
                if device == "main_light":
                    telemetry_state["main_light"] = turn_on
                elif device == "left_light":
                    telemetry_state["left_light"] = turn_on
                elif device == "both":
                    telemetry_state["main_light"] = turn_on
                    telemetry_state["left_light"] = turn_on

                action_history.append(f"{device} -> {state}")
                if len(action_history) > 5:
                    action_history.pop(0)
    except json.JSONDecodeError:
        print(f"[WARN] Malformed JSON from Ollama: {ai_command}")

    return ai_command


def push_to_esp32(command_json_str: str):
    """Pushes a voice-triggered command directly to the ESP32 HTTP server (<5ms)."""
    with state_lock:
        target_ip = esp32_ip

    if not target_ip:
        print("[WARN] Voice command ready, but ESP32 IP is not registered yet.")
        return

    try:
        url = f"http://{target_ip}/execute"
        resp = esp_session.post(
            url,
            data=command_json_str,
            headers={"Content-Type": "application/json"},
            timeout=2.0
        )
        if resp.status_code == 200:
            with state_lock:
                telemetry_state.update(resp.json())
            print(f"[ESP32 PUSH OK] -> {command_json_str}")
        else:
            print(f"[ESP32 PUSH ERR] HTTP {resp.status_code}")
    except requests.RequestException as e:
        print(f"[ESP32 UNREACHABLE] {e}")


# --- FLASK ENDPOINTS ---

@app.route('/register', methods=['POST'])
def register_esp32():
    """Called by ESP32 on boot/heartbeat to register its IP and sync GPIO states."""
    global esp32_ip
    data = request.json or {}
    client_ip = data.get("ip") or request.remote_addr
    incoming_telemetry = data.get("telemetry", {})

    with state_lock:
        first_seen = (esp32_ip != client_ip)
        esp32_ip = client_ip
        if incoming_telemetry:
            telemetry_state.update(incoming_telemetry)

    if first_seen:
        print(f"[DISCOVERY] ESP32 registered at {esp32_ip} | Telemetry: {telemetry_state}")
    return jsonify({"status": "registered", "ip": esp32_ip})


@app.route('/aegis', methods=['POST'])
def aegis_endpoint():
    """Preserves full compatibility with ESP32 Serial Monitor text commands."""
    global esp32_ip
    data = request.json or {}
    user_input = data.get("input", "")
    incoming_telemetry = data.get("telemetry", {})

    with state_lock:
        if request.remote_addr:
            esp32_ip = request.remote_addr
        if incoming_telemetry:
            telemetry_state.update(incoming_telemetry)

    t0 = time.perf_counter()
    ai_command = query_aegis_brain(user_input)
    dt = (time.perf_counter() - t0) * 1000
    print(f"[TEXT CMD] '{user_input}' -> {ai_command} ({dt:.0f} ms)")

    return jsonify({"command": ai_command})


# --- DIRECT REAL-TIME VOICE PIPELINE ---

HALLUCINATION_BLACKLIST = {
    "thank you.", "thank you", "thanks for watching.", "bye.", "bye",
    "you", "you.", "okay.", "okay", "yeah.", "yeah", "hmm.", "hmm", "aegis.", "aegis"
}


def highpass_rms(chunk: np.ndarray) -> float:
    """Strips DC bias and low-frequency fan/AC rumble (<300Hz) before measuring energy."""
    hp = chunk[1:] - 0.95 * chunk[:-1]
    return float(np.sqrt(np.mean(hp ** 2)))


def prepare_audio_for_stt(audio_data: np.ndarray) -> np.ndarray:
    """Removes Realtek DC voltage offset and normalizes speech volume to -3 dBFS."""
    centered = audio_data - np.mean(audio_data)
    peak = float(np.max(np.abs(centered)))
    if peak > 0.008:
        centered = centered * (0.70 / peak)
    return centered.astype(np.float32)


def voice_worker():
    """Captures microphone audio, detects speech endpoints, transcribes, and executes."""
    audio_q = queue.Queue()
    mic_info = sd.query_devices(kind="input")
    print(f"[VOICE] Using microphone: {mic_info['name']}")

    def audio_callback(indata, frames, time_info, status):
        audio_q.put(indata[:, 0].copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=BLOCK_SIZE,
        callback=audio_callback
    ):
        # 1. Flush initial 300ms Windows audio driver startup transient
        for _ in range(int(0.3 * SAMPLE_RATE / BLOCK_SIZE)):
            audio_q.get()

        # 2. Calibrate ambient room noise floor (0.6s) on high-pass filtered audio
        print("[VOICE] Calibrating room noise floor...")
        noise_samples = []
        for _ in range(int(0.6 * SAMPLE_RATE / BLOCK_SIZE)):
            chunk = audio_q.get()
            noise_samples.append(highpass_rms(chunk))

        noise_floor = max(float(np.median(noise_samples)), 0.005)
        start_threshold = max(noise_floor * 3.5, 0.038)
        stop_threshold = max(noise_floor * 1.8, 0.018)
        print(f"[VOICE] Always-On Listening (Filtered noise: {noise_floor:.4f}, Start: {start_threshold:.4f})")

        speech_buffer = []
        pre_roll = []
        max_pre_roll = int(PRE_ROLL_SEC * SAMPLE_RATE / BLOCK_SIZE)
        in_speech = False
        peak_rms = 0.0
        silence_chunks = 0
        max_silence_chunks = int(SILENCE_HANGOVER_SEC * SAMPLE_RATE / BLOCK_SIZE)
        min_speech_chunks = int(MIN_SPEECH_SEC * SAMPLE_RATE / BLOCK_SIZE)
        max_utterance_chunks = int(MAX_UTTERANCE_SEC * SAMPLE_RATE / BLOCK_SIZE)

        while True:
            chunk = audio_q.get()
            rms = highpass_rms(chunk)

            if not in_speech:
                pre_roll.append(chunk)
                if len(pre_roll) > max_pre_roll:
                    pre_roll.pop(0)

                if rms > start_threshold:
                    in_speech = True
                    peak_rms = rms
                    silence_chunks = 0
                    speech_buffer = list(pre_roll) + [chunk]
                    print("[VOICE] Speech detected...   ", end="\r", flush=True)
                else:
                    noise_floor = 0.98 * noise_floor + 0.02 * rms
                    start_threshold = max(noise_floor * 3.5, 0.038)
                    stop_threshold = max(noise_floor * 1.8, 0.018)
            else:
                speech_buffer.append(chunk)
                if rms > peak_rms:
                    peak_rms = rms

                if rms <= stop_threshold:
                    silence_chunks += 1
                else:
                    silence_chunks = 0

                if silence_chunks >= max_silence_chunks or len(speech_buffer) >= max_utterance_chunks:
                    in_speech = False
                    print("                              ", end="\r", flush=True)
                    if len(speech_buffer) >= min_speech_chunks and peak_rms >= (start_threshold * 1.15):
                        raw_audio = np.concatenate(speech_buffer)
                        speech_buffer = []
                        clean_audio = prepare_audio_for_stt(raw_audio)
                        process_utterance(clean_audio)

                        # Drain any queued frames captured during inference
                        while not audio_q.empty():
                            try:
                                audio_q.get_nowait()
                            except queue.Empty:
                                break
                    else:
                        speech_buffer = []


def process_utterance(audio_data: np.ndarray):
    """Transcribes normalized speech slice and dispatches valid commands to Ollama -> ESP32."""
    t_start = time.perf_counter()

    segments, _ = stt_model.transcribe(
        audio_data,
        language="en",
        beam_size=1,
        best_of=1,
        without_timestamps=True,
        max_new_tokens=32,
        vad_filter=False,
        no_speech_threshold=0.60,
        log_prob_threshold=-0.80,
        compression_ratio_threshold=2.0,
        condition_on_previous_text=False
    )
    transcript = " ".join(seg.text for seg in segments).strip()
    t_stt = (time.perf_counter() - t_start) * 1000

    if not transcript or len(transcript) < 3 or transcript.lower() in HALLUCINATION_BLACKLIST:
        return

    ai_command = query_aegis_brain(transcript)
    t_total = (time.perf_counter() - t_start) * 1000
    print(f"[VOICE CMD] '{transcript}' -> {ai_command} (STT: {t_stt:.0f}ms | Total: {t_total:.0f}ms)")

    # Only push valid hardware actions to ESP32
    try:
        parsed = json.loads(ai_command)
        if parsed.get("device") not in ("main_light", "left_light", "both"):
            return
        if parsed.get("state") not in ("on", "off"):
            return
    except json.JSONDecodeError:
        return

    push_to_esp32(ai_command)


if __name__ == '__main__':
    local_ips = [
        ip for ip in socket.gethostbyname_ex(socket.gethostname())[2]
        if not ip.startswith("127.")
    ]
    print("\n[NETWORK] A.E.G.I.S. Connection URLs (set bridgeBaseUrl in ESP32 to match subnet):")
    for ip in local_ips:
        print(f"   -> http://{ip}:5000")
    print()

    voice_thread = threading.Thread(target=voice_worker, daemon=True)
    voice_thread.start()
    app.run(host='0.0.0.0', port=5000, threaded=True)