"""MiniCPM-o 4.5 (Q8_0) on the llama.cpp-omni engine built in ~/ai/voice/s2s; same API calls as its voice_agent.py."""
import atexit
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import soundfile as sf

S2S = Path.home() / "ai/voice/s2s"
MODELS = S2S / "models/MiniCPM-o-4_5-gguf"
VOICE = S2S / "MiniCPM-o-Demo/assets/ref_audio/ref_en_dlc_1.wav"  # the demo's stock voice, not a cloned person


class Engine:
    def __init__(self, port=19099):
        self.url, self.work, self.n = f"http://127.0.0.1:{port}", Path(tempfile.mkdtemp(prefix="minicpmo_")), 0
        proc = subprocess.Popen(
            [str(S2S / "llama.cpp-omni/build/bin/llama-server"), "--host", "127.0.0.1", "--port", str(port),
             "--model", str(MODELS / "MiniCPM-o-4_5-Q8_0.gguf"), "-ngl", "99", "--ctx-size", "4096", "--temp", "0",
             "-n", "768"],  # cap the reply: Tamil output sometimes loops until the context is full (~400 s)
            cwd=S2S / "llama.cpp-omni", stdout=open(self.work / "engine.log", "w"), stderr=subprocess.STDOUT)
        atexit.register(proc.terminate)
        for _ in range(240):
            try:
                urllib.request.urlopen(self.url + "/health", timeout=1)
                break
            except OSError:
                time.sleep(0.5)
        self.silence = self.work / "silence.wav"
        sf.write(self.silence, [0.0] * 1600, 16000, subtype="PCM_16")

    def post(self, path, body):
        req = urllib.request.Request(self.url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.read().decode("utf-8")  # the engine sends UTF-8 without a charset header

    def ask(self, wav_path, system, use_tts=False):
        """One single-turn exchange in a fresh context. Returns (reply text, reply wav path or None)."""
        self.n += 1
        out = self.work / f"out_{self.n:04d}"
        # The omni server has no reset call, so every exchange gets a fresh context from omni_init.
        self.post("/v1/stream/omni_init", {
            # media_type 2: the engine only applies custom prompts in this mode
            "media_type": 2, "use_tts": use_tts, "duplex_mode": False, "model_dir": str(MODELS),
            "tts_bin_dir": str(MODELS / "tts"), "tts_gpu_layers": 100, "token2wav_device": "gpu:0",
            "output_dir": str(out), "voice_audio": str(VOICE),
            "voice_clone_prompt": "<|im_start|>system\nClone the voice in the provided audio prompt.\n<|audio_start|>",
            "assistant_prompt": f"<|audio_end|>{system}<|im_end|>\n<|im_start|>user\n"})
        wav, sr = sf.read(wav_path, dtype="float32")
        assert sr == 16000, sr
        chunks = [self.silence]  # cnt 0 sets up the system prompt
        for i in range(0, len(wav), sr):  # the engine takes 1-second chunks
            chunks.append(self.work / f"in_{len(chunks):04d}.wav")
            sf.write(chunks[-1], wav[i:i + sr], sr, subtype="PCM_16")
        for cnt, c in enumerate(chunks):
            self.post("/v1/stream/prefill", {"audio_path_prefix": str(c), "img_path_prefix": "", "cnt": cnt})
        body = self.post("/v1/stream/decode", {"stream": True})
        text = "".join(json.loads(l[6:]).get("content", "") for l in body.splitlines()
                       if l.startswith("data: ") and l[6:] != "[DONE]").strip()
        if not use_tts:
            return text, None
        tts = out / "round_000/tts_wav"
        for _ in range(1200):  # rendering can lag behind the text
            if (tts / "generation_done.flag").exists():
                break
            time.sleep(0.1)
        files = sorted(tts.glob("wav_*.wav"), key=lambda f: int(f.stem.split("_")[1]))
        if not files:
            return text, None
        audio = [s for f in files for s in sf.read(f, dtype="float32")[0]]
        sf.write(out / "reply.wav", audio, sf.info(files[0]).samplerate)
        return text, out / "reply.wav"
