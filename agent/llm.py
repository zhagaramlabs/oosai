"""Sarvam 30B (official Q4_K_M GGUF) on mainline llama.cpp in ~/ai/llama.cpp, bound to 127.0.0.1.

Thinking is off by default (a voice reply can't wait for it): the model ignores the template's
<|nothink|> switch, so an empty <think></think> is pre-filled instead.
"""
import atexit
import json
import subprocess
import time
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

LLAMA = Path.home() / "ai/llama.cpp/build/bin/llama-server"


class Sarvam:
    def __init__(self, port=19110):
        self.url, self.calls = f"http://127.0.0.1:{port}", []  # calls: every /completion response, for timing
        model = Path(snapshot_download("sarvamai/sarvam-30b-gguf")) / "sarvam-30b-Q4_K_M.gguf-00001-of-00006.gguf"
        server = subprocess.Popen([str(LLAMA), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
                                   "-ngl", "99", "-c", "8192", "--jinja"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        atexit.register(server.terminate)
        for _ in range(300):
            try:
                urllib.request.urlopen(self.url + "/health", timeout=1)
                return
            except OSError:
                time.sleep(1)
        raise RuntimeError("llama-server did not start")

    def post(self, path, body):
        req = urllib.request.Request(self.url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.loads(r.read().decode("utf-8"))

    def generate(self, messages, tools=None, think=False, n_predict=300):
        """One reply at temperature 0; with tools, the raw text may hold a <tool_call>."""
        prompt = self.post("/apply-template", {"messages": messages, **({"tools": tools} if tools else {})})["prompt"]
        self.calls.append(self.post("/completion", {"prompt": prompt + ("" if think else "<think></think>"),
                                                    "n_predict": n_predict, "temperature": 0}))
        return self.calls[-1]["content"].split("</think>")[-1].strip()
