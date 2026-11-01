"""Push-to-talk Tamil voice loop (README §7, Phase 1 step 5):
mic -> Indic-Transcribe -> agent/rules.py + Sarvam 30B -> Indic-Speak, sentence by sentence -> speaker.

  .venv/bin/python -m agent.loop                       # Enter to talk, Enter to stop; Ctrl-C quits
  .venv/bin/python -m agent.loop --tts vani            # the macOS voice: fast, for trying the loop only
  .venv/bin/python -m agent.loop --wav a.wav b.wav     # recorded turns instead of the mic, no playback

Each turn prints what was heard, which rule or tool answered, the reply, and seconds per stage;
replies are saved as wav files. The first run asks macOS for microphone access for your terminal.
"""
import argparse
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from agent import rules  # noqa: E402
from agent.llm import Sarvam  # noqa: E402
from agent.numbers_ta import speak_digits  # noqa: E402


def vani():
    """macOS's Tamil voice. Apple's terms make it a test stand-in, not a voice for the product."""
    def run(text):
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            subprocess.run(["say", "-v", "Vani", "--file-format=WAVE", "--data-format=LEF32@22050", "-o", f.name, text],
                           check=True)
            return sf.read(f.name, dtype="float32")[0]
    return run, 22050


def load_tts(name, voice):
    if name == "vani":
        return vani()
    import tts_samples  # eval/tts_samples.py loads both models the way the TTS test does
    voices, sr = getattr(tts_samples, name)()
    return voices[voice] if voice in voices else next(iter(voices.values())), sr


def output_muted():
    r = subprocess.run(["osascript", "-e", "output muted of (get volume settings)"], capture_output=True, text=True)
    return r.stdout.strip() == "true"


def record(sr=16000):
    import sounddevice as sd
    input("\nPress Enter and speak Tamil; press Enter again when you finish. ")
    chunks = []
    with sd.InputStream(samplerate=sr, channels=1, dtype="float32", callback=lambda d, *_: chunks.append(d.copy())):
        input("  ... listening (Enter to stop)")
    return np.concatenate(chunks)[:, 0] if chunks else np.zeros(0, np.float32)


def speak(reply, tts, sr, play):
    """Synthesise sentence by sentence while earlier sentences play. Returns (audio, time first audio was ready)."""
    sentences = [speak_digits(s) for s in re.split(r"(?<=[.!?])\s+", reply) if s.strip()]
    ready, first = queue.Queue(), []

    def player():
        import sounddevice as sd
        while (w := ready.get()) is not None:
            if not first:
                first.append(time.perf_counter())
            if play:
                sd.play(w, sr)
                sd.wait()

    th = threading.Thread(target=player)
    th.start()
    audio = []
    for s in sentences:
        audio.append(tts(s))
        ready.put(audio[-1])
    ready.put(None)
    th.join()
    return (np.concatenate(audio) if audio else np.zeros(0, np.float32)), (first[0] if first else None)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tts", choices=["indicspeak", "indicf5", "vani"], default="indicspeak")
    p.add_argument("--voice", default="Anitha", help="Indic-Speak voice: Anitha or Arun")
    p.add_argument("--wav", nargs="+", type=Path, help="16 kHz mono recordings to use as turns, in order")
    p.add_argument("--out", type=Path, default=Path(tempfile.mkdtemp(prefix="oosai_")), help="where reply audio goes")
    a = p.parse_args()

    from run_asr import indictranscribe  # eval/run_asr.py
    print("Loading Sarvam 30B, Indic-Transcribe and the TTS ...")
    llm, asr = Sarvam(), indictranscribe()
    tts, sr = load_tts(a.tts, a.voice)
    style = (ROOT / "prompts/style_ta.txt").read_text().strip()
    a.out.mkdir(parents=True, exist_ok=True)
    if not a.wav and output_muted():
        print("\n*** The Mac's sound output is muted: unmute it to hear the replies. ***")
    turns = iter(a.wav) if a.wav else None
    pending, history = None, []
    for n in range(1, 10**6):
        try:
            if turns:
                path = next(turns, None)
                if path is None:
                    break
            else:
                path = a.out / f"you_{n:03d}.wav"
                sf.write(path, record(), 16000)
        except KeyboardInterrupt:
            break
        t0 = time.perf_counter()
        # under half a second is a tap, not speech; the recogniser invents words for those
        heard = asr(str(path)).strip() if sf.info(path).duration >= 0.5 else ""
        t_asr = time.perf_counter()
        llm.calls.clear()
        answer = rules.follow_up(heard, pending) if pending and heard else None
        if answer:
            (reply, route), pending = answer, None
        elif heard:
            reply, route, pending = rules.respond(heard, style, llm.generate, history)
        else:
            reply, route = "மன்னிக்கணும், கேக்கல. திரும்ப சொல்லுங்க.", "silence"
        if heard:
            history += [{"role": "user", "content": heard}, {"role": "assistant", "content": reply}]
        t_llm = time.perf_counter()
        audio, t_first = speak(reply, tts, sr, play=not a.wav)
        t_end = time.perf_counter()
        sf.write(a.out / f"oosai_{n:03d}.wav", audio, sr)
        print(f"\n[{n}] heard: {heard}\n    [{route}] {reply}\n"
              f"    speech recognition {t_asr - t0:.1f}s · brain {t_llm - t_asr:.1f}s ({len(llm.calls)} LLM calls) · "
              f"first audio after {(t_first or t_end) - t0:.1f}s · reply audio {len(audio) / sr:.1f}s, "
              f"all done after {t_end - t0:.1f}s -> {a.out / f'oosai_{n:03d}.wav'}")
