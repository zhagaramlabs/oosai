"""Speak the 30 Tamil test questions to MiniCPM-o 4.5 and check what it understands and says back.

  .venv/bin/python eval/ask_minicpmo.py [--limit N] [--speak]

Questions are voiced by the macOS Tamil voice Vani, a stand-in until native recordings exist.
MiniCPM-o replies in text, and with --speak also in speech. IndicConformer transcribes the question
audio (is the input clear?) and the reply audio (does the mouth speak Tamil?). Spoken replies are
saved for listening in eval/results/minicpmo_replies/. On 2026-10-29 the mouth produced 2-3 minutes
of unintelligible audio per Tamil reply (English: normal), so --speak is off by default.
"""
import argparse
import csv
import json
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import soundfile as sf

from minicpmo import Engine
from run_asr import indicconformer
from wer import score

HERE = Path(__file__).parent
SYSTEM = ("You are a helpful voice assistant for Tamil speakers. The user speaks Tamil. Always reply in spoken "
          "Tamil, in Tamil script, in one to three short sentences.")


def tamil_share(s):
    letters = [c for c in s if c.isalpha()]  # Tamil consonants and vowels count; vowel signs don't
    return sum("஀" <= c <= "௿" for c in letters) / len(letters) if letters else 0.0


def to_16k(src, dst):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(dst)], check=True)
    return dst


assert tamil_share("வணக்கம்") == 1.0 and tamil_share("hello") == 0.0 and tamil_share("") == 0.0

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int)
    p.add_argument("--speak", action="store_true", help="also render and check spoken replies (slow)")
    a = p.parse_args()

    rows = list(csv.DictReader(open(HERE / "own_questions.tsv"), delimiter="\t"))[: a.limit]
    asr, eng, work = indicconformer(), Engine(port=19100), Path(tempfile.mkdtemp(prefix="ask_"))
    replies = HERE / "results/minicpmo_replies"
    replies.mkdir(parents=True, exist_ok=True)
    out = HERE / "results/minicpmo_own_questions.jsonl"
    recs = []
    with open(out, "w") as f:
        for r in rows:
            subprocess.run(["say", "-v", "Vani", "-o", str(work / "q.aiff"), r["text"]], check=True)
            q = to_16k(work / "q.aiff", work / f"{r['id']}.wav")
            t = time.perf_counter()
            text, wav = eng.ask(q, SYSTEM, use_tts=a.speak)
            rec = {"id": r["id"], "domain": r["domain"], "question": r["text"], "question_asr": asr(q), "reply": text,
                   "tamil_share": round(tamil_share(text), 2), "reply_asr": "", "reply_sec": 0.0, "sec": round(time.perf_counter() - t, 1)}
            if wav:
                shutil.copy(wav, replies / f"{r['id']}.wav")
                rec["reply_sec"] = round(sf.info(wav).duration, 1)
                rec["reply_asr"] = asr(to_16k(wav, work / "reply16k.wav"))
            recs.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{r['id']} Q: {r['text']}\n     Q heard: {rec['question_asr']}\n      A: {text}\n"
                  + (f"  A audio: {rec['reply_sec']} s, heard: {rec['reply_asr']}\n" if wav else ""))

    with_audio = [x for x in recs if x["reply_sec"]]
    print(f"n={len(recs)}  question CER (Vani audio, IndicConformer)={score([(x['question'], x['question_asr']) for x in recs])[1]:.1%}"
          f"  replies in Tamil script={statistics.mean(x['tamil_share'] for x in recs):.0%}"
          + (f"  reply speech CER={score([(x['reply'], x['reply_asr']) for x in with_audio])[1]:.1%}"
             f" (median {statistics.median(x['reply_sec'] for x in with_audio)} s of audio)" if with_audio else "")
          + f"  median={statistics.median(x['sec'] for x in recs):.0f}s -> {out}")
