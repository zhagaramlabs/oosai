"""Write Uraikal hypotheses ({"id", "hyp"} JSONL) for a Uraikal manifest with one of run_asr.py's models.

  .venv/bin/python eval/uraikal_hyps.py indictranscribe ../uraikal/data/a1_kathbath_*.jsonl \
      --out ../uraikal/hyps/indic-transcribe-core.jsonl

Re-running with the same --out resumes. Clips are decoded to 16 kHz mono WAV first, since the loaders read WAV.
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import run_asr

p = argparse.ArgumentParser()
p.add_argument("model", choices=["indicconformer", "indictranscribe"])
p.add_argument("manifest", nargs="+")
p.add_argument("--out", required=True)
a = p.parse_args()

rows = [json.loads(line) for m in a.manifest for line in open(m, encoding="utf-8")]
out = Path(a.out)
done = {json.loads(line)["id"] for line in open(out, encoding="utf-8")} if out.exists() else set()
todo = [r for r in rows if r["id"] not in done]
print(f"{len(todo)} to transcribe, {len(done)} already done", flush=True)
run = getattr(run_asr, a.model)()
with tempfile.TemporaryDirectory() as tmp, open(out, "a", encoding="utf-8") as f:
    wav = str(Path(tmp) / "clip.wav")
    for i, r in enumerate(todo, 1):
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", r["audio"], "-ac", "1", "-ar", "16000", wav],
                       check=True)
        f.write(json.dumps({"id": r["id"], "hyp": run(wav)}, ensure_ascii=False) + "\n")
        f.flush()
        if i % 100 == 0:
            print(f"{i}/{len(todo)}", flush=True)
