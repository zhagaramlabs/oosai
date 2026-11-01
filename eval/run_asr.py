"""Transcribe a manifest (id, wav, reference) with one ASR model; print WER, CER and speed.

  .venv/bin/python eval/run_asr.py indicconformer --limit 20
  .venv/bin/python eval/run_asr.py indictranscribe
  .venv/bin/python eval/run_asr.py minicpmo        # needs the engine in ~/ai/voice/s2s
"""
import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

from wer import score

HERE = Path(__file__).parent


def indicconformer():
    from transformers import AutoModel

    m = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)

    def run(path):
        wav, sr = sf.read(path, dtype="float32")
        assert sr == 16000, sr
        return m(torch.from_numpy(wav).unsqueeze(0), "ta", "rnnt")

    return run


def indictranscribe():
    from huggingface_hub import snapshot_download

    d = snapshot_download("bodhan-ai/indic-transcribe-core", ignore_patterns=["nemo/*"])  # skip the 4.6 GB NeMo copy
    sys.path.insert(0, d)
    from indic_transcribe import IndicTranscribe
    from long_form import transcribe_long

    asr = IndicTranscribe.from_pretrained(d, device="mps" if torch.backends.mps.is_available() else "cpu")
    # Single pass refuses audio over 45 s; the shipped chunker handles the few long FLEURS clips.
    return lambda path: transcribe_long(asr, path, lang="ta") if sf.info(path).duration > 30 else asr(path, lang="ta")


def minicpmo():
    from minicpmo import Engine

    eng = Engine()
    return lambda path: eng.ask(path, ASR_PROMPT)[0]


ASR_PROMPT = ("You are a speech recognizer. The user speaks Tamil. Write down exactly what they say, in Tamil script, "
              "and nothing else. Do not translate, answer or comment.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=["indicconformer", "indictranscribe", "minicpmo"])
    p.add_argument("--manifest", default=HERE / "data/fleurs_ta/test.tsv")
    p.add_argument("--limit", type=int)
    a = p.parse_args()

    rows = list(csv.reader(open(a.manifest), delimiter="\t"))[: a.limit]
    run = globals()[a.model]()
    out = HERE / "results" / f"{a.model}_{Path(a.manifest).parent.name}.jsonl"
    out.parent.mkdir(exist_ok=True)
    pairs, secs, audio = [], [], 0.0
    with open(out, "w") as f:
        for i, (uid, wav, ref) in enumerate(rows):
            t = time.perf_counter()
            hyp = run(wav)
            secs.append(time.perf_counter() - t)
            audio += sf.info(wav).duration
            pairs.append((ref, hyp))
            f.write(json.dumps({"id": uid, "ref": ref, "hyp": hyp, "sec": round(secs[-1], 3)}, ensure_ascii=False) + "\n")
            if i < 3:
                print(f"REF {ref}\nHYP {hyp}\n")

    w, c = score(pairs)
    print(f"{a.model}: n={len(rows)} WER={w:.1%} CER={c:.1%} "
          f"median={statistics.median(secs):.2f}s/utt RTF={sum(secs) / audio:.2f} -> {out}")
