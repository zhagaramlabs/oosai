"""Compare two ASR result files: paired bootstrap on the WER/CER gap, and the suspect-clip check.

  .venv/bin/python eval/compare.py results/indicconformer_fleurs_ta.jsonl results/indictranscribe_fleurs_ta.jsonl
"""
import csv
import json
import random
import re
import statistics
import sys
from pathlib import Path

import soundfile as sf

from wer import edits, norm, score

HERE = Path(__file__).parent


def load(p):
    return {r["id"]: r for r in map(json.loads, open(HERE / p))}


def counts(r):
    ref, hyp = norm(r["ref"]), norm(r["hyp"])
    return edits(ref.split(), hyp.split()), len(ref.split()), edits(ref, hyp), len(ref)


A, B = load(sys.argv[1]), load(sys.argv[2])
ids = sorted(A)
assert ids == sorted(B)
ca, cb = {i: counts(A[i]) for i in ids}, {i: counts(B[i]) for i in ids}

# Suspect: both outputs > 1.3x the reference's words, i.e. both heard more speech than the reference holds.
long = lambda r: len(norm(r["hyp"]).split()) > 1.3 * len(norm(r["ref"]).split())
suspect = [i for i in ids if long(A[i]) and long(B[i])]

# Speaking-time check: seconds of audio per reference character, suspect vs the rest.
wav = {r[0]: r[1] for r in csv.reader(open(HERE / "data/fleurs_ta/test.tsv"), delimiter="\t")}
rate = {i: sf.info(wav[i]).duration / cb[i][3] for i in ids}
print(f"suspect clips: {len(suspect)}/{len(ids)}; median s/char suspect="
      f"{statistics.median(rate[i] for i in suspect):.3f} others="
      f"{statistics.median(rate[i] for i in ids if i not in suspect):.3f}")


def gap(sel):
    """Corpus-level (WER_A - WER_B, CER_A - CER_B) over the selected ids."""
    s = [sum(c[i][k] for i in sel) for c in (ca, cb) for k in range(4)]
    return s[0] / s[1] - s[4] / s[5], s[2] / s[3] - s[6] / s[7]


# Sandhi check: drop word-final doubling consonants (க் ச் த் ப்), which change spelling, not words.
sandhi = lambda s: re.sub(r"[கசதப]்(?= |$)", "", norm(s))
for n, R in (("A", A), ("B", B)):
    w, _ = score((sandhi(R[i]["ref"]), sandhi(R[i]["hyp"])) for i in ids)
    print(f"{n} WER ignoring word-final sandhi consonants: {w:.1%}")

random.seed(0)
for name, sel in (("all", ids), ("without suspect", [i for i in ids if i not in suspect])):
    w, c = gap(sel)
    boot = [gap(random.choices(sel, k=len(sel))) for _ in range(1000)]
    ws, cs = sorted(b[0] for b in boot), sorted(b[1] for b in boot)
    print(f"{name} (n={len(sel)}): WER gap A-B {w:+.1%} [95% CI {ws[25]:+.1%}, {ws[974]:+.1%}]  "
          f"CER gap {c:+.1%} [95% CI {cs[25]:+.1%}, {cs[974]:+.1%}]")
