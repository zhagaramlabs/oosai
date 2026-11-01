"""FLEURS-Tamil test parquet -> wav files + manifest (id, wav path, reference)."""
import csv
from pathlib import Path

import pyarrow.parquet as pq

d = Path(__file__).parent / "data" / "fleurs_ta"
(d / "wav").mkdir(exist_ok=True)
rows = pq.read_table(d / "test.parquet", columns=["id", "audio", "transcription"]).to_pylist()
with open(d / "test.tsv", "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    for i, r in enumerate(rows):
        # FLEURS repeats sentence ids across speakers, so number the rows.
        p = d / "wav" / f"{i:03d}_{r['id']}.wav"
        p.write_bytes(r["audio"]["bytes"])
        w.writerow([f"{i:03d}_{r['id']}", p, r["transcription"]])
print(len(rows), "utterances ->", d / "test.tsv")
