"""Uraikal track I (Thirukkural) for Sarvam 30B, served by agent/llm.py. Writes to ../uraikal/hyps and results.

  .venv/bin/python eval/uraikal_classics.py --system sarvam-30b
"""
import argparse
import os
import sys
from pathlib import Path

OOSAI = Path(__file__).resolve().parents[1]
URAIKAL = OOSAI.parent / "uraikal"
sys.path[:0] = [str(URAIKAL / "src"), str(OOSAI)]
from agent.llm import Sarvam  # noqa: E402
from uraikal import classics, cli  # noqa: E402

p = argparse.ArgumentParser()
cli.classics_args(p)
a = p.parse_args()
os.chdir(URAIKAL)  # data/, hyps/ and results/ are relative to it
items = classics.items(*classics.load(a.kural, a.topics), n=a.n)
llm = Sarvam()


def ask(prompt):
    reply = llm.generate([{"role": "user", "content": prompt}], n_predict=a.max_tokens)
    return reply, llm.calls[-1]["tokens_predicted"] >= a.max_tokens


cli.write_classics(a, classics.summarise(items, classics.run(items, ask, Path("hyps") / f"{a.system}.classics.jsonl")))
