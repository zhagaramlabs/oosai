"""Ask Sarvam 30B (agent/llm.py) the 30 Tamil test questions as text.

  .venv/bin/python eval/ask_sarvam.py [--limit N] [--think] [--prompt prompts/system_ta.txt] [--rules]

Default system prompt: the English one from ask_minicpmo.py. --rules wraps the model in
agent/rules.py (fixed safety replies, a tool-routing call, calculator, reply clean-up); the prompt
file is then only the speaking style, e.g. prompts/style_ta.txt.
"""
import argparse
import collections
import csv
import json
import statistics
import sys
import time
from pathlib import Path

from ask_minicpmo import SYSTEM, tamil_share

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from agent import rules  # noqa: E402
from agent.llm import Sarvam  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int)
    p.add_argument("--think", action="store_true", help="let the model reason before answering (slow)")
    p.add_argument("--prompt", type=Path, help="system prompt file (default: the English prompt of ask_minicpmo.py)")
    p.add_argument("--rules", action="store_true", help="enforce agent/rules.py around the model")
    a = p.parse_args()
    system = a.prompt.read_text().strip() if a.prompt else SYSTEM

    llm = Sarvam()
    rows = list(csv.DictReader(open(HERE / "own_questions.tsv"), delimiter="\t"))[: a.limit]
    tag = (f"_{a.prompt.stem}" if a.prompt else "") + ("_rules" if a.rules else "") + ("_think" if a.think else "")
    out = HERE / f"results/sarvam30b{tag}_own_questions.jsonl"
    recs = []
    with open(out, "w") as f:
        for r in rows:
            def generate(messages, tools=None):
                return llm.generate(messages, tools, think=a.think, n_predict=1500 if a.think else 300)

            llm.calls.clear()
            t = time.perf_counter()
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": r["text"]}]
            reply, route, _ = rules.respond(r["text"], system, generate) if a.rules else (generate(msgs), "llm", None)
            calls = llm.calls
            rec = {"id": r["id"], "domain": r["domain"], "note": r["note"], "question": r["text"], "reply": reply,
                   "route": route, "raw": [c["content"] for c in calls], "tamil_share": round(tamil_share(reply), 2),
                   "llm_calls": len(calls), "tokens": sum(c["tokens_predicted"] for c in calls),
                   "sec": round(time.perf_counter() - t, 1),
                   "prompt_ms": sum(round(c["timings"]["prompt_ms"]) for c in calls),
                   "tok_s": round(calls[-1]["timings"]["predicted_per_second"], 1) if calls else 0}
            recs.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{r['id']} ({r['note']}) [{route}] Q: {r['text']}\n      A: {reply}\n")

    llm_recs = [x for x in recs if x["tokens"]]
    print(f"n={len(recs)}  replies in Tamil script={statistics.mean(x['tamil_share'] for x in recs):.0%}"
          f"  median tokens={statistics.median(x['tokens'] for x in llm_recs)}"
          f"  median prompt={statistics.median(x['prompt_ms'] for x in llm_recs)}ms"
          f"  median speed={statistics.median(x['tok_s'] for x in llm_recs)} tok/s"
          f"  median={statistics.median(x['sec'] for x in recs):.1f}s"
          f"  routes={dict(collections.Counter(x['route'] for x in recs))} -> {out}")
