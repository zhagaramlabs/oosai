"""Speak eval/tts_sentences.tsv with a Tamil text-to-speech model; check clarity by transcribing it back.

  .venv/bin/python eval/tts_samples.py indicspeak
  .venv/bin/python eval/tts_samples.py indicf5
  KUYIL_CKPT=../kuyil/data/runs/<run>/best.pt .venv/bin/python eval/tts_samples.py kuyil

The 10 sentences are Sarvam 30B replies to the test questions (2026-10-29): Tanglish, quantities, a
date, digits, English in Latin script, a question. Audio goes to eval/results/tts/<model>_<voice>/
for native listeners. IndicConformer's transcript gives a rough clarity score (CER); digits and
Latin-script English inflate it, since the recognizer writes them as Tamil words.
"""
import argparse
import csv
import json
import statistics
import sys
import tempfile
import time
import types
from pathlib import Path

import soundfile as sf
import torch
from huggingface_hub import snapshot_download

from ask_minicpmo import to_16k
from run_asr import indicconformer
from wer import score

HERE = Path(__file__).parent


def indicspeak():
    d = snapshot_download("bodhan-ai/indic-speak")
    sys.path.insert(0, d)
    from inference import TTS

    tts = TTS(d, device="mps" if torch.backends.mps.is_available() else "cpu")
    # The two Tamil library voices (contracted voice artists who consented).
    return {v: (lambda text, v=v: tts(text, speaker=v, seed=1234)) for v in ["Anitha", "Arun"]}, tts.sample_rate


def indicf5():
    trainer = types.ModuleType("f5_tts.model.trainer")  # training only; it needs wandb, not installed
    trainer.Trainer = None
    sys.modules.setdefault("f5_tts.model.trainer", trainer)
    d = snapshot_download("ai4bharat/IndicF5")
    sys.path.insert(0, d)  # the repo ships its own copy of the f5_tts package
    import f5_tts.infer.utils_infer as U
    from f5_tts.model import CFM, DiT
    from safetensors.torch import load_file

    # The repo's AutoModel wrapper breaks under transformers 5 (it builds models on the meta device),
    # so this repeats what the wrapper does: same DiT config, weights from model.safetensors.
    sd = load_file(d + "/model.safetensors")
    part = lambda prefix: {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    vocab, n_vocab = U.get_tokenizer(d + "/checkpoints/vocab.txt", "custom")
    model = CFM(transformer=DiT(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4,
                                text_num_embeds=n_vocab, mel_dim=U.n_mel_channels),
                mel_spec_kwargs=dict(n_fft=U.n_fft, hop_length=U.hop_length, win_length=U.win_length,
                                     n_mel_channels=U.n_mel_channels, target_sample_rate=U.target_sample_rate,
                                     mel_spec_type="vocos"),
                odeint_kwargs=dict(method=U.ode_method), vocab_char_map=vocab)
    model.load_state_dict(part("ema_model._orig_mod."))
    vocoder = U.load_vocoder("vocos", device="cpu")
    vocoder.load_state_dict(part("vocoder._orig_mod."))
    model, vocoder = model.to(U.device).eval(), vocoder.to(U.device).eval()
    # IndicF5 has no default voice. This is the model card's own example prompt (a Punjabi speaker);
    # a Tamil voice needs a clip from a speaker who gave written consent (README §11 Q4).
    ref = U.preprocess_ref_audio_text(
        d + "/prompts/PAN_F_HAPPY_00001.wav",
        "ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ  ਹਨ।")

    def run(text):
        return U.infer_process(*ref, text, model, vocoder, mel_spec_type="vocos", device=U.device)[0]

    return {"PAN_F_example": run}, U.target_sample_rate


def kuyil():
    """Our PiVo-style model (the kuyil repo, checked out next to this one); the checkpoint comes from KUYIL_CKPT."""
    import os
    sys.path.insert(0, str(HERE.resolve().parent.parent / "kuyil"))
    from synth import load

    speak, voices = load(os.environ["KUYIL_CKPT"])
    return {v: (lambda text, v=v: speak(text, v)) for v in voices}, 16000


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=["indicspeak", "indicf5", "kuyil"])
    a = p.parse_args()

    rows = list(csv.DictReader(open(HERE / "tts_sentences.tsv"), delimiter="\t"))
    voices, sr = globals()[a.model]()
    asr, work = indicconformer(), Path(tempfile.mkdtemp(prefix="tts_"))
    out = HERE / f"results/tts_{a.model}.jsonl"
    recs = []
    with open(out, "w") as f:
        for voice, speak in voices.items():
            d = HERE / f"results/tts/{a.model}_{voice}"
            d.mkdir(parents=True, exist_ok=True)
            for r in rows:
                t = time.perf_counter()
                wav = speak(r["text"])
                sec = time.perf_counter() - t
                sf.write(d / f"{r['id']}.wav", wav, sr)
                heard = asr(to_16k(d / f"{r['id']}.wav", work / "16k.wav"))
                rec = {"model": a.model, "voice": voice, "id": r["id"], "text": r["text"], "heard": heard,
                       "audio_sec": round(len(wav) / sr, 1), "sec": round(sec, 1), "rtf": round(sec * sr / len(wav), 2)}
                recs.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"{voice} {r['id']} {rec['audio_sec']} s audio in {rec['sec']} s\n  said:  {r['text']}\n  heard: {heard}\n")

    for voice in voices:
        v = [x for x in recs if x["voice"] == voice]
        w, c = score([(x["text"], x["heard"]) for x in v])
        print(f"{a.model} {voice}: n={len(v)} heard-back WER={w:.1%} CER={c:.1%} "
              f"median RTF={statistics.median(x['rtf'] for x in v)} -> {out}")
