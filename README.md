# Oosai — a Tamil speech-to-speech voice agent

Updated: 2026-10-31 · Status: Phase 0 done except native ratings; Phase 1 step 5 done (a
push-to-talk Tamil voice loop runs on the Mac, with the safety rules in code; too slow so far)
· Git repo initialized, nothing committed yet

"Oosai" (ஓசை) means "sound" in Tamil. (Named Osai, briefly Oli, then Oosai on 2026-10-29. Project folder: `~/ai/zhagaramlabs/oosai` (moved from `~/ai/Tamil/Oli` on 2026-10-29). `.venv/bin` scripts and `eval/data/fleurs_ta/test.tsv` store absolute paths: after any move, rewrite the old path in both with `sed`, or recreate `.venv` and rerun `eval/prep_fleurs.py`.)

This document is the hand-off for continuing the project in a new session. It records what
was researched, decided and ruled out, so none of it has to be rediscovered. Items marked
**unverified** came from secondary sources or haven't been tested.

---

## 1. Goal

Build a conversational voice agent that **listens and speaks natural Tamil**, which you own
and can tune with your own audio and data: your instructions, your voice, and your tools.

**Done for phase 1 means:** a person speaks Tamil into the Mac's microphone and hears a correct,
natural-sounding spoken Tamil reply, with first audio in about 2 seconds or less. Accuracy is
measured on a fixed Tamil test set (see §8).

**Longer term:** possibly a single end-to-end Tamil speech-to-speech model ("Oosai model"), only
if the pipeline proves the need and real Tamil conversation data has been collected (§6).

---

## 2. Background: where this came from

This project grew out of an English speech-to-speech exploration in `~/ai/voice/s2s`:
- **NemotronLabs VoiceChat** was considered. It's English only and needs an 80 GB NVIDIA GPU.
- **MiniCPM-o 4.5** was set up and working locally on the Mac, with full-duplex speech and voice
  cloning.

The user wants:
- their own speech-to-speech agent, built on existing open models instead of trained from scratch;
- custom voices, good reasoning, and tool calls;
- a budget of a few hundred dollars for now;
- Tamil as the language for this project.

---

## 3. Key finding: there is no open speech-to-speech model built for Tamil

Checked in October 2026:

| Model | Tamil speech in | Tamil speech out | Open weights | Notes |
|---|---|---|---|---|
| MiniCPM-o 4.5 (OpenBMB, 9B) | ✗ | ✗ | Apache 2.0 | **Measured on this Mac (§7 step 2b):** asked to transcribe FLEURS-Tamil, 103% of characters wrong; 1 of 30 spoken Tamil questions answered usefully; Tamil speech output unintelligible. Conversation is English and Chinese only. |
| Qwen3-Omni (Alibaba) | ✗ | ✗ | yes | 19 input and 10 output speech languages; Tamil isn't in either list. |
| Qwen3.5-Omni (Alibaba) | ? | ✗ | hosted | 29 output languages (includes Hindi and Urdu, not Tamil). |
| NemotronLabs VoiceChat 11B (NVIDIA) | ✗ | ✗ | OpenMDW | English only. |
| PersonaPlex 7B (NVIDIA), Moshi (Kyutai) | ✗ | ✗ | yes | English (Moshi also French). |
| **Inya VoiceOS 5B (Gnani.ai)** | ? | ? | **no** | Voice-to-voice model for "15+ Indian languages", announced February 2026 under the IndiaAI Mission. Research preview with no public weights and no published language list. **Revisit if it's released openly.** |
| Sarvam speech models (Saaras, Saarika, Bulbul) | yes | yes | API only (**unverified**) | Separate speech recognition and text-to-speech services, not a single model. Useful as a quality baseline. |
| Bodhan AI Indic-Transcribe-core / Indic-Speak (with AI4Bharat) | yes | yes | Indic Open Model License | Released Aug 2026. Separate speech recognition and text-to-speech models, not a single model. AI4Bharat's own cards for IndicConformer and IndicF5 now recommend these instead (§4). |

**Conclusion:** to get a Tamil agent you control, build it from **Tamil-trained open parts** (§4).
A single Tamil model is possible, but it's a large training project (§6).

---

## 4. Recommended architecture (phase 1): a Tamil pipeline

```
mic → [turn detection] → [Tamil speech recognition] → [LLM brain + tools] → [Tamil text-to-speech] → speaker
        Silero VAD /         IndicConformer 600M        Sarvam 30B           IndicF5
        Smart Turn           (AI4Bharat)                (Sarvam AI)          (AI4Bharat)
                       ↘________ orchestrated by Pipecat (or a simple push-to-talk loop first) ________↗
```

| Part | Pick | Why | License | Status |
|---|---|---|---|---|
| Speech recognition | **AI4Bharat IndicConformer 600M multilingual** (`ai4bharat/indic-conformer-600m-multilingual`) | Trained for all 22 official Indian languages; a NeMo model, so it can be fine-tuned | MIT | **Measured on this Mac, FLEURS-Tamil test (591 clips): 27.1% of words / 11.1% of characters wrong; 0.91 s median per clip on CPU** (§7 step 2). |
| Speech recognition (contender, likely better) | **Bodhan AI Indic-Transcribe-core** (`bodhan-ai/indic-transcribe-core`, 1.2B, Canary architecture) | Successor recommended on IndicConformer's own card. Handles code-mixed speech (Tanglish). Plain transformers/torch code, no NeMo. Their "Voice of India" benchmark, Tamil: **9.0%** words wrong vs IndicConformer 16.2% and Saaras V3 9.1% (vendor's numbers, spelling-variant-tolerant metric) | Indic Open Model License (see below) | **Measured on this Mac, FLEURS-Tamil: 26.1% of words / 12.1% of characters wrong; 0.88 s median per clip on the Mac GPU.** The vendor's large lead didn't show on FLEURS (§7 step 2) |
| Speech recognition (alternative) | `osmapi/tamil-asr-qwen3` (Tamil-only, Qwen3-based) | Same third-party test: about 25% of words wrong | check card | unverified |
| Brain (LLM) | **Sarvam 30B** (`sarvamai/sarvam-30b`) | Indian lab, built for Indian languages. Mixture of experts: 32B total parameters, **2.4B active per word**, so fast. 65K context. Reasoning on/off modes | Apache 2.0 | **Runs on the Mac** (official Q4_K_M GGUF, 20 GB, mainline llama.cpp): 26.5 tokens/s, 1.4 s per reply. Natural spoken Tamil. Alone, about 11 of 30 answers wrong or unsafe, with or without a rules prompt; with the rules in code (`agent/rules.py`), about 6 wrong and no unsafe ones (§7 step 4). **Tool calling works**, but only with a bare system prompt, so routing is a separate call |
| Brain (alternatives) | Sarvam 105B (10B active, 128K context); Gemma / Qwen / Llama models with Tamil | Fallbacks if 30B is weak in Tamil | varies | not evaluated |
| Text-to-speech | **AI4Bharat IndicF5** (`ai4bharat/IndicF5`) | Tamil among 11 Indian languages; **clones a voice from a short sample** plus its transcript; trained on 1,417 h of Indian speech; 24 kHz output | MIT (official card, checked 2026-10-28) | **Tried on the Mac:** clear plain Tamil (4.0% of characters wrong when transcribed back) but garbles digits and Latin-script English; 3.6× slower than real time (§7 step 3) |
| Text-to-speech (contender) | **Bodhan AI Indic-Speak** (`bodhan-ai/indic-speak`) | Successor recommended on IndicF5's card. Tamil is "production" tier; reads English words and numbers inside Tamil sentences; several consented library voices. **No voice cloning** | Indic Open Model License | **Tried on the Mac, current pick:** Tamil voices Anitha and Arun; 2.9–3.5% of characters wrong when transcribed back, reads digits and English right; one runaway clip; 5–9× slower than real time (§7 step 3) |
| Text-to-speech (alternative) | AI4Bharat Indic Parler-TTS | Voice is described in words; no cloning | Apache 2.0, gated (sources conflict) | unverified |
| Turn detection | Silero VAD, or Pipecat Smart Turn | Silero responds faster; Smart Turn handles pauses better | open | — |
| Orchestration | Pipecat (or LiveKit Agents) | Streaming, interruptions, tool calls | open | — |

**Licenses ruled out for commercial use:**
- Dhee-Indic-F5 and F5-TTS weights: CC-BY-NC.
- Fish Speech and OmniVoice weights: non-commercial.
- Community Tamil fine-tunes of IndicF5 with no stated license.

**Indic Open Model License (Bodhan AI models), checked 2026-10-28:** allows commercial use
when self-hosted, with conditions: credit "Built with <model> from Bodhan AI / AI4Bharat";
**share-alike** (anything derived from the model, such as a fine-tune, must keep this license);
written sign-off from Bodhan AI before offering it as an API others call; no robocalls or
auto-dialers, no voice clones without consent, no romance-companion products; a separate license
above 500M monthly users or $250M revenue. **Consequence for §6:** a future Oosai model built from
these weights would inherit this license. Built from MIT parts (IndicConformer, IndicF5) it could
be released under any license. Full text: https://github.com/Bodhan-AI/bodhan-model-info

**Trade-offs of a pipeline compared with a single model:**
- **Lost:** no talking over each other (it's turn-based), about a one-second delay, and tone of
  voice is lost when speech becomes text.
- **Gained:** every part is already Tamil-trained, any part can be swapped later, and each part
  is cheap to tune.

---

## 5. Tamil-specific things to get right

1. **Spoken and written Tamil are very different.** LLMs tend to write formal literary Tamil,
   which sounds stilted read aloud. Prompt the brain for **colloquial spoken Tamil**, and test it
   with native listeners.
2. **People mix in English constantly** ("Tanglish": English words in Tamil sentences, or Tamil
   written in Latin letters). The speech recognition must handle it, and the text-to-speech must
   pronounce English words sensibly. Include mixed-language utterances in the test set.
3. **Dialects:** Chennai, Kongu, Madurai/southern, Sri Lankan (Jaffna) and others. Decide the
   target dialect(s) and record test audio for each.
4. **Numbers, dates, money and abbreviations** must be written out as Tamil words before
   text-to-speech ("₹1,250" → spoken form).
5. **Formal or familiar address** (நீங்கள் / நீ): set it in the system prompt to match the use case.
6. **Phone audio** (8 kHz, compressed) makes speech recognition noticeably worse. If the agent
   will take phone calls, test with phone-quality audio.

---

## 6. Later option: train a single Oosai speech-to-speech model

Only worth starting after the pipeline works and real conversation data exists. This is the
"ears + brain + mouth" design:

- **Ears:** a speech encoder retrained or fine-tuned on Tamil speech. IndicConformer's encoder
  is a candidate.
- **Brain:** a Tamil-capable LLM such as Sarvam 30B. Swapping the brain means redoing the
  ear-to-brain connection training.
- **Mouth:** a speech generator trained on Tamil. Starting from an existing model such as
  MiniCPM-o's means retraining its mouth, which has never produced Tamil, on roughly 1,000+ hours
  of clean Tamil speech. MiniCPM-o is no shortcut: measured on Tamil, its ear and brain fail too
  (§7 step 2b), so all three parts would need retraining.
- **Data** (licenses checked 2026-10-29 from Hugging Face cards and papers; recheck before
  training):

  | Dataset | License | Commercial | Tamil size | Use for |
  |---|---|---|---|---|
  | IndicVoices-R (`ai4bharat/indicvoices_r`) | CC-BY-4.0 | yes | 99.5 h, 1,084 speakers | mouth (TTS-quality) |
  | Rasa (`ai4bharat/Rasa`) | CC-BY-4.0 | yes | 56 h (F 29.9, M 26.2) | expressive mouth |
  | IndicTTS, IIT Madras (mirror `SPRINGLab/IndicTTS_Tamil`) | own license: royalty-free, sublicensable; mirrors say CC-BY-4.0 | probably; full text **unverified** | 20.3 h, 2 speakers, 48 kHz | clean mouth |
  | IndicVoices (`ai4bharat/IndicVoices`) | CC-BY-4.0 | yes | not checked (gated); all languages 23.7k h, 11.2k transcribed (Dec 2025) | ears |
  | Kathbath (`ai4bharat/Kathbath`) | CC-BY-4.0 | yes | not checked (gated); 1,684 h over 12 languages | ears, eval |
  | Vaani (`ARTPARK-IISc/Vaani`) | CC-BY-4.0 | yes | 848 h audio, 24 h transcribed | unlabelled pre-training |
  | Shrutilipi (`ai4bharat/Shrutilipi`) | CC-BY-4.0 | **risky**: audio is All India Radio news (Prasar Bharati copyright) | ~790 h | ears, only after clearing rights |
  | Mozilla Common Voice (ta) | CC0 | yes | not checked | ears; since Oct 2025 only on Mozilla Data Collective, not HF |
  | OpenSLR 65 | CC-BY-SA 4.0 | share-alike | small | skip |
  | Microsoft Speech Corpus (Indian languages) | research only | **no** | — | ruled out |
  | FLEURS ta_in | CC-BY-4.0 | yes | — | our test set; never train on it |

  - **Ears:** IndicVoices was used to build AI4Bharat's IndicASR, so IndicConformer has likely
    already seen most of these sets. Gains need our own domain Tamil (support, booking, health).
  - **Mouth:** open TTS-quality Tamil totals only ~175 h (IndicVoices-R + Rasa + IndicTTS),
    against the 1,000+ h option B needs. The rest has to be synthetic (IndicF5) or our own
    consented recordings.
  - **Full duplex:** no open Tamil conversation corpus near 10k h; Vaani is picture descriptions,
    not dialogue.
  - Also: the Vakyansh/EkStep Tamil models (one was fine-tuned on 900 h), and synthetic
    dialogues (an LLM writes Tamil conversations and IndicF5 voices them in many voices).
  - Local copy: IndicTTS Tamil goes in `data/indictts_ta/` (gitignored, ~8.4 GB).
- **Cost estimate (2026-10-29),** at about $2.50 per H100-hour (§9). These are my extrapolations
  from published runs, not quotes, and they include roughly 2–3× for failed and tuning runs:

  | Option | What it does | Data it needs | H100-hours | GPU cost |
  |---|---|---|---|---|
  | **A. Speech-in model** (ears + brain) | Hears Tamil directly and keeps tone of voice; still replies through a separate TTS | ~1–2k h of transcribed Tamil (open sets above) + ~50k synthetic spoken questions | 200–500 | **$500–1,500** |
  | **B. Turn-based speech-to-speech** (ears + brain + mouth; LLaMA-Omni design) | Tamil speech in, Tamil speech out, one model, takes turns | ~200k synthetic Tamil spoken question/answer pairs (~1,000 h of audio, made with the LLM + IndicF5) + 1,000+ h of clean Tamil speech for the mouth | 1,500–4,000 | **$4k–10k** |
  | **C. Full-duplex** (Moshi / J-Moshi design) | Listens while talking, interruptions, back-channels ("ம்ம்") | 10k–60k h of Tamil conversation, ideally two-channel, plus a few hundred hours of clean dialogue | 3,000–5,000 | **$10k–20k compute; data is the real blocker** |

  - **Anchors:** LLaMA-Omni (B's design) trained on 200k spoken instruction pairs in about 3 days
    on 4 GPUs (arXiv 2409.06666). J-Moshi (C, Japanese Moshi) pre-trained on 60k h of Japanese
    dialogue on 128 V100s for about 36 h (≈4,600 V100-hours, roughly 1,200–1,500 H100-hours), then
    fine-tuned on 344 h of stereo dialogue (arXiv 2506.02979). Indic Parler-TTS training used 32
    H100s for 1.5 days (arXiv 2505.18609).
  - **Not in the GPU cost:** engineer time (3–6 months for B), native-speaker evaluation, and
    for C, acquiring tens of thousands of hours of consented Tamil conversation. No open Tamil
    corpus of that size is known, and buying or recording it likely costs more than the compute.
  - **Fit with the current budget (a few hundred dollars):** only a first, small run of A.
  - **Where one model actually beats the pipeline:** natural turn-taking, tone of voice, and
    latency in casual chat. For support, booking and healthcare the pipeline is better: the text
    can be checked, tool calls are reliable, and each part can be fixed separately.
- **Cheap route there:** once the pipeline is live, record real Tamil conversations **with
  consent**. That is exactly the training data this phase needs.

---

## 7. Plan

**Phase 0: baseline and test set (on the Mac, free)**
1. `git init` this folder. Create `eval/` with a fixed Tamil test set:
   - FLEURS-Tamil test split (591 utterances, 2.13 h, 16 kHz mono) for speech recognition
     accuracy. **Done:** `eval/prep_fleurs.py` writes `eval/data/fleurs_ta/test.tsv`;
     `eval/wer.py` scores; `eval/run_asr.py indicconformer|indictranscribe` runs a model;
   - 20–30 of your own recorded questions (**30 drafted** in `eval/own_questions.tsv`, not yet recorded) covering colloquial speech, mixed English, numbers and
     your use case.
2. Run IndicConformer **and Indic-Transcribe-core** on the test set and record the share of
   words and characters wrong, plus seconds per utterance. **Done 2026-10-29**
   (`eval/results/*.jsonl` hold every reference and hypothesis):

   | Model | Words wrong | Characters wrong | Words / chars wrong without 22 suspect clips | Median s per clip |
   |---|---|---|---|---|
   | IndicConformer (RNNT, CPU) | 27.1% | 11.1% | 23.1% / **6.9%** | 0.91 |
   | Indic-Transcribe-core (Mac GPU) | **26.1%** | 12.1% | **21.3%** / 7.2% | 0.88 |

   - **A trade-off, not a winner** (`eval/compare.py`, paired bootstrap, 1,000 resamples):
     without the suspect clips, Indic-Transcribe makes significantly fewer word errors (1.8
     points, 95% CI 0.8–2.7); on all clips, IndicConformer makes significantly fewer character
     errors (1.0 point, CI 0.2–2.0), an advantage that comes from the suspect clips. Per clip
     (clean set): Indic-Transcribe better on 233, IndicConformer on 125, tie on 211. Both run
     about 12× faster than real time, so speed won't decide it.
   - **Sandhi:** ignoring word-final க் ச் த் ப் lowers word error by 2.6 points for both models
     (27.1→24.5%, 26.1→23.5%), about one word error in ten.
   - **Suspect clips confirmed by timing:** they hold 0.175 s of audio per reference character
     vs 0.080 s for the rest (2.2×), so the extra speech is really there. Still not listened to.
   - **Suspect clips:** in 22 clips both models' output is 30%+ longer than the reference, e.g.
     clip `002_1666` where both transcribe the sentence twice. The speaker probably restarted and
     FLEURS's reference omits it. This is a heuristic, not checked by listening.
   - **Read Tamil word error rates look high** because Tamil words are long and agglutinative:
     one sandhi or spelling difference (ஆலோசனையை பெற / ஆலோசனையைப் பெற, ph / பிஹெச்) makes the whole
     word wrong. Character error is the better guide.
   - FLEURS is read, formal Tamil. It doesn't test Indic-Transcribe's claimed strengths
     (code-mixed Tanglish, noisy and conversational speech), so **the choice is made on the own
     recorded set** (rest of step 1).

   2b. **MiniCPM-o 4.5 on Tamil, done 2026-10-29** (Q8_0 on the llama.cpp-omni engine in
   `~/ai/voice/s2s`, wrapped by `eval/minicpmo.py`). **Ruled out as a starting point:** ear, brain
   and mouth would all need retraining.
   - **Ear:** `eval/run_asr.py minicpmo` (prompted to transcribe): 140.1% of words / 103.1% of
     characters wrong on all 591 clips, 19 s median per clip (slower than real time). 49 replies
     empty, 36 mostly English, 52 over twice the reference length (it answers or keeps talking
     instead of transcribing). On the 301 clips with Tamil output of plausible length: 85.4% /
     53.0%, against 11–12% of characters for the two Tamil models.
   - **Brain:** `eval/ask_minicpmo.py` speaks the 30 questions with the Vani voice (IndicConformer
     hears them with 3.7% of characters wrong, so the input is clear). 90% of replies are in Tamil
     script, but by my reading (not a native review) 1 is useful, about 8 partly on topic and 21
     off-topic, empty or looping. The chest-pain case (hlt05) got no emergency advice. Results in
     `eval/results/minicpmo_own_questions.jsonl` (rebuilt from the run log; per-question seconds
     lost).
   - **Mouth:** `eval/ask_minicpmo.py --speak`: each Tamil reply came out as 2–3 minutes of
     unintelligible audio; the English control was a normal 2.8 s reply.
3. Generate 10 sample sentences with IndicF5 (default voice and a cloned voice) **and
   Indic-Speak** (library voices), and have a native speaker rate them. **Done 2026-10-29, first
   pass, native rating still to do:** `eval/tts_samples.py indicspeak|indicf5` reads the 10
   sentences in `eval/tts_sentences.tsv` (10 of Sarvam's replies: Tanglish, quantities, a date,
   the digits "35.75", English in Latin script, a question), saves them to
   `eval/results/tts/<model>_<voice>/` and has IndicConformer transcribe them back:

   | Voice | Characters wrong, 8 plain Tamil sentences | "35.75" | English in Latin script | Seconds per second of audio (M1 Max) |
   |---|---|---|---|---|
   | Indic-Speak Anitha (F) | **2.9%** | read as English "thirty five point seven five" | read right | 9.3 (partly under memory pressure) |
   | Indic-Speak Arun (M) | 13.2%; 3.5% without cht05 | same | read right | 5.1 |
   | IndicF5, model card's Punjabi example voice | 4.0% | garbled | recogniser heard nothing (7.7 s of sound) | 3.6 |

   - **Indic-Speak** handles the mixed sentences well: digits and English words come out right
     (the 70%+ error on those two sentences is only the recogniser writing English in Tamil
     letters). **Failure:** Arun's cht05 said the first of four sentences, then 27 s of silence up
     to the length cap. Speaking one sentence at a time should avoid this (untested).
   - **IndicF5** has no default voice and no Tamil example prompt, so it used the card's Punjabi
     one; a Tamil voice needs a reference clip from a speaker with written consent (§11 Q4). The
     cloned-voice part of this step waits for that. Digits must be written out as Tamil words
     first (§5 item 4).
   - **Neither is real time on the Mac** (3.6–9× slower). Indic-Speak runs through transformers
     on the Mac GPU; it is a Llama-3.2-3B, so a llama.cpp or MLX port should be much faster
     (untested).
   - **Setup:** IndicF5's AutoModel wrapper breaks under transformers 5 (it builds models on the
     meta device), so the script loads its F5 code and weights directly; its training module is
     stubbed to avoid installing wandb. Needs `torchcodec` for torchaudio 2.11.
4. Ask Sarvam 30B 20 Tamil questions in text, and rate how natural and correct the spoken-style
   answers are. **Done 2026-10-29, first pass:** `eval/ask_sarvam.py` asks the 30 questions in
   text (official Q4_K_M GGUF on mainline llama.cpp, same system prompt as the MiniCPM-o test, no
   tools or healthcare rules yet, thinking off, temperature 0). Replies in
   `eval/results/sarvam30b_own_questions.jsonl`.
   - **Language is the strength:** 97% Tamil script, natural colloquial spoken Tamil and Tanglish
     (பண்ணுங்க, செக் பண்ணுங்க, "ரௌட்டர்-அ ஆன் பண்ணி"). Median 27 tokens, so replies are short.
   - **Correctness is weak** (my reading, not a native review): about 13 good, 6 that only
     acknowledge or ask a follow-up (fair without tools), and 11 wrong or unsafe:
     - wrong facts: 15% of 300 = "35.75" (edu02); an invented first Thirukkural (edu03);
     - mis-heard details: alarm at 6 instead of 6:30 (ast01); asks for a date it was given (bkg03);
     - made-up tool results: rain forecast (ast03), a free salon slot (bkg05);
     - agrees to unblock a card without checking identity (sup04);
     - repeats the question until the 300-token cap (sup02); no joke (cht02); echoes instead of
       acting (ast02);
     - **unsafe:** paracetamol "4 to 6 times a day" (hlt04, a dose, and wrong). The chest-pain
       case (hlt05) says go to hospital at once but doesn't mention 108.
   - **Speed, quiet Mac (re-measured 2026-10-29, after stopping the English MiniCPM-o demo):**
     26.5 tokens/s, 0.3 s to read the prompt, 1.4 s median per whole reply. With streaming, the
     first sentence would be ready in well under a second, inside the 2 s budget. (The first run,
     with the demo still running and 28 GB of swap in use, gave 4.3 s.) At temperature 0 the rerun
     gave the same 30 replies word for word.
   - **Thinking switch:** the GGUF's template adds `<|nothink|>`, but the model still writes
     `<think>`; pre-filling an empty `<think></think>` turns it off (the script does this).
     `--think` lets it reason (slow); untested whether that fixes the arithmetic and the kural.
   - **With the Tamil system prompt** (`prompts/system_ta.txt`, first draft of Phase 2 step 7:
     persona, spoken register, polite நீங்க, numbers in words, don't invent, keep the user's
     details, confirm before acting, identity check, hand-off, the healthcare rules; not yet
     reviewed by a native speaker): `eval/ask_sarvam.py --prompt prompts/system_ta.txt`, results in
     `eval/results/sarvam30b_system_ta_own_questions.jsonl`. **Same overall score: about 13 good,
     6 partial, 11 wrong** (my reading), but different items:
     - **Fixed:** chest pain now says "உடனே நூத்தியெட்டுக்கு கால் பண்ணுங்க" in the first sentence
       (hlt05); alarm at 6:30 (ast01); no longer asks for a date it was given (bkg03); no longer
       invents a salon slot (bkg05); asks for the tracking number (sup03).
     - **Rules ignored:** still gives paracetamol "four times a day" (hlt04); still invents the
       rain forecast (ast03); says the booking is cancelled without confirming (bkg04); no
       identity check for the card (sup04); invents reasons for the bill (sup02); invented
       Thirukkural again (edu03); numbered lists in 3 replies.
     - **New failures:** loops until the 300-token cap on the arithmetic and the joke (edu02,
       cht02); echoes the user (cht01, ast02); wrong past-tense examples (edu04). Replies got
       longer: median 80 tokens against 27, so 2.6 s per reply.
     - **Conclusion:** prompting fixes some cases, but Sarvam 30B with thinking off doesn't follow
       most rules reliably, so the rules moved into code (below).
   - **Same rules in English** (`prompts/system_en.txt`, prompt only): same score, about 13 good,
     6 partial, 11 wrong. Still gives the paracetamol dose, invents the weather, skips the
     identity check; the chest-pain reply says நூத்தியெட்டுக்கு and then repeats it until the cap.
     The prompt's language isn't the problem.
   - **Rules in code** (`agent/rules.py`, Phase 2 step 8; `eval/ask_sarvam.py --prompt
     prompts/style_ta.txt --rules`, results in `eval/results/sarvam30b_style_ta_rules_own_questions.jsonl`):
     - **Before the LLM:** keyword rules give fixed replies for emergencies (108), medicine-dose
       questions (ask a doctor or pharmacist, follow the label) and requests for a person.
     - **Routing call:** a minimal English system prompt with 10 tools (alarm, reminder, list,
       call, book, cancel, account, live lookup, hand-off, and `answer` for everything else); the
       model must call exactly one. Code decides what each does: actions get a spoken read-back
       ending "சரியா?" and nothing runs before a yes; account requests ask for an OTP first; live
       lookups say they aren't connected yet. Routing picked the expected tool for 27 of 30.
     - **After the LLM:** replies are cut to three distinct sentences, without lists, markdown or
       repeated phrases; a dose that slips into a reply is replaced by the fixed refusal.
     - **Why two calls:** Sarvam stops calling tools as soon as the system prompt asks for spoken
       Tamil (0 tool calls with either the Tamil or a short English style prompt), and a required
       read-back argument cut tool calls from 3 of 3 to 1 of 3. With a bare prompt it calls tools
       reliably, even for Tamil questions with thinking off.
     - **Result: about 20 good, 4 partial, 6 wrong** (my reading). All safety cases pass: 108,
       no dose, OTP before the card, hand-off. No made-up weather, parcel status or free slots,
       and every action asks before acting.
     - **Still wrong:** 15% of 300 (edu02) and the first Thirukkural (edu03); a garbled answer
       about the bill (sup02); the sugar-level answer doesn't refer to a doctor (hlt02). Two
       read-backs repeat details the router invented: a date "2025-11-17" (bkg03) and a booking
       "12345" (bkg04). That's a safe failure (the user hears it and says no), but the router's
       arguments need checking. The hospital appointment was routed as a lookup (hlt03).
     - **Speed:** median 2.3 s per turn (two LLM calls for ordinary answers, one for tools, none
       for the fixed rules), 32.7 tokens/s.
     - **Caveats:** keyword rules only catch the phrasings listed in `agent/rules.py`; the fixed
       Tamil replies and both prompts still need a native review. The rules were tuned on these
       30 questions, so a fresh question set must confirm the gain.
   - **Calculator and read-back check (2026-10-29, same command, results overwrite the file):**
     - **Router arguments are unusable as they are:** asked to copy the user's exact words, the
       router still wrote English ("train ticket", "Madurai"), reordered words, and turned "the
       15th of November" into "November 17th" again. So read-backs are now written from the
       user's own words only (`CONFIRM_NOTE` no longer passes the arguments). A read-back with
       a number the user didn't say is replaced by an echo of their words ("நீங்க சொன்னது, ...
       இதைப் பண்ணட்டுமா?"). A yes runs nothing yet (tools are stubs); the arguments are kept,
       marked unverified, for real tools to check.
     - **Calculator:** Sarvam can't do sums from Tamil number words (1 of 4 right without
       thinking; with thinking 20+ s and it read முன்னூறுல பதினஞ்சு as 115%). `uraikal.numbers`
       turns number words into digits (and back, for the TTS; round-trip tested 0–120,000 and
       up to crores). With digits, a short prompt gets the sum right for 7 of 9 test questions.
       Code evaluates it (only + − × ÷) and says the sum as well as the answer, so a wrong sum
       can be heard: "முன்னூறு பெருக்கல் பதினஞ்சு வகுத்தல் நூறு, அதாவது நாப்பத்தி அஞ்சு".
     - **Result: about 19 good, 7 partial, 4 wrong** (was 20 / 4 / 6). Fixed: 15% of 300 = 45
       (edu02); no invented date or booking number in read-backs (bkg03, bkg04).
       **Regression:** adding the calculator tool changed routing, so the parcel question (sup03)
       now goes to `answer`, which invents a tracking result. Routing is fragile; a fresh
       question set and a sturdier router (thinking for unclear cases, or a small classifier)
       are needed. Still wrong: sup02, edu03, hlt02 as before. Median 2.9 s per turn.

**Phase 1: pipeline prototype (Mac)**

5. Build a push-to-talk loop: record → speech recognition → LLM → text-to-speech → play. Time
   each stage. **Done 2026-10-29, first version:** `agent/loop.py`, run from the repo root:
   `.venv/bin/python -m agent.loop` (Enter to talk, Enter to stop; the first run asks macOS for
   microphone access for the terminal). `--tts vani` uses the macOS voice for a fast loop (test
   only, not for the product); `--wav a.wav b.wav` runs recorded turns without the mic.
   - **Pipeline:** mic (sounddevice, 16 kHz) → Indic-Transcribe-core (Mac GPU) →
     `agent/rules.py` + Sarvam 30B (`agent/llm.py`) → digits as Tamil words
     (`uraikal.numbers`) → Indic-Speak Anitha, one sentence at a time while the previous
     one plays. It keeps a pending action between turns, so a read-back can be answered with
     ஆமா or வேண்டாம். Replies are saved as wav files.
   - **Tested with 6 recorded turns** (Vani voice: alarm, ஆமாம், 15% of 300, chest pain, card,
     Ooty or Kodaikanal). Everything behaved as designed: read-back then "not connected yet"
     after the yes, 45, OTP request, a normal answer. The recogniser misheard வை as பே in the
     alarm request, and the read-back made that audible.
   - **Safety fix from the test:** the recogniser heard நெஞ்சு வலி as நென்யு வலி, so the
     emergency keyword missed and Sarvam said "go to hospital" without 108. The keyword now
     tolerates that, and any answer that sends someone to hospital at once is replaced by the
     108 reply.
   - **Time to first audio:** with Indic-Speak 28–35 s on turns that use Sarvam (3.9–8.2 s for
     the fixed replies); with the macOS voice 1.5–8.8 s, about 3 s typical. Speech recognition
     takes 0.4–1.2 s once warm, and Sarvam takes about 4.7 s for its two calls (routing, then
     the reply), since the reply is only spoken when it's complete. **The 2 s goal needs a
     faster TTS** (Indic-Speak on llama.cpp or MLX, or a GPU) **and streaming** Sarvam's reply
     into the TTS sentence by sentence (step 6).
   - **First live try by the user (2026-10-29, `--tts vani`, 4 turns):** replies were generated
     and saved but nothing was heard, because the Mac's sound output was muted (the loop now
     warns about that at start). Asking for a Thirukkural failed three ways:
     - the recogniser heard குறள் (couplet) as குரல் (voice), since ற and ர sound alike;
     - the loop had no memory, so "say one" lost the earlier mention of Thirukkural (it now
       sends the last 8 messages with each answer);
     - asked plainly, Sarvam **invents** a couplet ("அறத்தாருக்குத் தலைவன் அறம்", not a real
       kural), as in every earlier test. Fix: a Thirukkural tool backed by the real 1,330
       couplets (public-domain text), not the model's memory.
     A 0.1 s tap made the recogniser invent a word; recordings under 0.5 s are now ignored.
6. Move to Pipecat with streaming and turn detection, and measure the time to first audio.

**Phase 2: tune without big training**

7. Write the system prompt in Tamil: persona, spoken register, how to address people, and rules
   for when to use tools.
8. Add 1–5 tools with simple inputs, have the agent read values back for confirmation, and
   validate the values in code.
9. Voice: record 5–10 seconds of the target voice (**with written consent**) for IndicF5 cloning.
10. Fine-tune the speech recognition on your own Tamil audio if accuracy on your domain is weak
    (a few GPU-hours on a rented NVIDIA GPU, roughly $10–50).
11. Add pronunciation fixes and text normalization for numbers, dates and English words.

**Phase 3: data and decision**

12. Collect consented conversations, re-measure, and decide whether a single model (§6) is
    justified.

---

## 8. How to measure

| What | Metric | Data |
|---|---|---|
| Speech recognition | Share of words and characters wrong (WER/CER, same text normalization every time) | FLEURS-Tamil plus your own set, including mixed-English and phone-quality audio |
| Brain | Correctness, spoken naturalness (1–5, native rater), tool chosen and inputs right | 20–50 scripted Tamil requests |
| Text-to-speech | Naturalness rated by native listeners; clarity checked by running speech recognition on the output; similarity to the cloned voice | 10–20 sentences |
| End to end | Time to first audio, total turn time, task success | Scripted spoken calls |

---

## 9. Environment and lessons learned (same Mac)

- **Mac:** Apple M1 Max, 64 GB memory, macOS 27, about 350 GB free disk. Homebrew, uv, Python
  3.10/3.11/3.12, Node, Bun, ffmpeg, cmake, git and the Hugging Face CLI (`hf`) are all installed.
- **Tamil test audio without recording:** `say -v Vani -o q.aiff "<Tamil text>"`, then
  `ffmpeg -i q.aiff -ar 16000 -ac 1 q.wav`.
- **macOS 27 and SciPy:** SciPy 1.15.x fails to load on macOS 27 (a `__thread_bss` dlopen
  error), which breaks librosa. Use Python 3.11+ with SciPy 1.17 or newer.
- **Creating virtual environments:** `python3.11` on the PATH is a uv-managed build, and
  `python3.11 -m venv` produces a broken environment with it. Use
  `uv venv -p python3.11 --seed <dir>`. Add `pip install "setuptools<81"` if an old library
  needs `pkg_resources`.
- **llama.cpp for Sarvam:** `/usr/local/bin/llama-server` is an old Intel Homebrew build ("bad
  CPU type"). Mainline llama.cpp is cloned and built in `~/ai/llama.cpp` (2026-10-29, supports
  `sarvam_moe`): `cmake -B build -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF && cmake --build build -j 10
  --target llama-server`. The model is in the Hugging Face cache (`hf download
  sarvamai/sarvam-30b-gguf`).
- **Building llama.cpp-based servers:** Homebrew's OpenSSL turns them HTTPS-only, and they then
  exit silently. Build with `-DLLAMA_OPENSSL=OFF`.
- **llama.cpp-omni and Tamil:** the server crashed on Tamil output, because a token can end in
  the middle of a 3-byte Tamil letter. Fixed in `tools/server/server-omni.cpp` by holding back
  incomplete UTF-8 bytes; `~/ai/voice/s2s/runpod/patches/llama.cpp-omni.patch` was regenerated.
  The server has no reset call, so `eval/minicpmo.py` calls `omni_init` for every clip. Tamil
  replies can loop until the context is full (~400 s), so it runs with `-n 768`.
- **Local servers:** bind every local service to `127.0.0.1`, never `0.0.0.0`.
- **Training** needs a rented NVIDIA GPU (an H100 cost about $2–3 an hour in August 2026 listings
  on RunPod and Vast).
- **IndicConformer no longer needs NeMo:** its Hugging Face repo ships ONNX files (including
  `joint_post_net_ta.onnx`) and loads with `transformers.AutoModel(..., trust_remote_code=True)`.
  Indic-Transcribe-core is plain torch code too. Neither has been run on the Mac yet.
- **Project environment:** `.venv` (Python 3.11) with torch 2.15, **onnxruntime 1.20.1**,
  transformers, sentencepiece, soundfile, pyarrow. Created with `uv venv -p python3.11 --seed .venv`.
  For the TTS tests (2026-10-29): `uv pip install --python .venv/bin/python snac pydub vocos
  accelerate datasets ema_pytorch librosa pypinyin torchdiffeq x_transformers jieba matplotlib
  torchcodec` (not wandb: it pulls in an unknown beta package, and only IndicF5's training code
  needs it). For the voice loop: `sounddevice` (mic and speaker).
- **Code layout:** `agent/` is the assistant (`rules.py`, `llm.py`, `loop.py`;
  run as modules from the repo root, e.g. `.venv/bin/python -m agent.rules` for its
  self-check); `eval/` holds the tests and benchmarks; `prompts/` the system prompts. Number
  words come from [`uraikal.numbers`](https://github.com/zhagaramlabs/uraikal), Zhagaram's Tamil text
  library: `uv pip install --python .venv/bin/python -e ../uraikal` from a lab checkout, or
  `uraikal` from PyPI once released.
- **IndicConformer needs onnxruntime 1.20.1** (the version its card pins). onnxruntime 1.31
  refuses to load it ("External data path escapes model directory") because the Hugging Face
  cache stores its weight files as symlinks into `blobs/`.
- **Indic-Transcribe defaults to CPU on a Mac.** `run_asr.py` passes `device="mps"`, which works
  in float32. Skip its `nemo/` folder when downloading (4.6 GB duplicate).
- **Gated models:** IndicConformer, IndicF5, Indic-Transcribe-core and Indic-Speak all require
  accepting terms on their Hugging Face page while logged in, then `hf auth login` on the Mac.
  Bodhan AI's gate asks for organization, country and intended use.
- **FLEURS-Tamil** downloads without login as one parquet file (467 MB):
  `datasets/google/fleurs/.../parquet-data/ta_in/test-00000-of-00001.parquet`. 9 clips are over
  30 s and 4 over 45 s (Indic-Transcribe refuses over 45 s in one pass; `run_asr.py` uses its
  chunker above 30 s).

---

## 10. Pointers

- **English project, for reusable pieces:** `~/ai/voice/s2s`
  - `ask.sh`: one spoken question → spoken reply with MiniCPM-o; a good pattern for a Tamil
    equivalent.
  - `make_test_session.py`, `probe_duplex.py`, `check_duplex_decode.sh`: test-harness patterns.
  - `llama.cpp-omni/`, `MiniCPM-o-Demo/`: the working English full-duplex demo, patched.
- **Memory notes for the English project:**
  `~/.claude/projects/-Users-gopalanj-ai-voice-s2s/memory/s2s-voice-agent-project.md`.
- **Sources:**
  - IndicConformer: https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual
  - Indic-Transcribe-core: https://huggingface.co/bodhan-ai/indic-transcribe-core · Indic-Speak:
    https://huggingface.co/bodhan-ai/indic-speak · IndicF5 (official): https://huggingface.co/ai4bharat/IndicF5
  - Tamil speech recognition comparison (FLEURS): https://huggingface.co/osmapi/tamil-asr-qwen3
  - Sarvam 30B and 105B: https://artificialanalysis.ai/articles/sarvam-105b-Sarvam-30b-everything-you-need-to-know
  - IndicF5: https://www.promptlayer.com/models/indicf5 (find the official AI4Bharat card)
  - Dhee-Indic-F5 (license note): https://huggingface.co/dheeyantra/dhee-indic-f5
  - IndicVoices: https://arxiv.org/pdf/2403.01926 · IndicVoices-R: https://arxiv.org/pdf/2409.05356
  - IndicTTS (IIT Madras): https://www.iitm.ac.in/donlab/indictts/database (license:
    https://www.iitm.ac.in/donlab/indictts/downloads/license.pdf; access by e-mail to
    smtiitm@gmail.com) · HF mirror: https://huggingface.co/datasets/SPRINGLab/IndicTTS_Tamil
  - Rasa: https://arxiv.org/pdf/2407.14056 · Vakyansh: https://github.com/Open-Speech-EkStep/vakyansh-models
  - Gnani Inya VoiceOS: https://www.opensourceforu.com/2026/02/india-unveils-first-5b-voice-ai/
  - Qwen3-Omni languages: https://github.com/QwenLM/Qwen3-Omni · Qwen3.5-Omni voices: https://help.aliyun.com/en/model-studio/omni-voice-list
  - India open-source voice overview (vendor blog, treat with caution): https://caller.digital/blog/open-source-voice-ai-india-sarvam-ai4bharat-bhasini-2026

---

## 11. Open questions for the user

1. **Use case: answered 2026-10-29: all of them.** Customer support, personal assistant,
   booking, education, healthcare and casual chat. So Oosai is a **general Tamil assistant**: the
   LLM covers knowledge and conversation, and each domain adds its own tools later (Phase 2).
   **Healthcare rules:** general information, appointment booking and pointing people to a
   doctor only. No diagnosis and no medicine doses, and for emergencies it says to call 108.
   (Bodhan AI's license says its models aren't certified for medical use and bans automated
   decisions about people's rights without human oversight.)
2. **Dialect and tone:** which Tamil dialect(s), and should it address people formally or
   familiarly?
3. **Channel:** web or app microphone, or phone calls (8 kHz)?
4. **Voice:** whose voice will it use, and is written consent in place?
5. **Data:** what Tamil audio or transcripts do you already have, and are they stereo or mono,
   and how many hours?
6. **Deployment:** local on the Mac only, or a cloud GPU later?

## 12. Next step

Hugging Face access is in place (account `gopalanj`) and the FLEURS speech recognition baseline is
done (§7 step 2): a trade-off between the two models, no overall winner. MiniCPM-o was measured on
Tamil and ruled out as a base (§7 step 2b), so the pipeline plan stands.

**User, to unblock fine-tuning** (none of these is needed for the pipeline itself):
- Answer §11 questions 2–6, and set a budget ceiling (option A in §6 needs $500–1,500).
- On Hugging Face, accept the terms for `ai4bharat/IndicVoices`, `ai4bharat/indicvoices_r`,
  `ai4bharat/Rasa`, `ai4bharat/Kathbath` and `ARTPARK-IISc/Vaani` (checked 2026-10-29: still
  gated; IndicF5, Indic-Speak, Sarvam 30B and the IndicTTS mirror are open).
- A RunPod or Vast account with credit (API key in an environment variable), and a Mozilla Data
  Collective account for Common Voice Tamil.
- E-mail smtiitm@gmail.com to confirm commercial use of IndicTTS, or drop it (20 h).
- A native reviewer for the TTS samples and LLM answers; your own Tamil audio, if any (§11 Q5).

**Paper draft:** `paper/main.tex` ("Towards Oosai: Open Building Blocks and Baselines for a Tamil
Speech-to-Speech Voice Agent", Jayakumar Gopalan, Zhagaram Labs; 10 pages). It reports only what
has been done; red TODOs mark the rest. Build: `cd paper && xelatex main && bibtex main && xelatex
main && xelatex main`. For arXiv: choose the **XeLaTeX** processor (supported since November
2025), upload `main.tex`, `refs.bib`, `main.bbl` and the `fonts/` folder (Noto Serif Tamil, SIL
OFL 1.1, loaded by file name as arXiv requires). Contact: jay@zhagaramlabs.com. Open TODOs: public code
repository URL, listening check of the 22 suspect clips, native review and recordings of the
test set, AI-use statement. First-time submitters to cs.CL may need an arXiv endorsement.
Suggested timing: submit after Phase 0 is complete, so the paper also has TTS and LLM results.
1. **User:** check the 30 drafted questions in `eval/own_questions.tsv` (5 per domain:
   colloquial, Tanglish, numbers, healthcare safety cases) and fix any wording that isn't how
   people really talk. Then record each one as `eval/data/own/<id>.wav` (any recorder; convert
   with `ffmpeg -i in.m4a -ar 16000 -ac 1 <id>.wav`). Build the manifest and score both models:
   `awk -F'\t' -v d="$PWD/eval/data/own" 'NR>1{print $1"\t"d"/"$1".wav\t"$3}' eval/own_questions.tsv > eval/data/own/test.tsv`
   then `eval/run_asr.py <model> --manifest eval/data/own/test.tsv`. Pick the speech
   recognition model from that. Until then, prototype with Indic-Transcribe-core (claims
   code-mixing; wins more clips), with IndicConformer (MIT) as the fallback.
2. **User:** have a native speaker listen to the 30 TTS clips in `eval/results/tts/*/` and read
   Sarvam's 30 replies (`eval/results/sarvam30b_own_questions.jsonl`), rating 1–5 for
   naturalness and correctness. Phase 0 steps 3–4 have a first pass (§7) but no native ratings.
3. **User:** try the voice loop again with the sound unmuted: `.venv/bin/python -m agent.loop
   --tts vani` for a fast loop, then without `--tts vani` for the real voice (slow: about 30 s
   before it speaks). Sarvam doesn't know Thirukkural or other exact texts; a grounded
   Thirukkural tool is the proposed fix.
   The rules live in code (`agent/rules.py`, §7 step 4): about 19 of 30 good, 4 wrong, every
   safety case passes. Native review needed for `prompts/style_ta.txt`, the fixed replies in
   `agent/rules.py` and the number words in `uraikal.numbers`.
4. **Speed (Phase 1 step 6):** run Indic-Speak's Llama-3.2-3B on llama.cpp or MLX (or a rented
   GPU) and stream Sarvam's reply into the TTS sentence by sentence; target first audio ≤2 s.
5. **Accuracy:** a fresh set of test questions (the rules were tuned on these 30), a sturdier
   router (sup03 flipped to an invented answer when the calculator tool was added), and real
   tools behind `check` and the actions.
