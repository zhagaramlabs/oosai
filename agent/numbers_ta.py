"""Tamil number words <-> digits (spoken forms first, formal ones accepted when reading).

to_digits: "முன்னூறுல பதினஞ்சு பெர்சென்ட்" -> "300ல 15 பெர்சென்ட்"   (Sarvam 30B can't do sums from Tamil words)
to_words:  2300 -> "ரெண்டாயிரத்தி முன்னூறு"                          (the TTS reads digits in English)

  .venv/bin/python agent/numbers_ta.py      # self-check
"""
import re

UNITS = ["பூஜ்யம்", "ஒண்ணு", "ரெண்டு", "மூணு", "நாலு", "அஞ்சு", "ஆறு", "ஏழு", "எட்டு", "ஒம்பது"]
TEENS = ["பத்து", "பதினொண்ணு", "பன்னெண்டு", "பதிமூணு", "பதினாலு", "பதினஞ்சு", "பதினாறு", "பதினேழு", "பதினெட்டு", "பத்தொம்பது"]
TENS = {2: "இருபது", 3: "முப்பது", 4: "நாப்பது", 5: "அம்பது", 6: "அறுபது", 7: "எழுபது", 8: "எம்பது", 9: "தொண்ணூறு"}
HUNDREDS = {1: "நூறு", 2: "இருநூறு", 3: "முன்னூறு", 4: "நானூறு", 5: "ஐநூறு", 6: "அறுநூறு", 7: "எழுநூறு", 8: "எண்ணூறு", 9: "தொள்ளாயிரம்"}
BIG = [(10**7, "கோடி", "கோடியே"), (10**5, "லட்சம்", "லட்சத்தி"), (1000, "ஆயிரம்", "ஆயிரத்தி")]


def comb(word):
    """The form before another number: நூறு -> நூத்தி, இருபது -> இருபத்தி, தொள்ளாயிரம் -> தொள்ளாயிரத்தி."""
    return word[:-2] + "த்தி" if word.endswith(("று", "து", "ம்")) else word


def below_1000(n):
    h, r = divmod(n, 100)
    parts = [comb(HUNDREDS[h]) if r else HUNDREDS[h]] if h else []
    if r:
        t, u = divmod(r, 10)
        parts.append(UNITS[r] if r < 10 else TEENS[u] if r < 20 else f"{comb(TENS[t])} {UNITS[u]}" if u else TENS[t])
    return " ".join(parts)


def to_words(x):
    """A number in spoken Tamil words: whole numbers below 100 crore, decimals digit by digit."""
    if x < 0:
        return "மைனஸ் " + to_words(-x)
    whole, _, frac = f"{x:.2f}".rstrip("0").rstrip(".").partition(".")
    n, parts = int(whole), []
    if n == 0:
        parts.append(UNITS[0])
    for size, word, joined in BIG:
        k, n = divmod(n, size)
        if k:
            lead = below_1000(k)
            if k == 1:  # ஆயிரம், ஒரு லட்சம்
                parts.append(("" if word == "ஆயிரம்" else "ஒரு ") + (joined if n else word))
            elif word == "ஆயிரம்" and lead.endswith("ு"):  # ரெண்டு + ஆயிரம் -> ரெண்டாயிரம்
                parts.append(lead[:-1] + "ா" + (joined if n else word)[1:])
            else:
                parts.append(f"{lead} " + (joined if n else word))
    if n:
        parts.append(below_1000(n))
    words = " ".join(parts)
    return words + (" புள்ளி " + " ".join(UNITS[int(d)] for d in frac) if frac else "")


# Reading: morphemes that can start at the front of a word, as (regex, value, multiplies?).
_M = []
for i, w in enumerate(UNITS[1:], 1):
    _M.append((w, i, False))
for i, w in enumerate(TEENS, 10):
    _M.append((w, i, False))
for t, w in TENS.items():
    _M += [(w, t * 10, False), (w[:-2] + "த்(?:தி|து)", t * 10, False)]
for h, w in HUNDREDS.items():
    _M += [(w, h * 100, False), (comb(w), h * 100, False)]
_M += [  # formal and other spoken variants
    ("பூஜ்யம்|ஜீரோ", 0, False), ("ஒன்று|ஒன்னு|ஒரு", 1, False), ("இரண்டு", 2, False), ("மூன்று", 3, False), ("நான்கு", 4, False),
    ("ஐந்து", 5, False), ("ஒன்பது", 9, False), ("பதினொன்று|பதினொன்னு", 11, False), ("பன்னிரண்டு|பனிரெண்டு", 12, False),
    ("பதின்மூன்று", 13, False), ("பதினான்கு", 14, False), ("பதினைந்து|பதினைஞ்சு", 15, False), ("பத்தொன்பது", 19, False),
    ("நாற்பது", 40, False), ("ஐம்பது", 50, False), ("எண்பது", 80, False),
    ("நாற்பத்(?:தி|து)", 40, False), ("ஐம்பத்(?:தி|து)", 50, False), ("எண்பத்(?:தி|து)", 80, False),
    ("நூற்று|நூத்து", 100, False), ("முந்நூறு", 300, False), ("முந்நூற்று|முன்னூற்று", 300, False),
    ("தொண்ணூற்று", 90, False), ("அஞ்ஞூறு", 500, False),
    # after a tens stem the next unit fuses: இருபத் + அஞ்சு -> இருபத்தஞ்சு
    ("(?<=த்)(?:தொண்ணு|தொன்று)", 1, False), ("(?<=த்)(?:தஞ்சு|தைந்து|தைஞ்சு)", 5, False),
    ("(?<=த்)தாறு", 6, False), ("(?<=த்)தேழு", 7, False), ("(?<=த்)தெட்டு", 8, False),
    ("(?<=த்)(?:தொம்பது|தொன்பது)", 9, False),
    ("ஆயிர(?:ம்|த்தி|த்து)", 1000, True), ("(?<=ா)யிர(?:ம்|த்தி|த்து)", 1000, True),
    ("லட்ச(?:ம்|த்தி|த்து)", 10**5, True), ("கோடி(?:யே)?", 10**7, True),
]
for w in UNITS[1:] + TEENS + list(TENS.values()) + ["மூவு"]:  # ரெண்டு -> ரெண்டா before யிரம்
    _M.append((w[:-1] + "ா(?=யிர)", (UNITS + TEENS).index(w) if w in UNITS + TEENS else
               3 if w == "மூவு" else 10 * next(t for t, v in TENS.items() if v == w), False))
# a fused tens stem with nothing after it in the word: இருபத் (+ தஞ்சு)
for t, w in TENS.items():
    _M.append((w[:-2] + "த்(?=த[ொாேெஞை])", t * 10, False))
for w, v in [("நாற்ப", 40), ("ஐம்ப", 50), ("எண்ப", 80)]:
    _M.append((w + "த்(?=த[ொாேெஞை])", v, False))
_M = [(re.compile(p), v, m) for p, v, m in _M]
SUFFIX = re.compile(r"(?:ல|ல்|லே|க்கு|ஆல|னா|ன்னா|ம்)?[.,!?;:\"'()]*$")


def _read_word(word):
    """Morphemes covering the front of a word, if the rest is only a case ending or punctuation."""
    pos, out = 0, []
    while True:
        best = max(((m.match(word, pos), v, mult) for m, v, mult in _M), key=lambda c: c[0].end() if c[0] else -1)
        if not best[0] or best[0].end() == pos:
            break
        out.append((best[1], best[2]))
        pos = best[0].end()
    return (out, word[pos:]) if out and SUFFIX.fullmatch(word[pos:]) else (None, word)


def _value(morphemes):
    total = cur = 0
    for v, mult in morphemes:
        if mult:
            total, cur = total + (cur or 1) * v, 0
        else:
            cur += v
    return total + cur


def to_digits(text):
    """Replace runs of Tamil number words with digits, keeping the last word's case ending."""
    out, run = [], []
    for word in text.split(" ") + [""]:
        morphemes, rest = _read_word(word) if word else (None, "")
        if morphemes:
            run += morphemes
            if rest:  # a case ending closes the number
                out.append(f"{_value(run)}{rest}")
                run = []
            continue
        if run:
            out.append(str(_value(run)))
            run = []
        out.append(word)
    return " ".join(out[:-1])


def speak_digits(text):
    """Digits in a reply as Tamil words, for the TTS (numbers of a billion or more are left alone)."""
    def words(m):
        x = float(m.group()) if "." in m.group() else int(m.group())
        return to_words(x) if x < 10**9 else m.group()
    return re.sub(r"\d+(?:\.\d+)?", words, text)


if __name__ == "__main__":
    assert speak_digits("அது 35.75, 108-க்கு கால்") == "அது முப்பத்தி அஞ்சு புள்ளி ஏழு அஞ்சு, நூத்தி எட்டு-க்கு கால்"
    assert to_words(45) == "நாப்பத்தி அஞ்சு", to_words(45)
    assert to_words(2300) == "ரெண்டாயிரத்தி முன்னூறு", to_words(2300)
    assert to_words(1000) == "ஆயிரம்" and to_words(1500) == "ஆயிரத்தி ஐநூறு", to_words(1500)
    assert to_words(140) == "நூத்தி நாப்பது" and to_words(108) == "நூத்தி எட்டு", to_words(140)
    assert to_words(1.5) == "ஒண்ணு புள்ளி அஞ்சு" and to_words(0) == "பூஜ்யம்"
    assert to_words(250000) == "ரெண்டு லட்சத்தி அம்பதாயிரம்", to_words(250000)
    for n in list(range(3000)) + list(range(3000, 120001, 97)) + [21000, 10**5, 345678, 10**7, 23456789]:
        assert to_digits(to_words(n)) == str(n), (n, to_words(n), to_digits(to_words(n)))
    assert to_digits("முன்னூறுல பதினஞ்சு பெர்சென்ட் எவ்ளோ?") == "300ல 15 பெர்சென்ட் எவ்ளோ?"
    assert to_digits("ரெண்டாயிரத்து முன்னூறு ரூபாய்க்கு பத்து பெர்சென்ட்") == "2300 ரூபாய்க்கு 10 பெர்சென்ட்"
    assert to_digits("கிலோ நாப்பது ரூபான்னா மூணு கிலோ") == "கிலோ 40 ரூபான்னா 3 கிலோ"
    assert to_digits("ஆயிரம் ரூபாய நாலு பேரு பிரிச்சா ஒருத்தருக்கு எவ்ளோ?") == "1000 ரூபாய 4 பேரு பிரிச்சா ஒருத்தருக்கு எவ்ளோ?"
    fused = "இருபத்தஞ்சு பேரு, நாப்பத்தொண்ணு ரூபா, தொண்ணூத்தொம்பது கிலோ"
    assert to_digits(fused) == "25 பேரு, 41 ரூபா, 99 கிலோ", to_digits(fused)
    assert to_digits("நான் நாளைக்கு பத்தாம் கிளாஸ்") == "நான் நாளைக்கு பத்தாம் கிளாஸ்"
    print("ok")
