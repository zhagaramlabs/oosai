"""Word and character error rates with one fixed Tamil normalization.

CER counts Unicode code points (a consonant plus vowel sign is 2), spaces included.
"""
import re
import unicodedata


def norm(s):
    s = unicodedata.normalize("NFC", s).lower()
    # Drop punctuation and symbols; keep letters, digits and Tamil vowel signs (Mn/Mc).
    s = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s)
    return re.sub(r"\s+", " ", s).strip()


def edits(r, h):
    d = list(range(len(h) + 1))
    for i, x in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


def score(pairs):
    """pairs: iterable of (reference, hypothesis). Returns corpus-level (wer, cer)."""
    we = wn = ce = cn = 0
    for r, h in pairs:
        r, h = norm(r), norm(h)
        we += edits(r.split(), h.split())
        wn += len(r.split())
        ce += edits(r, h)
        cn += len(r)
    return we / wn, ce / cn


if __name__ == "__main__":
    assert edits("kitten", "sitting") == 3
    assert norm("இது pH, என்று!") == "இது ph என்று"
    assert score([("ஒரு இரண்டு மூன்று", "ஒரு இரண்டு மூன்று.")]) == (0, 0)
    w, c = score([("ஒரு இரண்டு மூன்று நான்கு", "ஒரு மூன்று நான்கு ஐந்து")])
    assert w == 0.5, w
    print("ok")
