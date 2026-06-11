from collections import defaultdict

OMIT = frozenset("q")

def normalize(raw, omit=OMIT):
    out = []
    for ch in raw:
        if ch in (" ", "\n"):
            out.append(" ")
        elif ch.isalpha():
            low = ch.lower()
            if low not in omit:
                out.append(low)
        elif ch == ".":
            out.append(".")
    return out

def build_vocab(stream):
    alphabet = sorted(set(stream))
    char_to_idx = {ch: i for i, ch in enumerate(alphabet)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    return alphabet, char_to_idx, idx_to_char

def build_global_freq(stream, alphabet):
    freq = defaultdict(int)
    for ch in stream:
        if ch in alphabet:
            freq[ch] += 1
    total = sum(freq.values()) or 1
    return sorted(alphabet, key=lambda c: -freq[c] / total)
