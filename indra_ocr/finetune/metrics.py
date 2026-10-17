"""Unicode-explicit, micro-averaged transcription metrics."""

from __future__ import annotations

import unicodedata

try:
    from rapidfuzz.distance.Levenshtein import distance as edit_distance
except ImportError:
    def edit_distance(a, b):
        previous = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            current = [i]
            for j, cb in enumerate(b, 1):
                current.append(min(current[-1] + 1, previous[j] + 1,
                                   previous[j - 1] + (ca != cb)))
            previous = current
        return previous[-1]


class TranscriptionMetrics:
    def __init__(self, normalization="NFC"):
        if normalization not in ("NFC", "NFD"):
            raise ValueError("normalization must be NFC or NFD")
        self.normalization = normalization
        self.samples = self.exact = self.char_errors = self.word_errors = 0
        self.chars = self.words = 0
        self.macro_cer = 0.0

    def add(self, reference, prediction):
        # Preserve case, punctuation, inner whitespace and line breaks. Strip
        # only outer whitespace, with the same rule on both sides.
        ref, hyp = (unicodedata.normalize(self.normalization, t.replace("\r\n", "\n")).strip()
                    for t in (reference, prediction))
        ce, we = edit_distance(ref, hyp), edit_distance(ref.split(), hyp.split())
        self.samples += 1
        self.exact += ref == hyp
        self.char_errors += ce
        self.word_errors += we
        self.chars += len(ref)
        self.words += len(ref.split())
        self.macro_cer += ce / max(len(ref), 1)
        return {"char_errors": ce, "reference_chars": len(ref), "word_errors": we,
                "reference_words": len(ref.split()), "cer": ce / max(len(ref), 1),
                "wer": we / max(len(ref.split()), 1), "exact": ref == hyp}

    def result(self):
        if not self.samples:
            raise ValueError("cannot score zero samples")
        return {"normalization": self.normalization, "samples": self.samples,
                "cer": self.char_errors / max(self.chars, 1),
                "wer": self.word_errors / max(self.words, 1),
                "macro_cer": self.macro_cer / self.samples,
                "exact": self.exact, "accuracy": self.exact / self.samples,
                "char_errors": self.char_errors, "reference_chars": self.chars,
                "word_errors": self.word_errors, "reference_words": self.words}
