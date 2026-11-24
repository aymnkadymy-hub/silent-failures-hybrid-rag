"""Arabic normaliser shared by the lexical runs (BM25 and the independent normaliser of the controller)."""
import re
import unicodedata

_DIAC = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_NON_AR = re.compile("[^ء-غف-ي ]+")
_FOLD = str.maketrans({"آ": "ا", "أ": "ا", "إ": "ا", "ٱ": "ا",
                       "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي",
                       "ی": "ي", "ک": "ك"})


def norm_words(text, nfkc=True):
    """Retrieval and alignment normaliser: NFKC (folds presentation forms), strip diacritics and tatweel, fold alef,
    yaa, taa marbuta and hamza seats, keep Arabic letters only. Returns the list of words.
    nfkc=False is the pre-registered sensitivity variant N0 (a naive indexer: presentation forms are dropped)."""
    t = unicodedata.normalize("NFKC", text or "") if nfkc else (text or "")
    t = _DIAC.sub("", t).translate(_FOLD)
    t = _NON_AR.sub(" ", t)
    return t.split()
