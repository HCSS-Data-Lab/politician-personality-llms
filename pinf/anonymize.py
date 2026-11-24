import re
import unicodedata
from typing import Optional, Iterable
_NAME_STOPWORDS: frozenset[str] = frozenset({'de', 'del', 'della', 'der', 'den', 'van', 'von', 'la', 'le', 'el', 'al', 'bin', 'ibn', 'ben', 'dos', 'das', 'da', 'di', 'the', 'of', 'mr', 'mrs', 'ms', 'dr', 'sir', 'president', 'prime', 'minister', 'chancellor', 'premier', 'mister'})

def _strip_accents(s: str) -> str:
    return ''.join((c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c)))

def _norm_tokens(s: str) -> set[str]:
    if not s:
        return set()
    cleaned = re.sub('[^0-9a-z\\s]+', ' ', _strip_accents(s).lower())
    return {t for t in cleaned.split() if len(t) >= 2}

def _norm_name_tokens(name: str) -> list[str]:
    if not name:
        return []
    cleaned = re.sub('[^0-9a-z\\s]+', ' ', _strip_accents(name).lower())
    return [t for t in cleaned.split() if len(t) >= 2 and t not in _NAME_STOPWORDS]

def score_reid_guess(guess: str, true_name: str, *, country: Optional[str]=None, aliases: Iterable[str]=()) -> str:
    g = _norm_tokens(guess)
    if not g:
        return 'wrong'

    def _names_hit(name: str) -> bool:
        toks = _norm_name_tokens(name)
        if not toks:
            return False
        return toks[-1] in g or set(toks) <= g
    if _names_hit(true_name) or any((_names_hit(a) for a in aliases)):
        return 'correct'
    name_set = set(_norm_name_tokens(true_name))
    if name_set & g:
        return 'partial'
    if country and _norm_tokens(country) and (_norm_tokens(country) <= g):
        return 'partial'
    return 'wrong'
