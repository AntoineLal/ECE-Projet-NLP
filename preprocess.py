"""Prétraitement du texte libre : normalisation, tokenisation, stop-words, stemming."""
import re
import unicodedata

from nltk.stem.snowball import FrenchStemmer

STOPWORDS = set("""
a au aux avec ce ces dans de des du elle en et eux il je la le les leur lui ma mais me meme mes moi mon ne nos notre
nous on ou par pas pour qu que qui sa se ses son sur ta te tes toi ton tu un une vos votre vous c d j l m n s t y
est suis es sommes etes sont etais etait ete etre ai as avons avez ont avoir fait faire peux peut peuvent veux veut
voudrais aimerais souhaite assez tres plus moins trop bien aussi donc alors si ainsi cela ca cette cet ceci comme
lorsque quand puis car or ni soit tout tous toute toutes
""".split())

_stemmer = FrenchStemmer()


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def normalize(text: str) -> str:
    """Minuscules, Unicode NFC, apostrophes unifiées, espaces nettoyés (accents conservés)."""
    text = unicodedata.normalize("NFC", str(text or "")).lower()
    text = text.replace("’", "'").replace("`", "'")
    return re.sub(r"\s+", " ", text).strip()


def fold(text: str) -> str:
    """Normalisation sans accents, utilisée pour les lexiques (regex)."""
    return strip_accents(normalize(text))


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zà-ÿœ]+", normalize(text))


def preprocess(text: str) -> list[str]:
    """Pipeline complet : normalisation → tokens → suppression des stop-words → racinisation."""
    out = []
    for tok in tokenize(text):
        if strip_accents(tok) in STOPWORDS or len(tok) < 2:
            continue
        out.append(_stemmer.stem(tok))
    return out


def preprocess_str(text: str) -> str:
    return " ".join(preprocess(text))


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?;])\s+|\n+", normalize(text))
    return [p.strip() for p in parts if len(p.strip()) > 3]
