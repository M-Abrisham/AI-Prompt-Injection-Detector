"""Language detection for multilingual routing (D6 category).

Detects the primary language of input text and flags non-English or
mixed-language content.  Uses the ``langdetect`` library when available;
gracefully degrades to a built-in heuristic that uses Unicode script
analysis and non-English stopword detection.

Technique mapping:
    non_english_input    -> D6   (Multilingual Injection)
    mixed_language_input -> D6.3 (Chinese / mixed-language context)
"""

import logging
import re

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency -- langdetect
# ---------------------------------------------------------------------------
try:
    from langdetect import detect_langs, DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException

    # Make detection deterministic across runs
    DetectorFactory.seed = 0
    _HAS_LANGDETECT = True
except ImportError:
    _HAS_LANGDETECT = False
    _logger.debug("langdetect not installed; using built-in heuristic")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Minimum character count for reliable langdetect detection.  Below this
# threshold the library returns unreliable guesses so we use heuristics.
_MIN_CHARS_FOR_DETECTION = 20

# Confidence threshold -- below this we treat the result as unreliable.
_MIN_CONFIDENCE = 0.5

# Quick heuristic regex: text contains CJK, Arabic, Cyrillic, Devanagari,
# Thai, or other non-Latin script blocks -- used for mixed-language detection.
_NON_LATIN_RE = re.compile(
    "["
    "\u0600-\u06ff"   # Arabic
    "\u0400-\u04ff"   # Cyrillic
    "\u0900-\u097f"   # Devanagari
    "\u0e00-\u0e7f"   # Thai
    "\u3040-\u30ff"   # Hiragana + Katakana
    "\u4e00-\u9fff"   # CJK Unified
    "\uac00-\ud7af"   # Hangul Syllables
    "\u3400-\u4dbf"   # CJK Extension A
    "\U00020000-\U0002a6df"  # CJK Extension B
    "]"
)

_LATIN_LETTER_RE = re.compile(r"[a-zA-Z]")

# ---------------------------------------------------------------------------
# Latin-script non-English stopword detection
# ---------------------------------------------------------------------------
# Common function words in major non-English languages that use the Latin
# script.  These words are extremely common in their respective languages
# but virtually never appear in English text.  We require multiple hits
# to avoid false positives from loanwords or proper nouns.
#
# Covers: French, Spanish, Portuguese, German, Italian, Dutch, Turkish,
# Romanian, Polish, Czech, Swedish, Danish, Norwegian, Indonesian/Malay,
# Swahili, Estonian, Croatian, and transliterated Arabic/Japanese/Hindi.
_NON_ENGLISH_STOPWORDS = frozenset({
    # French
    "les", "des", "une", "est", "pas", "sont", "mais", "dans", "avec",
    "pour", "sur", "tout", "qui", "que", "cette", "ses", "ces", "vous",
    "nous", "ils", "ont", "aux", "oubliez", "affichez", "devoilez",
    "comme", "aucune", "votre", "vos", "tous", "sans", "aucun", "etre",
    "peut", "aussi", "chez", "entre", "depuis", "vers", "avant", "apres",
    "tres", "pouvez", "quelque", "chose", "bonjour", "merci", "puis",
    "allez", "comment", "aujourd", "puis",
    # Spanish
    "los", "las", "una", "del", "con", "por", "como", "pero", "sus",
    "mas", "hay", "sin", "puede", "esta", "este", "esto", "ese", "esa",
    "donde", "cuando", "porque", "cual", "entre", "desde", "todos",
    "otro", "otra", "otros", "cada", "algo", "sido", "tiene", "muy",
    "favor", "hola", "pueden", "ayudar",
    "olvidate", "revelame", "secretos", "guardados",
    # Portuguese
    "nao", "uma", "dos", "foi", "com", "por", "mais", "como", "pode",
    "tem", "para", "isso", "todo", "essa", "este", "tudo", "muito",
    "tambem", "receita", "fazer", "preciso",
    # German
    "und", "die", "der", "das", "ein", "eine", "ist", "nicht", "mit",
    "den", "von", "auf", "ich", "sie", "wir", "sind", "auch", "haben",
    "oder", "werden", "kann", "nach", "vor", "noch", "nur", "wie",
    "vergiss", "enthuelle", "geheimen", "wurde", "alles",
    # Italian
    "gli", "dei", "del", "che", "con", "sono", "una", "questo", "anche",
    "essere", "hanno", "suo", "sua", "suoi", "loro", "tutto", "tutti",
    # Dutch
    "het", "een", "van", "dat", "met", "zijn", "niet", "voor", "ook",
    "nog", "wel", "maar", "deze", "naar",
    # Turkish
    "bir", "ile", "olan", "gibi", "daha", "icin", "kadar", "sonra",
    "bana", "unut", "goster", "sifreleri", "hepsini",
    # Romanian
    "este", "sunt", "din", "pentru", "prin", "care", "dar",
    # Transliterated Arabic (Arabizi)
    "kull", "sabiqan", "ikshif", "qeel", "insa",
    # Transliterated Japanese (Romaji)
    "subete", "meirei", "wasurete", "oshiete", "himitsu",
    # Transliterated Chinese (Pinyin)
    "suoyou", "mingling", "wangji", "yiqian",
    # Transliterated Hindi (Hinglish)
    "sabhi", "aadesh", "bhool", "pehle", "diye", "gaye", "batao",
    # Transliterated Korean
    "modeun", "jisileul", "musihago", "ijeonui",
    # Transliterated Russian
    "predydushchie", "pravila", "pokashi", "skrytuyu", "otmeni",
    # Indonesian/Malay
    "dari", "atau", "juga", "akan", "sudah",
})

# Minimum number of non-English stopword hits required to flag text.
# Require >= 2 to avoid false positives from isolated loanwords.
_MIN_STOPWORD_HITS = 2

# Regex for Latin Extended characters (accented letters common in European
# languages but rare in English).  Covers Latin Extended-A, Extended-B,
# and Latin Extended Additional.
_LATIN_EXTENDED_RE = re.compile(
    "["
    "\u00c0-\u024f"  # Latin Extended-A + Extended-B (accented chars)
    "\u1e00-\u1eff"  # Latin Extended Additional
    "]"
)

# Minimum ratio of Latin Extended characters to total alpha characters
# to flag as likely non-English.  French/Spanish/German text typically
# has 3-8% accented characters; English text has near 0%.
_MIN_ACCENT_RATIO = 0.02  # 2% of alpha chars


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_language(text):
    """Detect the primary language of *text*.

    Parameters
    ----------
    text : str
        The input text to analyse.  Should be the **sanitized** (post-
        normalization) text so that invisible characters and encoding
        artefacts have already been removed.

    Returns
    -------
    dict
        ``detected_language`` : str
            ISO 639-1 language code (e.g. ``"en"``, ``"zh-cn"``, ``"ar"``),
            or ``"unknown"`` when detection is unreliable.
        ``language_confidence`` : float
            Confidence score 0.0 -- 1.0.
        ``is_non_english`` : bool
            ``True`` when the detected language is not English.
        ``anomaly_flags`` : list[str]
            May contain ``"non_english_input"`` and/or
            ``"mixed_language_input"``.
    """
    result = {
        "detected_language": "unknown",
        "language_confidence": 0.0,
        "is_non_english": False,
        "anomaly_flags": [],
    }

    # --- Guard: empty / whitespace-only text ---
    if not text or not text.strip():
        return result

    stripped = text.strip()

    # --- Short text: always use heuristic (langdetect unreliable) ---
    if len(stripped) < _MIN_CHARS_FOR_DETECTION:
        return _heuristic_detect(stripped)

    # --- langdetect not available: use enhanced heuristic fallback ---
    if not _HAS_LANGDETECT:
        return _heuristic_detect(stripped)

    # --- Primary detection via langdetect ---
    try:
        langs = detect_langs(stripped)
    except LangDetectException:
        _logger.debug("langdetect raised LangDetectException for input")
        return _heuristic_detect(stripped)

    if not langs:
        return _heuristic_detect(stripped)

    top = langs[0]
    # langdetect returns objects with .lang and .prob attributes
    lang_code = top.lang        # e.g. "en", "zh-cn", "ar"
    confidence = top.prob       # float 0.0 - 1.0

    result["detected_language"] = lang_code
    result["language_confidence"] = round(confidence, 4)

    # --- Determine non-English status ---
    is_english = lang_code.startswith("en")

    if not is_english and confidence >= _MIN_CONFIDENCE:
        result["is_non_english"] = True
        result["anomaly_flags"].append("non_english_input")

    # --- Mixed-language detection ---
    # Two signals: (1) langdetect returns multiple languages with
    # non-trivial probabilities, or (2) script-level heuristic detects
    # both Latin and non-Latin characters.
    if _has_mixed_scripts(stripped):
        if "mixed_language_input" not in result["anomaly_flags"]:
            result["anomaly_flags"].append("mixed_language_input")
        # If text is mixed but top detected language is English,
        # still flag as non-English since it contains non-English segments
        if is_english:
            result["is_non_english"] = True
            if "non_english_input" not in result["anomaly_flags"]:
                result["anomaly_flags"].append("non_english_input")

    # Also check langdetect multi-language output
    if len(langs) >= 2:
        second = langs[1]
        # If two languages both have significant probability, it is mixed
        if second.prob >= 0.2:
            if "mixed_language_input" not in result["anomaly_flags"]:
                result["anomaly_flags"].append("mixed_language_input")

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _heuristic_detect(text):
    """Fallback detection using Unicode script analysis and stopword matching.

    Uses three complementary signals to detect non-English content
    without requiring the ``langdetect`` library:

    1. **Non-Latin script detection**: CJK, Arabic, Cyrillic, Devanagari,
       Thai, Hangul characters are strong signals of non-English content.
    2. **Latin Extended / accented character detection**: Characters like
       accented vowels common in French, Spanish, German, Portuguese, etc.
       but rare in English.
    3. **Non-English stopword detection**: Matching common function words
       from major non-English languages (requires >= 2 hits to avoid
       false positives from loanwords).
    """
    result = {
        "detected_language": "unknown",
        "language_confidence": 0.0,
        "is_non_english": False,
        "anomaly_flags": [],
    }

    has_non_latin = bool(_NON_LATIN_RE.search(text))
    has_latin = bool(_LATIN_LETTER_RE.search(text))

    if has_non_latin and has_latin:
        # Mixed scripts -- both Latin letters and non-Latin characters
        result["is_non_english"] = True
        result["anomaly_flags"].append("non_english_input")
        result["anomaly_flags"].append("mixed_language_input")
        return result

    if has_non_latin and not has_latin:
        # Purely non-Latin text (Arabic, CJK, Cyrillic, etc.)
        result["is_non_english"] = True
        result["anomaly_flags"].append("non_english_input")
        return result

    # --- Latin-script text: check for non-English indicators ---
    # Signal 1: Accented / Latin Extended characters
    if _has_significant_accented_chars(text):
        result["is_non_english"] = True
        result["anomaly_flags"].append("non_english_input")
        return result

    # Signal 2: Non-English stopword frequency
    if _has_non_english_stopwords(text):
        result["is_non_english"] = True
        result["anomaly_flags"].append("non_english_input")
        return result

    return result


def _has_mixed_scripts(text):
    """Return True if text contains both Latin letters and non-Latin
    script characters.

    This is a strong heuristic for mixed-language content (e.g.
    English words mixed with CJK characters).
    """
    has_non_latin = bool(_NON_LATIN_RE.search(text))
    has_latin = bool(_LATIN_LETTER_RE.search(text))
    return has_non_latin and has_latin


def _has_significant_accented_chars(text):
    """Return True if text contains a significant proportion of accented
    Latin characters (common in European non-English languages)."""
    alpha_count = sum(1 for ch in text if ch.isalpha())
    if alpha_count < 10:
        return False
    accent_count = len(_LATIN_EXTENDED_RE.findall(text))
    return accent_count / alpha_count >= _MIN_ACCENT_RATIO


def _has_non_english_stopwords(text):
    """Return True if text contains >= _MIN_STOPWORD_HITS non-English
    stopwords.

    Tokenizes on whitespace and punctuation boundaries, lowercases,
    and checks against the stopword set.
    """
    # Split on non-alpha characters to get word tokens
    words = re.findall(r"[a-zA-Z\u00c0-\u024f]+", text.lower())
    hits = 0
    for word in words:
        if word in _NON_ENGLISH_STOPWORDS:
            hits += 1
            if hits >= _MIN_STOPWORD_HITS:
                return True
    return False
