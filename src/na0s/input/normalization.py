import logging
import math
import os
import re
import unicodedata

# ftfy fixes mojibake (encoding mix-ups) — e.g. UTF-8 decoded as latin-1.
# Graceful fallback: if not installed, mojibake repair is simply skipped.
try:
    import ftfy

    _HAS_FTFY = True
except ImportError:
    _HAS_FTFY = False

# Unicode whitespace variants that should become plain ASCII space.
# Does NOT include \n, \r, \t — those are handled separately.
# U+2800 (BRAILLE PATTERN BLANK) is included: it renders as a blank/space
# but has category So (Other Symbol) so is not caught by standard whitespace
# checks.  Attackers use it as an invisible word separator to evade regex
# matching (e.g. "Ignore⠀all⠀previous" with braille blanks between words).
_UNICODE_WHITESPACE_RE = re.compile(
    "[\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u2800\u3000\ufeff\x0b\x0c]"
)

# Normalize \r\n → \n and lone \r → \n to prevent parser differentials.
# Some parsers treat \r as whitespace (Python \s, .split()), while line-based
# parsers ignore it.  Lone \r can also overwrite displayed text via terminal
# carriage-return semantics, hiding malicious content from visual inspection.
_CR_NORMALIZE_RE = re.compile(r"\r\n?")


# Collapse runs of multiple ASCII spaces into one
_MULTI_SPACE_RE = re.compile(r" {2,}")

# Collapse 3+ consecutive newlines into 2 (preserves paragraph breaks)
_EXCESSIVE_NEWLINES_RE = re.compile(r"\n{3,}")

# Collapse 3+ consecutive tabs into 1
_EXCESSIVE_TABS_RE = re.compile(r"\t{3,}")

# Token splitter: split on whitespace but keep punctuation attached to the
# token for reconstruction.  We use re.split with a capturing group so that
# delimiters (whitespace runs) are preserved for lossless reassembly.
_TOKEN_SPLIT_RE = re.compile(r"(\s+)")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Character-level reassembly (anti-evasion for spaced/dotted chars)
# ---------------------------------------------------------------------------
# Matches a dot-separated token where every segment is a single alpha char
# and there are at least 3 segments: e.g. "i.g.n.o.r.e"
_DOT_CHAR_SPLIT_RE = re.compile(
    r"\b([A-Za-z](?:\.[A-Za-z]){2,})\b"
)

# Known abbreviations that should NOT be reassembled (lowercase, with dots)
_ABBREVIATION_ALLOWLIST = frozenset({
    "u.s.a.", "u.s.", "a.m.", "p.m.", "e.g.", "i.e.", "a.d.", "b.c.",
    "d.c.", "u.k.", "e.u.", "ph.d.", "m.d.", "b.a.", "m.a.", "b.s.",
    "m.s.", "j.d.", "r.n.", "d.d.s.", "o.k.", "a.k.a.", "e.t.a.",
    "r.s.v.p.", "a.s.a.p.", "d.i.y.", "f.y.i.", "t.b.d.", "n.a.",
    "c.e.o.", "c.f.o.", "c.t.o.", "v.p.", "l.l.c.", "inc.",
})

# Generic single-character-separator evasion (Pass 3).  Beyond spaces and
# dots, attackers split words with *any* repeated punctuation:
# ``i-g-n-o-r-e``, ``i_g_n_o_r_e``, ``i,g,n,o,r,e``, ``i=g=n``, ``i--g--n``,
# ``i—g—n`` (em-dash), ``i#g#n``, ``i@g@n``, etc.  Rather than enumerate
# separators (a whack-a-mole that left em-dash/=/#/@/\\ evading), we match a
# run of >= 4 single alpha chars joined by a *consistent* separator string
# (1-3 non-alphanumeric, **non-whitespace** chars; the ``\1`` backreference
# forces the same separator between every pair).
#
# The separator deliberately excludes whitespace: a whitespace-bearing
# separator like ``", "`` is the canonical *list* delimiter, so allowing it
# would reassemble benign enumerations ("a, b, c, d, e") into nonsense and
# flag them.  Pure single-space splits ("i g n o r e") are already handled by
# Pass 1 and newline stacks by Pass 4; space+punct ("i . g . n") and tab
# separators are left as (documented) residuals to preserve list safety.
#
# The separator also excludes ``/ | < > \``: these connect single chars in
# benign text (URL paths ``/a/b/c/d/``, regex alternation ``a|b|c``, CSS /
# breadcrumb chains ``a > b > c``, HTML, and backslash escape runs like
# ``\n\n\n\n`` which would otherwise reassemble to ``nnnn``), so reassembling
# across them would mangle code.  These separators are left as residuals.
#
# The single-alpha-only + run>=4 + no-whitespace + no-code-operator
# constraints keep benign text safe (``e-mail``, ``snake_case``, ``a, b, c``
# lists, ``1,000``, URL paths, ``\n`` escapes) -- empirically it adds 0/30000
# FPs over the prior space/dot baseline on real-world scrape.  Bounded ({1,3},
# single-char alternation, backref) so it is not a ReDoS vector (<9ms/50k).
_GENERIC_CHAR_SPLIT_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"[A-Za-z]([^A-Za-z0-9\s/|<>\\]{1,3})(?:[A-Za-z]\1){2,}[A-Za-z]"
)

# Reassembly always normalizes the text (so downstream rules see the word),
# but the SCORED ``char_level_reassembly`` flag is only emitted for a run this
# long (>= chars).  A bare 3-char run is the weakest signal and fires on
# incidental benign letter-sequences (~0.05% of genuine-benign text, e.g. an
# acronym or a list of single letters), so it must NOT add composite risk;
# real char-split attacks spell whole words (run >= 4).
_CHAR_SPLIT_MIN_SCORED = 4

# A single-char run this long (>= chars) is treated as *heavy* obfuscation:
# essentially never produced by benign text, so the scorer floors it to the
# decision threshold (see predict.py char-split block).
_CHAR_SPLIT_HEAVY_RUN = 8


def _reassemble_char_splits(text):
    """Reassemble words that have been split into individual characters.

    Handles four evasion patterns:
      1. Space-separated single chars: ``i g n o r e`` -> ``ignore``
      2. Dot-separated single chars: ``i.g.n.o.r.e`` -> ``ignore``
      3. Other-punctuation-separated single chars (``- _ , · = # @ : ; * + ~``
         em-/en-dash, etc.; excludes ``/ | < > \\`` and whitespace):
         ``i-g-n-o-r-e`` / ``i_g_n_o_r_e`` / ``i,g,n,o,r,e`` -> ``ignore``
      4. Vertical (newline-stacked) single chars: one alpha char per line.

    For space-separated chars, double-space boundaries (``"e  a"``) are
    treated as word separators so that ``"i g n o r e  a l l"`` becomes
    ``"ignore all"`` rather than ``"ignoreall"``.  When no double-space
    boundaries exist, consecutive single-char runs are joined as one word.

    Passes 1-2 reassemble runs of 3+ single alpha chars; passes 3-4 require
    4+ (stricter, since their separators are common in benign text).  This
    avoids false positives on legitimate text like "I am", "e-mail",
    "snake_case", "a, b, c" lists, "1,000" and URL paths -- empirically the
    generic passes fire 0/30000 on real-world scrape.

    Known abbreviations (U.S.A., e.g., etc.) are preserved.

    Returns ``(text, reassembled, max_run)`` where *reassembled* is True if
    any reassembly occurred and *max_run* is the length (in chars) of the
    longest single-char run that was collapsed -- a magnitude the scorer
    uses to grade obfuscation severity (see ``_CHAR_SPLIT_HEAVY_RUN``).
    """
    reassembled = False
    max_run = 0

    # --- Pass 1: Space-separated single alpha chars ---
    # Process each line independently to avoid cross-line reassembly.
    # Within a line, split on double-space boundaries first (word
    # separators), then reassemble single-char runs within each segment.
    out_lines = []
    for line in text.split("\n"):
        # Split on runs of 2+ spaces (preserving them as word boundaries)
        segments = re.split(r"( {2,})", line)
        rebuilt_segments = []
        for seg in segments:
            # If this segment is a multi-space delimiter, keep as-is
            if seg and not seg.strip():
                rebuilt_segments.append(seg)
                continue
            # Process tokens within this segment
            tokens = seg.split(" ")
            out_tokens = []
            i = 0
            while i < len(tokens):
                if len(tokens[i]) == 1 and tokens[i].isalpha():
                    run_start = i
                    while (i < len(tokens)
                           and len(tokens[i]) == 1
                           and tokens[i].isalpha()):
                        i += 1
                    run_len = i - run_start
                    if run_len >= 3:
                        word = "".join(tokens[run_start:i])
                        out_tokens.append(word)
                        reassembled = True
                        max_run = max(max_run, run_len)
                    else:
                        out_tokens.extend(tokens[run_start:i])
                else:
                    out_tokens.append(tokens[i])
                    i += 1
            rebuilt_segments.append(" ".join(out_tokens))
        out_lines.append("".join(rebuilt_segments))
    text = "\n".join(out_lines)

    # --- Pass 2: Dot-separated single alpha chars ---
    def _dot_replace(m):
        nonlocal reassembled, max_run
        matched = m.group(1)
        # Check if it's a known abbreviation (with or without trailing dot)
        lower_with_dot = matched.lower() + "."
        lower_no_dot = matched.lower()
        if (lower_with_dot in _ABBREVIATION_ALLOWLIST
                or lower_no_dot in _ABBREVIATION_ALLOWLIST):
            return matched
        # Reassemble: remove dots
        word = matched.replace(".", "")
        reassembled = True
        max_run = max(max_run, len(word))
        return word

    text = _DOT_CHAR_SPLIT_RE.sub(_dot_replace, text)

    # --- Pass 3: Generic consistent-separator single alpha chars ---
    def _generic_replace(m):
        nonlocal reassembled, max_run
        # The match is single-alpha chars joined by a consistent separator;
        # strip everything that is not a letter to recover the word.
        word = "".join(c for c in m.group(0) if c.isalpha())
        reassembled = True
        max_run = max(max_run, len(word))
        return word

    text = _GENERIC_CHAR_SPLIT_RE.sub(_generic_replace, text)

    # --- Pass 4: Vertical (newline-stacked) single alpha chars ---
    # Collapse runs of 4+ consecutive lines that are each a single alpha
    # char (e.g. an instruction stacked one letter per line).
    lines = text.split("\n")
    if len(lines) >= 4:
        rebuilt = []
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            if len(stripped) == 1 and stripped.isalpha():
                run_start = i
                while (i < len(lines)
                       and len(lines[i].strip()) == 1
                       and lines[i].strip().isalpha()):
                    i += 1
                run_len = i - run_start
                if run_len >= 4:
                    word = "".join(ln.strip() for ln in lines[run_start:i])
                    rebuilt.append(word)
                    reassembled = True
                    max_run = max(max_run, run_len)
                else:
                    rebuilt.extend(lines[run_start:i])
            else:
                rebuilt.append(lines[i])
                i += 1
        text = "\n".join(rebuilt)

    if reassembled:
        logger.debug("char-level reassembly applied (max_run=%d)", max_run)

    return text, reassembled, max_run


# ---------------------------------------------------------------------------
# Configurable thresholds (named constants, env-overridable)
# ---------------------------------------------------------------------------

def _safe_float_env(name, default, lo=0.0, hi=1.0):
    """Read a float from env, clamping to [lo, hi]. Falls back to *default*."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        val = float(raw)
    except (ValueError, TypeError):
        return default
    if not math.isfinite(val):
        return default
    if val < lo or val > hi:
        return default
    return val


def _safe_int_env(name, default, lo=0, hi=None):
    """Read an int from env, clamping to [lo, hi]. Falls back to *default*."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        val = int(raw)
    except (ValueError, TypeError):
        return default
    if val < lo:
        return default
    if hi is not None and val > hi:
        return default
    return val


# Fraction of original characters that must be compatibility-form (NFKC-
# decomposable) before the ``nfkc_changed`` flag is raised.  Low values
# fire on normal smart quotes / superscripts; high values miss evasion.
# Default 0.25 (25%).  Override: L0_NFKC_CHANGE_THRESHOLD
_NFKC_CHANGE_THRESHOLD = _safe_float_env(
    "L0_NFKC_CHANGE_THRESHOLD", 0.25, lo=0.0, hi=1.0
)

# Minimum number of invisible / control characters required before the
# ``invisible_chars_found`` flag is raised.  A single zero-width space
# from copy-paste is normal; a cluster indicates evasion.
# Default 2 (flag when count > 2, i.e. >= 3).  Override: L0_INVISIBLE_CHARS_THRESHOLD
_INVISIBLE_CHARS_THRESHOLD = _safe_int_env(
    "L0_INVISIBLE_CHARS_THRESHOLD", 2, lo=0
)


# ---------------------------------------------------------------------------
# Confusable homoglyph mapping (Unicode TR39 / UTS #39)
# ---------------------------------------------------------------------------
# Maps Cyrillic, Greek, and select Armenian characters that are visually
# identical (or near-identical) to Latin characters.  Derived from the
# Unicode Consortium's confusables.txt data file.
#
# DESIGN: Only applied to MIXED-SCRIPT tokens (tokens containing both
# Latin and Cyrillic/Greek/Armenian characters).  Pure non-Latin tokens
# are left untouched to preserve legitimate multilingual text.
#
# WHY NFKC DOESN'T HANDLE THIS: NFKC normalizes compatibility
# decompositions (e.g., fullwidth A -> A, ligature fi -> fi).  Cyrillic
# 'a' (U+0430) is a canonical character, NOT a compatibility form of
# Latin 'a' (U+0061).  They are separate characters in separate scripts
# that happen to look identical.  Unicode explicitly does NOT merge
# cross-script look-alikes in NFC/NFKC because that would destroy
# legitimate Cyrillic/Greek text.

# --- Cyrillic -> Latin confusables ---
_CYRILLIC_TO_LATIN = {
    # Uppercase Cyrillic -> Latin
    "\u0410": "A",   # А -> A
    "\u0412": "B",   # В -> B
    "\u0421": "C",   # С -> C
    "\u0415": "E",   # Е -> E
    "\u041D": "H",   # Н -> H
    "\u0406": "I",   # І -> I  (Ukrainian/Belarusian)
    "\u0408": "J",   # Ј -> J  (Serbian)
    "\u041A": "K",   # К -> K
    "\u041C": "M",   # М -> M
    "\u041E": "O",   # О -> O
    "\u0420": "P",   # Р -> P
    "\u0405": "S",   # Ѕ -> S  (Macedonian)
    "\u0422": "T",   # Т -> T
    "\u0425": "X",   # Х -> X
    "\u04AE": "Y",   # Ү -> Y  (Kazakh/Mongolian)
    # Lowercase Cyrillic -> Latin
    "\u0430": "a",   # а -> a
    "\u0441": "c",   # с -> c
    "\u0435": "e",   # е -> e
    "\u0456": "i",   # і -> i  (Ukrainian і)
    "\u0458": "j",   # ј -> j  (Serbian)
    "\u043E": "o",   # о -> o
    "\u0440": "p",   # р -> p
    "\u0455": "s",   # ѕ -> s  (Macedonian)
    "\u0443": "y",   # у -> y  (Cyrillic у looks like Latin y)
    "\u0445": "x",   # х -> x
    "\u04BB": "h",   # һ -> h  (Bashkir/Kazakh)
    "\u0501": "d",   # ԁ -> d  (Cyrillic Supplement, Komi)
    "\u051B": "q",   # ԛ -> q  (Cyrillic Supplement, Kurdish)
    "\u051D": "w",   # ԝ -> w  (Cyrillic Supplement, Abkhaz)
    # Extended / less common but exploitable
    "\u0454": "e",   # є -> e  (Ukrainian yest, close to epsilon/e)
    "\u0471": "v",   # ѱ -> v  (archaic psi, but rarely used for attack)
    "\u04CF": "l",   # ӏ -> l  (Cyrillic palochka, looks like l or I)
    "\u04C0": "I",   # Ӏ -> I  (Cyrillic palochka uppercase)
}

# --- Greek -> Latin confusables ---
_GREEK_TO_LATIN = {
    # Uppercase Greek -> Latin
    "\u0391": "A",   # Α -> A  (Alpha)
    "\u0392": "B",   # Β -> B  (Beta)
    "\u0395": "E",   # Ε -> E  (Epsilon)
    "\u0396": "Z",   # Ζ -> Z  (Zeta)
    "\u0397": "H",   # Η -> H  (Eta)
    "\u0399": "I",   # Ι -> I  (Iota)
    "\u039A": "K",   # Κ -> K  (Kappa)
    "\u039C": "M",   # Μ -> M  (Mu)
    "\u039D": "N",   # Ν -> N  (Nu)
    "\u039F": "O",   # Ο -> O  (Omicron)
    "\u03A1": "P",   # Ρ -> P  (Rho)
    "\u03A4": "T",   # Τ -> T  (Tau)
    "\u03A5": "Y",   # Υ -> Y  (Upsilon)
    "\u03A7": "X",   # Χ -> X  (Chi)
    # Lowercase Greek -> Latin
    "\u03BF": "o",   # ο -> o  (omicron)
    "\u03B9": "i",   # ι -> i  (iota — in many sans-serif fonts)
    "\u03BA": "k",   # κ -> k  (kappa — close in some fonts)
    "\u03BD": "v",   # ν -> v  (nu — visually identical to v)
    "\u03C1": "p",   # ρ -> p  (rho — descender differs but close)
    "\u03C5": "u",   # υ -> u  (upsilon — close in sans-serif)
    "\u03C7": "x",   # χ -> x  (chi — with descender but close)
}

# --- Armenian -> Latin confusables ---
_ARMENIAN_TO_LATIN = {
    "\u054D": "S",   # Ս -> S
    "\u054F": "T",   # Տ -> T  (close in some fonts)
    "\u0555": "O",   # Օ -> O
    "\u0585": "o",   # օ -> o
    "\u0570": "h",   # հ -> h  (close in some fonts)
    "\u0578": "n",   # ո -> n  (close in some fonts)
    "\u057D": "s",   # ս -> s
    "\u0575": "j",   # յ -> j  (close in some fonts)
}

# Combined mapping — all confusable scripts -> Latin
_CONFUSABLE_TO_LATIN = {}
_CONFUSABLE_TO_LATIN.update(_CYRILLIC_TO_LATIN)
_CONFUSABLE_TO_LATIN.update(_GREEK_TO_LATIN)
_CONFUSABLE_TO_LATIN.update(_ARMENIAN_TO_LATIN)

# Pre-build a frozenset of confusable codepoints for fast O(1) lookup
_CONFUSABLE_CODEPOINTS = frozenset(_CONFUSABLE_TO_LATIN.keys())

# Scripts that contain Latin-confusable characters
_CONFUSABLE_SCRIPTS = frozenset({"Cyrillic", "Greek", "Armenian"})


# ---------------------------------------------------------------------------
# Post-ftfy integrity validation (guards against ftfy #149, #202)
# ---------------------------------------------------------------------------

# Approximate Unicode script ranges for script-injection detection.
# Characters outside these ranges return "Common" (punctuation, symbols, etc.)
_SCRIPT_RANGES = (
    (0x0000, 0x007F, "Latin"),       # Basic Latin
    (0x0080, 0x024F, "Latin"),       # Latin Extended
    (0x0250, 0x02AF, "Latin"),       # IPA Extensions
    (0x0370, 0x03FF, "Greek"),
    (0x0400, 0x04FF, "Cyrillic"),
    (0x0500, 0x052F, "Cyrillic"),    # Cyrillic Supplement
    (0x0530, 0x058F, "Armenian"),
    (0x0590, 0x05FF, "Hebrew"),
    (0x0600, 0x06FF, "Arabic"),
    (0x0700, 0x074F, "Syriac"),
    (0x0900, 0x097F, "Devanagari"),
    (0x3040, 0x309F, "Hiragana"),
    (0x30A0, 0x30FF, "Katakana"),
    (0x3400, 0x9FFF, "CJK"),
    (0xAC00, 0xD7AF, "Hangul"),
    (0xF900, 0xFAFF, "CJK"),
    (0x10000, 0x1007F, "LinearB"),
    (0x1F600, 0x1F9FF, "Emoji"),
)


def _char_script(ch):
    """Return the approximate Unicode script for a character."""
    cp = ord(ch)
    for lo, hi, script in _SCRIPT_RANGES:
        if lo <= cp <= hi:
            return script
    return "Common"


def _script_inventory(text):
    """Return the set of non-Common scripts present in *text*."""
    return {_char_script(ch) for ch in text} - {"Common"}


# ---------------------------------------------------------------------------
# Homoglyph normalization (D5.3 — cross-script confusable detection)
# ---------------------------------------------------------------------------

def _has_mixed_scripts_for_homoglyphs(token):
    """Check if a token mixes Latin with Cyrillic/Greek/Armenian characters.

    Only considers alphabetic characters; digits, punctuation, and symbols
    are ignored.  Returns True if the token contains BOTH Latin letters
    AND letters from a confusable script (Cyrillic, Greek, or Armenian).

    This is the gate that prevents legitimate pure-Cyrillic (e.g., Russian)
    or pure-Greek text from being transliterated.
    """
    has_latin = False
    has_confusable = False
    for ch in token:
        if ch.isalpha():
            script = _char_script(ch)
            if script == "Latin":
                has_latin = True
            elif script in _CONFUSABLE_SCRIPTS:
                has_confusable = True
        if has_latin and has_confusable:
            return True
    return False


def normalize_homoglyphs(text):
    """Normalize Cyrillic/Greek/Armenian homoglyphs in mixed-script tokens.

    Only normalizes tokens that MIX Latin with confusable-script characters.
    Pure Cyrillic/Greek/Armenian tokens are left unchanged (legitimate text).

    Uses whitespace-preserving split so that the original spacing (including
    newlines and tabs) is preserved exactly.

    Parameters
    ----------
    text : str
        The input text (should already be NFKC-normalized).

    Returns
    -------
    tuple of (str, int)
        ``(normalized_text, homoglyph_count)`` where *homoglyph_count* is
        the number of confusable characters that were replaced.
    """
    # Split into tokens and whitespace delimiters for lossless reassembly
    parts = _TOKEN_SPLIT_RE.split(text)
    total_replaced = 0

    for i, part in enumerate(parts):
        # Whitespace delimiters (odd indices) are never modified
        if not part or part.isspace():
            continue
        if _has_mixed_scripts_for_homoglyphs(part):
            new_chars = []
            for ch in part:
                replacement = _CONFUSABLE_TO_LATIN.get(ch)
                if replacement is not None:
                    new_chars.append(replacement)
                    total_replaced += 1
                else:
                    new_chars.append(ch)
            parts[i] = "".join(new_chars)

    return "".join(parts), total_replaced


# ---------------------------------------------------------------------------
# Combining diacritical mark stripping (D5.6 — accent-based evasion)
# ---------------------------------------------------------------------------
# Combining marks (U+0300-U+036F, Unicode category Mn in this block) attach
# to preceding base characters.  NFKC composes them into precomposed forms
# (e.g. o + U+0300 -> U+00F2 ò), but the resulting accented characters are
# DIFFERENT from their unaccented base letters.  Rules matching "ignore"
# will NOT match "ignòre".
#
# To close this evasion gap, we decompose via NFD (splits precomposed chars
# back into base + combining mark), then strip all characters with Unicode
# category Mn (Nonspacing Mark) in the combining diacritical range.  This
# is more targeted than stripping ALL Mn characters, which would also remove
# legitimate diacritics in scripts like Hebrew/Arabic/Devanagari niqqud.
#
# DESIGN: We strip combining marks in the range U+0300-U+036F only.  These
# are the Latin/Cyrillic/Greek combining diacriticals.  Extended combining
# marks (U+0370+) for other scripts are left intact to preserve legitimate
# multilingual text.

def _strip_combining_diacriticals(text):
    """Strip combining diacritical marks (U+0300-U+036F) from text.

    Uses NFD decomposition to split precomposed characters (e.g. ò -> o +
    combining grave), then removes combining marks in the basic diacriticals
    block.  Finally re-applies NFC to re-compose any remaining sequences.

    Parameters
    ----------
    text : str
        Input text (should already be NFKC-normalized).

    Returns
    -------
    tuple of (str, int)
        ``(cleaned_text, marks_stripped)`` where *marks_stripped* is the count
        of combining diacritical marks that were removed.
    """
    # NFD decomposition: splits precomposed chars into base + combining marks
    decomposed = unicodedata.normalize("NFD", text)
    cleaned = []
    marks_stripped = 0
    for ch in decomposed:
        cp = ord(ch)
        # Strip combining diacritical marks (U+0300-U+036F)
        if 0x0300 <= cp <= 0x036F:
            marks_stripped += 1
        else:
            cleaned.append(ch)
    if marks_stripped == 0:
        return text, 0
    # NFC re-composition: re-compose any remaining valid sequences
    result = unicodedata.normalize("NFC", "".join(cleaned))
    return result, marks_stripped


def _validate_ftfy_output(original, fixed):
    """Check whether ftfy's correction is safe.

    Returns True if the correction is acceptable, False if ftfy introduced
    suspicious characters.

    Guards against:
    - ftfy #149: Dutch text producing Pallas symbol (U+26B4)
    - ftfy #202: en-dash mojibake producing Cyrillic (fixed in 6.2, but
      we keep the guard as defense-in-depth)

    Key insight: full-text mojibake repair (e.g., Latin garble → CJK) is
    legitimate and changes the entire script.  Isolated wrong corrections
    (e.g., one Pallas symbol or a few Cyrillic chars in Latin text) are
    suspicious.  We distinguish by checking the RATIO of new-script chars.
    """
    new_chars = set(fixed) - set(original)
    if not new_chars:
        return True

    # 0. Idempotency guard: reject if ftfy's output is NOT stable —
    #    i.e., running ftfy on the fixed text changes it AGAIN.
    #    Real mojibake repair is idempotent (the correctly decoded text
    #    won't be "re-fixed").  False-positive corrections like
    #    "ƒß"→"ħ" or "Ü¢"→"ܢ" are unstable — ftfy would try to
    #    "fix" the output again on a second pass.
    if _HAS_FTFY:
        refixed = _ftfy_fix_with_sentinel(fixed)
        if refixed != fixed:
            return False

    # 0b. Clean-text merge guard: reject if ftfy reduces length AND
    #     the original text has no "garble indicators" — characters
    #     that are typical artifacts of misinterpreted encodings.
    #     Real mojibake always includes at least one of:
    #       Sc (currency: €¤), So (symbols: ™©®), Sk (modifiers: ¯ˆ),
    #       Cc (control: U+0080-U+009F C1 controls), Cf (format),
    #       Cn (unassigned), Cs (surrogate)
    #     False positives like "ƒß"→"ħ" contain only letters, digits,
    #     and standard punctuation — no garble indicators.
    _GARBLE_CATEGORIES = {"Sc", "So", "Sk", "Cc", "Cf", "Cn", "Cs"}
    if len(fixed) < len(original) and not any(
        unicodedata.category(c) in _GARBLE_CATEGORIES for c in original
    ):
        return False

    # 1. Symbol injection: reject if new "Other Symbol" (So) chars appear
    #    when the original had none (catches Pallas symbol U+26B4, etc.)
    orig_has_so = any(unicodedata.category(ch) == "So" for ch in original)
    if not orig_has_so:
        for ch in new_chars:
            if unicodedata.category(ch) == "So":
                return False

    # 2. Partial script injection: reject if a SMALL number of chars from
    #    a new script appear (isolated wrong fix), but allow full-script
    #    changes (legitimate mojibake repair like Latin garble → CJK).
    orig_scripts = _script_inventory(original)
    new_script_chars = [
        ch for ch in new_chars
        if _char_script(ch) not in orig_scripts and _char_script(ch) != "Common"
    ]
    if new_script_chars:
        # Count how many chars in the FIXED text belong to the new script(s)
        new_scripts = {_char_script(ch) for ch in new_script_chars}
        new_script_count = sum(
            1 for ch in fixed if _char_script(ch) in new_scripts
        )
        # If less than 50% of the output is in the new script, it's an
        # isolated injection (suspicious).  Full mojibake repair changes
        # most of the text.
        if len(fixed) > 0 and new_script_count / len(fixed) < 0.5:
            return False
        # Short-text guard: for very short texts (≤4 printable chars),
        # a full script change is almost always a false positive —
        # ftfy misinterprets a couple of Latin/Common chars as an
        # encoded form.  Real mojibake at this length is extremely rare
        # and not worth the false-positive risk.
        orig_printable = sum(1 for c in original if c.isprintable())
        if orig_printable <= 4 and new_script_count > 0:
            return False

    return True


def has_invisible_chars(text):
    """Detect invisible characters, zero-width chars, RTL overrides, surrogates."""
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Cf":  # Format chars (zero-width, RTL override, etc.)
            return True
        if cat == "Cs":  # Lone surrogates — invalid in interchange
            return True
        if cat in ("Cc", "Cn") and char not in "\n\r\t":
            return True
    return False


def _count_invisible_chars(text):
    """Count invisible/control characters that strip_invisible_chars removes.

    Returns the number of characters that would be stripped (Cf, Cs, Cc, Cn
    excluding newlines, carriage returns, tabs, and spaces).
    """
    count = 0
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Cs":
            count += 1
        elif cat in ("Cf", "Cc", "Cn") and char not in "\n\t ":
            count += 1
    return count


def strip_invisible_chars(text):
    """Remove invisible/control Unicode characters. Preserves newlines, tabs.

    Also strips lone surrogates (category Cs) — these are invalid in UTF-8
    interchange and crash downstream encoders (hashlib, tiktoken).

    Word-boundary restoration (two-pass approach):
      Pass 1: Strip all invisible/control characters, producing a clean string.
      Pass 2: Where invisible chars were removed between two groups of 2+
              word-forming characters, insert a single space to restore the
              word boundary that the invisible char was replacing.

    This handles two distinct D5.2 evasion patterns correctly:
      - Per-letter splitting:  "i<ZWSP>g<ZWSP>n<ZWSP>o<ZWSP>r<ZWSP>e" -> "ignore"
        (invisible chars between single characters = intra-word, just strip)
      - Word-boundary hiding: "ignore<ZWSP>all<ZWSP>previous" -> "ignore all previous"
        (invisible chars between multi-char groups = inter-word, insert space)

    The heuristic: if a removed invisible char has >= 2 word-forming characters
    on BOTH sides before the next gap/space/non-word, it was likely a word
    boundary and gets replaced with a space.
    """
    # Build a list of (char, is_visible) pairs to analyze context.
    # First, categorize each character.
    chars_info = []  # list of (char, is_invisible_to_strip)
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Cs":
            chars_info.append((char, True))
        elif cat in ("Cf", "Cc", "Cn") and char not in "\n\t ":
            chars_info.append((char, True))
        else:
            chars_info.append((char, False))

    # Now build the result, deciding whether to insert spaces.
    # Strategy: scan segments between invisible-char gaps.
    # A "segment" is a run of visible characters.
    # If two adjacent segments both have length >= 2 (in word chars),
    # insert a space between them; otherwise just concatenate.
    segments = []
    current_segment = []
    had_invisible_between = False

    for char, is_invisible in chars_info:
        if is_invisible:
            if current_segment:
                segments.append(("".join(current_segment), had_invisible_between))
                current_segment = []
                had_invisible_between = False
            had_invisible_between = True
        else:
            current_segment.append(char)

    if current_segment:
        segments.append(("".join(current_segment), had_invisible_between))

    if not segments:
        return ""

    result_parts = [segments[0][0]]
    for i in range(1, len(segments)):
        seg_text, preceded_by_invisible = segments[i]
        if not preceded_by_invisible:
            result_parts.append(seg_text)
            continue

        prev_seg = segments[i - 1][0]
        # Count trailing word chars in previous segment
        prev_word_len = 0
        for ch in reversed(prev_seg):
            if ch.isalpha() or ch.isdigit():
                prev_word_len += 1
            else:
                break

        # Count leading word chars in current segment
        cur_word_len = 0
        for ch in seg_text:
            if ch.isalpha() or ch.isdigit():
                cur_word_len += 1
            else:
                break

        # Insert space only if both sides have 3+ word chars.
        if prev_word_len >= 3 and cur_word_len >= 3:
            if prev_seg and prev_seg[-1] not in (" ", "\n", "\r", "\t"):
                result_parts.append(" ")
        result_parts.append(seg_text)

    return "".join(result_parts)


def strip_invisible_chars_concat(text):
    """Remove invisible/control characters by simple concatenation.

    Unlike ``strip_invisible_chars()`` which attempts word-boundary
    restoration, this function simply removes all invisible characters
    and concatenates the remaining visible characters.
    """
    result = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Cs":
            continue
        if cat in ("Cf", "Cc", "Cn") and char not in "\n\t ":
            continue
        result.append(char)
    return "".join(result)


def _ftfy_fix_with_sentinel(text):
    """Run ftfy.fix_text with a workaround for issue #222 (string-start bug).

    ftfy's badness heuristic (BADNESS_RE) fails to detect certain mojibake
    patterns at position 0 because several patterns require preceding context
    (e.g., a lowercase letter before the garbled capital letter).  When
    mojibake begins at the very first character, that context is absent.

    Workaround: prepend a single ASCII space so the mojibake is no longer
    at position 0, then strip the sentinel after ftfy processes the text.

    See: https://github.com/rspeer/python-ftfy/issues/222
    """
    if not text:
        return text

    sentinel_added = False
    ftfy_input = text

    if not text[0:1].isspace():
        ftfy_input = " " + text
        sentinel_added = True

    fixed = ftfy.fix_text(ftfy_input, fix_character_width=False)

    if sentinel_added:
        if fixed.startswith(" "):
            fixed = fixed[1:]
        else:
            fixed = fixed.lstrip(" ")

    return fixed


def _count_compat_chars(text):
    """Count characters whose NFKC decomposition differs from themselves.

    This per-character check avoids false positives from positional shift
    (e.g. a ligature expanding fi→fi shifts all later positions).
    """
    count = 0
    for ch in text:
        if unicodedata.normalize("NFKC", ch) != ch:
            count += 1
    return count


def _extract_tag_stego(text):
    """Extract hidden ASCII from Unicode Tag Characters (U+E0001-U+E007F).

    Unicode Tag Characters map 1:1 to ASCII via ``chr(codepoint - 0xE0000)``.
    Attackers embed invisible instructions (e.g. "ignore all rules") as tag
    characters that are invisible in rendered text but processed by LLMs.

    Must be called BEFORE ``strip_invisible_chars()`` because tag chars have
    Unicode category Cf and would be silently removed, losing the hidden
    payload forever.

    References:
    - Cisco: Understanding and Mitigating Unicode Tag Prompt Injection
    - AWS: Defending LLM Applications Against Unicode Character Smuggling
    - Trend Micro: Invisible Prompt Injection (Jan 2025)
    - HackerOne #2372363: Invisible Prompt Injection via Unicode Tags

    Returns
    -------
    str
        The decoded ASCII message, or empty string if no tag chars found.
    """
    decoded = []
    for ch in text:
        cp = ord(ch)
        if 0xE0001 <= cp <= 0xE007F:
            decoded.append(chr(cp - 0xE0000))
    return "".join(decoded) if decoded else ""


def _extract_variation_selector_stego(text):
    """Extract hidden data from Variation Selector steganography.

    Variation Selectors (VS1-VS16: U+FE00-U+FE0F, VS17-VS256:
    U+E0100-U+E01EF) are invisible Unicode category Mn (Nonspacing Mark)
    characters.  The "Sneaky Bits" technique maps each byte (0-255) to one
    of the 256 variation selectors:

        byte 0-15   -> U+FE00 + byte      (VS1-VS16)
        byte 16-255 -> U+E0100 + (byte-16) (VS17-VS256)

    Attackers embed these after emoji or other base characters to hide
    arbitrary payloads (shell commands, prompt injections) that are
    invisible in rendered text but survive copy-paste and LLM tokenisation.

    Must be called BEFORE ``strip_invisible_chars()`` because once stripped,
    the hidden payload would be lost.  VS chars have Unicode category Mn
    which is NOT caught by the Cf/Cc/Cn/Cs filter in strip_invisible_chars.

    References:
    - Dawid Rylko: "Hiding Data in Emoji" (Unicode Stego via VS)
    - Veracode: NPM os-info-checker-es6 attack using VS steganography
    - Unicode Consortium: Variation Selectors chart (U+FE00-FE0F)

    Returns
    -------
    str
        Decoded text from variation selectors, or empty string if fewer
        than 1 byte could be decoded.
    """
    vs_codepoints = []
    for ch in text:
        cp = ord(ch)
        if 0xFE00 <= cp <= 0xFE0F or 0xE0100 <= cp <= 0xE01EF:
            vs_codepoints.append(cp)

    if not vs_codepoints:
        return ""

    # Decode: reverse the byte-to-VS mapping
    decoded_bytes = []
    for cp in vs_codepoints:
        if 0xFE00 <= cp <= 0xFE0F:
            decoded_bytes.append(cp - 0xFE00)        # byte 0-15
        else:
            decoded_bytes.append(cp - 0xE0100 + 16)   # byte 16-255

    # Try to decode as UTF-8 text; fall back to latin-1 for raw bytes
    raw = bytes(decoded_bytes)
    try:
        decoded_text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        decoded_text = raw.decode("latin-1")

    # Filter to printable content (keep ASCII printable + common whitespace)
    printable = []
    for ch in decoded_text:
        if ch in ("\n", "\r", "\t") or (0x20 <= ord(ch) <= 0x7E):
            printable.append(ch)
    return "".join(printable)


def _strip_variation_selectors(text):
    """Remove all variation selector characters from text.

    Strips both ranges:
    - VS1-VS16:   U+FE00 - U+FE0F   (Basic Multilingual Plane)
    - VS17-VS256: U+E0100 - U+E01EF  (Supplementary)

    Returns the cleaned text with all variation selectors removed.
    """
    return "".join(
        ch for ch in text
        if not (0xFE00 <= ord(ch) <= 0xFE0F or 0xE0100 <= ord(ch) <= 0xE01EF)
    )


def normalize_text(text, _idempotency_pass=False):
    """Run all Layer 0 normalization steps in order.

    Returns (normalized_text, chars_stripped, anomaly_flags).
    """
    flags = []
    original_len = len(text)

    # Step 0: Mojibake repair via ftfy (before NFKC)
    # Fixes encoding errors like UTF-8 decoded as latin-1:
    #   "â€™" → "'"   "Ã©" → "é"   "â€œhelloâ€\x9d" → ""hello""
    # Uses sentinel workaround for ftfy #222 (mojibake at position 0).
    # Post-fix validation guards against ftfy #149/#202 (wrong corrections).
    if _HAS_FTFY:
        fixed = _ftfy_fix_with_sentinel(text)
        if fixed != text:
            if _validate_ftfy_output(text, fixed):
                flags.append("mojibake_repaired")
                text = fixed
            else:
                # ftfy produced a suspicious correction (new scripts or
                # symbols not in original).  Revert and flag for review.
                # See: ftfy #149 (Pallas symbol), #202 (Cyrillic injection)
                flags.append("ftfy_suspicious_correction")

    # Step 1: NFKC normalization
    # Collapses fullwidth chars, ligatures, superscripts, compatibility forms
    compat_count = _count_compat_chars(text)
    text = unicodedata.normalize("NFKC", text)
    # Only flag if more than _NFKC_CHANGE_THRESHOLD of original chars are
    # compatibility forms — ligatures from Word, superscripts in math (x²),
    # smart quotes are all normal.  A wall of fullwidth chars (evasion)
    # typically hits 80%+.
    if compat_count > 0 and compat_count / max(original_len, 1) > _NFKC_CHANGE_THRESHOLD:
        flags.append("nfkc_changed")

    # Step 1.5: Cross-script homoglyph normalization (D5.3)
    # Cyrillic/Greek/Armenian characters that are visually identical to Latin
    # are normalized to their Latin equivalents, but ONLY in mixed-script
    # tokens.  Pure non-Latin tokens are preserved (legitimate multilingual
    # text).  This closes the D5.3 bypass where NFKC cannot help because
    # these are canonical characters, not compatibility forms.
    text, homoglyph_count = normalize_homoglyphs(text)
    if homoglyph_count > 0:
        flags.append("mixed_script_homoglyphs")

    # Step 1.7: Combining diacritical mark detection (D5.6)
    _pre_diacritics = text
    _check_text, diacritics_count = _strip_combining_diacriticals(text)
    if diacritics_count > 0:
        flags.append("combining_diacritics_stripped")

    # Step 1.9: Unicode Tag Character steganography extraction (D5.2+)
    # MUST run BEFORE Step 2 (invisible char stripping) because tag chars
    # have Unicode category Cf and would be silently destroyed.
    # The decoded payload is appended to the text so downstream layers
    # (L1 rules, L2 obfuscation, ML) automatically scan the hidden message.
    tag_stego_text = _extract_tag_stego(text)
    if tag_stego_text:
        flags.append("unicode_tag_stego")
        # Append decoded payload so downstream layers can detect it.
        # The visible text + hidden payload are separated by a newline.
        text = text + "\n" + tag_stego_text

    # Step 1.95: Variation Selector steganography extraction (D5.2+)
    # MUST run BEFORE Step 2 (invisible char stripping) to capture the
    # hidden payload.  VS chars are Unicode category Mn (Nonspacing Mark),
    # which is NOT stripped by the Cf/Cc/Cn/Cs filter, but extracting
    # early ensures no data loss regardless of future filter changes.
    # The decoded payload is appended to the text so downstream layers
    # (L1 rules, L2 obfuscation, ML) automatically scan the hidden message.
    vs_stego_text = _extract_variation_selector_stego(text)
    if vs_stego_text:
        flags.append("variation_selector_stego")
    # Always strip variation selectors from text (they are invisible noise)
    text = _strip_variation_selectors(text)

    # If VS stego decoded a hidden message, append it for downstream scanning
    if vs_stego_text:
        text = text + "\n" + vs_stego_text

    # Step 2: Invisible character stripping
    if has_invisible_chars(text):
        # Count actual invisible chars before stripping.  Cannot use
        # length difference because strip_invisible_chars() may INSERT
        # spaces at word boundaries, offsetting the count.
        invisible_count = _count_invisible_chars(text)
        text = strip_invisible_chars(text)
        # Only flag if more than _INVISIBLE_CHARS_THRESHOLD invisible chars
        # — a single zero-width space from copy-paste is normal; a cluster
        # of them is evasion
        if invisible_count > _INVISIBLE_CHARS_THRESHOLD:
            flags.append("invisible_chars_found")

    # Step 3: Whitespace canonicalization
    # Normalize carriage returns first to prevent parser differentials
    text = _CR_NORMALIZE_RE.sub("\n", text)

    # Replace Unicode whitespace variants with ASCII space
    cleaned, count = _UNICODE_WHITESPACE_RE.subn(" ", text)
    if count > 0:
        flags.append("unicode_whitespace_normalized")
        text = cleaned

    # Step 3.5: Character-level reassembly (anti-evasion)
    # Detect and reassemble words split into individual characters via
    # spaces ("i g n o r e") or dots ("i.g.n.o.r.e").  Must run BEFORE
    # multi-space collapse so that double-space word boundaries are
    # preserved for accurate reassembly.
    text, char_reassembled, char_run = _reassemble_char_splits(text)
    if char_reassembled and char_run >= _CHAR_SPLIT_MIN_SCORED:
        # Emit the SCORED flag only for a run >= 4 (a bare 3-char run is too
        # weak and fires on incidental benign letter-sequences).  The text is
        # still reassembled above regardless, so rules see the de-split word.
        flags.append("char_level_reassembly")
        # Grade severity: a long single-char run is essentially never benign,
        # so tag it for the scorer's heavy-obfuscation floor.
        if char_run >= _CHAR_SPLIT_HEAVY_RUN:
            flags.append("char_level_reassembly_heavy")

    # Collapse multiple spaces into one, strip leading/trailing
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    # Collapse excessive newlines and tabs (prevents padding attacks)
    text = _EXCESSIVE_NEWLINES_RE.sub("\n\n", text)
    text = _EXCESSIVE_TABS_RE.sub("\t", text)

    chars_stripped = original_len - len(text)

    # Idempotency guard: earlier steps (invisible-char stripping, NFKC,
    # etc.) can create character sequences that ftfy misinterprets as
    # mojibake on a subsequent call.  We detect this by running the
    # pipeline once more.  If the second pass changes the text, we
    # return the second-pass result (which is guaranteed to be stable
    # because we only do one re-run).
    if not _idempotency_pass and _HAS_FTFY:
        text2, _, flags2 = normalize_text(text, _idempotency_pass=True)
        if text2 != text:
            # Second pass changed the text — use the stable result.
            # Merge flags from both passes (deduplicated).
            seen = set(flags)
            for f in flags2:
                if f not in seen:
                    flags.append(f)
                    seen.add(f)
            text = text2
            chars_stripped = original_len - len(text)

    return text, chars_stripped, flags


_LITERAL_UNICODE_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")


def _decode_literal_escapes(text):
    """Decode literal \\uXXXX escape sequences into actual Unicode characters.

    Attackers embed literal escape sequences like ``Ign\\u200bore`` to evade
    keyword matching.  The text contains the ASCII characters ``\\``, ``u``,
    ``2``, ``0``, ``0``, ``b`` — not the actual zero-width space.

    Returns ``(decoded_text, n_decoded)`` where *n_decoded* is the count of
    escape sequences replaced.
    """
    count = 0

    def _replace(m):
        nonlocal count
        count += 1
        return chr(int(m.group(1), 16))

    decoded = _LITERAL_UNICODE_ESCAPE.sub(_replace, text)
    return decoded, count


def quick_normalize_concat(text):
    """Fast concat-based normalization for use as an additional rule surface.

    Produces a normalized view of *text* where invisible characters are
    stripped by simple concatenation (no space insertion at removal
    points).  This complements ``normalize_text()``'s heuristic
    word-boundary restoration which can incorrectly split words at
    syllable boundaries (e.g. "Ign<ZWS>ore" -> "Ign ore").

    Steps applied (subset of ``normalize_text``):
      1. NFKC normalization
      2. Combining diacritical mark stripping (U+0300-U+036F)
      3. Invisible character removal (concat, no space insertion)
      4. Whitespace canonicalization (braille blank, exotic spaces)
      5. Multi-space collapse

    This is intentionally lightweight: no ftfy, no stego extraction,
    no homoglyph normalization (those are already handled by the
    primary ``normalize_text`` pipeline).  Its purpose is to provide
    a clean rule-matching surface for cases where the heuristic
    space-insertion creates false word breaks.

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    str
        Normalized text with invisible chars stripped by concatenation.
    """
    # Decode literal \uXXXX escape sequences (evasion via ASCII escapes)
    text, _ = _decode_literal_escapes(text)
    # NFKC normalization
    text = unicodedata.normalize("NFKC", text)
    # Strip combining diacritical marks
    text, _ = _strip_combining_diacriticals(text)
    # Strip invisible chars by concatenation (no space insertion)
    text = strip_invisible_chars_concat(text)
    # Whitespace canonicalization (braille blank, exotic spaces -> ASCII space)
    text = _UNICODE_WHITESPACE_RE.sub(" ", text)
    # Collapse multiple spaces and strip
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text
