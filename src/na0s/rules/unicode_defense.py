"""Unicode evasion defenses — homoglyph folding and Zalgo stripping.

Protects L1 rules from bypass via:
- Angle bracket look-alikes (12 Unicode variants → ASCII < >)
- Zalgo text (stacked combining marks stripped to reveal hidden text)
"""

import unicodedata

# ---------------------------------------------------------------------------
# Angle-bracket homoglyph folding
# ---------------------------------------------------------------------------
# Unicode has 12+ characters that LOOK like < and > but are different
# codepoints, allowing attackers to write ＜system＞ or 〈system〉 to
# bypass rules that only match ASCII < and >.
#
# We fold all visual equivalents to ASCII before running rules.
# This protects: xml_role_tags, fake_system_prompt, chat_template_injection.
# ---------------------------------------------------------------------------
_LEFT_ANGLE_HOMOGLYPHS = (
    "\u3008",  # 〈 LEFT ANGLE BRACKET
    "\uFF1C",  # ＜ FULLWIDTH LESS-THAN SIGN
    "\u2039",  # ‹ SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    "\u276C",  # ❬ MEDIUM LEFT-POINTING ANGLE BRACKET ORNAMENT
    "\u27E8",  # ⟨ MATHEMATICAL LEFT ANGLE BRACKET
    "\uFE64",  # ﹤ SMALL LESS-THAN SIGN
)
_RIGHT_ANGLE_HOMOGLYPHS = (
    "\u3009",  # 〉 RIGHT ANGLE BRACKET
    "\uFF1E",  # ＞ FULLWIDTH GREATER-THAN SIGN
    "\u203A",  # › SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    "\u276D",  # ❭ MEDIUM RIGHT-POINTING ANGLE BRACKET ORNAMENT
    "\u27E9",  # ⟩ MATHEMATICAL RIGHT ANGLE BRACKET
    "\uFE65",  # ﹥ SMALL GREATER-THAN SIGN
)
_ANGLE_FOLD_TABLE = str.maketrans(
    "".join(_LEFT_ANGLE_HOMOGLYPHS) + "".join(_RIGHT_ANGLE_HOMOGLYPHS),
    "<" * len(_LEFT_ANGLE_HOMOGLYPHS) + ">" * len(_RIGHT_ANGLE_HOMOGLYPHS),
)


def _fold_angle_homoglyphs(text: str) -> str:
    """Fold Unicode angle bracket look-alikes to ASCII < and >."""
    return text.translate(_ANGLE_FOLD_TABLE)


def _strip_combining_marks(text: str) -> str:
    """Strip combining diacritical marks (Zalgo text defense).

    Zalgo text uses stacked combining marks (U+0300–U+036F, etc.) on
    each letter, creating visually distorted text that bypasses regex
    rules.  After NFKC normalization, some combining marks are composed
    into precomposed characters (e.g., i + U+0300 -> U+00EC), but
    additional stacked marks remain as separate codepoints.

    This function decomposes to NFD (canonical decomposition), strips
    all combining marks (Unicode category 'M'), then recomposes to NFC.
    The result is plain ASCII-like text that rules can match.

    Example: "ì́̂̃ḡ̅n̆̇ö̉r̊̋ě̍" -> "ignore"
    """
    # NFD decompose -> filter out combining marks -> NFC recompose
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed
                       if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", stripped)
