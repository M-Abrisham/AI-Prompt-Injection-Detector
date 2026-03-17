# D5 Unicode Evasion -- Deep Research Notes

## Research Date: 2026-02-17

## Layer 0 Normalization Pipeline (normalization.py)

The normalization pipeline runs three steps in order:
1. **NFKC normalization** -- collapses fullwidth, math alphanumeric, ligatures, superscripts
2. **Invisible char stripping** -- removes category Cf (ZWSP, RTL override, etc.) and Cs (surrogates)
3. **Whitespace canonicalization** -- replaces Unicode whitespace (Zs category) with ASCII space

### What NFKC handles (effectively detected):
- Fullwidth Latin (U+FF00+) -> ASCII
- Mathematical Bold (U+1D400+) -> ASCII
- Mathematical Italic (U+1D434+) -> ASCII
- Mathematical Fraktur (U+1D504+) -> ASCII
- Most Mathematical Script (U+1D49C+) -> ASCII (with some exceptions)
- Ligatures (fi, fl, etc.) -> decomposed
- Superscripts/subscripts -> base chars

### What NFKC does NOT handle:
- **Cyrillic homoglyphs** -- canonical characters, not compatibility forms
  - U+043E (o), U+0435 (e), U+0430 (a), U+0456 (i) survive NFKC unchanged
- **Combining diacritics** -- compose with base char, not stripped
  - i + U+0300 -> U+00EC (i-grave), NOT stripped to plain 'i'
- **Some Math Script letters** that have preexisting codepoints (e.g., script B = U+212C)

### What invisible char stripping handles:
- ZWSP (U+200B), ZWNJ (U+200C), ZWJ (U+200D) -- all category Cf
- RTL Override (U+202E), Pop Directional (U+202C) -- category Cf
- RTL Mark (U+200F), LTR Mark (U+200E) -- category Cf
- BOM (U+FEFF) -- category Cf
- Lone surrogates (category Cs)

### Anomaly flags produced:
- `nfkc_changed` -- when >25% of chars are compatibility forms (catches fullwidth, math)
- `invisible_chars_found` -- when >2 invisible chars stripped
- `unicode_whitespace_normalized` -- when any Unicode whitespace replaced

## D5 Sub-technique Detection Matrix

| Sub-technique | Defense Layer | Detection Rate | Anomaly Flags | Notes |
|---|---|---|---|---|
| D5.1 Fullwidth | NFKC | 100% | nfkc_changed | Normalizes to ASCII, then rules+ML catch payload |
| D5.2 Zero-width | Cf stripping | 100% | invisible_chars_found | Chars removed, words concatenated or rejoined |
| D5.3 Cyrillic | Language det. | ~60% | mixed_language_input, non_english_input | Light sub detected; heavy sub defeats ML (GAP) |
| D5.4 Math Bold/Italic | NFKC | 100% | (none unless >25%) | NFKC maps to ASCII reliably |
| D5.4 Math Script | NFKC | ~80% | varies | Some script letters have normalization gaps |
| D5.5 RTL Override | Cf stripping | 100% | invisible_chars_found | Control chars stripped, underlying text exposed |
| D5.6 Combining | NFKC compose | ~90% | (none) | Composed chars differ from plain; ML still catches |
| D5.6 Zalgo | high_entropy | 100% | high_entropy | Heavy stacking triggers obfuscation scanner |
| D5.7 Whitespace | Regex replace | 100% | unicode_whitespace_normalized | All Zs-category chars replaced with ASCII space |

## Key Security Gap: D5.3 Cyrillic Homoglyphs

**Problem**: When an attacker replaces many Latin chars with Cyrillic homoglyphs:
- NFKC normalization has no effect (Cyrillic chars are canonical)
- TF-IDF vectorizer sees unknown tokens (Cyrillic-containing words)
- ML model loses classification signal
- Only the language_detector's mixed_language_input flag fires
- The weighted_decision trusts ML confidence when it says "safe" with >0.8

**Recommended fix**: Add a Unicode confusable normalization step to Layer 0:
1. Use Unicode UTS #39 skeleton algorithm (confusables.txt)
2. Map Cyrillic/Greek/Armenian homoglyphs to Latin equivalents
3. Fire a new `confusable_chars_normalized` anomaly flag
4. Implement BEFORE NFKC normalization in the pipeline

**Python implementation options**:
- `confusable_homoglyphs` library (pip install confusable-homoglyphs)
- Manual mapping table for most common homoglyphs (faster, no dependency)
- ICU-based detection via `icu` or `PyICU` library

## Attack Sources and References

1. HackerOne #2372363 -- Invisible Prompt Injection on Hai using Unicode tags (U+E0000-E007F)
2. Meta LLaMA Issue #1382 -- Homoglyph substitution bypass of safety filters
3. Mindgard -- Azure AI Content Safety bypass via zero-width characters (Feb 2024)
4. Cisco/Robust Intelligence -- Unicode Tag Prompt Injection mitigation guide
5. AWS Security Blog -- Defending LLM apps against Unicode character smuggling
6. Embrace The Red -- ASCII Smuggling for prompt injection (2024-2025)
7. arXiv 2504.11168 -- Bypassing LLM Guardrail Jailbreak Detection (Apr 2025)
8. Unit42 (Palo Alto) -- The Homograph Illusion
9. AppCheck -- Unicode Normalization Vulnerabilities (Special K Polyglot)
10. Promptfoo -- Invisible Unicode Threats in AI-Generated Code
11. UAX #15 -- Unicode Normalization Forms specification
12. UTS #39 -- Unicode Security Mechanisms (confusables detection)

## Unicode Tag Characters (U+E0000-E007F) -- ASCII Smuggling

Not tested in D5 tests because these are a distinct category (more like invisible
prompt injection than Unicode evasion). Key facts:
- Tags U+E0001-E007E map to ASCII 0x01-0x7E (invisible versions of ASCII)
- LLM tokenizers reconstruct the ASCII text from tag characters
- Human reviewers see nothing (characters are invisible)
- Layer 0 should strip these as category Cf (they are format characters)
- Riley Goodside disclosed this technique in January 2024
- Used in HackerOne #2372363 to manipulate Hai's severity assessment
