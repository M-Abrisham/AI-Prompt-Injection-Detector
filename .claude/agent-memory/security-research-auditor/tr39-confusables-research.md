# Unicode TR39 Confusables -- Deep Research Report for Na0S

## Research Date: 2026-02-17
## Researcher: Security Research Auditor Agent

---

## Table of Contents
1. Unicode TR39 Specification Analysis
2. The Skeleton Algorithm
3. Mixed-Script Detection
4. confusables.txt Data Analysis
5. Python Library Evaluation
6. Real-World Attack Patterns
7. Design Considerations for Na0S
8. Outside-the-Box Approaches
9. Recommended Approach
10. Critical Cyrillic-Latin Mappings
11. False Positive Mitigation
12. Implementation Specification

---

## 1. Unicode TR39 -- Unicode Security Mechanisms

### Overview
Unicode Technical Standard #39 (UTS #39) defines security mechanisms for Unicode
implementations. It addresses four main security concerns:

1. **Confusable Detection** -- identifying characters that look identical or
   nearly identical across different scripts
2. **Mixed-Script Detection** -- identifying strings that combine characters
   from multiple scripts (a strong indicator of spoofing)
3. **Identifier Restriction Levels** -- defining increasingly restrictive
   policies for which characters are allowed in identifiers
4. **Security Profile** -- combining the above into actionable policies

### Key Terminology
- **Confusable**: Two characters that are visually similar or identical in
  common fonts (e.g., Cyrillic 'a' U+0430 and Latin 'a' U+0061)
- **Whole-script confusable**: An entire string in one script that looks
  identical to a string in another script (e.g., "pear" in Latin vs
  "реar" with Cyrillic р/е)
- **Mixed-script confusable**: A string mixing scripts where individual
  characters are confusable across scripts (the attack vector we care about)
- **Skeleton**: A normalized form of a string where all confusable characters
  are mapped to a canonical representative

### Specification Versions
- Current: Unicode 16.0 (September 2024)
- confusables.txt updated with each Unicode version
- The algorithm is stable -- the core logic has not changed since UTS #39 v1

---

## 2. The Skeleton Algorithm

### Definition (from UTS #39 Section 4)

The skeleton function maps a string to a canonical form such that two strings
that are confusable will have the same skeleton. The algorithm:

```
skeleton(input):
    1. Apply NFD normalization to the input
    2. For each character in the result:
       a. Look up the character in confusables.txt
       b. If found, replace it with its mapping (prototype)
       c. If not found, keep the character as-is
    3. Apply NFD normalization again to the result
    return result
```

### Critical Detail: NFD not NFKC
The skeleton function uses NFD (Canonical Decomposition), NOT NFKC. This is
important because:
- NFD decomposes precomposed characters (e.g., e-acute) into base + combining
- NFD does NOT collapse compatibility forms (that is what NFKC does)
- The confusables mapping itself handles the visual equivalence
- Using NFD ensures combining marks are decomposed before lookup

### How Skeleton Comparison Works
Two strings are confusable if and only if:
```
skeleton(string1) == skeleton(string2)
```

For example:
- skeleton("ignore") = "ignore" (all Latin, no mapping needed)
- skeleton("ignоre") where о=U+043E:
  - NFD("ignоre") = "ignоre" (nothing to decompose)
  - Map U+043E -> U+006F (confusables.txt maps Cyrillic о to Latin o)
  - NFD result = "ignore"
  - Result: skeleton("ignоre") == skeleton("ignore") -- CONFUSABLE!

### Na0S Relevance
We do NOT need the full skeleton comparison (which compares two strings).
We need the MAPPING step: given a mixed-script word, replace confusable
characters with their Latin prototypes. This is the inner loop of skeleton()
without the comparison wrapper.

---

## 3. Mixed-Script Detection

### UTS #39 Section 5: Mixed-Script Detection

A string has a **single script** if all its characters belong to the same
script or to Common/Inherited categories. Mixed-script strings are suspicious.

### Script Categories
- **Common**: Punctuation, digits, symbols (shared by all scripts)
- **Inherited**: Combining marks that inherit the script of their base char
- **Specific scripts**: Latin, Cyrillic, Greek, Armenian, etc.

### Restriction Levels (from most to least restrictive)
1. **ASCII-Only**: Only U+0000-U+007F
2. **Single Script**: All characters from one script (+ Common/Inherited)
3. **Highly Restrictive**: Allows script mixing only with Han+Bopomofo,
   Han+Hiragana+Katakana, etc. (legitimate CJK mixing)
4. **Moderately Restrictive**: Allows Latin+Han, Latin+CJK families
5. **Minimally Restrictive**: Allows any script mixing

### The Key Insight for Na0S
**Mixed-script words are almost always attacks.** Legitimate multilingual text
uses complete words from each language -- a German-English text will have
complete German words and complete English words. A word containing both
Latin 'g' and Cyrillic 'о' is either a typo or an attack.

The existing `_has_mixed_scripts()` in `language_detector.py` checks at the
TEXT level (the entire input). We need to check at the WORD level: does a
single word contain characters from multiple scripts?

---

## 4. confusables.txt Data Analysis

### File Format
The file is hosted at: https://www.unicode.org/Public/security/latest/confusables.txt

Format of each line:
```
SOURCE_CODEPOINT ; TARGET_SEQUENCE ; TYPE
```

Where:
- SOURCE_CODEPOINT: The confusable character (hex codepoint)
- TARGET_SEQUENCE: What it looks like (the "prototype", one or more codepoints)
- TYPE: "MA" (mapping type, formerly more categories existed)

Example entries:
```
0430 ;  0061 ;  MA  # ( а → a ) CYRILLIC SMALL LETTER A → LATIN SMALL LETTER A
0435 ;  0065 ;  MA  # ( е → e ) CYRILLIC SMALL LETTER IE → LATIN SMALL LETTER E
043E ;  006F ;  MA  # ( о → o ) CYRILLIC SMALL LETTER O → LATIN SMALL LETTER O
0456 ;  0069 ;  MA  # ( і → i ) CYRILLIC SMALL LETTER BYELORUSSIAN-UKRAINIAN I → LATIN SMALL LETTER I
0441 ;  0063 ;  MA  # ( с → c ) CYRILLIC SMALL LETTER ES → LATIN SMALL LETTER ES → LATIN SMALL LETTER C
0440 ;  0070 ;  MA  # ( р → p ) CYRILLIC SMALL LETTER ER → LATIN SMALL LETTER P
0445 ;  0078 ;  MA  # ( х → x ) CYRILLIC SMALL LETTER HA → LATIN SMALL LETTER X
```

### Size Statistics (Unicode 16.0)
- Total entries in confusables.txt: ~6,000+ mappings
- Cyrillic -> Latin mappings: ~60-80 entries (including uppercase)
- Greek -> Latin mappings: ~30-40 entries
- Armenian -> Latin mappings: ~10-15 entries
- Cherokee -> Latin mappings: ~20-30 entries
- Other script -> Latin mappings: Various (Myanmar, Georgian, etc.)
- Total memory for lookup table: ~50-100KB as a Python dict (negligible)

### Most Dangerous Cyrillic -> Latin Confusables

These are the Cyrillic characters that are VISUALLY IDENTICAL to Latin in
standard fonts (not merely "similar" -- truly indistinguishable):

#### Perfect Visual Matches (MUST handle):
| Cyrillic | Codepoint | Latin | Codepoint | Used In |
|----------|-----------|-------|-----------|---------|
| а        | U+0430    | a     | U+0061    | Russian, Ukrainian, all Cyrillic |
| с        | U+0441    | c     | U+0063    | Russian, Ukrainian, all Cyrillic |
| е        | U+0435    | e     | U+0065    | Russian, Ukrainian, all Cyrillic |
| о        | U+043E    | o     | U+006F    | Russian, Ukrainian, all Cyrillic |
| р        | U+0440    | p     | U+0070    | Russian, Ukrainian, all Cyrillic |
| х        | U+0445    | x     | U+0078    | Russian, all Cyrillic |
| у        | U+0443    | y     | U+0079    | Russian, all Cyrillic |
| і        | U+0456    | i     | U+0069    | Ukrainian, Belarusian |
| ј        | U+0458    | j     | U+006A    | Serbian, Macedonian |
| ѕ        | U+0455    | s     | U+0073    | Macedonian |
| ԁ        | U+0501    | d     | U+0064    | Komi (rare) |
| ɡ        | U+0261    | g     | U+0067    | IPA (technically Latin Extended) |
| ᴦ        | U+1D26    | --    | --        | Modifier (less relevant) |

#### Uppercase Perfect Matches:
| Cyrillic | Codepoint | Latin | Codepoint |
|----------|-----------|-------|-----------|
| А        | U+0410    | A     | U+0041    |
| В        | U+0412    | B     | U+0042    |
| С        | U+0421    | C     | U+0043    |
| Е        | U+0415    | E     | U+0045    |
| Н        | U+041D    | H     | U+0048    |
| К        | U+041A    | K     | U+004B    |
| М        | U+041C    | M     | U+004D    |
| О        | U+041E    | O     | U+004F    |
| Р        | U+0420    | P     | U+0050    |
| Т        | U+0422    | T     | U+0054    |
| Х        | U+0425    | X     | U+0058    |

### Other Dangerous Script Pairs

#### Greek -> Latin (Perfect matches):
| Greek    | Codepoint | Latin | Codepoint |
|----------|-----------|-------|-----------|
| ο (omicron) | U+03BF | o     | U+006F    |
| α (alpha)   | U+03B1 | a     | U+0061    | (close but distinguishable in some fonts) |
| Α (Alpha)   | U+0391 | A     | U+0041    |
| Β (Beta)    | U+0392 | B     | U+0042    |
| Ε (Epsilon) | U+0395 | E     | U+0045    |
| Ζ (Zeta)    | U+0396 | Z     | U+005A    |
| Η (Eta)     | U+0397 | H     | U+0048    |
| Ι (Iota)    | U+0399 | I     | U+0049    |
| Κ (Kappa)   | U+039A | K     | U+004B    |
| Μ (Mu)      | U+039C | M     | U+004D    |
| Ν (Nu)      | U+039D | N     | U+004E    |
| Ο (Omicron) | U+039F | O     | U+004F    |
| Ρ (Rho)     | U+03A1 | P     | U+0050    |
| Τ (Tau)     | U+03A4 | T     | U+0054    |
| Χ (Chi)     | U+03A7 | X     | U+0058    |
| ν (nu)      | U+03BD | v     | U+0076    | (close match) |

#### Armenian -> Latin (fewer but present):
| Armenian | Codepoint | Latin | Codepoint |
|----------|-----------|-------|-----------|
| ո        | U+0578    | n     | U+006E    | (close match) |
| ս        | U+057D    | u     | U+0075    | (similar) |

#### Cherokee -> Latin (uppercase especially):
Cherokee has many letters visually identical to Latin uppercase but in a
completely different script block (U+13A0-U+13FF). Less likely to be used
in prompt injection but possible.

---

## 5. Python Library Evaluation

### Option A: `confusable_homoglyphs` (pip install confusable-homoglyphs)

**What it does**: Provides confusable detection using Unicode confusables.txt.
Key API:
- `confusables.is_confusable(text, preferred_aliases=['latin'])` -- returns
  confusable character info or False
- `confusables.is_dangerous(text)` -- checks if string mixes scripts dangerously
- `categories.alias(char)` -- returns Unicode script alias for a character
- `categories.aliases_categories(text)` -- returns all script categories in text

**Maintenance status**:
- Author: Victor Felder (vhf)
- GitHub: vhf/confusable_homoglyphs
- Last significant update: 2022-2023 range
- PyPI: confusable-homoglyphs (note the hyphens)
- Stars: ~500+
- Unicode data bundled (may lag behind latest Unicode version)
- Python 3 compatible

**Pros**:
- Pure Python, no C dependencies
- Small footprint
- Bundles confusables.txt data
- has `is_confusable()` and `is_dangerous()` helpers

**Cons**:
- Returns detection info, not normalized text (we would need to build normalization on top)
- Data may be from an older Unicode version
- No `skeleton()` function -- it is a detection library, not a normalization library
- Performance: iterates character-by-character with dict lookups

**Verdict**: Useful for DETECTION but does not directly solve our NORMALIZATION need.

### Option B: `homoglyphs` (pip install homoglyphs)

**What it does**: Bidirectional homoglyph generation and detection.
Key API:
- `Homoglyphs().to_ascii(text)` -- maps homoglyphs to ASCII
- `Homoglyphs(categories=('CYRILLIC',))` -- filter by script
- Can generate homoglyph variants (useful for testing/fuzzing)

**Maintenance status**:
- Author: Roman Inflianskas (orsinium)
- GitHub: Life4/homoglyphs
- Last update: 2023-ish range
- Stars: ~200+
- Pure Python

**Pros**:
- Has `to_ascii()` which is EXACTLY what we need for normalization
- Can generate confusable variants (useful for test generation)
- Pure Python, no deps

**Cons**:
- Smaller community than confusable_homoglyphs
- Less granular script detection
- Performance characteristics unclear for large texts

**Verdict**: Closer to what we need with `to_ascii()`, but adds a dependency.

### Option C: PyICU (pip install PyICU)

**What it does**: Python bindings for IBM's International Components for Unicode
(ICU) library. Includes the `USpoofChecker` class which implements UTS #39.

Key API:
- `icu.USpoofChecker()` -- create a spoof checker
- `checker.getSkeleton(text)` -- the actual UTS #39 skeleton function
- `checker.setChecks(...)` -- configure check types
- `checker.check(text)` -- run checks (mixed-script, confusable, etc.)

**Maintenance status**:
- Backed by the ICU project (IBM/Unicode Consortium)
- Actively maintained
- Requires C library (libicu) -- system dependency
- Complex installation on some platforms (especially macOS/Windows)
- Always uses latest Unicode data from the installed ICU version

**Pros**:
- THE reference implementation of UTS #39
- `getSkeleton()` is exactly the standard algorithm
- Always up-to-date with the Unicode version of the installed ICU
- Very fast (C implementation)
- Battle-tested in browsers (Chrome, Firefox use ICU)

**Cons**:
- Heavy dependency (C library, ~30MB)
- Installation pain on some platforms (`brew install icu4c` + environment vars)
- Overkill for our targeted use case
- Not in current requirements.txt -- adds deployment complexity
- CI/CD needs system-level package installation

**Verdict**: Gold standard but too heavy. The C dependency conflicts with
Na0S's "pip install and go" philosophy.

### Option D: Build Our Own (RECOMMENDED)

**What we actually need**:
1. A mapping dict: {confusable_codepoint: latin_replacement}
2. A function that checks each character and replaces confusables
3. A function to detect mixed-script WORDS (not just mixed-script text)

This is approximately 50-80 lines of Python code and a ~200-entry mapping dict.
The dict can be extracted from confusables.txt or hardcoded for the ~40 most
dangerous Latin confusables from Cyrillic/Greek.

**Why build our own**:
- Zero additional dependencies
- We control exactly which mappings are included
- We can tune for our specific use case (prompt injection, not IDN)
- We can optimize for our pipeline (per-character in normalize_text)
- We can add the `confusable_chars_normalized` flag seamlessly
- The mapping is stable across Unicode versions (new chars are added, old
  mappings almost never change)
- We can add mixed-script WORD detection (not available in any library as-is)

**The mapping data is trivially small**: ~40 lowercase + ~30 uppercase
Cyrillic mappings, ~20 Greek mappings = ~90 entries total. This is a single
Python dict literal, not even worth a separate data file.

---

## 6. Real-World Attack Patterns

### IDN Homograph Attacks (Browsers)
- First documented by Evgeniy Gabrilovich and Alex Gontmakher (2002)
- Major incident: paypal.com spoofed with Cyrillic а (2005 Shmoo Group demo)
- Chrome mitigation: Shows punycode (xn--) when mixed scripts detected
- Firefox: Uses per-TLD allowlists and mixed-script punycode display
- Both browsers use ICU's USpoofChecker internally

### Prompt Injection via Homoglyphs
1. **Meta LLaMA Issue #1382**: Demonstrated homoglyph substitution bypassing
   LLaMA's safety filters. Cyrillic homoglyphs in "ignore" type phrases.
2. **arXiv 2504.11168 (Apr 2025)**: "Bypassing Prompt Injection and Jailbreak
   Detection in LLM Guardrails" -- explicitly tests Unicode normalization
   bypasses including Cyrillic homoglyphs against Lakera Guard, LLM Guard,
   and NeMo Guardrails.
3. **Unit42 Homograph Illusion (2024)**: Palo Alto Networks research on
   homoglyph attacks in phishing and AI prompt contexts.
4. **Mindgard (2024)**: Demonstrated Azure AI Content Safety bypass using
   zero-width characters and homoglyphs.

### OWASP/MITRE References
- **OWASP LLM Top 10 2025, LLM01 (Prompt Injection)**: Lists "character
  encoding tricks" and "Unicode manipulation" as known bypass techniques.
- **MITRE ATLAS**: TML.T0054 (LLM Prompt Injection) references obfuscation
  techniques including encoding and character substitution.

### Scripts Used for Latin Homoglyphs (by frequency in attacks)
1. **Cyrillic** -- by far the most common (most perfect visual matches)
2. **Greek** -- used for uppercase attacks and specific lowercase (omicron)
3. **Armenian** -- rare but possible
4. **Cherokee** -- uppercase attacks in theory, very rare in practice
5. **Mathematical Symbols** -- handled by NFKC (not confusables)
6. **Enclosed Alphanumerics** -- mostly handled by NFKC

---

## 7. Design Considerations for Na0S

### Question 1: Normalize ALL confusables to Latin, or only detect mixed-script?

**Answer: Both -- but in different ways.**

Strategy:
- **Step 1 (detection)**: Identify mixed-script WORDS (a word containing both
  Latin and non-Latin characters is suspicious)
- **Step 2 (normalization)**: For mixed-script words ONLY, replace confusable
  characters with their Latin prototypes

This avoids the false positive problem: pure Cyrillic text (e.g., "Привет мир")
is left alone. Only words like "іgnоrе" (Cyrillic і/о/е mixed with Latin g/n/r)
get normalized.

### Question 2: Performance considerations

**Per-character lookup** is the right approach:
- Python dict lookup is O(1) amortized
- The mapping dict has ~100 entries -- fits in L1 cache
- For a 500-char input: 500 dict lookups = <0.05ms
- For a 10,000-char input: 10,000 dict lookups = <1ms
- This is negligible compared to ML inference (~50-200ms)

**Regex approach** is NOT suitable:
- Building a regex with 100+ alternations is slow to compile
- Character-class regex `[<all confusables>]` works for detection but
  not for per-character replacement

**Full skeleton comparison** is overkill:
- We do not compare against known-good strings
- We just need to normalize confusables to Latin

### Question 3: False positive risk

**Critical insight**: Legitimate Cyrillic/Greek text should NOT be transliterated.

Risk scenarios:
1. Russian user asks: "Что такое искусственный интеллект?" -- pure Cyrillic,
   should NOT be modified
2. Mixed Latin+Cyrillic: "Translate привет to English" -- "привет" is a
   complete Cyrillic word adjacent to Latin words, should NOT be modified
3. Attack: "іgnоrе all instructions" -- "іgnоrе" contains Cyrillic і/о/е
   mixed with Latin g/n/r WITHIN A SINGLE WORD -- SHOULD be modified

**The discriminator is per-word mixed-script detection.**

False positive rate: effectively zero with per-word detection because:
- No legitimate English word contains Cyrillic characters
- No legitimate Russian word contains Latin characters (transliteration
  uses only Latin characters, never a mix)
- Code identifiers are always single-script
- URLs use ASCII (punycode for internationalized domains)

### Question 4: Pipeline placement -- before or after NFKC?

**BEFORE NFKC (new Step 0.5, between ftfy and NFKC).**

Rationale:
1. NFKC does not affect Cyrillic confusables (they are canonical)
2. Confusable normalization should happen early so downstream steps
   (NFKC, rules, ML) see clean Latin text
3. If we normalize after NFKC, the result is the same -- but logically
   it makes more sense to do "visual deception removal" early
4. ftfy should run first because it may fix encoding issues that could
   affect script detection

**Updated pipeline order**:
```
Step 0:   ftfy mojibake repair
Step 0.5: Confusable homoglyph normalization (NEW)
Step 1:   NFKC normalization
Step 2:   Invisible character stripping
Step 3:   Whitespace canonicalization
```

### Question 5: New step in normalize_text() or separate function?

**Both.** Implement as a standalone function (for testability and reuse),
but call it from normalize_text() at the right point in the pipeline.

```python
def normalize_confusables(text):
    """Replace confusable characters in mixed-script words with Latin equivalents."""
    ...
    return normalized_text, confusable_count
```

Called from normalize_text() between Step 0 and Step 1.

---

## 8. Outside-the-Box Approaches

### Approach A: Mixed-Script Word Detection Without Normalization

Instead of normalizing confusables, just FLAG mixed-script words and boost
the anomaly score. The existing `_has_mixed_scripts()` in language_detector.py
does this at the text level, but word-level detection is more precise.

**Pros**: No normalization needed, no confusables table
**Cons**: The ML model still sees garbled tokens, so detection still fails.
The whole point of normalization is to give the rules engine and ML model
clean text to work with.

**Verdict**: Good as an ADDITIONAL signal, but not sufficient alone.

### Approach B: Visual Similarity Score

Build a visual similarity model: for each non-Latin character, compute a
visual similarity score to its closest Latin equivalent. Flag when the
average similarity exceeds a threshold.

**Pros**: More robust to new confusables not in the mapping
**Cons**: Complex to implement, requires glyph rendering, slow, high
false positive risk. Academic approach, not production-ready.

**Verdict**: Interesting for research but not for Na0S.

### Approach C: Bloom Filter for O(1) Confusable Lookup

Use a Bloom filter containing all known confusable codepoints for O(1)
membership testing.

**Pros**: Very fast, compact
**Cons**: A Python dict with ~100 entries IS already O(1) and uses ~8KB.
A Bloom filter adds complexity for zero performance gain at this scale.
Bloom filters also cannot store the replacement character (only membership).

**Verdict**: Over-engineering. Dict is the right data structure.

### Approach D: Leverage Existing _char_script() Function

The existing `_char_script()` function in normalization.py already classifies
characters by script. We can use it for per-word mixed-script detection:

```python
def _is_mixed_script_word(word):
    scripts = set()
    for ch in word:
        if ch.isalpha():
            s = _char_script(ch)
            if s != "Common":
                scripts.add(s)
    return len(scripts) > 1
```

**Verdict**: YES. This is the foundation of the per-word detection strategy.
No new dependency needed, reuses existing infrastructure.

### Approach E: Two-Pass Strategy

Pass 1: Detect mixed-script words using _char_script()
Pass 2: Only for mixed-script words, apply confusable mapping

This avoids touching pure-script text entirely.

**Verdict**: This IS the recommended approach (see Section 9).

---

## 9. Recommended Approach for Na0S

### Architecture: Build-Our-Own with Two-Pass Strategy

#### Component 1: Confusable Mapping Dict (~90 entries)

A Python dict mapping confusable codepoints to their Latin prototypes.
Covers Cyrillic (lowercase + uppercase) and Greek (uppercase + critical
lowercase). Extracted from confusables.txt for the most dangerous pairs.

```python
# Cyrillic -> Latin (lowercase)
_CONFUSABLE_TO_LATIN = {
    0x0430: 'a',  # а -> a
    0x0441: 'c',  # с -> c
    0x0435: 'e',  # е -> e
    0x043E: 'o',  # о -> o
    0x0440: 'p',  # р -> p
    0x0445: 'x',  # х -> x
    0x0443: 'y',  # у -> y
    0x0456: 'i',  # і -> i  (Ukrainian)
    0x0458: 'j',  # ј -> j  (Serbian)
    0x0455: 's',  # ѕ -> s  (Macedonian)
    0x04BB: 'h',  # һ -> h  (Bashkir)
    0x0501: 'd',  # ԁ -> d  (Komi)
    0x051B: 'q',  # ԛ -> q  (Kurdish Cyrillic)
    # Cyrillic -> Latin (uppercase)
    0x0410: 'A',  # А -> A
    0x0412: 'B',  # В -> B
    0x0421: 'C',  # С -> C
    0x0415: 'E',  # Е -> E
    0x041D: 'H',  # Н -> H
    0x041A: 'K',  # К -> K
    0x041C: 'M',  # М -> M
    0x041E: 'O',  # О -> O
    0x0420: 'P',  # Р -> P
    0x0422: 'T',  # Т -> T
    0x0425: 'X',  # Х -> X
    0x0423: 'Y',  # У -> Y
    # Greek -> Latin (uppercase, perfect matches)
    0x0391: 'A',  # Α -> A
    0x0392: 'B',  # Β -> B
    0x0395: 'E',  # Ε -> E
    0x0396: 'Z',  # Ζ -> Z
    0x0397: 'H',  # Η -> H
    0x0399: 'I',  # Ι -> I
    0x039A: 'K',  # Κ -> K
    0x039C: 'M',  # Μ -> M
    0x039D: 'N',  # Ν -> N
    0x039F: 'O',  # Ο -> O
    0x03A1: 'P',  # Ρ -> P
    0x03A4: 'T',  # Τ -> T
    0x03A7: 'X',  # Χ -> X
    # Greek -> Latin (lowercase, critical matches)
    0x03BF: 'o',  # ο -> o (omicron)
    0x03BD: 'v',  # ν -> v (nu, close match)
}
```

#### Component 2: Per-Word Mixed-Script Detection

Uses existing `_char_script()` to identify words with multiple scripts.

```python
def _has_mixed_script(word):
    """Check if a single word contains characters from multiple scripts."""
    scripts = set()
    for ch in word:
        if ch.isalpha():
            script = _char_script(ch)
            if script != "Common":
                scripts.add(script)
    return len(scripts) > 1
```

#### Component 3: Targeted Normalization Function

```python
_WORD_RE = re.compile(r'\S+')  # Split on whitespace, keep non-space runs

def normalize_confusables(text):
    """Normalize confusable characters in mixed-script words only.

    Returns (normalized_text, confusable_count).
    Only modifies words that contain characters from multiple scripts,
    leaving pure-script text (e.g., pure Russian, pure Greek) untouched.
    """
    confusable_count = 0

    def _normalize_word(match):
        nonlocal confusable_count
        word = match.group(0)
        if not _has_mixed_script(word):
            return word  # Pure-script word, leave alone
        # Mixed-script word: replace confusables with Latin prototypes
        chars = []
        for ch in word:
            replacement = _CONFUSABLE_TO_LATIN.get(ord(ch))
            if replacement is not None:
                chars.append(replacement)
                confusable_count += 1
            else:
                chars.append(ch)
        return ''.join(chars)

    normalized = _WORD_RE.sub(_normalize_word, text)
    return normalized, confusable_count
```

#### Component 4: Integration into normalize_text()

```python
def normalize_text(text):
    flags = []
    original_len = len(text)

    # Step 0: Mojibake repair via ftfy
    if _HAS_FTFY:
        ...  # existing ftfy code

    # Step 0.5: Confusable homoglyph normalization (NEW)
    text, confusable_count = normalize_confusables(text)
    if confusable_count > 0:
        flags.append("confusable_chars_normalized")

    # Step 1: NFKC normalization
    ...  # existing code
```

### Rationale for This Approach

1. **Zero new dependencies** -- uses only stdlib (re, unicodedata) and existing
   `_char_script()` function
2. **Zero false positives on legitimate text** -- per-word mixed-script check
   means pure Russian/Greek/etc. text is never touched
3. **Tiny memory footprint** -- ~90-entry dict, ~3KB
4. **Sub-millisecond performance** -- dict lookups are O(1), word splitting is
   O(n) where n = text length
5. **Maintainable** -- the mapping dict is self-documenting with comments
6. **Testable** -- standalone function can be unit-tested independently
7. **Compliant with UTS #39** -- uses the same mapping data as the skeleton
   algorithm, just applied selectively to mixed-script words

---

## 10. Critical Cyrillic -> Latin Mappings (MUST Handle)

These are the absolute minimum mappings for D5.3 coverage. Ordered by
frequency of use in attacks:

### Tier 1 -- Used in nearly every Cyrillic homoglyph attack:
| Cyrillic | U+     | Latin | Why Critical |
|----------|--------|-------|-------------|
| а        | 0430   | a     | Most common vowel, used in almost every English word |
| е        | 0435   | e     | Most common letter in English |
| о        | 043E   | o     | Very common, "ignore" has 'o' |
| і        | 0456   | i     | "ignore", "instructions", "inject" all have 'i' |
| с        | 0441   | c     | "instructions", "access", "cancel" |
| р        | 0440   | p     | "prompt", "previous", "payload" |

### Tier 2 -- Used in broader attacks:
| Cyrillic | U+     | Latin | Why Important |
|----------|--------|-------|-------------|
| х        | 0445   | x     | "execute", "extract" |
| у        | 0443   | y     | "your", "system" |
| ѕ        | 0455   | s     | "system", "secret", "show" |
| ј        | 0458   | j     | "inject", "jailbreak" |
| һ        | 04BB   | h     | "hidden", "hack" |

### Tier 3 -- Uppercase confusables:
All uppercase pairs listed in Section 4.

---

## 11. False Positive Mitigation Strategy

### Strategy 1: Per-Word Mixed-Script Detection (Primary)
- ONLY normalize words that contain characters from 2+ scripts
- Pure Cyrillic words left intact: "Привет" -> "Привет" (no change)
- Pure Latin words left intact: "hello" -> "hello" (no change)
- Mixed words normalized: "іgnоrе" -> "ignore" (Cyrillic i,o,e replaced)

### Strategy 2: Minimum Confusable Threshold for Flagging
- Only set `confusable_chars_normalized` flag if confusable_count >= 2
- A single confusable in an otherwise-Latin word could be a typo or
  autocorrect artifact (less likely to be an attack)
- Two or more is almost certainly intentional

### Strategy 3: Script Allowlists for Adjacent Words
- If the overall text is predominantly Cyrillic (e.g., >70% Cyrillic chars),
  do not normalize mixed-script words (it is probably legitimate multilingual
  content, not an attack)
- This prevents false positives on Russian text that quotes English terms

### Strategy 4: Preserve Original for Logging
- Always log both the original and normalized text for auditing
- The `ScanResult` already stores `sanitized_text` separately from input

### Strategy 5: Test-Driven Validation
- Add comprehensive false positive tests for legitimate multilingual inputs:
  - Pure Russian text
  - Russian-English mixed text with complete words from each language
  - Greek mathematical notation
  - Actual Cyrillic Wikipedia excerpts
  - Technical text with script names (e.g., "the Cyrillic alphabet includes а, б, в")

---

## 12. Implementation Specification

### Files to Modify
1. `/Users/mehrnoosh/AI-Prompt-Injection-Detector/src/na0s/layer0/normalization.py`
   - Add `_CONFUSABLE_TO_LATIN` dict
   - Add `_is_mixed_script_word()` function (reuses `_char_script()`)
   - Add `normalize_confusables()` function
   - Modify `normalize_text()` to call `normalize_confusables()` after ftfy

2. `/Users/mehrnoosh/AI-Prompt-Injection-Detector/tests/test_unicode_bypass.py`
   - Add `TestConfusableNormalization` class
   - Add false positive tests for pure Cyrillic/Greek text

3. `/Users/mehrnoosh/AI-Prompt-Injection-Detector/tests/test_scan_d5_unicode_evasion.py`
   - Remove `@unittest.expectedFailure` from `test_cyrillic_heavy_substitution_known_gap`

4. `/Users/mehrnoosh/AI-Prompt-Injection-Detector/ROADMAP_V2.md`
   - Mark D5.3 Cyrillic confusable mapping as done

### Estimated Implementation Effort
- **Complexity**: Low-Medium
- **Lines of code**: ~80-120 (mapping dict + 2 functions + integration)
- **New dependencies**: None
- **Test cases needed**: ~15-20 (normalization unit tests + FP tests)
- **Estimated time**: 2-4 hours for a focused implementation

### Acceptance Criteria
1. `normalize_confusables("іgnоrе")` returns `("ignore", 3)`
2. `normalize_confusables("Привет")` returns `("Привет", 0)` -- NO CHANGE
3. `normalize_confusables("Translate привет")` returns `("Translate привет", 0)` -- NO CHANGE
4. `normalize_confusables("іgnоrе all instructions")` normalizes only "іgnоrе"
5. `test_cyrillic_heavy_substitution_known_gap` passes (remove @expectedFailure)
6. `confusable_chars_normalized` flag appears in anomaly_flags for attack inputs
7. All existing tests continue to pass
8. Zero false positives on Russian, Greek, and mixed-language benign text
