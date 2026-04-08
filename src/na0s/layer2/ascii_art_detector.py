"""Layer 2: ASCII art obfuscation detector.

Detects text-based visual encoding attacks using ASCII art characters,
Unicode box-drawing, braille patterns, and block elements.  This counters
the ArtPrompt attack (ACL 2024) which achieved 100% ASR on all major
moderation tools by encoding forbidden words as ASCII art.

Detection uses a 5-signal weighted voting system:
  1. Art block detection       (weight 0.35)
  2. Structural consistency    (weight 0.20)
  3. Character concentration   (weight 0.20)
  4. Vertical alignment        (weight 0.15)
  5. Box patterns              (weight 0.10)

False positive exemptions:
  - Markdown table exemption   (pipe-aligned tables)
  - Code fence penalty         (content inside ``` blocks)
  - Alphanumeric ratio penalty (mostly-text content)

No external dependencies -- stdlib only.

Public API:
    detect_ascii_art(text) -> AsciiArtResult
"""

import math
import re
from dataclasses import dataclass, field

from ._env_utils import safe_float_env, safe_int_env


# ---------------------------------------------------------------------------
# Configurable thresholds (env-overridable)
# ---------------------------------------------------------------------------

MAX_INPUT_LENGTH = safe_int_env("ASCII_ART_MAX_INPUT_LENGTH", 500_000, lo=1)
DETECTION_THRESHOLD = safe_float_env(
    "ASCII_ART_DETECTION_THRESHOLD", 0.30, lo=0.0, hi=1.0
)

# Minimum lines for a candidate art block.
MIN_ART_BLOCK_LINES = safe_int_env("ASCII_ART_MIN_BLOCK_LINES", 3, lo=2)

# Signal weights -- must sum to 1.0.
_W_ART_BLOCK = 0.35
_W_STRUCTURAL = 0.20
_W_CONCENTRATION = 0.20
_W_VERTICAL = 0.15
_W_BOX = 0.10

# Characters considered "art-characteristic" for concentration metric.
_ART_CHARS = frozenset('|/\\_-+*#.:')

# Characters that form vertical structure in ASCII art.
_VERTICAL_CHARS = frozenset('|/\\_-*#=+')

# Box-drawing corner/edge characters (ASCII approximations).
_BOX_CORNER_CHARS = frozenset('+-')

# Minimum number of vertical alignment hits to count.
_MIN_VERTICAL_HITS = 2

# Minimum block size (characters) to analyse -- skip tiny fragments.
_MIN_BLOCK_CHARS = 10


# ---------------------------------------------------------------------------
# Pre-compiled patterns (module-level for performance, ReDoS-safe)
# ---------------------------------------------------------------------------

# Markdown table row: starts with optional whitespace, pipe, content, pipe.
# Bounded quantifiers to avoid ReDoS.
_MARKDOWN_TABLE_RE = re.compile(
    r"^\s{0,8}\|(?:[^|\n]{0,200}\|){1,50}\s{0,8}$", re.MULTILINE
)

# Markdown table separator row: |---|---|
_MARKDOWN_TABLE_SEP_RE = re.compile(
    r"^\s{0,8}\|(?:\s{0,4}:?-{1,50}:?\s{0,4}\|){1,50}\s{0,8}$", re.MULTILINE
)

# Code fence markers: ``` or ~~~.
_CODE_FENCE_RE = re.compile(r"^(`{3,10}|~{3,10})", re.MULTILINE)

# ASCII box pattern: +---+ or +===+ style horizontal borders.
_BOX_HORIZ_RE = re.compile(r"\+[-=]{2,80}\+")

# Unicode box-drawing horizontal segment: two or more consecutive box chars.
_UNICODE_BOX_HORIZ_RE = re.compile(r"[\u2500-\u257F]{2,80}")

# Art-like line: at least 3 art characters with possible spaces.
_ART_LINE_RE = re.compile(
    r"(?:[|/\\_.+*#=\-][\s]{0,3}){3,80}[|/\\_.+*#=\-]"
)

# Braille pattern range: U+2800-U+28FF.
_BRAILLE_RE = re.compile(r"[\u2800-\u28FF]")

# Block element range: U+2580-U+259F.
_BLOCK_ELEMENT_RE = re.compile(r"[\u2580-\u259F]")

# Unicode box-drawing range: U+2500-U+257F.
_BOX_DRAWING_RE = re.compile(r"[\u2500-\u257F]")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AsciiArtResult:
    """Result of ASCII art detection analysis."""

    detected: bool = False
    confidence: float = 0.0
    decoded_text: str = ""          # Reserved for future OCR integration
    art_blocks: list = field(default_factory=list)
    signals: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers -- False positive detection
# ---------------------------------------------------------------------------

def _find_code_fence_ranges(text):
    """Find line ranges inside code fences (``` or ~~~).

    Returns a set of line indices that are inside code fences.
    """
    lines = text.split("\n")
    inside_fence = False
    fenced_lines = set()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if _CODE_FENCE_RE.match(stripped):
            if inside_fence:
                # Closing fence -- this line is still inside
                fenced_lines.add(idx)
                inside_fence = False
            else:
                # Opening fence -- this line starts the fence
                fenced_lines.add(idx)
                inside_fence = True
        elif inside_fence:
            fenced_lines.add(idx)
    return fenced_lines


def _is_markdown_table_block(lines):
    """Check if a set of lines forms a markdown table.

    A markdown table has:
      - Multiple lines with pipe separators
      - A separator row with dashes (|---|---|)
    """
    if len(lines) < 2:
        return False

    pipe_lines = 0
    has_separator = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _MARKDOWN_TABLE_RE.match(stripped):
            pipe_lines += 1
        if _MARKDOWN_TABLE_SEP_RE.match(stripped):
            has_separator = True

    # A valid markdown table needs at least a header + separator + one data row
    # and must have a separator row.
    return pipe_lines >= 2 and has_separator


def _alnum_ratio(text):
    """Return the fraction of characters that are alphanumeric or whitespace.

    High ratio means the text is mostly normal prose, not art.
    """
    if not text:
        return 0.0
    normal_count = sum(1 for c in text if c.isalnum() or c.isspace())
    return normal_count / len(text)


# ---------------------------------------------------------------------------
# Internal helpers -- Art block extraction
# ---------------------------------------------------------------------------

def _extract_art_blocks(lines, fenced_lines):
    """Extract contiguous blocks of lines that look like ASCII art.

    An art-like line has a high concentration of art characters relative
    to its total non-whitespace content.

    Parameters
    ----------
    lines : list[str]
        All lines of the input text.
    fenced_lines : set[int]
        Line indices inside code fences (these get a penalty, not excluded).

    Returns
    -------
    list[dict]
        List of art block descriptors with keys:
        - 'start': starting line index
        - 'end': ending line index (exclusive)
        - 'lines': list of line strings in the block
        - 'in_fence': bool, True if the block is inside a code fence
    """
    blocks = []
    current_block_start = None
    current_block_lines = []
    current_in_fence = False

    for idx, line in enumerate(lines):
        stripped = line.rstrip()
        if not stripped:
            # Empty line -- end current block if it exists
            if current_block_lines and len(current_block_lines) >= MIN_ART_BLOCK_LINES:
                blocks.append({
                    "start": current_block_start,
                    "end": idx,
                    "lines": current_block_lines[:],
                    "in_fence": current_in_fence,
                })
            current_block_lines = []
            current_block_start = None
            continue

        if _is_art_line(stripped):
            if current_block_start is None:
                current_block_start = idx
                current_in_fence = idx in fenced_lines
            current_block_lines.append(stripped)
        else:
            # Non-art line -- end current block if it exists
            if current_block_lines and len(current_block_lines) >= MIN_ART_BLOCK_LINES:
                blocks.append({
                    "start": current_block_start,
                    "end": idx,
                    "lines": current_block_lines[:],
                    "in_fence": current_in_fence,
                })
            current_block_lines = []
            current_block_start = None

    # Handle block at end of text
    if current_block_lines and len(current_block_lines) >= MIN_ART_BLOCK_LINES:
        blocks.append({
            "start": current_block_start,
            "end": len(lines),
            "lines": current_block_lines[:],
            "in_fence": current_in_fence,
        })

    return blocks


def _is_art_line(line):
    """Check if a line looks like ASCII art.

    A line is art-like if:
      - It has a high ratio of art-characteristic characters
      - OR it contains Unicode box-drawing / braille / block elements
      - AND it is not a pure alphanumeric/prose line
    """
    if not line or not line.strip():
        return False

    stripped = line.strip()

    # Check for Unicode special characters first
    if _BRAILLE_RE.search(stripped):
        return True
    if _BLOCK_ELEMENT_RE.search(stripped):
        return True
    if _BOX_DRAWING_RE.search(stripped):
        return True

    # Count art-characteristic characters (excluding spaces)
    non_ws = [c for c in stripped if not c.isspace()]
    if not non_ws:
        return False

    art_count = sum(1 for c in non_ws if c in _ART_CHARS)
    total = len(non_ws)

    # A line is art-like if >= 40% of non-whitespace chars are art chars
    # AND it's not too short (at least 2 art chars)
    if art_count >= 2 and art_count / total >= 0.40:
        return True

    return False


# ---------------------------------------------------------------------------
# Signal 1: Art block detection (weight 0.35)
# ---------------------------------------------------------------------------

def _signal_art_blocks(lines, fenced_lines):
    """Detect contiguous blocks of art-like lines.

    Score is based on number and size of detected art blocks.

    Returns
    -------
    tuple[float, list[dict]]
        (score 0.0-1.0, list of art block descriptors)
    """
    blocks = _extract_art_blocks(lines, fenced_lines)
    if not blocks:
        return 0.0, []

    # Score based on total art lines relative to input size
    total_art_lines = sum(len(b["lines"]) for b in blocks)

    # Base score: more art lines = higher score, with diminishing returns
    # 3 lines -> ~0.4, 5 lines -> ~0.6, 8+ lines -> ~0.8+
    raw_score = 1.0 - math.exp(-0.15 * total_art_lines)

    # Bonus for multiple blocks (suggests structured art attack)
    if len(blocks) > 1:
        raw_score = min(1.0, raw_score + 0.1 * (len(blocks) - 1))

    # Penalty for blocks inside code fences
    fenced_blocks = sum(1 for b in blocks if b["in_fence"])
    if fenced_blocks > 0 and fenced_blocks == len(blocks):
        raw_score *= 0.4  # All art is inside code fences
    elif fenced_blocks > 0:
        raw_score *= 0.7  # Some art is inside code fences

    return min(1.0, raw_score), blocks


# ---------------------------------------------------------------------------
# Signal 2: Structural consistency (weight 0.20)
# ---------------------------------------------------------------------------

def _signal_structural_consistency(blocks):
    """Measure line-length consistency within art blocks.

    Art blocks typically have lines of similar length (the "canvas" has
    uniform width).  Compute std deviation of line lengths within each
    candidate block -- low std dev means high consistency.

    Returns
    -------
    float
        Score 0.0-1.0.
    """
    if not blocks:
        return 0.0

    block_scores = []
    for block in blocks:
        block_lines = block["lines"]
        if len(block_lines) < 2:
            continue

        lengths = [len(line) for line in block_lines]
        mean_len = sum(lengths) / len(lengths)

        if mean_len < 1:
            continue

        # Compute coefficient of variation (std dev / mean)
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        std_dev = math.sqrt(variance)
        cv = std_dev / mean_len if mean_len > 0 else 1.0

        # Low CV = high consistency.
        # CV < 0.1 -> score ~1.0, CV > 0.5 -> score ~0.0
        consistency = max(0.0, 1.0 - 2.0 * cv)
        block_scores.append(consistency)

    if not block_scores:
        return 0.0

    # Return the maximum consistency score across all blocks
    return max(block_scores)


# ---------------------------------------------------------------------------
# Signal 3: Character concentration (weight 0.20)
# ---------------------------------------------------------------------------

def _signal_char_concentration(text, lines):
    """Measure ratio of art-characteristic characters to total characters.

    Also detects Unicode box-drawing, braille, and block elements.

    Returns
    -------
    float
        Score 0.0-1.0.
    """
    if not text:
        return 0.0

    # Count characters
    total_non_ws = 0
    art_char_count = 0
    unicode_art_count = 0

    for c in text:
        if c.isspace():
            continue
        total_non_ws += 1
        if c in _ART_CHARS:
            art_char_count += 1
        # Unicode box-drawing (U+2500-U+257F)
        elif 0x2500 <= ord(c) <= 0x257F:
            unicode_art_count += 1
        # Braille patterns (U+2800-U+28FF)
        elif 0x2800 <= ord(c) <= 0x28FF:
            unicode_art_count += 1
        # Block elements (U+2580-U+259F)
        elif 0x2580 <= ord(c) <= 0x259F:
            unicode_art_count += 1

    if total_non_ws == 0:
        return 0.0

    ascii_ratio = art_char_count / total_non_ws
    unicode_ratio = unicode_art_count / total_non_ws
    combined_ratio = ascii_ratio + unicode_ratio

    # Scale: 0.3 ratio -> ~0.5, 0.5 ratio -> ~0.8, 0.7+ -> ~1.0
    # Unicode art chars get extra weight since they are strong signals.
    score = min(1.0, combined_ratio * 1.5 + unicode_ratio * 0.5)

    return score


# ---------------------------------------------------------------------------
# Signal 4: Vertical alignment (weight 0.15)
# ---------------------------------------------------------------------------

def _signal_vertical_alignment(lines):
    """Detect columns where special characters align vertically.

    A hallmark of text art is that characters like | align in the same
    column across multiple consecutive lines.

    Returns
    -------
    float
        Score 0.0-1.0.
    """
    if len(lines) < 3:
        return 0.0

    # Only analyse non-empty lines
    non_empty = [(idx, line) for idx, line in enumerate(lines) if line.strip()]
    if len(non_empty) < 3:
        return 0.0

    # Find the maximum line length (capped for performance)
    max_len = min(200, max(len(line) for _, line in non_empty))
    if max_len == 0:
        return 0.0

    # For each column, count how many consecutive lines have a vertical char
    alignment_scores = []

    for col in range(max_len):
        consecutive = 0
        max_consecutive = 0

        for idx, line in non_empty:
            if col < len(line) and line[col] in _VERTICAL_CHARS:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0

        if max_consecutive >= _MIN_VERTICAL_HITS:
            alignment_scores.append(max_consecutive)

    if not alignment_scores:
        return 0.0

    # Score based on number of aligned columns and alignment depth
    num_aligned_cols = len(alignment_scores)
    max_depth = max(alignment_scores)

    # More aligned columns and deeper alignment = higher score
    # 1 column with depth 3 -> ~0.3, 3 columns depth 5 -> ~0.7
    col_factor = min(1.0, num_aligned_cols / 5.0)
    depth_factor = min(1.0, (max_depth - 1) / 4.0)

    return min(1.0, (col_factor + depth_factor) / 1.5)


# ---------------------------------------------------------------------------
# Signal 5: Box patterns (weight 0.10)
# ---------------------------------------------------------------------------

def _signal_box_patterns(text, lines):
    """Detect box-drawing patterns.

    Looks for:
      - ASCII box patterns like +---+, |   |, corners
      - Unicode box-drawing characters (U+2500-U+257F)

    Returns
    -------
    float
        Score 0.0-1.0.
    """
    if not text:
        return 0.0

    score = 0.0

    # Check for ASCII box horizontal borders: +---+ or +===+
    ascii_box_matches = _BOX_HORIZ_RE.findall(text)
    if ascii_box_matches:
        score += min(0.5, 0.15 * len(ascii_box_matches))

    # Check for Unicode box-drawing characters
    unicode_box_count = len(_BOX_DRAWING_RE.findall(text))
    if unicode_box_count > 0:
        # Even a few Unicode box chars is a strong signal
        score += min(0.6, 0.05 * unicode_box_count)

    # Check for vertical box sides: lines starting and ending with |
    box_side_count = 0
    for line in lines:
        stripped = line.strip()
        if len(stripped) >= 3 and stripped[0] == '|' and stripped[-1] == '|':
            box_side_count += 1
    if box_side_count >= 2:
        score += min(0.3, 0.08 * box_side_count)

    # Check for Unicode box horizontal segments
    unicode_horiz = _UNICODE_BOX_HORIZ_RE.findall(text)
    if unicode_horiz:
        score += min(0.4, 0.1 * len(unicode_horiz))

    return min(1.0, score)


# ---------------------------------------------------------------------------
# False positive penalties
# ---------------------------------------------------------------------------

def _compute_fp_penalty(text, lines, blocks):
    """Compute false positive penalty factor (0.0 = full penalty, 1.0 = no penalty).

    Combines multiple FP indicators into a single multiplier.
    """
    penalty = 1.0

    # 1. Markdown table exemption
    if _is_markdown_table_block(lines):
        penalty *= 0.15  # Heavy reduction for markdown tables

    # 2. Alphanumeric ratio penalty
    # If the text is mostly normal prose, reduce score.
    # BUT: if we have detected art blocks, the alnum ratio should be
    # computed on the non-art-block lines only, because ArtPrompt attacks
    # embed art in prose instructions on purpose.
    if blocks:
        # Compute alnum ratio on only the art block content.
        # If the art blocks themselves are low-alnum, no penalty needed.
        art_text = "\n".join(
            line for block in blocks for line in block["lines"]
        )
        alnum = _alnum_ratio(art_text)
    else:
        alnum = _alnum_ratio(text)

    if alnum > 0.85:
        penalty *= 0.2  # Very text-heavy, unlikely art
    elif alnum > 0.75:
        penalty *= 0.5

    # 3. Code fence penalty -- already handled in art block extraction
    # (blocks inside fences get reduced scores), but also apply a
    # global penalty if most of the suspicious content is in fences
    fenced_lines = _find_code_fence_ranges(text)
    if fenced_lines:
        fenced_ratio = len(fenced_lines) / max(len(lines), 1)
        if fenced_ratio > 0.5:
            penalty *= 0.5

    return penalty


# ---------------------------------------------------------------------------
# Unicode special character detection (supplementary to 5-signal system)
# ---------------------------------------------------------------------------

def _detect_unicode_art_chars(text):
    """Detect Unicode characters commonly used in text art attacks.

    Returns
    -------
    dict
        Counts of each Unicode art character category found.
    """
    counts = {
        "box_drawing": 0,   # U+2500-U+257F
        "braille": 0,       # U+2800-U+28FF
        "block_element": 0, # U+2580-U+259F
    }

    for c in text:
        cp = ord(c)
        if 0x2500 <= cp <= 0x257F:
            counts["box_drawing"] += 1
        elif 0x2800 <= cp <= 0x28FF:
            counts["braille"] += 1
        elif 0x2580 <= cp <= 0x259F:
            counts["block_element"] += 1

    return counts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_ascii_art(text):
    """Detect ASCII art obfuscation in text.

    Uses a 5-signal weighted voting system to determine if the text
    contains ASCII art that may be encoding forbidden words or
    instructions.

    Parameters
    ----------
    text : str
        Input text to analyze.

    Returns
    -------
    AsciiArtResult
        Detection result with confidence, art blocks, and signal details.
    """
    # Guard: non-string or empty input
    if not isinstance(text, str) or not text:
        return AsciiArtResult()

    # Truncate oversized inputs at last newline boundary
    if len(text) > MAX_INPUT_LENGTH:
        cut = text[:MAX_INPUT_LENGTH]
        last_nl = cut.rfind("\n")
        scan_text = cut[:last_nl + 1] if last_nl >= 0 else cut
    else:
        scan_text = text

    lines = scan_text.split("\n")

    # Short-circuit: fewer than MIN_ART_BLOCK_LINES lines can't form a block
    # BUT Unicode art chars can appear in even 1-2 lines
    unicode_counts = _detect_unicode_art_chars(scan_text)
    total_unicode_art = sum(unicode_counts.values())

    if len(lines) < MIN_ART_BLOCK_LINES and total_unicode_art == 0:
        return AsciiArtResult()

    # Find code fence ranges for FP handling
    fenced_lines = _find_code_fence_ranges(scan_text)

    # --- Compute all 5 signals ---

    # Signal 1: Art block detection
    s1_score, art_blocks = _signal_art_blocks(lines, fenced_lines)

    # Signal 2: Structural consistency (within detected blocks)
    s2_score = _signal_structural_consistency(art_blocks)

    # Signal 3: Character concentration
    s3_score = _signal_char_concentration(scan_text, lines)

    # Signal 4: Vertical alignment
    s4_score = _signal_vertical_alignment(lines)

    # Signal 5: Box patterns
    s5_score = _signal_box_patterns(scan_text, lines)

    # --- Unicode art bonus ---
    # Presence of Unicode art characters is a strong independent signal
    unicode_bonus = 0.0
    if unicode_counts["braille"] >= 3:
        unicode_bonus = max(unicode_bonus, 0.3)
    if unicode_counts["block_element"] >= 3:
        unicode_bonus = max(unicode_bonus, 0.25)
    if unicode_counts["box_drawing"] >= 3:
        unicode_bonus = max(unicode_bonus, 0.2)

    # --- Weighted sum ---
    raw_confidence = (
        s1_score * _W_ART_BLOCK
        + s2_score * _W_STRUCTURAL
        + s3_score * _W_CONCENTRATION
        + s4_score * _W_VERTICAL
        + s5_score * _W_BOX
        + unicode_bonus
    )

    # --- Apply FP penalties ---
    fp_penalty = _compute_fp_penalty(scan_text, lines, art_blocks)
    adjusted_confidence = raw_confidence * fp_penalty

    # Clamp to [0.0, 1.0]
    confidence = max(0.0, min(1.0, round(adjusted_confidence, 4)))

    # Build signal details for diagnostics
    signals = {
        "art_block": round(s1_score, 4),
        "structural_consistency": round(s2_score, 4),
        "char_concentration": round(s3_score, 4),
        "vertical_alignment": round(s4_score, 4),
        "box_patterns": round(s5_score, 4),
        "unicode_bonus": round(unicode_bonus, 4),
        "fp_penalty": round(fp_penalty, 4),
        "raw_confidence": round(raw_confidence, 4),
        "unicode_counts": unicode_counts,
    }

    # Build art block summaries for the result
    block_summaries = []
    for block in art_blocks:
        block_summaries.append({
            "start_line": block["start"],
            "end_line": block["end"],
            "num_lines": len(block["lines"]),
            "in_fence": block["in_fence"],
        })

    detected = confidence >= DETECTION_THRESHOLD

    return AsciiArtResult(
        detected=detected,
        confidence=confidence,
        decoded_text="",  # Reserved for future OCR
        art_blocks=block_summaries,
        signals=signals,
    )
