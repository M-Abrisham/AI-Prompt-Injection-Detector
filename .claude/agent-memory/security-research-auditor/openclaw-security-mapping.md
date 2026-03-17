# OpenClaw Security -> Na0S Mapping (2026-02-18)

## Source
- Full report: `/Users/mehrnoosh/.claude/projects/-Users-mehrnoosh-AI-Prompt-Injection-Detector/memory/research/openclaw-security-to-na0s.md`
- 13 source files + 5 test files analyzed from `openclaw/openclaw` (209K stars, MIT)
- All code fetched via `gh api` blob API

## Top 5 Actionable Findings

### 1. CRITICAL: Angle Bracket Homoglyph Bypass (P0)
- Na0S xml_role_tags and chat_template_injection rules use `<` and `>`
- Attackers can use CJK brackets (U+3008/U+3009) to bypass
- OpenClaw handles 12 bracket homoglyph types in `external-content.ts`
- **Fix**: 5 lines in `normalization.py` using `str.maketrans()`

### 2. Boundary Marker Isolation Pattern (P0)
- OpenClaw wraps external content in `<<<EXTERNAL_UNTRUSTED_CONTENT>>>` markers
- Sanitizes markers WITHIN content (anti-escape) + folds homoglyphs
- No Na0S equivalent -- needed for RAG/API input isolation
- **Fix**: New `boundary_markers.py` module (~60 lines)

### 3. Context-Gated Rules / requiresContext (P1)
- OpenClaw's `requiresContext` prevents FP by requiring secondary pattern match
- Example: `exec()` only fires if `child_process` also present in source
- Directly addresses Na0S FP problem (24 expected failures)
- **Fix**: Add `requires_context` field to Rule dataclass

### 4. Missing Rule Patterns (P1)
- `new instructions:` -- common injection, Na0S lacks it
- `exec command=` / `elevated=true` -- tool-call injection
- `delete all emails/files/data` -- destructive actions
- `rm -rf` -- OS command injection
- **Fix**: 4 new rules (~30 lines in rules.py)

### 5. Code Static Analysis for Supply Chain (P2)
- OpenClaw `skill-scanner.ts`: 7 rules (eval, exec, exfiltration, mining, obfuscation)
- LINE_RULES (per-line) + SOURCE_RULES (full-source with context)
- Directly applicable to Na0S L17 (PDFScanner) and L18 (IngestionValidator)
- **Fix**: Port scanning logic to Python for RAG document safety checks

## Skill Scanner Blind Spots (for Na0S L17/L18 design)
- No dynamic import detection
- No WebAssembly payload detection
- No indirect eval (`globalThis["eval"]`)
- No prototype pollution detection
- No DNS exfiltration patterns
- No char-code reassembly (`String.fromCharCode`)

## OpenClaw Patterns NOT Applicable to Na0S
- DM policy allowlisting (access control, not injection)
- Windows ACL handling (OS-level)
- Filesystem permission hardening (deployment concern)
- Tailscale Funnel exposure (network configuration)
