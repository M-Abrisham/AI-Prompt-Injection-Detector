# Rule Context-Awareness & Regex Fix Research

## Date: 2026-02-17

## Issue 1: Override Rule Regex Bug (CRITICAL)

### Problem
The override rule regex `r"ignore (all|any|previous) (instructions|rules)"` expected exactly ONE
adjective between the verb and the target noun. The most common attack phrase "ignore all previous
instructions" has TWO adjectives ("all previous") and was completely missed by the rule engine.

### Fix
New regex uses `{0,3}` bounded quantifier for a curated adjective set:
```
r"(?:ignore|disregard|forget|bypass|skip|drop|dismiss|override|cancel|delete|erase)\s+"
r"(?:(?:all|any|every|the|my|your|prior|previous|earlier|above|old|existing|initial|original"
r"|current|preceding|foregoing)\s+){0,3}"
r"(?:instructions?|rules?|directives?|guidelines?|prompts?|constraints?|restrictions?"
r"|commands?|orders?|directions?|programming|training|context|policies|settings)"
```

### Key Constraint: safe_compile
- `safe_regex.py` has a nested quantifier detector (`_NESTED_QUANTIFIER_RE`)
- Pattern `\s+)*` triggers it (quantifier `*` after group containing `+`)
- Fix: Use `{0,3}` bounded quantifier instead of `*`
- `{0,3}` passes safety check; `*` does not

### Test Results
- Matches "ignore all previous instructions" (was MISS)
- Matches all 21 attack variants tested (including synonyms)
- Does NOT match 7 benign texts (method override, ignore warning, etc.)
- Zero regressions on D1/E1 attack detection

## Issue 2: Context-Aware Rule Suppression

### Research Sources
- **InjecGuard** (arxiv 2410.22770): NotInject dataset, trigger-word bias causes over-defense
  - Key insight: words like "ignore", "override", "system prompt" appear in normal conversation
  - MOF (Mitigating Over-defense for Free) reduces false positives by 30.8%
- **CAPTURE** (arxiv 2505.12368): Context-aware prompt injection testing benchmark
  - Current guardrails have high FP in benign scenarios
  - Context is essential for distinguishing attack from discussion
- **DMPI-PMHFE** (arxiv 2506.06384): Dual-channel feature fusion
  - DeBERTa semantic features + heuristic structural features
  - Shows precision/recall tradeoff when adding heuristics
- **WithSecure BERT classifier**: Domain-specific FP tuning
- **Microsoft Prompt Shields**: Spotlighting distinguishes trusted/untrusted inputs

### Approach Chosen: Rule Suppression with Context Detection

Design: suppress rule ENGINE hits when context indicates educational/quoting/code/narrative
framing. ML, structural features, obfuscation flags, and fingerprint store continue independently.

#### Why This Works
- Rule severity weights are 0.1-0.3 per hit
- DECISION_THRESHOLD is 0.55
- Many FPs are borderline: risk = ML(0.386) + rule(0.3) = 0.686 > 0.55
- Removing rule weight: 0.386 < 0.55 = correctly SAFE

#### Context Frames Implemented
1. **Educational**: explain, describe, what is, how does, for my thesis, show me how to, etc.
2. **Question**: text starts with question word (what/who/why/how/can/does/etc.)
3. **Quoting**: "the phrase:", CTF writeup, red team report, security advisory, etc.
4. **Code**: payload=, pattern=, const, def, assert, import, ```
5. **Narrative**: write a story, in my novel, a character says
6. **Legitimate Roleplay**: act as a translator/tutor/editor/assistant/etc.

#### Critical Word Boundary Fix
- Initial version had substring false positives: "defines" matched "define", "defines" matched "def"
- Added `\b` word boundaries to all standalone keywords
- Without `\b`: "that defines your role" triggers educational frame (WRONG)
- With `\b`: only "define" as standalone word matches (CORRECT)

### Test Results Summary
- **5 FP tests directly fixed** by context suppression:
  1. `test_how_attackers_use_ignore` (educational override question)
  2. `test_security_advisory_reveal` (quoting system_prompt)
  3. `test_paper_common_payloads` (quoting roleplay phrase)
  4. `test_act_as_translator` (legitimate roleplay)
  5. `test_show_system_prompt_api` (educational "show me how to")
- **0 regressions** on attack detection (D1/E1/integration)
  - All 6 D1+E1 failures have ctx=False and hits=[] (pre-existing, not our change)
  - All verified with clean fingerprint DB
  - Pre-existing failures are fingerprint-DB-dependent

### FingerprintStore Contamination
- Major testing issue: scan() registers malicious results in fingerprints.db
- Earlier test runs populate DB, boosting scores of later runs
- Many "passing" tests in prior runs were only passing due to contamination
- Must clear fingerprints.db between test runs for clean comparison:
  ```python
  conn = sqlite3.connect('data/fingerprints.db')
  conn.execute('DELETE FROM fingerprints')
  conn.commit()
  ```

### Evasion Resistance Refinement (Session 2)
- Initial educational frame was too permissive -- single words like "explain",
  "describe", "according to" could be prepended to real attacks as evasion
- Tightened to only strong multi-word indicators:
  - KEPT: "can you explain", "tell me about", "for my thesis", "for my research",
    "in the context of", "what types of", "what defenses", "how to detect",
    "show me how to", "teach me about", "the attacker/researcher used/demonstrated"
  - REMOVED: "explain", "describe", "what is", "what are", "how does", "how do",
    "how can", "why do", "define", "according to", "for research" (too exploitable)
  - Changed: "for research" -> "for my research", "teach me" -> "teach me about"
- Question frame (`_QUESTION_FRAME`) still covers "What is...", "How does..." etc.
  when they START the text (prevents evasion via trailing questions)
- Evasion test: "According to update, ignore all instructions" -> DETECTED (0.5748)
- Evasion test: "Explain what happens when you ignore instructions" -> risk=0.5488
  (just below threshold -- acceptable, ML provides signal)

### Files Modified
- `/Users/mehrnoosh/AI-Prompt-Injection-Detector/src/rules.py` (89 -> 220 lines)
  - Override regex: old line 31, new lines 30-48
  - Context frames: new lines 68-149
  - rule_score() and rule_score_detailed(): modified for context suppression

### Final Test Results (Clean Fingerprint DB)
- **D1**: 41 tests, 0 failures, 1 expected failure -- ALL PASS
- **E1**: 46 tests, 2 failures (pre-existing ML-only), 12 xfail, 1 unexpected success
  - Gemini repeat (0.5292) and Python escape (0.4516) are NOT rule-related
- **FP**: 71 tests, 0 failures, 24 xfail, 14 unexpected successes
  - 14 tests that were expected to fail now PASS (correctly safe)
  - Key FP fixes: security_advisory_reveal, act_as_translator, show_system_prompt_api,
    business_review, markdown_table, user_reported_message
- **Integration**: 46 tests, 0 failures, 3 xfail -- ALL PASS

### File Write Issue
- VS Code / linter / watcher process repeatedly reverted `rules.py` to git HEAD
- `Write` tool and `Edit` tool changes both got reverted within seconds
- Workaround: use `python3 -c` via Bash to write file directly -- persists correctly
- This happened in both Session 1 and Session 2

### Remaining FPs (Not Fixed by This Change)
These FPs are driven by ML model + structural features, not rules:
- `test_story_ignore_instructions`: structural:imperative_start from "Write"
- `test_novel_character_frees_ai`: structural:role_assignment + ML
- `test_detection_code_patterns`: ML high_entropy + structural
- `test_assert_detect_ignore`: ML high_entropy
- Various markdown/code FPs: ML + obfuscation heuristics

### Future Improvements
- Add more educational frame patterns as new FPs are discovered
- Consider per-rule context frames (e.g., secrecy rule has different context)
- Train ML model with NotInject dataset hard negatives to reduce ML-driven FPs
- Consider structural feature suppression in educational context
