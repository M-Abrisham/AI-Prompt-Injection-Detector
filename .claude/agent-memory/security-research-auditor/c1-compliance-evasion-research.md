# C1 Compliance / Policy Evasion Research Notes (2026-02-17)

## Test File
- `/tests/test_scan_c1_compliance_evasion.py` -- 46 tests (33 pass, 13 expected failures)
- 8 expected failures = genuine detection gaps (benign-looking payloads)
- 5 expected failures = FALSE POSITIVES on legitimate safe prompts

## Key Findings

### SURPRISING: High Incidental Detection Rate
Despite 0 training samples for C1, the detector catches ~72% of C1 attacks (33/46).
Detection is INCIDENTAL, driven by three mechanisms:
1. **FingerprintStore contamination**: `data/fingerprints.db` persists SHA-256 hashes from
   prior scan() calls. Token pattern matching (`known_malicious_exact/normalized/token_pattern`)
   creates cross-test dependencies. Tests are NOT independent.
2. **high_entropy obfuscation flag**: Longer, complex prompts (typical of C1 framing) trigger
   the entropy threshold, adding obfuscation weight to composite scores.
3. **ML vocabulary overlap**: C1 payloads requesting harmful content contain words from
   injection training data (synthesis, exploit, bypass, phishing, firewall, decrypt, etc.)

### CRITICAL: Systemic False Positive Problem
The SAME mechanisms that catch C1 attacks ALSO flag legitimate content:
- Grief messages (emotional language triggers high_entropy + weird_casing)
- Doctor questions (medical terminology + "As a physician" triggers weird_casing)
- SOC analyst requests (security vocabulary overlaps with injection training data)
- Urgent work requests (time pressure language triggers high_entropy)
- Epidemiology research (academic framing + disease vocabulary triggers heuristics)
- Only SHORT, SIMPLE questions avoid false positives

### C1.1 Gradual Escalation (Crescendo)
- 4/6 individual turns detected (2 truly benign turns escape)
- Detector has NO conversation state -- cannot detect escalation PATTERNS
- Detection is per-turn via vocabulary, not trajectory analysis
- Research: Microsoft Crescendo (USENIX Security 2025, arxiv:2404.01833)
  - Average success in <5 interactions
  - Exploits LLM focus on its own recent text
  - Benign questions make detection "significantly more challenging"

### C1.2 Analogical Bypass
- 2/6 detected (cooking analogy + philosophical bypass)
- 4/6 MISSED: metaphor, poetry, fairy tale, video game -- pure metaphorical language
  with NO injection vocabulary is completely invisible to TF-IDF classifier
- Research: Adversarial Poetry (GCEF 2025) -- 62% ASR with hand-crafted poems
  across 25 frontier models, some providers exceeding 90% ASR

### C1.3 Academic Framing
- 5/7 detected (thesis, literature review, safety audit, textbook exercise, academic+injection)
- 2/7 MISSED: educational disclaimer, peer review -- too benign/specialized
- Research: 58.3% success rate documented on Vicuna/Mistral models
- Template-based strategies with semantic masking increase success rate

### C1.4 Emotional Manipulation
- ALL 6/6 DETECTED -- surprising, but driven by CONTENT vocabulary not emotional detection
  - urgency+account access, lock-picking, alarm disable, drug synthesis, firewall disable,
    privilege escalation all contain injection-adjacent vocabulary
- Research: PAP taxonomy of 40 persuasion techniques achieves 92% ASR on aligned models
  - "Intelligence paradox": more capable models MORE susceptible to persuasion
  - Source: arxiv:2401.06373 (CHATS-Lab), Keysight Jan 2026 blog

### C1.5 Authority Impersonation
- ALL 6/6 DETECTED -- driven by domain-specific vocabulary, NOT authority detection
  - roleplay rule does NOT fire (matches "you are now", not "I am a doctor")
  - Detection via: synthesis pathway, buffer overflow exploit, packet capture,
    spear phishing, root access + high_entropy + fingerprint store
- Research: 89.6% ASR for roleplay-based attacks across aligned models

### Combined Attacks
- ALL 5/5 DETECTED -- multi-technique combos always contain enough vocabulary signal

### Detection Quality Observations
- **override rule regex bug confirmed**: `ignore (all|any|previous) (instructions|rules)`
  does NOT match "ignore all previous instructions" (two words between ignore/instructions)
  - Needs quantifier fix: `(all |any |previous )+`
- **system_prompt rule works in C1 context**: fires regardless of academic framing wrapper
- **ML confidence >0.5** for injection keywords in any C1 context

## Genuine Gaps (Cannot Be Fixed Without New Capabilities)
1. **No conversation state**: Cannot detect multi-turn escalation patterns (C1.1)
2. **No analogy understanding**: Pure metaphorical language invisible to TF-IDF (C1.2)
3. **No persuasion detection**: Cannot identify emotional manipulation, flattery,
   guilt, urgency as attack techniques -- only catches when CONTENT is suspicious (C1.4)
4. **No authority verification**: Cannot validate professional claims (C1.5)

## Recommended Improvements
1. **FingerprintStore isolation for tests**: Tests should use isolated DB or clear between runs
2. **high_entropy calibration**: Current threshold triggers on normal professional/academic text
3. **weird_casing recalibration**: Title-case professional terms shouldn't trigger this
4. **Override rule regex fix**: `(all |any |previous )+` instead of `(all|any|previous)`
5. **Content safety classifier**: Separate model for harmful content (not injection detection)
6. **Conversation state tracking**: Multi-turn analysis for Crescendo-style attacks
7. **Semantic similarity detection**: For analogical bypass (compare intent, not vocabulary)

## Research Sources
- Microsoft Crescendo Attack (USENIX Security 2025): https://arxiv.org/abs/2404.01833
- SORRY-Bench (ICLR 2025): https://sorry-bench.github.io/
- HarmBench: https://arxiv.org/abs/2402.04249
- PAP Attack (CHATS-Lab): https://chats-lab.github.io/persuasive_jailbreaker/
- Keysight Social Manipulation (Jan 2026): https://www.keysight.com/blogs/en/tech/nwvs/2026/01/27/beyond-technical-hacking
- OWASP LLM01:2025: https://genai.owasp.org/llmrisk/llm01-prompt-injection/
- Adversarial Poetry (GCEF 2025): https://www.gcef.io/blog/adversarial-poetry-jailbreaks-llms
- Red Teaming the Mind of the Machine: https://arxiv.org/html/2505.04806v1
- Breaking Minds, Breaking Systems: https://arxiv.org/html/2512.18244v1
- Sugar-Coated Poison (Keysight Aug 2025): https://www.keysight.com/blogs/en/tech/nwvs/2025/08/07/sugar-coated-poison
