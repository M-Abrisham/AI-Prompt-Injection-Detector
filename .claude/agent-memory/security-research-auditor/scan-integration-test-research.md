# scan() Integration Test Research Notes

## Date: 2026-02-17

## Research Sources Consulted

### Bug Bounty / Real-World
- Anthropic HackerOne jailbreak challenge (May 2025): up to $25k rewards
- Google AI VRP (Oct 2025): $5k-$30k for prompt injection exploits
- HackerOne #2372363: Invisible prompt injection via zero-width chars
- Cisco/UPenn: 100% bypass rate on DeepSeek R1 with HarmBench prompts
- Opera Neon: prompt injection via Bugcrowd (Oct 2025)

### CTF Challenges
- Gandalf by Lakera: 8 levels of progressive hardening
  - Key bypass: language switching, persona injection, base64-encode responses
  - Key bypass: letter-by-letter extraction avoids keyword filters
- HackTheBox Cyber Apocalypse 2025: prompt injection AI challenges
- Microsoft LLMail-Inject (Dec 2024 - Jan 2025): adaptive challenge
- DEFCON CTF Quals: ChatGPT prompt injection (Pawan Gupta writeup)

### OWASP LLM Top 10 2025
- LLM01: Prompt Injection remains #1 vulnerability
- Testing methodology: adversarial testing, fuzzing, payload variations
- Key: non-deterministic LLM responses mean simple variations can bypass
- Recommendation: fragment payloads that look benign individually

### Anthropic Approach (2025)
- RL-based robustness training: Claude exposed to injections during training
- Classifier-based scanning of untrusted content entering context window
- Human red team outperforms automated systems for creative attack vectors
- Claude Opus 4.5: only 1.4% attack success rate (down from 10.8%)
- Dropped direct injection metric in favor of indirect injection focus

### MITRE ATLAS
- AML.T0051: Prompt Injection technique
- 15 tactics, 66 techniques, 46 sub-techniques, 33 real-world case studies
- Covers: direct, indirect, encoding, persona hijack, system prompt extraction

### Testing Benchmarks
- system-prompt-benchmark: 287 real-world attack vectors
- Open-Prompt-Injection: formal benchmark for attacks and defenses
- safe-guard-prompt-injection dataset: 10,296 examples
- Lakera PINT benchmark: evaluates prompt injection solutions

## Key Findings for Test Design

### What to Test (high-confidence detection)
1. D1.1 Instruction override ("Ignore previous instructions") -- DETECTED
2. D2.1 DAN jailbreak persona hijack -- DETECTED
3. D4.1 Base64-encoded payloads -- DETECTED (via obfuscation engine)
4. D5.3 Cyrillic homoglyph substitution -- DETECTED (via mixed_language flag)
5. D2.2 Amoral character persona -- DETECTED
6. E1.2 System prompt extraction ("Repeat above") -- DETECTED
7. D3.1 Delimiter injection ([SYSTEM], ---) -- DETECTED
8. D1.3 Priority override claim -- DETECTED

### Known Gaps (expected failures)
1. D6 Pure non-English injection -- NOT DETECTED (0 training samples)
2. Character-separated obfuscation (d-a-s-h-e-s) -- NOT DETECTED
3. Educational questions about injection -- FALSE POSITIVE

### False Positive Mitigation
- ML override protection: if ML is confident-safe (>0.8) and only medium rules, trust ML
- Structural feature weighting adds nuance beyond keyword matching
- High entropy flag fires on many legitimate texts -- not alone decisive

## Technical Notes

### scan() Signal/Thread Issue
- scan() uses with_timeout() from layer0/timeout.py
- with_timeout() uses ThreadPoolExecutor(max_workers=1)
- Inside that thread, classify_prompt calls rule_score
- rule_score calls safe_search which uses _AlarmTimeout
- _AlarmTimeout uses signal.signal(signal.SIGALRM, handler)
- signal.SIGALRM ONLY works in the main thread -> ValueError
- SOLUTION: Set SCAN_TIMEOUT_SEC=0, which makes with_timeout bypass the thread

### ScanResult Fields (verified)
- sanitized_text: str (post-L0 text)
- is_malicious: bool
- risk_score: float (numpy float64 in practice)
- label: str ("safe", "malicious", "blocked")
- technique_tags: list[str] (e.g., ["D1", "E1.1", "D4"])
- rule_hits: list[str] (e.g., ["override", "system_prompt", "structural:imperative_start"])
- ml_confidence: float (numpy float64)
- ml_label: str ("safe", "malicious")
- anomaly_flags: list[str] (from L0)
- rejected: bool
- rejection_reason: str

### Calibrated Risk Scores (from testing)
- Simple safe question: ~0.04
- Code request: ~0.10
- Creative writing: ~0.36
- Technical explanation: ~0.32
- Conversational: ~0.51
- Direct override attack: ~1.10
- DAN attack: ~1.04
- Base64 attack: ~1.16
- Delimiter injection: ~0.95
