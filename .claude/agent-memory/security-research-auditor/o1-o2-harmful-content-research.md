# O1 Harmful Content + O2 Output Format Exploitation - Research Notes

## Research Date: 2026-02-17

## Sources Investigated

### Primary References
1. **HarmBench** (arxiv:2402.04249) - Standardized Evaluation Framework for Automated Red Teaming
   - 400+ adversarial attacks across 7 semantic categories
   - Categories: Cybercrime, Chemical/Bio Weapons, Copyright, Misinfo, Harassment, Illegal, General Harm
   - Used for O1.1-O1.4 test payload design

2. **OWASP Gen AI Red Teaming Guide** (Jan 2025)
   - Formalized red teaming methodology for generative models
   - Covers model-level vulnerabilities, prompt injection, system integration
   - URL: https://genai.owasp.org/2025/01/22/announcing-the-owasp-gen-ai-red-teaming-guide/

3. **EchoLeak** (arxiv:2509.10540, Jun 2025)
   - First real-world zero-click prompt injection exploit
   - Microsoft 365 Copilot: email implants hidden instructions
   - Markdown image auto-fetch = data exfiltration
   - Critical for O2.1 test design

4. **ToxicChat** (Hugging Face: lmsys/toxic-chat)
   - 10K user prompts from Vicuna demo with toxicity annotations
   - Human-AI collaborative annotation framework
   - Overlap with prompt injection research (LeakSealer)

5. **SORRY-Bench** (ICLR 2025)
   - Safety Refusal Evaluation benchmark
   - Tests LLM refusal behavior across safety categories

### Secondary References
- Johann Rehberger / Embrace The Red - Markdown image exfiltration (known since mid-2023)
- Simon Willison - exfiltration attacks via rendered markdown
- HackerOne - Prompt injection leading to data exfiltration
- Palo Alto Unit 42 - Dual-use dilemma of malicious LLMs
- SentinelOne - LLM-enabled malware (MalTerminal, BlackMamba, QuietVault)
- Immersive Labs - Weaponizing LLMs via indirect prompt injection
- StrongestLayer - LLM-powered social engineering in enterprise scams

## Key Findings

### 1. Incidental Detection is Real
The detector catches many O1/O2 payloads despite having 0 training samples.
Three contributing factors:

**ML Vocabulary Overlap**: The TF-IDF + ML classifier learned features from
injection training data that overlap with harmful content vocabulary:
- "exfiltrate/exfiltrates" -> very strong signal (ML conf >0.97)
- "credentials" -> strong overlap with E1 extraction vocabulary
- "phishing" / "scam" / "tricks" -> social engineering shares vocabulary
- "instructions for" / "step-by-step" -> imperative structure overlap

**FingerprintStore**: data/fingerprints.db persists SHA-256 hashes. Once a
payload triggers detection, it's registered. Future scans match via:
- known_malicious_exact (content_hash)
- known_malicious_normalized (normalized_hash)
- known_malicious_token_pattern (token_hash)

**Structural Features**: imperative_start, high_entropy, and obfuscation
flags contribute additional weight to composite scores.

### 2. Detection Boundary Analysis
Payloads with injection-adjacent vocabulary ARE detected.
Payloads with domain-specific vocabulary (media, crypto, politics) are NOT.

**Detected vocabulary**: exfiltrate, credentials, phishing, scam, tricks,
backdoor, master password, malicious link, tracking pixel

**Undetected vocabulary**: encrypts, cryptocurrency, reverse shell, deepfake,
propaganda, campaign plan, grassroots activism

### 3. O2 is Structurally Closer to Injection than O1
O2 attacks INSTRUCT the LLM to modify its output format. This makes them
structurally similar to prompt injection (D1 instruction override):
- "Include this in your response" -> instruction pattern
- "Always add this link" -> persistent instruction override
- "Replace all URLs" -> output manipulation instruction

This explains why ALL O2 payloads are detected (ML 0.72-0.84) while
some pure O1 payloads are not.

### 4. FingerprintStore Concern
The FingerprintStore may create artificially high detection rates on
O1/O2 payloads that are really only caught because of prior test runs.
A fresh deployment without fingerprint data would have lower recall.
This is documented in the test docstrings but could be misleading.

## Test Architecture

### Test Classes (9 classes, 44 tests)
1. TestO1_1_ViolenceInstructions: 5 tests (3 pure + 2 hybrid)
2. TestO1_2_MalwareGeneration: 6 tests (1 pure pass + 3 xfail + 2 hybrid)
3. TestO1_3_SocialEngineeringScripts: 6 tests (4 pure + 1 hybrid + 1 observational)
4. TestO1_4_DisinformationGeneration: 5 tests (1 pure pass + 3 xfail + 1 hybrid)
5. TestO2_1_MarkdownInjection: 5 tests (3 pure + 2 hybrid)
6. TestO2_2_LinkInjection: 5 tests (3 pure + 1 hybrid + 1 observational)
7. TestO1O2_SafePrompts: 5 tests (false positive guards)
8. TestO1_HybridAttacks: 4 tests (multi-technique combinations)
9. TestO2_3_CodeInjectionOutput: 3 tests (2 pure + 1 hybrid)

### Expected Failures (6)
- O1.2: keylogger, ransomware, reverse shell
- O1.4: hate propaganda, deepfake script, disinfo campaign plan

### Key Design Decisions
- Used @expectedFailure for genuine gaps, not for "we don't know" cases
- Documented WHY each payload is/isn't detected in detailed docstrings
- Included ML confidence values to track detection stability over time
- Added _fmt() helper for rich diagnostic output on assertion failures
- Separated pure O1/O2 tests from hybrid (O + D technique) tests
