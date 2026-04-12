[← Back to main README](../README.md)

# Standards & Compliance Mapping

Na0S is a defensive prompt injection detection library. This document maps its detection capabilities to industry security frameworks, enabling organizations to assess Na0S against their compliance requirements.

---

## Framework Coverage Summary

| Framework | Version | Coverage | Integration |
|-----------|---------|----------|-------------|
| [OWASP LLM Top 10](https://genai.owasp.org/) | 2025 | 9 of 10 items | 62 taxonomy tags, active monitoring via `layer15/owasp_sync.py` |
| [MITRE ATLAS](https://atlas.mitre.org/) | Current | Adversarial ML techniques | Live sync via `layer15/atlas_sync.py` |
| [AVID](https://avidml.org/) | Current | Security + Ethics taxonomy | 39 taxonomy tags |
| [LMRC](https://github.com/leondz/lm-risk-cards) | Current | 11 risk card types | 14 taxonomy tags |
| [NIST AI RMF](https://www.nist.gov/artificial-intelligence/executive-order-safe-secure-and-trustworthy-artificial-intelligence) | 1.0 | Govern, Map, Measure, Manage | Architectural alignment (see below) |

---

## OWASP LLM Top 10 (2025) Mapping

Na0S provides detection coverage for 9 of the 10 OWASP LLM Top 10 risk categories. Each mapping shows which Na0S detection layers address the risk.

| OWASP ID | Risk | Na0S Coverage | Detection Layers |
|----------|------|---------------|------------------|
| **LLM01** | Prompt Injection | **Primary focus** — 21 detection layers, 29 attack categories, 276+ techniques | L0-L10, L15-L16, all detectors |
| **LLM02** | Sensitive Information Disclosure | Output scanning for secrets, PII, system prompt leakage | L9 (output_scanner), `rag/`, `detectors/privacy_probe` |
| **LLM03** | Supply Chain Vulnerabilities | Model integrity verification, dependency scanning, SBOM | L11 (`integrity/safe_pickle`, `integrity/sbom`, `integrity/dep_scanner`) |
| **LLM04** | Data Poisoning | Indirect injection detection in RAG pipelines, document parsing | `rag/poison_detector`, `rag/propagation`, `parsers/office/` |
| **LLM05** | Improper Output Handling | Output scanning for role breaks, instruction leakage | L9 (`rag/output_scanner`), L10 (`canary/`) |
| **LLM06** | Excessive Agency | Agent/tool abuse detection, MCP tool injection | `detectors/mcp_tool`, `detectors/payload_assembly` |
| **LLM07** | System Prompt Leakage | Canary token detection, extraction attack patterns | L10 (`canary/`), `detectors/extraction`, L1 rules |
| **LLM08** | Vector and Embedding Weaknesses | Adversarial embedding detection, embedding drift monitoring | L5 (`ml/embedding_classifier`), L16 embedding drift detector |
| **LLM09** | Misinformation | Groundedness checking, hallucination indicators | `groundedness.py`, L8 positive validation |
| **LLM10** | Unbounded Consumption | Rate limiting, resource guards, input size caps | L0 (`layer0/resource_guard`), `judge/rate_limiter` |

### Taxonomy Tag Distribution

The Na0S threat taxonomy (`data/taxonomy.yaml`) embeds OWASP mappings directly via `owasp-llm:2025:llmXX` tags:

| OWASP ID | Taxonomy Tags | Example Categories |
|----------|---------------|-------------------|
| LLM01 | 20 tags | D1 (Override), D2 (Roleplay), D3 (Boundary), D4 (Obfuscation) |
| LLM02 | 8 tags | E (Exfiltration), P (Privacy), P2 (Privacy Extraction) |
| LLM04 | 6 tags | I1 (Data Source Poisoning), I2 (Markup Injection) |
| LLM07 | 9 tags | E (Exfiltration), D8 (Context Window) |
| LLM08 | 4 tags | A (Adversarial ML) |
| Others | 15 tags | T (Agent/Tool), R (Resource), O (Output), S (Supply Chain) |

---

## MITRE ATLAS Mapping

Na0S integrates with [MITRE ATLAS](https://atlas.mitre.org/) (Adversarial Threat Landscape for AI Systems) through `layer15/atlas_sync.py`, which fetches the latest ATLAS technique definitions from the ATLAS GitHub repository.

| ATLAS Tactic | Na0S Detection |
|--------------|----------------|
| **Reconnaissance** | `detectors/recon` — probing, capability testing, boundary exploration |
| **Initial Access** | L1 rules — prompt injection patterns, social engineering |
| **ML Model Access** | `integrity/model_provenance` — model artifact verification |
| **Execution** | `detectors/payload_assembly` — fragmented payload reconstruction |
| **Persistence** | `worm/detector` — self-replicating injection patterns |
| **Exfiltration** | `detectors/extraction`, `canary/` — system prompt extraction, data leakage |
| **Impact** | `rag/output_scanner` — role breaks, unauthorized actions |

### Living Sync

The `layer15/atlas_sync.py` module monitors the ATLAS GitHub repository for new technique definitions and maps them to Na0S taxonomy categories. This ensures coverage stays current as new adversarial ML techniques are documented.

---

## AVID (AI Vulnerability Database) Mapping

Na0S taxonomy tags reference AVID effect categories:

| AVID Category | Code | Na0S Coverage |
|---------------|------|---------------|
| Security: Integrity | S0403 | Evasion attacks (D4-D5 obfuscation, D6 multilingual) |
| Security: Availability | S0301 | Resource exhaustion (R category) |
| Security: Confidentiality | S0100 | Data exfiltration (E, P, P2 categories) |
| Ethics: Fairness | E0100 | Bias-based evasion (C1 compliance evasion) |
| Ethics: Performance | E0200 | Model degradation attacks (A adversarial ML) |

---

## NIST AI Risk Management Framework Alignment

Na0S architectural decisions align with the four functions of the [NIST AI RMF 1.0](https://www.nist.gov/artificial-intelligence/executive-order-safe-secure-and-trustworthy-artificial-intelligence):

### GOVERN — Establish AI risk management culture

| NIST Subcategory | Na0S Implementation |
|------------------|---------------------|
| GOVERN 1.1: Legal and regulatory requirements | OWASP LLM Top 10 compliance mapping (this document) |
| GOVERN 1.5: Risk assessment processes | 29-category threat taxonomy with severity ratings |
| GOVERN 4.1: Organizational practices for AI risk | `SECURITY.md` with 90-day coordinated disclosure policy |

### MAP — Contextualize AI risks

| NIST Subcategory | Na0S Implementation |
|------------------|---------------------|
| MAP 1.1: Intended purpose and context | Na0S is explicitly scoped to prompt injection detection for LLM applications |
| MAP 2.3: Scientific integrity of AI system | Reproducible evaluation via `scripts/evaluate_probes.py`, taxonomy-driven test coverage |
| MAP 5.1: Likelihood of AI risks | Risk scoring (0.0-1.0) with configurable thresholds per detection layer |

### MEASURE — Analyze and monitor AI risks

| NIST Subcategory | Na0S Implementation |
|------------------|---------------------|
| MEASURE 1.1: Approaches for measurement | Multi-layer detection with 21 independent signals; false-positive rate monitoring |
| MEASURE 2.5: Evaluation of AI system performance | 8,500+ automated tests, regression testing via CI, probe-based evaluation |
| MEASURE 2.6: Structured testing | 29-category adversarial probe framework with 8 mutation strategies (Base64, ROT13, Leet, Fullwidth, ZeroWidth, Homoglyph, Reverse, CaseAlternating) |
| MEASURE 4.2: Measurement approaches for trustworthiness | Cascade voting with explainability (signal_boost reasons trace, judge audit logs) |

### MANAGE — Prioritize and act on AI risks

| NIST Subcategory | Na0S Implementation |
|------------------|---------------------|
| MANAGE 1.1: Risk prioritization | Severity-based rule weighting (critical/high/medium/low); configurable cascade thresholds |
| MANAGE 2.1: Risk response | Graduated response system (`layer16/graduated_response.py`); block/flag/allow actions |
| MANAGE 2.4: Residual risk documentation | Threat taxonomy gaps tracked in `ROADMAP_V2.md`; known limitations in README disclaimer |
| MANAGE 4.2: Incident response | `SECURITY.md` vulnerability disclosure process; canary token breach detection |

---

## Detection Architecture vs. Standards

This table shows how Na0S's layered architecture maps to security standards requirements:

| Na0S Layer | What It Does | OWASP | ATLAS | NIST AI RMF |
|------------|-------------|-------|-------|-------------|
| L0 Input | Sanitize, normalize, parse | LLM01 | Initial Access | MEASURE 2.6 |
| L1 Rules | Pattern matching (98 rules) | LLM01 | Reconnaissance | MEASURE 1.1 |
| L2 Obfuscation | Decode evasion (base64, hex, ROT13, etc.) | LLM01 | Execution | MEASURE 2.6 |
| L3 Structural | 29 numeric features | LLM01 | — | MEASURE 2.5 |
| L4 ML (TF-IDF) | Logistic regression classifier | LLM01 | — | MEASURE 1.1 |
| L5 ML (Embedding) | Sentence-transformer similarity | LLM01, LLM08 | ML Model Access | MEASURE 1.1 |
| L6 Cascade | Weighted voting, RRF fusion | LLM01 | — | MEASURE 4.2 |
| L7 LLM Judge | GPT-4o / Llama second opinion | LLM01 | — | MANAGE 1.1 |
| L8 Validation | False-positive reduction | LLM09 | — | MEASURE 2.5 |
| L9 Output | Scan responses for leaks | LLM02, LLM05, LLM07 | Exfiltration | MANAGE 2.1 |
| L10 Canary | Honeytokens for leak detection | LLM07 | Exfiltration | MANAGE 4.2 |
| L11 Integrity | Model/supply chain verification | LLM03 | Persistence | GOVERN 4.1 |
| L15 Threat Intel | ATLAS/OWASP live sync | All | All | MAP 2.3 |
| L16 Conversation | Multi-turn monitoring, escalation | LLM01, LLM06 | Persistence | MANAGE 2.1 |

---

## Threat Taxonomy Categories

Na0S defines 29 attack categories in [`data/taxonomy.yaml`](../data/taxonomy.yaml):

| ID | Category | Type | Techniques | Severity | OWASP |
|----|----------|------|:----------:|----------|-------|
| D1 | Instruction Override | Direct | 22 | Critical | LLM01 |
| D2 | Persona/Roleplay Hijack | Direct | 4 | High | LLM01 |
| D3 | Structural Boundary | Direct | 4 | High | LLM01 |
| D4 | Obfuscation/Encoding | Direct | 6 | High | LLM01 |
| D5 | Unicode Evasion | Direct | 7 | High | LLM01 |
| D6 | Multilingual Injection | Direct | 6 | Medium | LLM01 |
| D7 | Payload Delivery | Direct | 6 | High | LLM01 |
| D8 | Context Window Manipulation | Direct | 6 | High | LLM01, LLM07 |
| I1 | Data Source Poisoning | Indirect | 8 | Critical | LLM04 |
| I2 | HTML/Markup Injection | Indirect | 3 | High | LLM04 |
| E | Exfiltration | Extraction | 11 | Critical | LLM02, LLM07 |
| A | Adversarial ML | Evasion | 5 | High | LLM08 |
| O | Output Manipulation | Output | 11 | High | LLM05 |
| T | Agent/Tool Abuse | Agent | 7 | Critical | LLM06 |
| R | Resource/Availability | DoS | 5 | Medium | LLM10 |
| P | Privacy/Data Leakage | Privacy | 6 | Critical | LLM02 |
| P2 | Privacy Extraction | Privacy | 4 | Critical | LLM02 |
| P3 | Malicious Code Generation | Code | 4 | High | LLM05 |
| M | Multimodal Injection | Multimodal | 14 | High | LLM01 |
| S | Supply Chain/Integrity | Supply Chain | 8 | Critical | LLM03 |
| C | Compliance/Policy Evasion | Compliance | 8 | Medium | — |
| C1 | Compliance (Advanced) | Compliance | 8 | Medium | — |
| IM | Inter-Model Propagation | Propagation | 29 | Critical | LLM04 |
| IG | Ingestion Manipulation | Ingestion | 12 | High | LLM04 |
| AD | Altered Delivery | Delivery | 19 | High | LLM01 |
| CT | Combo Techniques | Combined | 20 | Critical | LLM01 |
| AB | Adversarial Benchmarks | Benchmark | 12 | High | — |
| MB | Multi-Buff Combos | Combined | 15 | High | LLM01 |
| C1MT | Compliance Multi-Turn | Multi-Turn | 6 | Medium | — |

---

## Live Standards Monitoring

Na0S Layer 15 (`layer15/`) includes automated threat intelligence synchronization:

| Source | Module | What It Monitors |
|--------|--------|------------------|
| MITRE ATLAS | `atlas_sync.py` | New adversarial ML techniques, tactic updates |
| OWASP LLM Top 10 | `owasp_sync.py` | Version changes, new risk categories |
| Garak | `garak_sync.py` | New probe types and detector patterns |
| AIID | `aiid_sync.py` | AI incident reports relevant to prompt injection |
| JailbreakBench | `jailbreakbench_sync.py` | New jailbreak techniques and defense evaluations |

This ensures Na0S taxonomy coverage evolves with the threat landscape rather than becoming stale.

---

## Compliance Checklist for Integrators

Organizations integrating Na0S into their AI applications can use this checklist:

- [ ] **OWASP LLM01**: Na0S `scan()` called on all user inputs before LLM processing
- [ ] **OWASP LLM02**: Na0S output scanner enabled on LLM responses (`scan_output()`)
- [ ] **OWASP LLM03**: Model integrity verification enabled (`integrity/safe_pickle`)
- [ ] **OWASP LLM04**: RAG pipeline inputs scanned (`rag/poison_detector`)
- [ ] **OWASP LLM05**: Output scanning configured for role breaks and leaks
- [ ] **OWASP LLM06**: Agent/tool call inputs scanned before execution
- [ ] **OWASP LLM07**: Canary tokens injected in system prompts (`canary/`)
- [ ] **OWASP LLM10**: Rate limiting enabled on LLM judge calls (`judge/rate_limiter`)
- [ ] **Detection thresholds**: Reviewed and tuned for your risk tolerance
- [ ] **Logging**: Audit logs enabled for detection decisions
- [ ] **Updates**: Na0S version pinned and update process defined

---

*This document covers Na0S v0.1.x. Standards mappings are updated with each minor release. Last reviewed: April 2026.*
