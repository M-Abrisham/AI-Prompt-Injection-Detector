[← Back to main README](../README.md)

# Threat Taxonomy Coverage

This page documents the **30** categories (29 attack + 1 benign `BEN` sentinel) and **278** techniques covered by the AI Prompt Injection Detector. For the complete taxonomy definition including all technique IDs, descriptions, and example payloads, see [`THREAT_TAXONOMY.md`](../THREAT_TAXONOMY.md).

---

## <img src="https://img.shields.io/badge/-THREAT_TAXONOMY-457B9D?style=for-the-badge&labelColor=FF6B35" alt="" /> Threat Taxonomy Coverage

**30** categories (29 attack + 1 benign `BEN` sentinel) with **278 techniques**, mapped to [OWASP LLM Top 10 2025](https://genai.owasp.org/), AVID, and LMRC frameworks.

| Category | Name | Techniques | Coverage |
|:--------:|------|:----------:|:--------:|
| **D1** | Instruction Override | 22 | ![Full](https://img.shields.io/badge/FULL-3fb950?style=flat-square) |
| **D2** | Persona / Roleplay Hijack | 4 | ![Full](https://img.shields.io/badge/FULL-3fb950?style=flat-square) |
| **D3** | Structural Boundary Injection | 4 | ![Full](https://img.shields.io/badge/FULL-3fb950?style=flat-square) |
| **D4** | Obfuscation / Encoding | 6 | ![Partial](https://img.shields.io/badge/PARTIAL-F1C40F?style=flat-square) |
| **D5** | Unicode Evasion | 7 | ![Full](https://img.shields.io/badge/FULL-3fb950?style=flat-square) |
| **D6** | Multilingual Injection | 6 | ![Full](https://img.shields.io/badge/FULL-3fb950?style=flat-square) |
| **D7** | Payload Delivery Tricks | 6 | ![Full](https://img.shields.io/badge/FULL-3fb950?style=flat-square) |
| **D8** | Context Window Manipulation | 6 | ![Full](https://img.shields.io/badge/FULL-3fb950?style=flat-square) |
| **I1** | Data Source Poisoning | 8 | ![Full](https://img.shields.io/badge/FULL-3fb950?style=flat-square) |
| **I2** | HTML / Markup Injection | 3 | ![Full](https://img.shields.io/badge/FULL-3fb950?style=flat-square) |
| **E** | Exfiltration | 11 | ![Full](https://img.shields.io/badge/FULL-3fb950?style=flat-square) |
| **A** | Adversarial ML | 5 | ![Full](https://img.shields.io/badge/FULL-3fb950?style=flat-square) |
| **O** | Output Manipulation | 11 | ![Full](https://img.shields.io/badge/FULL-3fb950?style=flat-square) |
| **T** | Agent / Tool Abuse | 7 | ![Full](https://img.shields.io/badge/FULL-3fb950?style=flat-square) |
| **R** | Resource / Availability | 5 | ![Full](https://img.shields.io/badge/FULL-3fb950?style=flat-square) |
| **P** | Privacy / Data Leakage | 6 | ![Full](https://img.shields.io/badge/FULL-3fb950?style=flat-square) |
| **P2** | Privacy Extraction Attacks | 4 | ![Full](https://img.shields.io/badge/FULL-3fb950?style=flat-square) |
| **P3** | Malicious Code Generation | 4 | ![Full](https://img.shields.io/badge/FULL-3fb950?style=flat-square) |
| **M** | Multimodal Injection | 14 | ![Full](https://img.shields.io/badge/FULL-3fb950?style=flat-square) |
| **S** | Supply Chain / Integrity | 8 | ![Full](https://img.shields.io/badge/FULL-3fb950?style=flat-square) |
| **C** | Compliance / Policy Evasion | 8 | ![Full](https://img.shields.io/badge/FULL-3fb950?style=flat-square) |
| **C1** | Compliance / Policy Evasion (C1) | 8 | ![Full](https://img.shields.io/badge/FULL-3fb950?style=flat-square) |
| **IM** | Inter-Model Propagation | 29 | ![Full](https://img.shields.io/badge/FULL-3fb950?style=flat-square) |
| **IG** | Ingestion Manipulation | 12 | ![Full](https://img.shields.io/badge/FULL-3fb950?style=flat-square) |
| **AD** | Altered Delivery | 19 | ![Full](https://img.shields.io/badge/FULL-3fb950?style=flat-square) |
| **CT** | Combo Techniques | 20 | ![Full](https://img.shields.io/badge/FULL-3fb950?style=flat-square) |
| **AB** | Adversarial Benchmarks | 12 | ![Full](https://img.shields.io/badge/FULL-3fb950?style=flat-square) |
| **MB** | Multi-Buff Combos | 15 | ![Full](https://img.shields.io/badge/FULL-3fb950?style=flat-square) |
| **C1MT** | Compliance Multi-Turn Probes | 6 | ![Full](https://img.shields.io/badge/FULL-3fb950?style=flat-square) |
| **BEN** | Benign Control (FPR sentinel) | 2 | ![Benign](https://img.shields.io/badge/BENIGN-457B9D?style=flat-square) |

<details>
<summary><strong>Attack Category Distribution (click to expand)</strong></summary>

<br/>

```mermaid
%%{init: {'theme':'neutral'}}%%
pie title Attack Category Distribution (278 Techniques)
    "IM: Inter-Model Propagation" : 29
    "D1: Instruction Override" : 22
    "CT: Combo Techniques" : 20
    "AD: Altered Delivery" : 19
    "MB: Multi-Buff Combos" : 15
    "M: Multimodal Injection" : 14
    "IG: Ingestion Manipulation" : 12
    "AB: Adversarial Benchmarks" : 12
    "E: Exfiltration" : 11
    "O: Output Manipulation" : 11
    "Others (20 categories)" : 113
```

</details>

<details>
<summary><strong>Detection Flow — Sequence Diagram (click to expand)</strong></summary>

<br/>

```mermaid
%%{init: {'theme':'neutral'}}%%
sequenceDiagram
    participant U as User Input
    participant L0 as L0: Sanitizer
    participant L1 as L1: Rules
    participant L4 as L4: ML-TFIDF
    participant L5 as L5: Embeddings
    participant L6 as L6: Cascade
    participant L7 as L7: LLM Judge
    participant V as Verdict

    U->>L0: "Ignore all previous instructions"
    L0->>L0: NFKC normalize, validate
    L0->>L1: Sanitized text
    L1->>L1: Pattern match: "override"
    L0->>L4: Feature extraction
    L4->>L4: TF-IDF score: 0.945
    L0->>L5: Embedding extraction
    L5->>L5: Cosine score: 0.912
    L1->>L6: Rule: override (critical)
    L4->>L6: ML score: 0.945
    L5->>L6: Embed score: 0.912
    L6->>L6: Weighted: 0.931 (HIGH)
    Note over L6,L7: Confidence > 0.85 - Skip LLM Judge
    L6->>V: MALICIOUS (93.1%)
```

</details>
