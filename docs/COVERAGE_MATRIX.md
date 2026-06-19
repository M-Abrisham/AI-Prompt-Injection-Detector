[← Back to main README](../README.md)

# Prompt-Injection Coverage Matrix

Authoritative, audit-ready mapping of Na0S's internal injection-method (IM) taxonomy to the OWASP Top 10 for LLM Applications (2025) and MITRE ATLAS, with a per-class coverage score, the implementing components, backing tests, and the residual detection gap.

---

| Field | Value |
|-------|-------|
| **Version** | v1.0-draft |
| **Last reviewed** | 2026-06-03 (initial agent-assisted build) |
| **Owner** | @M-Abrisham |
| **Source catalog — MITRE ATLAS** | v2026.05 (format-version 6.0.0; collection version 2026.05; modified 2026-05-27) — https://github.com/mitre-atlas/atlas-data/releases/download/v2026.05/ATLAS-2026.05.yaml |
| **Source catalog — OWASP** | OWASP Top 10 for LLM Applications 2025 (version 2025 / v2.0; `LLMxx:2025` label form) — https://genai.owasp.org/llm-top-10/ |
| **Review cadence** | Quarterly (re-confirm IDs against the then-current ATLAS / OWASP releases) |

---

## Scoring rubric

Each class is scored 0–100 on how completely Na0S detects it today. The score reflects three signals in order of weight: (1) a named implementing component exists in `src/`, (2) dedicated tests exercise it, and (3) the coverage spans the whole class rather than a sub-scope. Bands:

| Band | Score | Meaning |
|------|-------|---------|
| **Validated** | 85–100 | Full class coverage, implementing component verified, **and** a dedicated/targeted pytest backs it. Scores stay below 95 when no concrete pass-rate is captured. |
| **Asserted** | 70–84 | Component(s) verified and tests located, but tests are generic (probe-contract / shared-rule) rather than per-technique assertions, or coverage is a verified sub-scope of the class. |
| **Partial** | 45–69 | A real slice is covered and tested, but a named, material portion of the class has no detection path. |
| **Taxonomy-only** | 15–44 | Class is named in `data/taxonomy.yaml` and may have probe/sample assets, but has **no** dedicated runtime detector and **no** test; credit only for adjacent/overlapping detection. |
| **Not modeled** | 0–14 | No dedicated detector, samples, or taxonomy entry for the class; only incidental downstream mitigation. |

**Confidence** (`high` / `med` / `low`) qualifies how firmly the score is evidenced. **Validation** distinguishes `validated` (a named test was located) from `asserted` (claimed but no test located). The **FP/recall** column records whether benign/contrast samples exist for false-positive control; it does not assert a measured rate.

---

## Coverage matrix

| Row | OWASP LLM 2025 | MITRE ATLAS | Class (subtype) | Gate | Coverage | Conf. | Components | Validation | FP/recall | Severity | Residual gap |
|-----|----------------|-------------|-----------------|------|----------|-------|------------|------------|-----------|----------|--------------|
| **INJ-0001** (IM0001) | LLM01:2025 — Prompt Injection | AML.T0051.000 — LLM Prompt Injection: Direct | Direct Prompt Injection | input | **92** | high | `layer1/rules_registry.py` (real engine; `rules.py` is a 29-line shim), `layer0/`, `cascade.py`, `scripts/taxonomy/instruction_override.py` (D1.x), `data/taxonomy.yaml` D1–D8 | validated: `test_scan_d1_instruction_override.py`, d3–d8 scan suites, `test_rules.py`, `test_cascade.py`, `test_l12_probe_validation.py` | D1.x benign pairs present for FP control | critical | None material; cited `rules.py` is a compat shim, not the engine. |
| **INJ-0002** (IM0002) | LLM01:2025 — Prompt Injection | AML.T0051.000 — LLM Prompt Injection: Direct | Prompt Body Injection (in-body / structural-boundary) | input | **88** | high | `scripts/taxonomy/structural_boundary.py` (D3.4), `scripts/taxonomy/payload_delivery.py` (D7.3), `layer0/html_extractor.py`, `data/taxonomy.yaml` D3.4/D7.3 | validated: `test_scan_d3_structural_boundary.py`, `test_scan_d7_payload_delivery.py`, `test_html_extractor.py`, `test_l12_probe_validation.py` | D3.4/D7.3 benign samples in probe files | high | None material; delimiter/code-block subset is narrower than the full class. |
| **INJ-0003** (IM0003) | LLM01:2025 — Prompt Injection | AML.T0051.001 — LLM Prompt Injection: Indirect | Multimodal Prompt Injection (document/image attachments) | input | **68** | med | `scripts/taxonomy/multimodal_injection.py` (M3.1 PDF/DOCX hidden text, M1.2 SVG), `layer0/content_type.py`, `layer0/doc_extractor.py`, `layer0/ocr_extractor.py`, `layer0/image_threat.py` | validated: `test_doc_extractor.py`, `test_pdf_javascript.py`, `test_layer0_image_threat.py`, `test_ocr_extractor.py` | extractor tests cover benign docs/images | high | No audio (M2.x) path; OCR extractor exists but no OCR-injection classifier; cited M1.4/M1.5 IDs are fictional (actual: M3.1/M1.2). |
| **INJ-0004** (IM0004) | LLM01:2025 — Prompt Injection | *No ATLAS equivalent* — human-relay vector; closest `AML.T0011` User Execution (not exact) | Social-engineering delivery vector (user induced to relay payload) | input | **8** | low | None found modeling the social-engineering delivery vector as a distinct class. | asserted: no test located | n/a | medium | Not modeled. Submitted payload still hits IM0001 input detection, but the delivery class has no dedicated detector, samples, or taxonomy entry. |
| **INJ-0005** (IM0005) | LLM01:2025 — Prompt Injection | AML.T0011 — User Execution *(closest, not exact; human-delivery vector)* | Social-engineering delivery vector (user tricked into submitting payload) | input | **8** | low | None found. | asserted: no test located | n/a | medium | Not modeled. Only the resulting payload (if submitted) is caught by IM0001 input detectors. |
| **INJ-0006** (IM0006) | LLM05:2025 — Improper Output Handling | AML.T0061 — LLM Prompt Self-Replication | Output-channel injection / prompt self-replication (worm) | output | **78** | high | `src/na0s/output/scanner.py` (32 KB; `output_scanner.py` is a shim), `src/na0s/output/propagation.py` (`PropagationScanner`, `worm_propagation` tag, single-hop output→input rescan) | validated: `tests/output/test_propagation.py`, `test_scanner.py`, `test_scanner_redaction.py`, `test_advanced.py` | scanner redaction test covers benign outputs | high | Single-hop rescan present; multi-hop cross-**model** lineage / propagation across distinct systems is absent. |
| **INJ-0007** (IM0007) | LLM01:2025 — Prompt Injection | AML.T0080.001 — AI Agent Context Poisoning: Thread *(corrected from AML.T0092)* | Context manipulation / multi-turn splitting / middleware tampering | mixed | **58** | med | `scripts/taxonomy/context_overflow.py` (D8.x), `scripts/taxonomy/payload_delivery.py` (D7.2), `data/taxonomy.yaml` Category AD (AD1.1–AD3.6, 19 techniques) + IM4.x, `context_manipulation_detector.py` (592-byte shim) | validated: `test_scan_d8_context_manipulation.py`, `tests/test_layer16/test_payload_splitting.py` | D7.2/D8 benign samples in probe files | high | D8/D7.2 covered+tested, but Category AD (middleware/proxy tampering) and IM4.x are taxonomy-only — no probe, samples, or detector. |
| **INJ-0008** (IM0008) | LLM01:2025 — Prompt Injection | AML.T0051.001 — LLM Prompt Injection: Indirect | Indirect Prompt Injection via ingested data | input | **84** | high | `scripts/taxonomy/data_source_poisoning.py` (I1.x, incl. I1.7 email-sig, I1.8 broad-distribution), `scripts/taxonomy/html_markup_injection.py` (I2), `layer0/html_extractor.py`, `layer1/rules_registry.py` (RAG rules R1.1–R1.4) | validated: `test_rag_injection_rules.py`, `tests/rag/test_rag_poison_detector.py`, `tests/rag/test_rag_position_scanner.py`, `test_html_extractor.py` | I1.7/I1.8 benign samples present | critical | None material; tests are RAG-rule/probe-contract rather than per-I1.x assertions. |
| **INJ-0009** (IM0009) | LLM08:2025 — Vector and Embedding Weaknesses | AML.T0070 — RAG Poisoning | Indirect Injection via internal docs / knowledge-base poisoning | input | **76** | med | `scripts/taxonomy/data_source_poisoning.py` (I1.2 doc-injection, I1.4 database/KB poisoning incl. internal wiki/API-doc), `data/taxonomy.yaml` I1.2/I1.4, layer1 RAG rules | validated: `test_rag_injection_rules.py`, `tests/rag/test_rag_poison_detector.py` | data_source_poisoning benign samples cover internal-doc FPs | critical | No I1.4-specific named pytest; internal-KB poisoning exercised only via generic RAG-poison + probe-contract tests. |
| **INJ-0010** (IM0010) | LLM01:2025 — Prompt Injection | AML.T0051.001 — LLM Prompt Injection: Indirect | Indirect Injection via external web/email/feeds | input | **76** | med | `scripts/taxonomy/data_source_poisoning.py` (I1.1 web incl. Wikipedia, I1.3 email), `data/taxonomy.yaml` I1.1/I1.3 (RSS/API roll up under these) | validated: `test_rag_injection_rules.py`, `test_html_extractor.py` (DataSourcePoisoningProbe contract) | I1.1/I1.3 benign samples present | critical | No external-feed-specific pytest; RSS/API have no separate technique ID. |
| **INJ-0011** (IM0011) | LLM01:2025 — Prompt Injection | AML.T0051.001 — LLM Prompt Injection: Indirect | Indirect Injection from attacker-controlled source | input | **70** | med | `scripts/taxonomy/data_source_poisoning.py` (I1.1 attacker-controlled web), `data/taxonomy.yaml` AD2.4 webhook (taxonomy-only) | validated: `test_rag_injection_rules.py` (DataSourcePoisoningProbe contract) | shares I1.1 benign samples | high | No probe distinguishes "attacker-owned" from other external sub-flavors; webhook callback (AD2.4) is taxonomy-only. |
| **INJ-0012** (IM0012) | LLM01:2025 — Prompt Injection | AML.T0051.001 — LLM Prompt Injection: Indirect | Indirect Injection from compromised trusted source (GitHub/SO/npm) | input | **72** | med | `scripts/taxonomy/data_source_poisoning.py` (I1.1 explicit Stack Overflow / GitHub README samples); npm conceptually under same I1.1 block | validated: `test_rag_injection_rules.py` (DataSourcePoisoningProbe contract) | shares I1.1 benign samples | high | Explicit GitHub/SO samples exist but no sub-class pytest; npm-package poisoning is conceptual, not a distinct sample/detector. |
| **INJ-0013** (IM0013) | LLM01:2025 — Prompt Injection | AML.T0051.001 — LLM Prompt Injection: Indirect | Indirect Injection via attacker-influenced UGC (reviews/comments/wiki) | input | **72** | med | `scripts/taxonomy/data_source_poisoning.py` (I1.1 explicit Yelp/Reddit/wiki-edit samples) | validated: `test_rag_injection_rules.py` (DataSourcePoisoningProbe contract) | shares I1.1 benign samples | high | Explicit UGC samples exist but no sub-class pytest; detection is generic I1.1, not a distinct UGC-influence detector. |
| **INJ-0014** (IM0014) | LLM03:2025 — Supply Chain | AML.T0010.002 — AI Supply Chain Compromise: Data | Supply-chain / ETL ingestion-pipeline compromise | mixed | **52** | med | `scripts/taxonomy/supply_chain.py` (S1.3–S1.5 samples), `scripts/taxonomy/ingestion_manipulation.py` (51 KB, IG1.x incl. IG1.7 ETL), `data/taxonomy.yaml` S1.3–S1.5 + IG | validated: `test_l12_probe_validation.py` (SupplyChainProbe, IngestionManipulationProbe contract); asserted: no technique-specific detector test | supply_chain/ingestion probes include benign pairs | high | Samples + probes exist, but **no** dedicated runtime detector module and only generic contract tests — no technique-specific assertions. |
| **INJ-0015** (IM0015) | LLM01:2025 — Prompt Injection | AML.T0080.001 — AI Agent Context Poisoning: Thread | Context-memory poisoning via prior model output | mixed | **86** | high | `src/na0s/layer16/detectors/context_poisoning.py` (`ContextPoisoningDetector`, `taxonomy_ids==['D1.20']`), `scripts/taxonomy/instruction_override.py` (D1.20 samples), `data/taxonomy.yaml` D1.20 | validated: `tests/test_layer16/test_context_poisoning.py` (~20+ test fns) | D1.20_benign + false-agreement benign cases in test | high | None material; samples + dedicated detector + dedicated test all exist. |
| **INJ-0016** (IM0016) | LLM01:2025 — Prompt Injection *(corrected from LLM08:2025)* | AML.T0080.000 — AI Agent Context Poisoning: Memory | Persistent agent-memory poisoning | agent-tool | **40** | low | `data/taxonomy.yaml` I1.6 / IG1.8 (named); `scripts/taxonomy/data_source_poisoning.py` (I1.4/I1.5 vector/cache-DB samples — adjacent, not the named technique). No dedicated agent-memory detector in `src/`. | asserted: no test located | n/a for the named class (no dedicated samples) | high | No explicit agent-memory-attack detector and no named pytest; named technique (I1.6/IG1.8) is taxonomy-only. |
| **INJ-0017** (IM0017) | LLM06:2025 — Excessive Agency | AML.T0080.000 — AI Agent Context Poisoning: Memory *(corrected from AML.T0080.001 Thread)* | Multi-agent / inter-model propagation | agent-tool | **90** | high | `src/na0s/detectors/inter_model.py` (`detect_inter_model` / `InterModelResult` / `get_inter_model_weight`; 76-pattern matching layer across 6 fabricated-cross-model-authority families IM-FAM-1..6, covering all 29 IM techniques) — wired **input-side** into `predict.py` composite scoring and `cascade.py` (parity block + positive-validation veto + whitelist tripwire); `scripts/taxonomy/inter_model_propagation.py` (516 mal / 55 benign), `data/taxonomy.yaml` IM1–IM6 | validated: `tests/detectors/test_inter_model.py` (recall + benign-FP TDD), `tests/detectors/test_inter_model_wiring.py` (per-technique coverage + predict/cascade integration) | detector 516/516 recall, 0/55 probe benign + 0/500 safe_holdout + 0/71 adversarial hard-negatives; `predict.scan` 25.78%→**100%**, cascade 12.4%→**99.81%**, benign FP unchanged (1.82%, pre-existing) | critical | Validated on the synthetic IM probe corpus; real-world attack-traffic + paraphrase-robustness validation remain, and cascade recall was measured with the L7 judge unavailable. |

---

## Summary rollup

**17 classes mapped.** Mean (equal-weight) coverage: **65.8%**. Severity-weighted coverage (critical×4, high×3, medium×2, low×1): **69.6%**. *(INJ-0017 inter-model propagation lifted 44→90 when the `detect_inter_model` matching layer landed and was wired into both decision paths.)*

**By score band**

| Band | Count | Rows |
|------|-------|------|
| Validated (85–100) | 3 | INJ-0001, INJ-0002, INJ-0015 |
| Asserted (70–84) | 7 | INJ-0006, INJ-0008, INJ-0009, INJ-0010, INJ-0011, INJ-0012, INJ-0013 |
| Partial (45–69) | 3 | INJ-0003, INJ-0007, INJ-0014 |
| Taxonomy-only (15–44) | 2 | INJ-0016, INJ-0017 |
| Not modeled (0–14) | 2 | INJ-0004, INJ-0005 |

INJ-0002 scores 88 and sits in the Validated band; INJ-0008 scores 84 (top of Asserted).

**By severity**

| Severity | Count | Rows |
|----------|-------|------|
| Critical | 5 | INJ-0001, INJ-0008, INJ-0009, INJ-0010, INJ-0017 |
| High | 10 | INJ-0002, INJ-0003, INJ-0006, INJ-0007, INJ-0011, INJ-0012, INJ-0013, INJ-0014, INJ-0015, INJ-0016 |
| Medium | 2 | INJ-0004, INJ-0005 |
| Low | 0 | — |

**Priority signal.** With INJ-0017 now closed (44→90 — `detect_inter_model` wired input-side into both decision paths), the most actionable remaining gaps are **INJ-0016** (persistent agent-memory poisoning, 40 — taxonomy-only, no detector, no test) and the supply-chain row **INJ-0014** (52 — samples/probes but no runtime detector). The agent-tool / supply-chain surface remains the weakest-covered overall.

> **Row-key note.** The parenthetical `(IM####)` after each `INJ-####` id is the legacy *injection-method* row alias and is **distinct from** the `data/taxonomy.yaml` category **IM = Inter-Model Propagation** (techniques IM1.1–IM6.6, the subject of row INJ-0017). The `INJ-####` id is authoritative; the `IM####` alias is retained only for backward reference.

---

## Rows needing external-taxonomy judgment

The following rows carry `no_clean_atlas_match = true`: no MITRE ATLAS technique cleanly describes the class, and the listed ID is the closest available approximation. These mappings are deliberately conservative and should be re-checked each review cycle against new ATLAS releases.

- **INJ-0004 — human-relay delivery (user induced to paste/relay the prompt).** **Decision: no clean ATLAS equivalent.** Verified against MITRE ATLAS **v2026.05** (all candidate IDs confirmed present: `T0011`, `T0051.000/.001/.002`, `T0052/.000`, `T0065`, `T0093`). No technique describes a human socially-engineered into relaying an attacker's prompt into an AI. The closest neighbor, `AML.T0011` User Execution (*"rely upon specific actions by a user… social engineering to get them to… open a malicious document file or link"*; maps to ATT&CK T1204), is scoped to gaining **execution** (code, packages, tools, links) — not relaying a text prompt, which produces inference, not execution. Note the row's legacy "Indirect PI" title is a misnomer in the ATLAS sense (ATLAS "Indirect" = payload pulled from a separate **data** channel and hidden from the user; here the user consciously pastes it).
- **INJ-0005 — deceived human as unwitting courier (user tricked into forwarding/submitting a payload).** **Decision: keep `AML.T0011` User Execution as "closest, not exact."** Same catalog verification. T0011 is the nearest match by tactic (Execution / `AML.TA0005`) and actor framing (adversary relies on a deceived user's action), but ATLAS scopes the action to executing code / invoking a tool / clicking a link rather than couriering a text payload. Rejected by catalog text: `T0093` (adversary plants the payload — wrong actor), `T0052/.000` (adversary/LLM phishes the victim — inverse direction), `T0051.001/.002` (the "unwitting user" is the injection victim, not the courier). The two rows differ deliberately: INJ-0004 is pure prompt-relay (no artifact), whereas INJ-0005 couriers an artifact, which sits one step closer to T0011's "open a malicious document."

No other rows are flagged `no_clean_atlas_match`. INJ-0017 originally read as borderline but resolves cleanly to `AML.T0080.000 (Memory)` after the verification correction. *(Source note: verify ATLAS IDs against the **v2026.05** release file pinned in the header — the `main/dist/ATLAS.yaml` branch currently serves an older 5.6.0 build that lacks `AML.T0093` and the `T0011` sub-techniques.)*

---

## How this maps back (reverse traceability)

Read the matrix in either direction:

- **From a framework finding to Na0S.** Given an OWASP `LLMxx:2025` item or a MITRE ATLAS `AML.Txxxx` technique, scan the matching column to find every `INJ-####` / `IM####` class that addresses it, then jump to the **Components** column for the implementing module(s) and the **Validation** column for the backing test(s). Note that LLM01:2025 and `AML.T0051.001 (Indirect)` are many-to-one anchors (INJ-0008 through INJ-0013 all specialize the indirect-injection surface by data source).
- **From Na0S code to a framework.** Each `INJ-####` row keys to an internal `IM####` taxonomy code (1:1) and to the `D*/I*/M*/S*/IG*/IM*/AD/T*` technique families in `data/taxonomy.yaml` and `scripts/taxonomy/`. A code change in any listed component traces forward to its OWASP + ATLAS obligations via that row.
- **Audit trail.** Four mapping corrections versus the initial draft — INJ-0007 (ATLAS `AML.T0092` → `AML.T0080.001`), INJ-0016 (OWASP `LLM08:2025` → `LLM01:2025`), INJ-0017 (ATLAS `AML.T0080.001` → `AML.T0080.000`), and INJ-0004 (ATLAS `AML.T0011` closest-fit → *no ATLAS equivalent*, after per-row analysis of the human-relay vector) — each made to match the authoritative ATLAS v2026.05 / OWASP 2025 catalog semantics. Re-validate all IDs at the next quarterly review.

---

*Owner: @M-Abrisham. Last validated: 2026-06-03 (initial agent-assisted build). Re-confirm every quarter against the then-current MITRE ATLAS and OWASP LLM Top 10 releases.*
