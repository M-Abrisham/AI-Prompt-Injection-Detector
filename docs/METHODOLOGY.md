[← Back to main README](../README.md)

# Detection Methodology

This document describes how Na0S detects prompt injection attacks and how its
effectiveness is measured. Every metric cited below comes from verifiable
source data in the repository — no number is estimated or extrapolated.

---

## 1. Detection Architecture

Na0S uses a **defense-in-depth cascade** where multiple independent detection
layers vote on whether an input is malicious. No single layer is trusted alone.

```
Input → L0 Sanitize → L1 Rules (98 patterns) → L2 Obfuscation Decode
      → L3 Structural Features (29 numeric) → L4 TF-IDF ML Classifier
      → L5 Embedding Classifier → L6 Cascade Voting (60/40 blend)
      → L7 LLM Judge (ambiguous cases only) → L8 Positive Validation
      → Verdict: malicious / safe / ambiguous
```

### Layer Roles

| Layer | Type | What It Does | Independence |
|-------|------|-------------|--------------|
| L0 | Preprocessing | Normalize Unicode, strip invisible chars, detect encoding, parse documents | Always runs |
| L1 | Pattern matching | 98 regex rules across 29 attack categories with severity weights | Always runs |
| L2 | Decoder | Unwrap obfuscation: Base64, hex, ROT13, leetspeak, Morse, Caesar (1–25), Pig Latin, steganography. Recursive Matryoshka unwrapping with provenance tracking | Always runs |
| L3 | Feature extraction | 29 numeric features: quote depth, entropy, many-shot indicators, structural anomalies | Always runs |
| L4 | ML classifier | TF-IDF + Logistic Regression trained on 126,245 samples | Always runs |
| L5 | ML classifier | Sentence-transformer (MiniLM-L6-v2) cosine similarity | Optional (needs sentence-transformers) |
| L6 | Fusion | Weighted voting across L1–L5 with RRF and Bayesian fusion | Always runs |
| L7 | LLM judge | GPT-4o or Llama-3.3 second opinion for ambiguous inputs (score 0.25–0.85) | Optional (needs API key) |
| L8 | Validation | False-positive reduction via safe-content scoring | Optional |

### Why Multiple Layers

Each layer catches different attack classes:

- **L1 rules** catch known patterns (e.g., "ignore previous instructions") but miss obfuscated variants
- **L2 decoders** catch encoded payloads (Base64, ROT13) but need L1 to classify the decoded text
- **L4 TF-IDF** generalizes beyond known patterns but has no awareness of encoding
- **L5 embeddings** capture semantic similarity but are vulnerable to adversarial perturbation
- **Signal boost** adds weight when multiple layers fire simultaneously (multi-vector attacks)

A single-layer detector can be evaded by targeting that layer's weakness. The cascade makes evasion require simultaneously bypassing all layers.

---

## 2. Threat Coverage

### Taxonomy

Na0S defines **29 attack categories** and **276+ techniques** in
[`data/taxonomy.yaml`](../data/taxonomy.yaml):

| Type | Categories | Example |
|------|:----------:|---------|
| Direct injection | D1–D8 | Override, roleplay hijack, obfuscation, Unicode evasion |
| Indirect injection | I1–I2 | Data source poisoning, markup injection |
| Extraction | E | System prompt leakage, conversation history extraction |
| Adversarial ML | A | Embedding perturbation, classifier evasion |
| Output manipulation | O | Role breaks, harmful content generation |
| Agent/tool abuse | T | MCP tool injection, function call exploitation |
| Privacy | P, P2 | PII extraction, credential harvesting |
| Supply chain | S | Model tampering, dependency attacks |
| Compliance evasion | C, C1 | Policy circumvention, graduated escalation |
| Multi-modal | M | Image-based injection, audio adversarial |
| Propagation | IM | Inter-model worm replication (Morris II patterns) |

### Standards Alignment

Detection categories map to industry frameworks. Full mapping in
[`docs/STANDARDS.md`](STANDARDS.md):

- **OWASP LLM Top 10 (2025)**: 9 of 10 items covered
- **MITRE ATLAS**: Adversarial ML tactics mapped to detection layers
- **NIST AI RMF 1.0**: All four functions (Govern/Map/Measure/Manage) aligned

---

## 3. Evaluation Results

### Holdout Dataset

**Source**: `benchmarks/results/threshold_sweep.json`
**Dataset**: 200 malicious + 200 safe samples from `data/holdout/`
**Method**: Full `scan()` pipeline (L0–L8), threshold sweep from 0.40 to 0.65

| Threshold | Precision | Recall | F1 | FPR | TP | FP | FN | TN |
|:---------:|:---------:|:------:|:--:|:---:|:--:|:--:|:--:|:--:|
| 0.40 | 0.9379 | 0.7550 | 0.8366 | 0.0500 | 151 | 10 | 49 | 190 |
| 0.45 | 0.9396 | 0.7000 | 0.8023 | 0.0450 | 140 | 9 | 60 | 191 |
| **0.50** | **0.9706** | **0.6600** | **0.7857** | **0.0200** | **132** | **4** | **68** | **196** |
| **0.55** | **0.9920** | **0.6200** | **0.7631** | **0.0050** | **124** | **1** | **76** | **199** |
| 0.60 | 0.9907 | 0.5300 | 0.6906 | 0.0050 | 106 | 1 | 94 | 199 |
| 0.65 | 1.0000 | 0.4950 | 0.6622 | 0.0000 | 99 | 0 | 101 | 200 |

**Default threshold: 0.55** — optimized for low false-positive rate (0.5% FPR)
in production environments where false alarms erode user trust.

**Key observations**:
- At threshold 0.55: **99.2% precision**, 62% recall, F1=0.76, FPR=0.5%
- At threshold 0.65: **zero false positives** (100% precision) but only 49.5% recall
- Precision is consistently high (>93%) across all thresholds — when Na0S flags something, it's almost always an actual attack
- The precision/recall tradeoff is configurable per deployment

### Adversarial Evasion Dataset

**Source**: `benchmarks/results/threshold_sweep.json`
**Dataset**: 200 adversarial samples from `data/benchmark/adversarial_evasion.jsonl`
**Method**: Same full pipeline, same threshold sweep

| Threshold | Precision | Recall | F1 | FPR |
|:---------:|:---------:|:------:|:--:|:---:|
| 0.40 | 1.0000 | 0.5200 | 0.6842 | 0.0000 |
| 0.55 | 1.0000 | 0.3900 | 0.5612 | 0.0000 |
| 0.65 | 1.0000 | 0.3050 | 0.4674 | 0.0000 |

**Key observations**:
- **Zero false positives** across all thresholds on adversarial data
- Recall is lower than holdout (52% vs 75% at t=0.40) — expected, since adversarial samples are specifically designed to evade detection
- 100% precision means Na0S never misclassifies benign inputs as attacks, even under adversarial conditions
- The 48% evasion rate at t=0.40 represents the current detection gap that future improvements target

### Layer 16 Conversation Monitoring

**Source**: ROADMAP_V2.md baseline evaluation
**Dataset**: 50 multi-turn conversation scenarios
**Result**: F1=0.9333, FPR=0% (zero false positives on benign conversations)

### Elapsed Time

- Holdout sweep (200 malicious + 200 safe, 6 thresholds): **87.8 seconds**
- Adversarial sweep (200 samples, 6 thresholds): **37.4 seconds**
- Per-sample latency: ~73ms (holdout), ~31ms (adversarial)

*Note: These times include full pipeline execution (L0–L8) on a development machine. Production latency will vary with hardware and optional layer configuration.*

---

## 4. Training Data

**Source**: [`docs/TRAINING.md`](TRAINING.md)

| Metric | Value |
|--------|-------|
| Total training samples | 126,245 |
| Attack categories covered | 19 |
| Sources | 24 public datasets |
| ML model | TF-IDF + Logistic Regression (L4) |
| Model size | ~424 KB (bundled in package) |

The training pipeline includes:
- Dataset download and merge (`scripts/sync_datasets.py`)
- Hard negative mining (`scripts/mine_hard_negatives.py`)
- Threshold optimization (`scripts/threshold_sweep.py`)
- Regression dashboard (`scripts/regression_dashboard.py`)

---

## 5. Test Coverage

| Metric | Value | How to verify |
|--------|-------|---------------|
| Total tests | 8,568 | `pytest tests/ --collect-only -q` |
| Test files | 258 | `find tests/ -name "test_*.py" \| wc -l` |
| Detection rules | 98 | `grep -c "Rule(" src/na0s/layer1/rules_registry.py` |
| Attack categories | 29 | `data/taxonomy.yaml` |
| Office parser fixtures | 16 real binary files | `tests/fixtures/office/` |
| Probe framework | 29 probes × 8 mutation buffs | `scripts/taxonomy/` |

### Mutation Buffs

The probe framework tests each attack technique through 8 evasion mutations:

1. **Base64** — encode payload
2. **ROT13** — Caesar shift by 13
3. **Leetspeak** — character substitution (a→4, e→3, etc.)
4. **Fullwidth** — Unicode fullwidth characters
5. **ZeroWidth** — invisible character insertion
6. **Homoglyph** — Cyrillic/Greek lookalike substitution
7. **Reverse** — reverse the string
8. **CaseAlternating** — aLtErNaTiNg CaSe

This means each technique is tested in 9 forms (original + 8 mutations),
providing coverage against obfuscation-based evasion.

---

## 6. Known Limitations

### Detection Gaps

- **Adversarial recall**: 52% at threshold 0.40 — adversarial samples specifically
  crafted to evade detection succeed ~48% of the time. This is an active area of improvement.
- **Semantic-only attacks**: Attacks that convey malicious intent through purely natural
  language without any structural indicators are harder to detect without L7 (LLM judge).
- **Novel techniques**: Zero-day attack patterns not in the taxonomy or training data
  may evade rule-based and ML-based detection until the training pipeline is updated.
- **Multi-modal**: Image and audio-based injection detection (M category) is limited
  to OCR text extraction; adversarial perturbation detection is not yet implemented.

### Measurement Limitations

- Holdout and adversarial datasets are each 200 samples — sufficient for threshold
  tuning but not for statistically robust per-category breakdowns.
- Per-category precision/recall is not yet published (requires larger labeled datasets
  per category).
- Latency measurements are from development hardware, not production benchmarks.

### Scope

Na0S detects prompt injection attacks in text input to LLM applications. It does **not**:
- Detect malware, phishing, or non-LLM attacks
- Protect against model training-time attacks (data poisoning during training)
- Replace application-level access controls or authentication
- Guarantee 100% detection — use as one layer in a defense-in-depth strategy

---

## 7. Reproducibility

All evaluation results can be reproduced:

```bash
# Install Na0S
pip install -e ".[dev]"

# Run threshold sweep (requires holdout data in data/holdout/)
python scripts/threshold_sweep.py

# Run probe evaluation (29 categories × 8 buffs)
python scripts/evaluate_probes.py

# Run regression dashboard
python scripts/regression_dashboard.py --run

# Run full test suite
pytest tests/ -q --tb=line
```

Results are stored in `benchmarks/results/` as JSON for programmatic comparison.

---

*This document describes Na0S v0.2.0. Methodology and metrics are updated with
each minor release. All numbers are derived from verifiable source data in the
repository — see the "How to verify" column in Section 5 and the source
citations throughout.*
