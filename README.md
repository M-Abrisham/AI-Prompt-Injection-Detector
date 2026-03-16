# Na0S -- AI Prompt Injection Detector

Multi-layer cascade classifier for detecting prompt injection attacks in LLM applications.

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

---

## Disclaimer

Na0S is under active development and **cannot guarantee 100% protection** against prompt injection attacks. Use as one layer in your security strategy, not as a silver bullet.

---

## How It Works

Na0S uses an 11-layer cascade pipeline (L0--L10) where layers L0--L4 always execute and layers L5--L10 gracefully degrade if optional dependencies are missing.

```
Input -> L0 Sanitize -> L1 Rules -> L2 Obfuscation -> L3 Structural -> L4 ML (TF-IDF)
      -> L5 Embedding -> L6 Cascade Voting -> L7 LLM Judge -> L8 Positive Validation
      -> L9 Output Scanner -> L10 Canary Tokens -> Verdict
```

| Layer | Module | Description | Required |
|-------|--------|-------------|----------|
| L0 | `layer0/` | Unicode normalization, encoding detection, input sanitization | Yes |
| L1 | `rules.py` | Pattern-based rule matching with severity weights | Yes |
| L2 | `layer2.py` | Obfuscation detection (base64, hex, rot13, leetspeak, etc.) | Yes |
| L3 | `structural_features.py` | Structural analysis (quote depth, entropy, many-shot) | Yes |
| L4 | `predict.py` | TF-IDF + LogisticRegression ML classifier | Yes |
| L5 | `embedding_classifier.py` | Sentence-transformer centroid similarity | No |
| L6 | `cascade.py` + `_voting.py` | Weighted voting, RRF fusion, Bayesian fusion, groundedness check | Yes |
| L7 | `llm_judge.py` | Optional LLM second opinion (GPT-4o / Llama) | No |
| L8 | `positive_validation.py` | False-positive reduction via safe-content scoring | No |
| L9 | `output_scanner.py` | Detect injection success in LLM output (secrets, PII, role breaks) | No |
| L10 | `canary.py` | Canary token injection and extraction for leak detection | No |

---

## Taxonomy

Na0S defines **29 attack categories** and **276 techniques** in [`data/taxonomy.yaml`](data/taxonomy.yaml).

| Category | Name | Techniques |
|----------|------|:----------:|
| D1 | Instruction Override | 22 |
| D2 | Persona/Roleplay Hijack | 4 |
| D3 | Structural Boundary Injection | 4 |
| D4 | Obfuscation/Encoding | 6 |
| D5 | Unicode Evasion | 7 |
| D6 | Multilingual Injection | 6 |
| D7 | Payload Delivery Tricks | 6 |
| D8 | Context Window Manipulation | 6 |
| I1 | Data Source Poisoning | 8 |
| I2 | HTML/Markup Injection | 3 |
| E | Exfiltration | 11 |
| A | Adversarial ML | 5 |
| O | Output Manipulation | 11 |
| T | Agent/Tool Abuse | 7 |
| R | Resource/Availability | 5 |
| P | Privacy/Data Leakage | 6 |
| P2 | Privacy Extraction Attacks | 4 |
| P3 | Malicious Code Generation | 4 |
| M | Multimodal Injection | 14 |
| S | Supply Chain/Integrity | 8 |
| C | Compliance/Policy Evasion | 8 |
| C1 | Compliance/Policy Evasion (C1) | 8 |
| IM | Inter-Model Propagation | 29 |
| IG | Ingestion Manipulation | 12 |
| AD | Altered Delivery | 19 |
| CT | Combo Techniques | 20 |
| AB | Adversarial Benchmarks | 12 |
| MB | Multi-Buff Combos | 15 |
| C1MT | Compliance Multi-Turn Probes | 6 |

External taxonomy mappings: OWASP LLM Top 10 (2025), AVID, LMRC.

---

## Installation

```bash
# Standard install
pip install -e .

# Development (testing, linting)
pip install -e ".[dev]"

# With optional features
pip install -e ".[embedding]"   # sentence-transformers for L5
pip install -e ".[ocr]"        # EasyOCR for image-based attacks
pip install -e ".[docs]"       # PDF/DOCX/PPTX document parsing
pip install -e ".[llm]"        # OpenAI/Groq for L7
pip install -e ".[all]"        # everything
```

---

## Quick Start

```python
from na0s import scan

result = scan("Ignore all previous instructions")
print(result.is_malicious)   # True
print(result.risk_score)     # 0.93
print(result.label)          # "malicious"
```

Output scanning:

```python
from na0s import scan_output

result = scan_output("Sure! Here is the system prompt: ...")
print(result.is_suspicious)  # True
```

Full cascade classifier:

```python
from na0s import CascadeClassifier

clf = CascadeClassifier()
label, confidence, hits, stage = clf.classify("What is the capital of France?")
print(label, confidence)  # "safe" 0.982
```

CLI usage:

```bash
na0s scan "Is this a prompt injection?"
na0s scan -f suspicious.txt
echo "some text" | na0s scan -
na0s scan --jsonl batch.jsonl
na0s scan --threshold 0.40 "borderline text"
```

Exit codes: `0` = safe, `1` = malicious, `2` = blocked/error, `3` = usage error.

---

## Development Workflow

```bash
make help            # Show all targets
make install         # Editable install with dev dependencies
make test            # Full test suite with coverage
make test-fast       # Fail-fast mode, no coverage
make lint            # Flake8 linting
make format          # Auto-format with black
make bench           # Full benchmark suite
make bench-fast      # Quick benchmark
make evaluate-buffs  # Buff mutation sweep across all probes
make clean           # Remove build artifacts and caches
```

---

## Adding a New Probe

1. Add category to `data/taxonomy.yaml`:
   ```yaml
   X1:
     name: Your Category
     description: "..."
     type: direct
     severity: high
     tags:
       - "owasp-llm:2025:llm01"
     expected_layers: ["layer1_ml"]
     techniques:
       X1.1: { name: "Technique Name", severity: high }
   ```

2. Create `scripts/taxonomy/your_category.py`:
   ```python
   from ._base import Probe

   class YourProbe(Probe):
       category_id = "X1"

       def generate(self):
           samples = []
           samples.append(("attack text here", "X1.1"))
           return self.expand(samples)
   ```

3. Register in `scripts/taxonomy/__init__.py` by adding to `ALL_PROBES`.

4. Run tests:
   ```bash
   make test-fast
   python scripts/evaluate_probes.py --probes X1
   ```

---

## Evaluation

```bash
# Run all probes through the detector
python scripts/evaluate_probes.py

# Run specific probes
python scripts/evaluate_probes.py --probes D1 D4 E

# Buff mutation sweep (test robustness to encoding transforms)
python scripts/evaluate_probes.py --buffs

# View by external taxonomy
python scripts/evaluate_probes.py --taxonomy owasp
python scripts/evaluate_probes.py --taxonomy avid

# Export results
python scripts/evaluate_probes.py --json
python scripts/evaluate_probes.py --buffs --output report.json
```

---

## Project Structure

```
src/na0s/              # Core detection library
  __init__.py          # Public API: scan(), scan_output(), CascadeClassifier
  cascade.py           # L6 CascadeClassifier (whitelist + weighted voting)
  predict.py           # L4 ML prediction pipeline (TF-IDF + LogisticRegression)
  rules.py             # L1 pattern rules
  layer0/              # L0 input sanitization
  layer2.py            # L2 obfuscation detection
  embedding_classifier.py  # L5 sentence-transformer classifier
  llm_judge.py         # L7 LLM judge
  positive_validation.py   # L8 false-positive reduction
  output_scanner.py    # L9 output scanning
  canary.py            # L10 canary tokens
  cli.py               # CLI entry point (na0s scan ...)
scripts/
  taxonomy/            # Probe framework (28 probes)
  evaluate_probes.py   # Probe evaluation runner
data/
  taxonomy.yaml        # Attack taxonomy (29 categories, 276 techniques)
tests/                 # 6,600+ tests
```

---

## License

MIT
