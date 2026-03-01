# Na0S Holdout & Adversarial Benchmark Datasets

**Generated**: 2026-02-28
**Generator**: `scripts/gen_all_datasets.py` (malicious + adversarial), inline Python (safe)
**License**: MIT (all content is synthetic; no third-party datasets included)
**Purpose**: Out-of-sample evaluation of Na0S prompt injection detection

---

## Files

### `data/holdout/safe_holdout.jsonl`

**Samples**: 502
**Label**: 0 (benign)
**Format**: `{"text": "...", "label": 0, "source": "generated", "category": "..."}`

| Category       | Count | Description                                                    |
|----------------|-------|----------------------------------------------------------------|
| instructional  | 102   | Cooking recipes, how-to guides, tutorials, DIY instructions    |
| code           | 98    | Python, JavaScript, SQL snippets (not security/injection related) |
| support        | 100   | Customer support conversations (orders, billing, returns)      |
| creative       | 101   | Story starters, poetry prompts, character descriptions         |
| technical      | 101   | API docs, config guides, architecture descriptions             |

Each sample is 1-3 sentences with diverse vocabulary. Samples are shuffled (seed=42).

### `data/holdout/malicious_holdout.jsonl`

**Samples**: 200
**Label**: 1 (malicious)
**Format**: `{"text": "...", "label": 1, "source": "generated", "category": "..."}`

| Category | Count | Attack Type                                                      |
|----------|-------|------------------------------------------------------------------|
| D1       | 40    | Instruction Override ("ignore previous", "disregard all", etc.)  |
| D2       | 30    | Role Hijacking ("you are now DAN", "pretend to be", "roleplay")  |
| D3       | 30    | Structural Boundary ("---SYSTEM---", "```system", delimiters)    |
| E1       | 25    | Prompt Extraction ("what is your system prompt", "reveal")       |
| D4       | 25    | Obfuscation (base64, ROT13, hex encoded injections)              |
| D5       | 25    | Unicode Evasion (homoglyphs, invisible chars, combining marks)   |
| D8       | 25    | Context Manipulation (many-shot, context flooding, filler)       |

Samples are shuffled (seed=42). Category codes follow the Na0S taxonomy.

### `data/benchmark/adversarial_evasion.jsonl`

**Samples**: 574
**Label**: 1 (malicious)
**Format**: `{"text": "...", "label": 1, "source": "generated", "evasion_type": "...", "original": "..."}`

82 core malicious prompts, each transformed with 7 evasion techniques:

| Evasion Type   | Count | Technique                                                |
|----------------|-------|----------------------------------------------------------|
| base64         | 82    | Base64 encoding (`base64.b64encode`)                     |
| rot13          | 82    | ROT13 substitution (`codecs.encode(s, 'rot13')`)         |
| leet           | 82    | Leetspeak (a->4, e->3, i->1, o->0, s->5, t->7)         |
| reversed       | 82    | Full text reversal (`text[::-1]`)                        |
| mixed_case     | 82    | Alternating caps (HeLlO wOrLd)                          |
| word_reverse   | 82    | Word-level reversal ("prompt system your reveal")        |
| padded         | 82    | Prefix padding with benign text before injection         |

Samples are shuffled (seed=42).

---

## How Samples Were Generated

All samples are **synthetic** -- written by hand or generated programmatically:

1. **Safe samples** (`safe_holdout.jsonl`): Hand-written across 5 categories to represent
   realistic production-like text that should NOT trigger prompt injection detection.

2. **Malicious samples** (`malicious_holdout.jsonl`): Hand-written attack patterns across
   7 Na0S taxonomy categories (D1-D8, E1). Templates use combinatorial generation
   (verb x target x payload) for D1; named-role templates for D2; structural markers
   with payloads for D3; encoding transforms for D4; Unicode substitutions for D5;
   context-flooding patterns for D8.

3. **Adversarial evasion** (`adversarial_evasion.jsonl`): 82 core malicious prompts
   transformed through 7 evasion functions. Each sample retains the `original` field
   for traceability.

---

## Usage

```bash
# Run Na0S benchmark on holdout data
python scripts/benchmark.py --dataset data/holdout/safe_holdout.jsonl --tool na0s
python scripts/benchmark.py --dataset data/holdout/malicious_holdout.jsonl --tool na0s
python scripts/benchmark.py --dataset data/benchmark/adversarial_evasion.jsonl --tool na0s
```

```python
import json

# Load a dataset
with open("data/holdout/safe_holdout.jsonl") as f:
    samples = [json.loads(line) for line in f]

# Each sample has: text, label, source, category
for sample in samples[:5]:
    print(sample["category"], sample["label"], sample["text"][:80])
```

---

## Important Notes

- These datasets are **holdout** sets -- they must NOT be used for training or threshold tuning.
- The safe holdout is designed to measure **false positive rate (FPR)**.
- The malicious holdout measures **recall** (true positive rate).
- The adversarial evasion dataset measures **evasion resistance**.
- All data is deterministically generated with `random.seed(42)` for reproducibility.
- No external datasets (HuggingFace, PINT, etc.) are included; all content is original.

---

## License

MIT License. All content is synthetic and original. No third-party datasets or
copyrighted material is included.
