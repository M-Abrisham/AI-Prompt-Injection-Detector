# Holdout Corpus

Curated evaluation datasets for Na0S benchmark. These are **never** used for
training or threshold tuning — only for final honest metric reporting.

## Files

| File | Samples | Labels | Purpose |
|------|---------|--------|---------|
| `safe_holdout.jsonl` | 500+ | all `0` | FPR measurement on production-like benign text |
| `malicious_holdout.jsonl` | 200+ | all `1` | Recall measurement across technique taxonomy |

## JSONL Schema

```jsonl
{"text": "...", "label": 0|1, "source": "holdout", "category": "..."}
```

- **text** (str): The input prompt.
- **label** (int): `0` = benign, `1` = malicious.
- **source** (str): Always `"holdout"` for provenance tracking.
- **category** (str): Sub-category describing the sample type.

## Safe Holdout Categories

| Category | Count | Description |
|----------|-------|-------------|
| `instructional` | 100 | Cooking recipes, how-to guides, tutorials |
| `code` | 100 | Python, JS, SQL snippets — legitimate programming |
| `customer_support` | 100 | Normal service interactions and ticket exchanges |
| `creative_writing` | 100 | Story prompts, poetry, creative exercises |
| `technical_docs` | 100 | API docs, architecture notes, config references |

Includes deliberate hard negatives: benign text containing trigger-like words
("ignore", "system", "override", "instructions") in non-malicious context.

## Malicious Holdout Categories

| Category | Count | Description |
|----------|-------|-------------|
| `D1` | 15+ | Instruction override attacks |
| `D2` | 15+ | Role hijacking / jailbreaks |
| `D3` | 15+ | Structural boundary manipulation |
| `D4` | 15+ | Obfuscation (base64, ROT13, leet) |
| `D5` | 15+ | Unicode evasion (homoglyphs, invisible chars) |
| `D6` | 15+ | Multilingual injection |
| `D7` | 15+ | Payload delivery / multi-turn |
| `D8` | 15+ | Context manipulation / many-shot |
| `E1` | 15+ | Prompt extraction |
| `E2` | 15+ | Reconnaissance |
| `C1` | 15+ | Compliance evasion |
| `O1` | 10+ | Harmful content / output manipulation |

## Provenance

- **Safe samples**: Hand-crafted by contributors. Inspired by production-like
  patterns from alpaca, dolly, and hard-negative templates
  (`scripts/mine_hard_negatives.py`). No samples copied from training data.
- **Malicious samples**: Hand-crafted attack prompts covering the Na0S technique
  taxonomy. Includes both obvious and subtle injection attempts.
- **Generation date**: 2026-02-28
- **License**: Same as Na0S project license.

## Integrity Rules

1. **No data leakage**: These files must NEVER be used for training or tuning.
2. **No cherry-picking**: Report all results, not just favorable ones.
3. **Strict separation**: `data/holdout/` is separate from `data/benchmark/`
   (which contains tune-split data).
4. **Reproducibility**: All samples are deterministic (no random generation
   without fixed seeds).
