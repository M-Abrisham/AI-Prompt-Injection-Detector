"""
Merge taxonomy samples into the combined training dataset.

Reads:  data/raw/taxonomy_samples.csv    (text, label, technique_id, category, + metadata)
Reads:  data/processed/combined_data.csv (text, label, ...)
Writes: data/processed/combined_data.csv (text, label, technique_id, category, + metadata)

Existing samples get technique_id="" and category="" (untagged).
New taxonomy samples are appended with their computed metadata.
"""

import csv
import os

_DEFAULT_CSV_FIELD_LIMIT = 5_000_000  # 5 MB per field
_field_limit_env = os.environ.get("NA0S_CSV_FIELD_LIMIT", "").strip()
if _field_limit_env:
    try:
        _limit = max(1024, int(_field_limit_env))
    except ValueError:
        _limit = _DEFAULT_CSV_FIELD_LIMIT
else:
    _limit = _DEFAULT_CSV_FIELD_LIMIT
csv.field_size_limit(_limit)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

combined_path = os.path.join(project_root, "data", "processed", "combined_data.csv")
taxonomy_path = os.path.join(project_root, "data", "raw", "taxonomy_samples.csv")
output_path = combined_path  # overwrite in place

# Metadata columns added by the taxonomy generator.
_META_COLS = [
    "length_chars", "length_bytes", "token_count",
    "compression_ratio", "has_reset_claim", "has_override_language",
]

_FIELDNAMES = ["text", "label", "technique_id", "category"] + _META_COLS

# 1. Read existing data
existing = []
with open(combined_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        out = {
            "text": row["text"],
            "label": row["label"],
            "technique_id": row.get("technique_id", ""),
            "category": row.get("category", ""),
        }
        for col in _META_COLS:
            out[col] = row.get(col, "")
        existing.append(out)

print("Existing samples: {}".format(len(existing)))

# 2. Read taxonomy samples
taxonomy = []
with open(taxonomy_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        out = {
            "text": row["text"],
            "label": row["label"],
            "technique_id": row["technique_id"],
            "category": row["category"],
        }
        for col in _META_COLS:
            out[col] = row.get(col, "")
        taxonomy.append(out)

print("Taxonomy samples: {}".format(len(taxonomy)))

# 3. Build taxonomy lookup for enrichment and dedup
taxonomy_by_text = {row["text"]: row for row in taxonomy}

# 4. Merge: enrich existing rows that match taxonomy, append truly new ones
merged = []
seen_texts = set()
enriched = 0
for row in existing:
    if row["text"] in taxonomy_by_text:
        # Replace with taxonomy version (has metadata + technique_id)
        merged.append(taxonomy_by_text[row["text"]])
        enriched += 1
    else:
        merged.append(row)
    seen_texts.add(row["text"])

new_count = 0
for row in taxonomy:
    if row["text"] not in seen_texts:
        merged.append(row)
        seen_texts.add(row["text"])
        new_count += 1

if enriched:
    print("Enriched {} existing samples with taxonomy metadata".format(enriched))
if new_count:
    print("Added {} new samples".format(new_count))

# 5. Count stats
safe = sum(1 for r in merged if r["label"] == "0")
mal = sum(1 for r in merged if r["label"] == "1")
print("\nMerged dataset:")
print("  Safe:      {}".format(safe))
print("  Malicious: {}".format(mal))
print("  Total:     {}".format(len(merged)))
print("  Split:     {:.1f}% safe / {:.1f}% malicious".format(
    safe * 100 / len(merged), mal * 100 / len(merged)))

# 6. Write
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
    writer.writeheader()
    for row in merged:
        writer.writerow(row)

print("\nWritten to: {}".format(output_path))
