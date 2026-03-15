# Na0S Benchmark Results

**Version**: 0.1.0
**Date**: 2026-03-15
**Threshold**: 0.55 (default)
**Methodology**: Each dataset run through `scripts/benchmark.py` with default settings. No cherry-picking, no threshold tuning.

---

## BM-1: Real Dataset Benchmarks

### Summary

| Dataset | Samples | Malicious | Safe | Precision | Recall | F1 | FPR | Accuracy | AUC-ROC | AUC-PR | Avg Latency |
|---------|---------|-----------|------|-----------|--------|----|-----|----------|---------|--------|-------------|
| deepset/prompt-injections | 546 | 203 | 343 | **0.9286** | **0.3202** | **0.4762** | 0.0146 | 0.7381 | 0.7796 | 0.7300 | 10.6ms |
| tatsu-lab/alpaca (benign) | 2,000 | 0 | 2,000 | — | — | — | **0.0215** | 0.9785 | — | — | 5.2ms |
| databricks/dolly (benign) | 2,000 | 0 | 2,000 | — | — | — | **0.1045** | 0.8955 | — | — | 11.2ms |

### Detailed Results

#### deepset/prompt-injections (546 samples: 203 malicious, 343 benign)

This is the primary benchmark — a third-party labelled prompt injection dataset from HuggingFace.

| Metric | Value |
|--------|-------|
| True Positives | 65 |
| True Negatives | 338 |
| False Positives | 5 |
| False Negatives | 138 |
| Precision | 0.9286 |
| Recall | 0.3202 |
| F1 | 0.4762 |
| FPR | 0.0146 |
| Accuracy | 0.7381 |
| AUC-ROC | 0.7796 |
| AUC-PR | 0.7300 |
| Avg Latency | 10.60ms |
| P50 Latency | 4.42ms |
| P95 Latency | 8.78ms |
| P99 Latency | 23.87ms |
| Throughput | 94.33 samples/sec |

**Interpretation**: Very high precision (93%) — when Na0S flags something, it's almost certainly an injection. But recall is low (32%) — Na0S misses ~68% of injections in this dataset. The deepset dataset contains many subtle, indirect, and novel attack patterns that the current rule-based + ML ensemble doesn't catch.

#### tatsu-lab/alpaca (2,000 benign instruction-following samples)

| Metric | Value |
|--------|-------|
| False Positives | 43 |
| FPR | 0.0215 (2.15%) |
| Accuracy | 0.9785 |
| Avg Latency | 5.21ms |
| P50 Latency | 4.01ms |
| P95 Latency | 6.17ms |
| P99 Latency | 8.91ms |
| Throughput | 191.86 samples/sec |

**Interpretation**: Low FPR on clean instructional text. 43 out of 2,000 benign prompts falsely flagged (2.15%). This is acceptable for many use cases but may need improvement for high-throughput production systems.

#### databricks/dolly-15k (2,000 benign human-written instruction samples)

| Metric | Value |
|--------|-------|
| False Positives | 209 |
| FPR | 0.1045 (10.45%) |
| Accuracy | 0.8955 |
| Avg Latency | 11.23ms |
| P50 Latency | 4.36ms |
| P95 Latency | 29.98ms |
| P99 Latency | 138.17ms |
| Throughput | 89.04 samples/sec |

**Interpretation**: Higher FPR on Dolly (10.45%) — significantly worse than Alpaca (2.15%). Dolly includes context-heavy prompts (instruction + context paragraphs) that likely trigger structural/boundary detection rules. The latency tail is also heavier (P99=138ms), likely due to longer inputs triggering more detection layers. This is a known weakness that needs investigation.

---

## Key Takeaways

1. **Precision is strong (93%)** — false alarms are rare when Na0S does flag
2. **Recall is weak (32%)** — the biggest gap; most real-world injections in deepset go undetected
3. **FPR varies by content type** — 1.5% on injection-mixed data, 2.2% on clean instructional, but 10.5% on context-heavy text
4. **Latency is fast** — median ~4ms, well under the claimed <10ms for typical prompts
5. **Dolly FPR is a red flag** — 10.45% false positive rate on benign text with context paragraphs suggests over-triggering on structural patterns

## Known Limitations

- Only tested on 3 datasets (1 injection, 2 benign)
- No competitor comparison yet (BM-3)
- No per-attack-category breakdown yet (BM-4)
- No Recall@FPR metric yet (BM-6)
- Threshold not optimized — 0.55 is the default, not necessarily optimal for these datasets

---

## Reproducibility

```bash
# Download datasets
python scripts/download_datasets.py --force

# Run benchmarks
python scripts/benchmark.py --dataset data/benchmark/deepset_pi.jsonl \
    --summary benchmarks/results/bm1/bench_deepset.json \
    --output benchmarks/results/bm1/bench_deepset_samples.jsonl

python scripts/benchmark.py --dataset data/benchmark/benign_alpaca.jsonl \
    --summary benchmarks/results/bm1/bench_alpaca.json \
    --output benchmarks/results/bm1/bench_alpaca_samples.jsonl

python scripts/benchmark.py --dataset data/benchmark/benign_dolly.jsonl \
    --summary benchmarks/results/bm1/bench_dolly.json \
    --output benchmarks/results/bm1/bench_dolly_samples.jsonl
```

Raw results stored in `benchmarks/results/bm1/`.
