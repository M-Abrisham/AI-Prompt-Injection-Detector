# Model Selection & Architecture Audit Report
## Na0s AI Prompt Injection Detector - February 18, 2026

---

## Executive Summary

The previous research agent's recommendations were **partially correct but significantly outdated** in two critical areas. This audit identified:

1. **CRITICAL UPDATE**: Meta Llama Prompt Guard 2 (released April 5, 2025) dramatically outperforms all previously recommended models. ProtectAI DeBERTa v2 scored only 22.2% APR on AgentDojo vs Prompt Guard 2's 81.2%.
2. **NEW CONTENDER**: Qualifire Sentinel v2 (Qwen3-0.6B, June 2025) claims 95.7% avg F1, significantly outperforming ProtectAI (75.0%) and Vijil (79.9%) on their benchmark suite.
3. **PEFT recommendation (LoRA) is CORRECT** but should be upgraded to DoRA for marginal gains at no inference cost.
4. **Unsloth recommendation is CORRECT** and remains the leading single-GPU training framework (52.4k stars, active Feb 2026).
5. **Two-track architecture is VALIDATED** by Meta's own approach (separate classifier + judge).

---

## Finding 1: Drop-in Classifier Audit

### Previous Recommendation
- Primary: ProtectAI DeBERTa v2 (184M, 95.25% accuracy, Apache-2.0)
- Alternative: Vijil Dome (ModernBERT, 100M, 96.15% accuracy)

### Audit Results: RECOMMENDATION REVISED

#### Current Landscape (Feb 2026, sorted by real-world effectiveness)

| Model | Architecture | Params | Accuracy/F1 | APR@3%* | Downloads/mo | License | Latency |
|-------|-------------|--------|-------------|---------|-------------|---------|---------|
| **Llama Prompt Guard 2 86M** | mDeBERTa-v3 | 86M | AUC 0.998 | **81.2%** | 19,765 | Llama 4 | 92.4ms |
| **Llama Prompt Guard 2 22M** | DeBERTa-xsmall | 22M | AUC 0.995 | **78.4%** | 84,522 | Llama 4 | **19.3ms** |
| **Qualifire Sentinel v2** | Qwen3-0.6B | 596M | F1 0.957 | N/A | 2,371 | Elastic | 38ms |
| ProtectAI DeBERTa v2 | DeBERTa-v3 | 184M | 95.25% acc | **22.2%** | 140,939 | Apache-2.0 | ~30ms |
| Vijil Dome | ModernBERT | 100M | 96.15% acc | N/A | 24,209 | Apache-2.0 | ~25ms |
| Lilbullet Qwen3-0.6B | Qwen3-0.6B | 596M | **98.00%** acc | N/A | 6 | Apache-2.0 | ~38ms |

*APR@3% = Attack Prevention Rate at 3% utility reduction on AgentDojo benchmark (real-world agentic scenarios)

#### Critical Finding: ProtectAI Dramatically Underperforms in Real-World Conditions

Meta's AgentDojo benchmark reveals that **ProtectAI v2 scores only 22.2% APR** -- meaning it blocks only ~1 in 5 real-world injection attacks in agentic pipelines. This is a **59 percentage point gap** vs Prompt Guard 2 86M (81.2%). The high accuracy numbers (95.25%) on static benchmarks do NOT translate to real-world effectiveness.

#### New Primary Recommendation: Llama Prompt Guard 2

**For Na0s, recommended configuration:**
- **Production (low-latency)**: Prompt Guard 2 22M (19.3ms, 78.4% APR, 84.5k downloads = battle-tested)
- **Maximum accuracy**: Prompt Guard 2 86M (92.4ms, 81.2% APR, multilingual)
- **License concern**: Llama 4 Community License -- permissive for most uses but NOT Apache-2.0

**Key advantages over ProtectAI v2:**
- 3.6x better real-world attack prevention (81.2% vs 22.2% APR)
- Adversarial-resistant tokenization (mitigates whitespace/fragmentation attacks)
- Energy-based loss function (lower FPR on out-of-distribution data)
- 8-language multilingual support (addresses Na0s D6 gap)
- Binary classification (simpler: benign/malicious vs 3-way)

**Concerns:**
- Llama 4 license is NOT Apache-2.0 (check compatibility with Na0s licensing)
- 512 token context window (same limitation as DeBERTa)
- Open-source model = adaptive attacks possible (same as all open models)

#### Alternative: Qualifire Sentinel v2 (Worth Watching)

- Uses Qwen3-0.6B (decoder model repurposed as classifier)
- Claims 27.6% higher F1 than ProtectAI across 5 benchmarks
- 32K context window (major advantage for long-prompt attacks)
- BUT: Elastic License (not Apache-2.0), no AgentDojo benchmark, only 2.4k downloads
- AND: 596M params / 1.2GB is 4x larger than Prompt Guard 2 22M

#### Verdict for Na0s

**PRIMARY**: Integrate Prompt Guard 2 22M as default classifier (speed), with 86M as high-accuracy option
**SECONDARY**: Keep ProtectAI DeBERTa v2 as fallback / ensemble member (Apache-2.0 license, different architecture = diversity)
**WATCH**: Qualifire Sentinel v2 (if they publish AgentDojo numbers and it performs well, consider)
**SKIP**: Vijil Dome, Lilbullet (insufficient adoption/validation)

**Confidence: HIGH** -- Meta's benchmarks are rigorous and AgentDojo is the most realistic evaluation framework available.

---

## Finding 2: PEFT Method (LoRA) Audit

### Previous Recommendation
- LoRA with r=16 on DeBERTa-v3-base
- ~0.32% trainable params, ~6 MB adapter

### Audit Results: RECOMMENDATION MOSTLY VALID, UPGRADE TO DoRA

#### PEFT Method Landscape (Feb 2026)

| Method | Description | vs LoRA | Inference Cost | Maturity |
|--------|-------------|---------|----------------|----------|
| **LoRA** | Low-rank decomposition of weight updates | Baseline | Zero (mergeable) | Production |
| **DoRA** (ICML 2024) | Weight-decomposed LoRA (magnitude + direction) | +1-3% on NLU tasks | **Zero** (mergeable) | Production |
| **AdaLoRA** | Adaptive rank allocation via SVD | +0.5-1% variable | Zero (mergeable) | Production |
| **LoHa** | Hadamard product decomposition | Higher expressivity | Zero (mergeable) | Experimental (vision) |
| **LoKr** | Kronecker product decomposition | Faster computation | Zero (mergeable) | Experimental (vision) |
| **IA3** | Learned vectors multiply activations | Fewer params | Zero (mergeable) | Production |
| **MiSS** (2024) | Matrix shard sharing, single matrix | 50% fewer params | Zero (mergeable) | Experimental |
| **X-LoRA** | Mixture of LoRA experts with gating | Dynamic adaptation | 2x forward pass | Experimental |

#### DoRA vs LoRA: The Upgrade Path

DoRA (Weight-Decomposed Low-Rank Adaptation) was presented as an **ICML 2024 Oral** paper:
- Decomposes weights into magnitude and direction components
- Applies LoRA specifically to directional updates
- **"Consistently outperforms LoRA"** on LLaMA, LLaVA, VL-BART
- **No additional inference overhead** (can be merged like LoRA)
- Supported in PEFT library (HuggingFace), Unsloth, and LLaMA-Factory

**For Na0s: Switch from LoRA to DoRA.** The change is a single parameter:
```python
# Before (LoRA)
peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["query_proj", "value_proj"])

# After (DoRA) -- just add use_dora=True
peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["query_proj", "value_proj"], use_dora=True)
```

#### Base Model Decision: DeBERTa-v3 vs ModernBERT vs Qwen3

For PEFT fine-tuning of a custom classifier:
- **DeBERTa-v3-base**: Still the best encoder for NLU tasks. Disentangled attention mechanism. Proven for prompt injection. 184M params.
- **ModernBERT-base**: Faster inference, longer context (8192 tokens), but fewer proven results for security classification. 100M params.
- **Qwen3-0.6B**: Decoder model, 6x larger. Reasoning capability but massive overkill for binary classification.

**Recommendation**: Fine-tune from **Prompt Guard 2 86M** (already mDeBERTa-v3, already trained on injection data). This gives you the best starting point -- you're adapting an expert rather than training a generalist.

**Confidence: HIGH** for DoRA upgrade. **MEDIUM** for base model (Prompt Guard 2 as starting point is logical but requires license review).

---

## Finding 3: Unsloth + LLM Judge Audit

### Previous Recommendation
- Fine-tune Llama 3.2 3B-Instruct with QLoRA via Unsloth
- Export to GGUF q4_k_m (1.8 GB), deploy via Ollama

### Audit Results: RECOMMENDATION VALID WITH MODEL UPGRADE

#### Unsloth Status (Feb 2026)
- **GitHub Stars**: 52.4k (up from ~30k in 2024)
- **Last Update**: Feb 10, 2026 (MoE training)
- **Key 2025 Features**: Qwen3 support, Llama 4 support, QAT, 500K context, 3x faster packing, FP8 RL, GRPO/GSPO
- **Supported Models**: Llama 3.2, Llama 3.3, Llama 4, Qwen 2.5, Qwen 3, DeepSeek R1, Gemma 3, Phi-4
- **Training Speed**: Claims 2x faster + 70% less VRAM vs standard HuggingFace

#### LLaMA-Factory as Alternative
- **GitHub Stars**: 67.4k (larger community than Unsloth)
- **Last Update**: Feb 18, 2026
- **Models**: 100+ supported, including all Unsloth models
- **Methods**: LoRA, QLoRA, DoRA, GaLore, PiSSA, DPO, PPO, GRPO
- **Key Feature**: WebUI for training (lower barrier)
- **Performance**: Integrates Unsloth for 170% speed boost, vLLM for 270% inference speed
- **License**: Apache-2.0

**LLaMA-Factory is actually the better choice for Na0s** because:
1. Larger community (67.4k vs 52.4k stars)
2. More training methods (DoRA, GaLore, etc.)
3. Integrates Unsloth internally for speed
4. WebUI for easier experimentation
5. Apache-2.0 license
6. Better documentation and more examples

#### Model Size Decision: 3B vs 8B vs 70B

| Model | GGUF Size (q4) | RAM Required | Quality for Judge | Inference Speed |
|-------|----------------|-------------|-------------------|-----------------|
| Llama 3.2 1B | ~0.6 GB | 1 GB | Insufficient | Very fast |
| Llama 3.2 3B | ~1.8 GB | 3 GB | Adequate | Fast |
| Qwen 2.5 3B | ~1.8 GB | 3 GB | Better reasoning | Fast |
| Llama 3.3 8B* | ~4.5 GB | 6 GB | Good | Moderate |
| Qwen 2.5 7B | ~4.0 GB | 6 GB | Good reasoning | Moderate |
| Qwen 3 0.6B | ~0.4 GB | 1 GB | Surprisingly good | Very fast |

*Note: Llama 3.3 is 70B only (not 8B). Llama 3.1 8B or Llama 3.2 3B are the small options.

**Updated Recommendation:**
- **For Na0s self-hosted judge**: Use **Qwen 2.5 3B-Instruct** instead of Llama 3.2 3B
  - Better reasoning capability for the same size
  - Apache-2.0 license (vs Llama Community License)
  - Strong multilingual support (addresses D6 gap)
  - Qwen 3 0.6B is also worth testing -- surprisingly capable for its size
- **Training framework**: LLaMA-Factory (integrates Unsloth internally)
- **Export**: GGUF q4_k_m via Unsloth/llama.cpp, deploy via Ollama

**Confidence: HIGH** for framework choice. **MEDIUM** for Qwen vs Llama (need to benchmark on injection judgment task specifically).

---

## Finding 4: Meta Prompt Guard 2 Audit

### Previous Research Question
- Has it been released publicly? Are there benchmarks? License suitable?

### Audit Results: FULLY RELEASED, EXCELLENT BENCHMARKS

#### Availability
- **Released**: April 5, 2025
- **HuggingFace**: `meta-llama/Llama-Prompt-Guard-2-86M` and `meta-llama/Llama-Prompt-Guard-2-22M`
- **Downloads**: 22M variant: 84,522/month; 86M variant: 19,765/month
- **Status**: Production-ready, actively used

#### Benchmarks (from model card)

**Direct Jailbreak Detection:**
| Metric | PG1 | PG2-22M | PG2-86M |
|--------|-----|---------|---------|
| AUC (English) | 0.987 | 0.995 | **0.998** |
| Recall @1% FPR (English) | 21.2% | 88.7% | **97.5%** |
| AUC (Multilingual) | 0.983 | 0.942 | **0.995** |
| Latency (A100, 512 tok) | 92.4ms | **19.3ms** | 92.4ms |

**AgentDojo Real-World Benchmark:**
| Model | APR @3% Utility |
|-------|----------------|
| Prompt Guard 2 86M | **81.2%** |
| Prompt Guard 2 22M | 78.4% |
| Prompt Guard 1 | 67.6% |
| ProtectAI DeBERTa v2 | 22.2% |
| Deepset | 13.5% |
| LLM Warden | 12.9% |

#### Key Technical Improvements in v2
1. **Energy-based loss function** -- reduces FPR on out-of-distribution data
2. **Adversarial-resistant tokenization** -- mitigates whitespace/fragmentation evasion
3. **Binary classification** -- simplified from 3-way (benign/injection/jailbreak) to 2-way (benign/malicious)
4. **Expanded training data** -- synthetic injections + red-team data

#### License Analysis
- **License**: Llama 4 Community License Agreement
- **Permissive for**: Research, development, commercial products with <700M MAU
- **Restrictions**: Cannot use outputs to train competing models, Meta attribution required
- **Na0s Impact**: Acceptable for an open-source detector. The 700M MAU threshold is unlikely to be hit. Attribution is minor. The "no competing model training" clause does NOT apply to using the model as a component.

**Confidence: HIGH** -- Prompt Guard 2 is the clear leader for drop-in prompt injection classification.

---

## Finding 5: Two-Track Architecture Audit

### Previous Recommendation
- Track 1 (Classifier): PEFT LoRA on encoder models (DeBERTa/ModernBERT)
- Track 2 (LLM Judge): QLoRA on causal LMs (Llama 3.2 3B)

### Audit Results: ARCHITECTURE VALIDATED WITH REFINEMENTS

#### Evidence Supporting Two-Track

1. **Meta uses the same pattern**: Prompt Guard (classifier) + Llama model (judge) are separate components in their safety stack
2. **LLM Guard architecture**: DeBERTa classifier + separate scanners for different threat types
3. **Speed vs accuracy tradeoff**: Classifiers (20-90ms) for real-time gating, LLM judges (1-5s) for ambiguous cases
4. **Different failure modes**: Classifier misses novel attacks but is consistent; LLM judge catches novel patterns but is non-deterministic

#### Refined Architecture for Na0s

```
Input Text
    |
    v
[Layer 0: Sanitization & Content Type] (existing, ~1ms)
    |
    v
[Layer 1: Rules Engine] (existing, 23 rules, ~5ms)
    |
    v
[Track A: Fast Classifier] ---------> [Track B: LLM Judge]
   Prompt Guard 2 22M                   Qwen 2.5 3B (self-hosted)
   ~20ms, binary                        ~500ms-2s, reasoning
   Always runs                          Only for score 0.3-0.7 (ambiguous)
    |                                        |
    v                                        v
[Ensemble Decision Layer]
   - If classifier score > 0.9: BLOCK (fast path)
   - If classifier score < 0.1: ALLOW (fast path)
   - If 0.1-0.9: Consult LLM judge (slow path)
   - Combine: weighted vote + rule engine signals
```

#### Alternative: Could a Single Approach Handle Both?

**No, for these reasons:**
1. **Latency**: A single LLM judge for all traffic adds 1-5s per request. Unacceptable.
2. **Cost**: Self-hosted LLM at scale is 10-50x more expensive than classifier inference.
3. **Reliability**: Classifiers are deterministic; LLMs can be jailbroken themselves.
4. **Coverage**: Classifiers handle known patterns; LLMs handle novel/reasoning-dependent attacks.

**The two-track architecture with an ensemble decision layer is the correct approach.**

#### Ensemble Methods Worth Considering
- **Weighted voting**: Classifier (0.6) + LLM Judge (0.3) + Rules (0.1)
- **Cascading**: Classifier first, LLM only for ambiguous (saves compute)
- **Multi-classifier**: Run both Prompt Guard 2 22M AND ProtectAI v2 (different architectures = lower correlated errors)

**Confidence: HIGH** -- industry consensus supports this architecture.

---

## Finding 6: Benchmarks and Leaderboards

### Current Benchmark Landscape (Feb 2026)

| Benchmark | Maintainer | Focus | Status |
|-----------|-----------|-------|--------|
| **AgentDojo** | ETH Zurich / Invariant Labs | Real-world agentic injection attacks | **Active** (435 stars, MIT) |
| **CyberSecEval** | Meta (PurpleLlama) | LLM security incl. injection | Active |
| **Guardrails Index** | Guardrails AI | 24 guardrails across 6 categories | Active |
| **HackAPrompt** | EMNLP 2023 | CTF-style injection challenges | Static (historical) |
| **Qualifire Benchmark** | Qualifire | 5 injection/jailbreak test sets | Active |
| OWASP LLM Top 10 | OWASP | Guidance (not a benchmark) | v2025 released |

### Key Finding: No Unified Leaderboard Exists

There is **NO single authoritative prompt injection detection leaderboard** analogous to the Open LLM Leaderboard for generation quality. This is a significant gap. The closest is AgentDojo, but it evaluates defenses in an agentic context, not standalone classifiers.

### Recommended Evaluation Strategy for Na0s

Build an internal benchmark combining:
1. **AgentDojo** scenarios for real-world attack simulation
2. **Qualifire's 5 test sets**: qualifire benchmark, WildJailbreak, deepset prompt-injections, safe-guard prompt-injection, jailbreak-classification
3. **Na0s taxonomy**: The existing 108-technique taxonomy already covers more attack vectors than any public benchmark
4. **Multilingual**: XSAFETY or custom multilingual injection corpus

**Confidence: HIGH** that no single external benchmark is sufficient; Na0s should build its own.

---

## Summary: Updated Recommendation Matrix

| Component | Previous Rec | Updated Rec | Change | Confidence |
|-----------|-------------|-------------|--------|------------|
| **Drop-in Classifier** | ProtectAI DeBERTa v2 | **Prompt Guard 2 22M/86M** | MAJOR CHANGE | HIGH |
| **Fallback Classifier** | Vijil Dome | **ProtectAI DeBERTa v2** (ensemble member) | Demoted | HIGH |
| **PEFT Method** | LoRA r=16 | **DoRA r=16** (use_dora=True) | Minor upgrade | HIGH |
| **PEFT Base Model** | DeBERTa-v3-base | **Prompt Guard 2 86M** (fine-tune from expert) | Model change | MEDIUM |
| **LLM Judge Model** | Llama 3.2 3B | **Qwen 2.5 3B-Instruct** | Model change | MEDIUM |
| **Training Framework** | Unsloth | **LLaMA-Factory** (integrates Unsloth) | Framework change | MEDIUM |
| **Architecture** | Two-track | **Two-track + ensemble decision layer** | Refinement | HIGH |
| **Benchmark** | (none specified) | **AgentDojo + Qualifire + Na0s taxonomy** | New addition | HIGH |

---

## Implementation Priority

### P0 (This Sprint)
1. Integrate Prompt Guard 2 22M as primary classifier (drop-in, 19.3ms)
2. Add ProtectAI DeBERTa v2 as secondary classifier (ensemble diversity)
3. Run both against Na0s test suite to establish internal baselines

### P1 (Next Sprint)
4. Fine-tune Prompt Guard 2 86M with DoRA on Na0s taxonomy data
5. Set up LLaMA-Factory for Qwen 2.5 3B judge training
6. Design ensemble decision layer (threshold-based cascading)

### P2 (Future)
7. Build comprehensive benchmark suite (AgentDojo + Qualifire + taxonomy)
8. Multi-classifier voting (PG2 22M + PG2 86M + ProtectAI v2)
9. Deploy self-hosted Qwen 2.5 3B judge via Ollama

---

## Appendix: Data Sources

| Source | URL | Fetch Date |
|--------|-----|-----------|
| ProtectAI DeBERTa v2 | huggingface.co/protectai/deberta-v3-base-prompt-injection-v2 | 2026-02-18 |
| Vijil Dome | huggingface.co/vijil/vijil_dome_prompt_injection_detection | 2026-02-18 |
| Prompt Guard 2 86M | huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M | 2026-02-18 |
| Prompt Guard 2 22M | huggingface.co/meta-llama/Llama-Prompt-Guard-2-22M | 2026-02-18 |
| Qualifire Sentinel v2 | huggingface.co/qualifire/prompt-injection-jailbreak-sentinel-v2 | 2026-02-18 |
| Lilbullet Qwen3 | huggingface.co/Lilbullet/Prompt-Injection-classifier-complex-Qwen3-0.6B | 2026-02-18 |
| DoRA paper | arxiv.org/abs/2402.09353 (ICML 2024 Oral) | 2026-02-18 |
| PEFT docs | huggingface.co/docs/peft/conceptual_guides/adapter | 2026-02-18 |
| Unsloth repo | github.com/unslothai/unsloth (52.4k stars) | 2026-02-18 |
| LLaMA-Factory repo | github.com/hiyouga/LLaMA-Factory (67.4k stars) | 2026-02-18 |
| AgentDojo repo | github.com/ethz-spylab/agentdojo (435 stars) | 2026-02-18 |
| LLM Guard repo | github.com/protectai/llm-guard (2.5k stars) | 2026-02-18 |
| HuggingFace model search | huggingface.co/models?search=prompt+injection+detection | 2026-02-18 |
