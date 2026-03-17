# Competitive Landscape Research - Prompt Injection Detectors (2026-02-15)

## Tools Investigated (with live data from GitHub/docs)

### 1. LLM Guard (protectai/llm-guard)
- **Version**: Active development, 517+ commits on main
- **Architecture**: 15 input scanners + 21 output scanners, modular pipeline
- **Prompt Injection Scanner**: Uses DeBERTa-v3-base fine-tuned models (v1, v2, v2-small)
  - Default model: `protectai/deberta-v3-base-prompt-injection-v2`
  - Default threshold: 0.92
  - Match types: SENTENCE, FULL, TRUNCATE_TOKEN_HEAD_TAIL, TRUNCATE_HEAD_TAIL, CHUNKS
  - Chunks: 256-char segments with 25-char overlap
  - Max length: 512 tokens with truncation
- **Other Scanners**: InvisibleText (Unicode Cf/Co/Cn), Gibberish, Secrets (regex), BanSubstrings, TokenLimit, Language, Toxicity, Sentiment, Code detection
- **Content Type Handling**: NONE -- text only
- **Strengths**: Production-grade, well-maintained, ONNX optimization
- **Weakness**: Single-model dependency, no file/binary handling, no obfuscation decoding

### 2. Vigil (deadbits/vigil-llm)
- **Architecture**: 5 scanner types + auto-updating vector DB + Streamlit UI
- **Detection**: VectorDB similarity, YARA heuristics, Transformer classifier, Prompt-response similarity, Canary tokens
- **Content Type Handling**: NONE -- text only
- **Strengths**: Multi-method detection, learns from detections, YARA rules
- **Weakness**: Smaller community, less maintained than LLM Guard

### 3. Rebuff (protectai/rebuff)
- **Status**: ARCHIVED (May 16, 2025) -- READ ONLY, no longer maintained
- **Last version**: 0.1.1 (Jan 20, 2024)
- **Architecture**: 4 layers -- Heuristic, LLM-based, VectorDB, Canary tokens
- **Conclusion**: Dead project, do not recommend

### 4. NeMo Guardrails (NVIDIA)
- **Version**: 0.20.0 on main, 3462+ commits, actively developed
- **Architecture**: Colang 2.0 dialog flow language, 5 rail types (input/dialog/retrieval/execution/output)
- **Detection**: Jailbreak detection, content safety (NVIDIA safety models), LLM self-check, fact-checking
- **Content Type Handling**: NONE -- designed for Chat Completions API (role/content pairs)
- **Strengths**: Enterprise backing (NVIDIA), Colang 2.0 programmable guardrails, async architecture
- **Weakness**: Heavy dependency (requires LLM for guardrailing), no file handling

### 5. Guardrails AI
- **Version**: 0.8.1 (Feb 13, 2026), 61 releases, actively developed
- **Architecture**: Pydantic-based validators + hub ecosystem
- **Detection**: Validator-based guards for input/output, structured output enforcement
- **Launched**: Guardrails Index -- benchmark comparing 24 guardrails across 6 categories
- **Content Type Handling**: NONE
- **Strengths**: Ecosystem, hub model, structured output validation
- **Weakness**: No dedicated prompt injection detector, more about output validation

### 6. last_layer
- **Version**: 0.1.33 (Apr 12, 2024), STALE (no updates in ~2 years)
- **Architecture**: Closed-source ML model + heuristics + regex signatures
- **13 Threat Categories**: MixedLangMarker, InvisibleUnicodeDetector, MarkdownLinkDetector, HiddenTextDetector, Base64Detector, SecretsMarker, ProfanityDetector, PiiMarker, ExploitClassifier, ObfuscationDetector, CodeFilter, GibberishDetector, IntellectualPropertyLeak
- **Performance**: >=2ms CPU, 92% accuracy, <50MB, fully offline
- **Content Type Handling**: Limited -- Base64Detector, HiddenTextDetector
- **Weakness**: Closed source core, stale, can't inspect/verify

### 7. Meta Prompt Guard 2 (PurpleLlama)
- **Architecture**: mDeBERTa-based sequence classifier
- **Models**: Prompt-Guard-2-22M (22M params), Prompt-Guard-2-86M (86M params)
- **Detection**: Jailbreaking + indirect injection in third-party data
- **Strengths**: Multilingual (mDeBERTa), Meta backing, small model size
- **Weakness**: Single-model binary classifier, no layered defense

### 8. Microsoft PyRIT (Python Risk Identification Toolkit)
- **Purpose**: Red teaming tool (attack generation), not defense
- **Use**: Generates attack prompts to test defenses
- **Relevant for**: Evaluating our detector's robustness

### 9. NVIDIA Garak
- **Purpose**: LLM vulnerability scanner (attack probes + detectors)
- **Use**: Automated vulnerability assessment
- **Relevant for**: Testing our detector against known attack patterns

## Key Insight: No Competitor Has Our Depth
- Our 20-layer cascade architecture is unique
- No tool combines: Unicode normalization + magic byte detection + obfuscation decoding + ML + structural analysis + LLM judge + output scanning + canary tokens
- Most tools are single-method (classifier) or limited multi-method (3-4 techniques)
- NONE handle content type / file format detection as part of injection detection
