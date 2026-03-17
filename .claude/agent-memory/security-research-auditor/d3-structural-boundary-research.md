# D3 Structural Boundary Injection -- Research Notes (2026-02-17)

## Attack Overview

D3 attacks inject fake system/role delimiters to confuse the LLM's instruction
hierarchy. They exploit the "semantic gap" where system prompts and user input
share the same natural-language format.

## Sub-techniques

### D3.1 -- Fake System Prompt
Injects delimiters like `### System:`, `<<SYS>>`, `[SYSTEM]`, `--- SYSTEM ---`
to trick the LLM into believing a new system prompt is being issued.

### D3.2 -- Chat Template Injection
Injects model-specific chat tokens:
- OpenAI ChatML: `<|im_start|>`, `<|im_end|>`
- Llama 2: `[INST]`, `<<SYS>>`, `<</SYS>>`, `</s>`
- Llama 3: `<|begin_of_text|>`, `<|start_header_id|>`, `<|end_header_id|>`
- Phi-3: `<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>`
- Gemma: `<start_of_turn>`, `<end_of_turn>`
- Mistral: `[INST]`, `[/INST]`
- Anthropic: `Human:`, `Assistant:`

### D3.3 -- XML Role Tags
Injects `<system>`, `<admin>`, `<developer>`, `<root>`, `<instructions>` tags.

### D3.4 -- Markdown Delimiter Abuse
Uses code blocks, headers, blockquotes, tables, horizontal rules, bold labels.

## Detection Mechanisms in Current Codebase

### structural_features.py -- _BOUNDARY_PATTERNS
```python
_BOUNDARY_PATTERNS = re.compile(
    r"(?:^|\s)(?:---+|===+|\*\*\*+|###+)(?:\s|$)"
    r"|\[SYSTEM\]|\[INST\]|<<SYS>>",
    re.IGNORECASE,
)
```
Detects: `---`, `===`, `***`, `###`, `[SYSTEM]`, `[INST]`, `<<SYS>>`
Does NOT detect: `<|im_start|>`, `<system>`, `<admin>`, `<start_of_turn>`, etc.

### positive_validation.py -- _SYSTEM_PROMPT_MARKERS
```python
_SYSTEM_PROMPT_MARKERS = [
    "[SYSTEM]", "<<SYS>>", "[INST]", "</s>", "<|im_start|>",
    "<|im_end|>", "[/INST]", "<</SYS>>", "### System:",
    "### Human:", "### Assistant:", "<|system|>", "<|user|>",
    "<|assistant|>",
]
```
Broader coverage but this is in positive_validation, not the main scan() path.

### cascade.py -- BOUNDARY_MARKERS
```python
r"---|===|\*\*\*|\[SYSTEM\]|\[INST\]|<<SYS>>|</s>"
```
Used in whitelist filter to prevent safe-classification of boundary-containing input.

### rules.py
- `override` rule: `r"ignore (all|any|previous) (instructions|rules)"` (critical)
- `system_prompt` rule: `r"(reveal|show|print).{0,40}(system prompt|developer message)"` (high)
- `roleplay` rule: `r"\byou are now\b|\bpretend to be\b|\bact as\b"` (medium)

## CRITICAL FINDING: Layer 0 Sanitization

Layer 0 sanitizes chat template tokens BEFORE structural features are computed:
- `<<SYS>>` becomes `< >` (angle brackets/pipes stripped)
- `<|im_start|>` becomes `< >` (pipes stripped)
- `[SYSTEM]` survives sanitization (brackets preserved)
- `###` survives sanitization (hash preserved)
- `---` survives sanitization (dashes preserved)

This means instruction_boundary can only detect `[SYSTEM]`, `[INST]`, `###`,
`---`, `===`, `***` after sanitization. Chat template tokens are invisible to
the structural feature extractor.

## Detection Fallback Chain
For D3 payloads, detection relies on this cascade:
1. Fingerprint store (known_malicious_exact) -- catches previously-seen payloads
2. ML classifier -- trained on D2 persona attacks, some D1 overrides
3. Obfuscation heuristics (high_entropy, weird_casing) -- catches unusual text
4. Rule engine (override, system_prompt) -- catches keyword patterns
5. Structural features (instruction_boundary) -- only works post-sanitization

## Test Results Summary

| Sub-technique | Tests | Passing | Expected Fail | Detection Mechanism |
|--------------|-------|---------|---------------|-------------------|
| D3.1 Fake System Prompt | 7 | 7 | 0 | boundary + rules + ML |
| D3.2 Chat Template | 8 | 8 | 0 | ML + fingerprint + obfuscation |
| D3.3 XML Role Tags | 7 | 7 | 0 | rules + ML + fingerprint |
| D3.4 Markdown Delimiter | 8 | 8 | 0 | boundary + rules + ML |
| D3 Subtle Variants | 4 | 4 | 0 | fingerprint + ML |
| D3 Combined Attacks | 4 | 4 | 0 | boundary + rules + ML |
| Safe Prompts (FP tests) | 6 | 2 | 4 | FP: fingerprint + ML |
| **Total** | **44** | **40** | **4** | |

## False Positive Analysis

4 benign prompts trigger false positives:
1. Shopping list with dashes -- fingerprint store + high_entropy + weird_casing
2. Markdown table -- punctuation_flood + weird_casing + mixed_language_input
3. Horizontal rule article -- instruction_boundary + ML = FP
4. XML tags in code question -- high_entropy + weird_casing + fingerprint

Root cause: The fingerprint store has been seeded aggressively, causing benign
markdown content to match. The ML model also has a high FP rate for structured
text (trained mostly on D2 persona attacks).

## Recommendations

1. **Add XML role tags to _BOUNDARY_PATTERNS**: `<system>`, `<admin>`, etc.
2. **Add chat template tokens to _BOUNDARY_PATTERNS**: `<|im_start|>`, etc.
   (but must operate on RAW input before Layer 0 sanitization)
3. **Add D3 training samples**: 0 samples currently; need at least 50-100
4. **Reduce FP rate**: fingerprint store matching is too aggressive for
   markdown-formatted content
5. **Pre-sanitization structural check**: Run _BOUNDARY_PATTERNS on raw input
   BEFORE Layer 0 sanitization strips the tokens

## Research Sources

- OWASP LLM01:2025: https://genai.owasp.org/llmrisk/llm01-prompt-injection/
- PayloadsAllTheThings: https://github.com/swisskyrepo/PayloadsAllTheThings
- Meta Llama 2 docs: https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-2/
- Phi-3 model card: https://huggingface.co/microsoft/Phi-3.5-mini-instruct
- Gemma docs: https://ai.google.dev/gemma/docs/core/prompt-structure
- Mistral tokenization: https://docs.mistral.ai/cookbooks/concept-deep-dive-tokenization-chat_templates
- OWASP Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html
- BSI Evasion Attacks: https://www.bsi.bund.de/SharedDocs/Downloads/EN/BSI/KI/Evasion_Attacks_on_LLMs-Countermeasures.pdf
- ICLR 2025 Role Separation: https://arxiv.org/html/2505.00626v2
- Snyk PI Techniques: https://snyk.io/articles/understanding-prompt-injection-techniques-challenges-and-risks/
