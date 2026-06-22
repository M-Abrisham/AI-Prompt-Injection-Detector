[← Back to main README](../README.md)

# Prompt-Injection Coverage Matrix

Audit-ready, two-dimensional mapping of Na0S's internal injection taxonomy to the OWASP Top 10 for LLM Applications (2025) and MITRE ATLAS. The matrix is **2D**: **attack-class rows** (the *what*) × an orthogonal **evasion-modifier axis** (the *how it's disguised*). Scores are graded against the **runtime detection path**, and — for every category the recall harness covers — against **measured recall** rather than expert estimate.

**v2.1 grounds the scores in measured data.** The two-sided recall harness (`scripts/technique_analysis.py` → `benchmarks/results/technique_analysis.json`) is now the single source of truth for coverage. Where it measures a category, the row's score **is** that measured recall (`Basis = measured`); elsewhere the score is rubric-estimated from runtime wiring (`Basis = estimated`). **The measured numbers came in materially below the v1.0/v2.0 estimates** (e.g. D1 direct-injection 72.5% measured vs 92 estimated; compliance 24% vs 72; exfiltration 48% vs 78) — confirming the estimates were optimistic. Treat the harness artifact, not the prose estimates, as authoritative.

---

| Field | Value |
|-------|-------|
| **Version** | v2.1 (2D, measured-grounded) |
| **Last reviewed** | 2026-06-20 (T/INJ-0026 & IM/INJ-0017: detector LANDED on branch `hardening/decompose-im-detection` — IM matchers + free-text tool-abuse + goal-decomposition wired into `scan()`/`cascade.py`; GTG-1002 xfails flipped to strict, 6/6 attacks BLOCK, 6/6 benign siblings ALLOWED, IM 100% recall on 8 zero-recall techniques / 0% benign FP) · 2026-06-16 (harness-grounded rebuild on branch `feat/inj0017-inter-model-detector`; D8 hardening in-flight) |
| **Owner** | @M-Abrisham |
| **Coverage source of truth** | `scripts/technique_analysis.py` → `benchmarks/results/technique_analysis.json` (schema v2; threshold **0.55**; measured 2026-06-17, uncommitted). **Not** the per-row "samples" columns and **not** `BENCHMARK_RESULTS.md` (whose 100% D8 row is known-mislabeled and unfixed). |
| **Source catalog — MITRE ATLAS** | v2026.05 (format-version 6.0.0; collection 2026.05; modified 2026-05-27) — https://github.com/mitre-atlas/atlas-data/releases/download/v2026.05/ATLAS-2026.05.yaml |
| **Source catalog — OWASP** | OWASP Top 10 for LLM Applications 2025 (`LLMxx:2025`) — https://genai.owasp.org/llm-top-10/ |
| **Internal taxonomy** | `data/taxonomy.yaml` v1.0 — 31 categories, 284 techniques. (2026-06-22: added the mid-level `E1` exfiltration key so the live `v0.1/` prompt-exfiltration scenarios filed under bare `attack_category: E1` validate; `BEN` confirmed canonical.) |
| **ATLAS anchoring** | The weekly threat-intel harvester is **taxonomy-aware and ATLAS-anchored**: `data/threat_intel_snapshots/atlas_to_na0s_mapping.yaml` (human-reviewed, 42 entries) maps input/output-boundary ATLAS techniques → canonical Na0S codes; `na0s.eval.harvest.tag_discovery` tags each discovery record (ATLAS id wins, else conservative keyword table; never invents, never drops). Attacker-side ATLAS techniques are deliberately unmapped. Offline/keyless. |
| **Review cadence** | Quarterly + re-run the harness; upgrade `estimated` rows to `measured` as the holdout grows. |

**Harness headline (measured @ 0.55):** overall malicious recall **57.1%** (194/340, CI 51.8–62.2) · overall benign FPR **1.2%** (6/500, CI 0.6–2.6) · overall evasion **47.6%** (270/567). The recall gate **fails** on C1, D2, E1, E2, O1, P1.

---

## Scoring rubric

Each class is scored 0–100 on how completely Na0S **detects it at runtime today**. For categories the harness covers, the score **is the measured recall** (rounded), `Basis = measured`. Otherwise the score is assigned by band from runtime **wiring** + **test depth**, `Basis = estimated`; `mixed` means part of the row is measured and part estimated. Sample-generator existence in `scripts/taxonomy/` earns **no** score on its own.

| Band | Score | Meaning |
|------|-------|---------|
| **Validated** | 85–100 | Wired into the default scan path with a dedicated test; measured recall ≥85% **or** (no harness slice) full wiring + targeted test. |
| **Asserted** | 70–84 | Wired + tested, but tests generic or coverage a verified sub-scope; or measured recall 70–84%. |
| **Partial** | 45–69 | A real slice wired/measured, but a material portion has no detection path, or the detector is gated/opt-in; or measured recall 45–69%. |
| **Taxonomy-only** | 15–44 | Named in `data/taxonomy.yaml`, no wired detector / failing recall; or measured recall 15–44%. |
| **Not modeled** | 0–14 | No detector, samples, or taxonomy entry. |

> **Wiring is load-bearing; measurement overrides estimate.** A detector that exists but is never invoked by the default `scan()` scores Partial at best. And where the harness has measured a category, that number wins over any estimate — even when it is lower.

---

## Coverage matrix — attack-class rows

| Row | Cat | OWASP | ATLAS | Class (subtype) | Gate | Score | Basis | Measured recall (harness @0.55) | Components (runtime) | Severity | Residual gap |
|-----|-----|-------|-------|-----------------|------|-------|-------|------------------------------|----------------------|----------|--------------|
| **INJ-0001** | D1 | LLM01 | T0051.000 Direct | Direct prompt injection / instruction override | input | **73** | measured | **72.5%** (29/40) | `layer1/rules_registry.py`, `structural/`, ML — WIRED; `test_scan_d1…` (38 pass) | critical | 27% of direct-override holdout missed; estimate (92) was +20pts optimistic. |
| **INJ-0002** | D3 | LLM01 | T0051.000 Direct | Structural-boundary / in-body injection | input | **83** | measured | **83.3%** (25/30) | `rules_registry.py` (D3 rules), `html_extractor.py` — WIRED | high | Strongest measured class after D8-adjacent; delimiter subset. |
| **INJ-0003** | M | LLM01 | T0051.001 Indirect | Multimodal injection (doc/image/audio) | input | **68** | estimated | — (not in harness) | `layer0/content_type.py`, `doc_extractor.py`, `ocr_extractor.py`, `detectors/visual_injection.py` (gated) — PARTIAL | high | No audio analysis; visual detector L0-gated; **unmeasured** (likely <68 by analogy to other measured drops). |
| **INJ-0004** | — | LLM01 | *no ATLAS* (closest T0011) | Social-engineering relay (user induced) | input | **8** | estimated | — | None | medium | Not modeled. |
| **INJ-0005** | — | LLM01 | T0011 User Execution *(closest)* | Social-engineering courier | input | **8** | estimated | — | None | medium | Not modeled. |
| **INJ-0006** | O | LLM05 | T0061 Self-Replication | Output-channel injection / worm + harmful-output | output | **50** | mixed | **O1 30%** (6/20) measured; worm est. | `output/scanner.py`, `output/propagation.py` (worm, wired+tested); `detectors/harmful_intent.py` (O1, WIRED) | high | O1 harmful-output intent measured **30%**; worm single-hop only; O2 output-injection via `cascade.scan_output()` (separate call). |
| **INJ-0007** | D8 | LLM01 | T0080.001 Thread | Context-window manipulation / multi-turn | mixed | **64** | measured | **64%** (16/25) | `detectors/context_manipulation.py` (**now wired** — was dead code), `detectors/token_budget.py`, `detectors/state_confusion.py`, per-segment ML max-pool, `rag/position_scanner.py`, multi-turn fold-in — all WIRED *(D8 hardening in-flight, pending commit)* | high | Wiring now complete (all 6 D8.x have a path; 14/14 scan tests green) — residual is **recall calibration**: ~7 round-number gate thresholds unvalidated; multi-turn needs `session_id`; detectors gate at >512 words. |
| **INJ-0008** | I1 | LLM01 | T0051.001 Indirect | Indirect injection via ingested data | input | **84** | estimated | — (not in harness) | `data_source_poisoning.py`, `rag/poison_detector.py` (WIRED), R1.x rules | critical | **Unmeasured** — estimate likely optimistic given measured drops elsewhere; add I1 to the holdout next cycle. |
| **INJ-0009** | I1 | LLM08 | T0070 RAG Poisoning | Indirect via internal docs / KB poisoning | input | **76** | estimated | — | `data_source_poisoning.py` (I1.2/I1.4), `rag/poison_detector.py` | critical | Unmeasured; generic RAG-poison path. |
| **INJ-0010** | I1 | LLM01 | T0051.001 Indirect | Indirect via external web/email/feeds | input | **76** | estimated | — | `data_source_poisoning.py` (I1.1/I1.3), `html_extractor.py` | critical | Unmeasured. |
| **INJ-0011** | I1 | LLM01 | T0051.001 Indirect | Indirect from attacker-controlled source | input | **70** | estimated | — | `data_source_poisoning.py` (I1.1); AD2.4 taxonomy-only | high | Unmeasured. |
| **INJ-0012** | I1 | LLM01 | T0051.001 Indirect | Indirect from compromised trusted source | input | **72** | estimated | — | `data_source_poisoning.py` (I1.1 SO/GitHub) | high | Unmeasured. |
| **INJ-0013** | I1 | LLM01 | T0051.001 Indirect | Indirect via attacker-influenced UGC | input | **72** | estimated | — | `data_source_poisoning.py` (I1.1 UGC) | high | Unmeasured. |
| **INJ-0014** | S, IG | LLM03 | T0010.002 Supply Chain: Data | Supply-chain / ETL ingestion compromise | mixed | **52** | estimated | — | `supply_chain.py`, `ingestion_manipulation.py`, `safe_pickle.py`+`KNOWN_HASHES` (load-time) | high | Load-time integrity only; IG1.x no `src/` detector; unmeasured. |
| **INJ-0015** | D1.20 | LLM01 | T0080.001 Thread | Context-memory poisoning via prior output | mixed | **86** | estimated | — (not in harness) | `layer16/detectors/context_poisoning.py` (WIRED) | high | Dedicated detector + 40-test suite; **unmeasured by the recall harness** — only Validated row not measured. |
| **INJ-0016** | I1.6, IG1.8 | LLM01 | T0080.000 Memory | Persistent agent-memory poisoning | agent-tool | **40** | estimated | — | taxonomy-only (I1.6/IG1.8) | high | No detector, no test. |
| **INJ-0017** | IM, IG, AD | LLM06 | T0080.000 Memory *(dual: T0061)* | Inter-model / cross-system propagation | agent-tool | **60** | measured (GTG IM 3/3 block; matchers 100% on 8 zero-recall / ~33% broad; 0% benign FP) | GTG IM 3/3; 100% (144/144) on 8 zero-recall techniques | `detectors/inter_model.py` now implements 24 self-anchored co-occurrence matchers across 6 `FAMILIES`, WIRED into `scan()`+`cascade.py` behind `_HAS_INTER_MODEL` (capped, corroborating); GTG-1002 IM scenarios (IM3.1/IM3.4/IM5.3) BLOCK via IM + goal-decomposition; xfail flipped strict | critical | Wired + GTG-blocked. Residual: broad IM recall ~33% across all 516 IM samples (matchers target the co-occurrence families); benign FP 0%. |
| **INJ-0018** | D2 | LLM01 | T0054 Jailbreak | Persona / roleplay jailbreak (DAN) | input | **57** | measured | **56.7%** (17/30) | `persona_roleplay.py` (D2), rules + `structural/` (role→D2.1) — WIRED | high | **Gate FAILS** (CI-low 0.39); estimate (70) was +13 optimistic. |
| **INJ-0019** | E | LLM07 | T0056 Extract System Prompt; T0057 Data Leakage | Exfiltration / system-prompt & secret extraction | mixed | **48** | measured | **E1 48%** (12/25); **E2 0%** (0/20) | `detectors/extraction.py` (WIRED), E1/E2 rules, `output/scanner.py`, `canary/` | critical | **E2 active-recon measured 0%** — wired but ineffective (gate FAILS). Estimate (78) was +30 optimistic. |
| **INJ-0020** | P, P2 | LLM02 | T0057 Data Leakage; T0024 Exfil via API | Privacy / PII / training-data extraction | input | **35** | measured | **P1 35%** (7/20) | `detectors/privacy_probe.py` (WIRED), P1 rules | high | **Gate FAILS**; P2 membership-inference untagged. Estimate (74) was +39 optimistic. |
| **INJ-0021** | C, C1 | LLM01 | T0054 Jailbreak | Compliance / policy / safety evasion | input | **24** | measured | **C1 24%** (6/25) | `detectors/fictional_frame.py` (WIRED), compliance rules | high | **Gate FAILS, lowest measured class.** Estimate (72) was +48 optimistic — the largest estimate error. |
| **INJ-0022** | C1MT | LLM01 | T0080.001 Thread | Multi-turn escalation / Crescendo | mixed | **56** | estimated | — (C1MT not in harness) | `layer16/detectors/escalation.py` + `ConversationSecurityMonitor`; **now folds into `scan()`** when `session_id` set (D8-G02) | high | Multi-turn fold-in is opt-in via `session_id`; default `scan(text)` provides none; unmeasured. |
| **INJ-0023** | A | LLM01 | T0054 Jailbreak *(automated)* | Adversarial-ML / automated jailbreak (GCG/PAIR/TAP) | input | **48** | estimated | — | `layer0/tokenization.py` (A1.1), `perplexity.py` (gated) — PARTIAL | high | GCG-suffix only; PAIR/TAP/AutoDAN/LatentBreak (fluent, evade perplexity) undetected. |
| **INJ-0024** | P3 | LLM05 *(adjacent)* | *no clean ATLAS* | Malicious code generation | input | **42** | estimated | — | `malicious_code_gen.py` (P3); overlaps `harmful_intent.py` O1.2 only | high | No P3 detector; samples-only for the named technique. |
| **INJ-0025** | R | LLM10 | T0034 Cost Harvesting; T0029 DoS | Resource exhaustion / DoS / cost-amplification | mixed | **46** | estimated | — | `layer0/resource_guard.py`, length/chunk caps, R1 rules — PARTIAL | medium | Input-side R1.1 only; output-side recursive/cost not detected. |
| **INJ-0026** | T | LLM06 | T0053 Tool Invocation; T0011.002 Poisoned Tool | Agent / tool / MCP abuse | agent-tool | **62** | measured (GTG T 3/3 block; benign 0 FP) | GTG T 3/3 | New `detectors/tool_abuse.py` (in-prose T1.x free-text matcher) + `scan(tool_calls=…)` autoroute through `mcp_tool` manifest scan, WIRED behind `_HAS_TOOL_ABUSE` into `scan()`+`cascade.py`; GTG-1002 T scenarios (T1.1/T1.3/T2.3) BLOCK via tool-abuse + goal-decomposition; xfail flipped strict | critical | On the default path now. Residual: not yet benchmarked vs MCPTox/MCPSecBench; manifest path needs structured tool traces supplied via `tool_calls=`. |
| **INJ-0027** | T (T1.2 / T1.4 / T2.3) | LLM06 | T0053 Tool Invocation; T0011.002 Poisoned Tool | MCP supply-chain lifecycle (tool-poisoning / rug-pull / typosquat) | agent-tool | **55** | estimated (64 unit tests pass; FP-safe on legit manifests; not in recall harness) | — (not in harness) | New `detectors/mcp_supply_chain.py` (`scan_tool_supply_chain` + capped `get_mcp_supply_chain_weight` ≤0.30) reached via the new `na0s.mcp` FastMCP **guard server** (`scan_text` / `check_tool_call` / `check_tool_response` over `na0s.scan`/`scan_output`/`predict.scan_tools`); SERVER-ONLY (behind `_HAS_MCP_SUPPLY_CHAIN`, NOT on the default free-text `scan()` path). Emits canonical T1.2 (description injection / OWASP-MCP01), T1.4 (schema poisoning / OWASP-MCP03, rug-pull / OWASP-MCP06, typosquat plugin-confusion), T2.3 (exfil / OWASP-MCP10). MCP SDK imported lazily (`pip install na0s[mcp]`). | critical | Local/keyless; reachable only when a host runs the guard server (opt-in), not from `scan(text)`. Rug-pull needs an approved `ToolBaselineStore`. Unmeasured by the recall harness; not yet benchmarked vs MCPTox/MCPSecBench. |

---

## Coverage matrix — evasion-modifier axis (measured detection rates)

Evasion is centralized preprocessing applied to every input before class detection, so class rows inherit it. The harness measures each modifier directly (`per_evasion_type`, n=63 each unless noted):

| Modifier | Cat | Mechanism (runtime) | Measured detection | Status |
|----------|-----|---------------------|--------------------|--------|
| **Base64** | D4 | decode-and-rescan ([obfuscation.py](src/na0s/layer2/obfuscation.py)) | **90.5%** (57/63) | strong |
| **Whitespace stego** | D4/D5 | SNOW/statistical/trailing-WS detectors | **71.4%** (45/63) | good |
| **Hex (incl. spaced)** | D4 | `_EMBEDDED_HEX_RE` + new `_SPACED_HEX_RE` (≥8 pairs, keyword-gated) | **69.8%** (44/63) | **0%→70% (spaced-hex fix)** |
| **Syllable-split** | D4 | syllable normalizer | **60.3%** (38/63) | partial |
| **Unicode homoglyph** | D5 | NFKC + homoglyph fold | **57.1%** (36/63) | partial |
| **ROT13 / Caesar** | D4 | brute-force decode (keyword-gated) | **31.7%** (20/63) | weak |
| **Mixed-encoding** | CT | — | **20.6%** (13/63) | weak |
| **Reversed** | D4 | full + per-word reverse (keyword-gated) | **15.9%** (10/63) | weak |
| **Leetspeak** | D4 | leet normalizer (keyword-gated) | **11.1%** (7/63) | weakest |
| **Multi-buff stacking** | MB, CT | recursive decode (depth-4) | **33%** (measured via `na0s.scan`, not harness) | **OPEN GAP** |

Category-level measured (per-category holdout, different pool): **D4 80%, D5 80%, D6 multilingual 67%, D7 delivery 64%**.

**The stacking gap, confirmed by measurement.** [layer2/obfuscation.py](src/na0s/layer2/obfuscation.py) recurses, but its ROT13/Reverse/Leet/Caesar decoders are **keyword-gated per layer** — a chain whose *inner* transform is keyword-gated never peels (the intermediate layer has no plaintext keywords to satisfy the gate). Measured: **33%** overall MB recall, **0%** on every chain with a keyword-gated inner transform (Base64+ROT13, ROT13+Leet, Reverse+ROT13, …), **100%** only when both buffs are non-gated codecs/unicode. This is also visible above: base64 90% vs rot13 32%, leetspeak 11%, reversed 16%, mixed-encoding 21%. **Per-row exceptions:** multimodal (INJ-0003) — decode doesn't reach inside image/audio; samples-only rows (INJ-0016/0017/0024) — no base detector to evade.

> **Buff provenance.** Na0S's 8 `_buffs.py` transforms are a superset of garak's 6 built-in buffs; the "29 probes × 8 buffs" framing is inspired-by, not parity-with, garak.

---

## Summary rollup

**26 classes mapped** (17 original + 9 new `INJ-0018…0026`). **8 rows now measured/mixed; 18 estimated.** Mean (equal-weight) coverage: **55.1%**. Severity-weighted (critical×4, high×3, medium×2): **57.1%** — coincident with the harness's overall 57.1% recall.

The mean fell from v2.0's 61.8% **because measurement replaced optimistic estimates**, not because coverage regressed. Net real movement is *up* (D8 hardening landed; D4 spaced-hex 0→70%) but the measured numbers expose that estimates ran **+13 to +48 points high** on every category the harness touches.

**By score band**

| Band | Count | Rows |
|------|-------|------|
| Validated (85–100) | 1 | INJ-0015 *(estimated — only Validated row not yet harness-measured)* |
| Asserted (70–84) | 8 | INJ-0001, INJ-0002, INJ-0008, INJ-0009, INJ-0010, INJ-0011, INJ-0012, INJ-0013 |
| Partial (45–69) | 12 | INJ-0003, INJ-0006, INJ-0007, INJ-0014, INJ-0017, INJ-0018, INJ-0019, INJ-0022, INJ-0023, INJ-0025, INJ-0026, INJ-0027 |
| Taxonomy-only (15–44) | 4 | INJ-0016, INJ-0020, INJ-0021, INJ-0024 |
| Not modeled (0–14) | 2 | INJ-0004, INJ-0005 |

**Measured vs estimated — the honesty gap.** Where both exist: D1 92→**72.5**, D3 88→**83.3**, D2 70→**56.7**, exfil 78→**48** (E2 **0**), privacy 74→**35**, compliance 72→**24**. The 8 estimated Asserted rows (INJ-0008…0013 indirect-injection, all unmeasured) are the **highest-risk overstatements** — by the observed pattern they likely measure 15–25 points lower; adding I1 to the holdout is the top measurement priority.

**Priority signal (measured + structural):**
1. **INJ-0021 compliance (24%)** and **INJ-0019 E2 recon (0%)** — lowest measured, gate-failing, wired-but-ineffective. Detection logic, not wiring, is the gap.
2. **INJ-0020 privacy (35%)**, **INJ-0018 jailbreak (57%)**, **O1 (30%)** — gate-failing measured slices.
3. **Evasion stacking (33%)** — keyword-gated chaining; affects every class.
4. **INJ-0017 (60)** and **INJ-0026 (62)** — agent-tool surface, now WIRED (2026-06-20, branch `hardening/decompose-im-detection`): IM matchers + free-text tool-abuse + goal-decomposition on the default `scan()` path; GTG-1002 6/6 block, 0 benign FP. Residual: broad IM recall ~33%; MCPTox/MCPSecBench benchmarking pending.
5. **INJ-0027 (55)** — MCP supply-chain lifecycle (tool-poisoning / rug-pull / typosquat), branch `feat/mcp-guard`: new `na0s.mcp` FastMCP guard server + `detectors/mcp_supply_chain.py` (T1.2/T1.4/T2.3 ↔ OWASP-MCP01/03/06/10). SERVER-ONLY (opt-in; behind `_HAS_MCP_SUPPLY_CHAIN`, NOT on the default free-text `scan()` path), local/keyless, FP-safe on legit manifests. Residual: unmeasured by the recall harness; not yet benchmarked vs MCPTox/MCPSecBench.

---

## Framework reverse-traceability

Maps **8 of 10** OWASP LLM 2025 items: LLM01, LLM02 (INJ-0020), LLM03, LLM05, LLM06, LLM07 (INJ-0019), LLM08, LLM10 (INJ-0025).

**Deliberately out of scope** (a prompt-injection matrix): **LLM04** Data & Model Poisoning (training-time/model-weight; runtime RAG/ingestion poisoning *is* covered via INJ-0008/0009/0014) and **LLM09** Misinformation (intrinsic hallucination; injection-induced misinfo covered transitively).

---

## Rows needing external-taxonomy judgment

- **INJ-0004 — human-relay delivery:** no clean ATLAS equivalent (T0011 User Execution is execution-scoped, not prompt-relay).
- **INJ-0005 — deceived courier:** `AML.T0011` as "closest, not exact."
- **INJ-0024 — malicious code generation:** no clean ATLAS; OWASP LLM05 (adjacent).
- **INJ-0017 dual-map:** `AML.T0080.000 (Memory)` for persistence + `AML.T0061 (Self-Replication)` for cross-model propagation.

*(Verify ATLAS IDs against the v2026.05 release pinned in the header.)*

---

## Known doc-hygiene issues (referenced, not owned by this file)

These live in sibling docs that are under active edit; flagged here for the next sync rather than silently depended on:

- **`THREAT_TAXONOMY.md`** — IM category still tagged `OWASP LLM01/LLM15` (**LLM15 does not exist** in OWASP 2025; should be LLM06 per this matrix). Internal count drift: "20 categories" vs "eighteen" in adjacent lines. The "zero training samples" line is now stale for D4/D5/D6 (the file's own measured-coverage note supersedes it).
- **`data/taxonomy.yaml`** — IM category carries the invalid `owasp-llm:2025:llm15` tag (source of the THREAT_TAXONOMY error).
- **`docs/TAXONOMY.md`** — says "19 categories / 103+ techniques"; authoritative count is **29 / 276**; disagrees with THREAT_TAXONOMY's "20 / 108".

---

## Audit trail (→ v2.1)

- **Measured grounding.** Scores for the 13 harness-covered categories now use measured recall @0.55 (`Basis = measured`/`mixed`); the rest stay `estimated`. Harness artifact is the declared source of truth, over the per-row "samples" columns and over `BENCHMARK_RESULTS.md` (whose 100% D8 row is known-mislabeled and unfixed — **not** used here).
- **D8 hardening reflected (in-flight, pending commit).** INJ-0007 rises 58→64 (measured): `context_manipulation.py` no longer dead code, multi-turn verdict folds into `scan()`, new `token_budget.py`/`state_confusion.py`, per-segment max-pool, cascade parity — all six D8.x techniques wired, 14/14 scan tests green. Residual is recall calibration (~7 round-number gates unvalidated), not wiring.
- **D4 spaced-hex reflected.** Evasion axis hex 0%→69.8% (measured). Multi-buff stacking remains **OPEN** (measured 33%, keyword-gated chaining untouched).
- **MCP guard server added (2026-06-22, branch `feat/mcp-guard`).** New INJ-0027 row: the `na0s.mcp` FastMCP guard server (`scan_text` / `check_tool_call` / `check_tool_response`) + `detectors/mcp_supply_chain.py` (tool-poisoning / rug-pull / typosquat). Emits canonical taxonomy codes only — T1.2 / T1.4 / T2.3 — with OWASP-MCP-Top-10 codes (MCP01/03/06/10) carried as external reference labels, NOT taxonomy codes. SERVER-ONLY (opt-in; does not change the default `scan()` path); local/keyless; the MCP SDK is an optional `na0s[mcp]` extra.
- **2D structure + 9 new rows + Category column** retained from v2.0. Four v1.0 ATLAS/OWASP mapping corrections retained.
- **Sibling-doc edits left untouched** — ROADMAP_V2.md and THREAT_TAXONOMY.md carry the author's own in-flight measured-coverage notes; this file is kept consistent with them, not overwritten onto them.

> **Scoring caveat.** `estimated` rows express relative ordering, not measured detection; the measured-vs-estimated gap above shows estimates run optimistic. Upgrading `estimated` → `measured` (extend the holdout to I1/M/E2/T/etc., re-run the harness, record per-row pass-rate) is the standing next-quarter obligation.

---

*Owner: @M-Abrisham. Last validated: 2026-06-16 (harness-grounded v2.1). Re-run the recall harness each cycle and re-confirm IDs against the then-current MITRE ATLAS / OWASP LLM Top 10.*
