# Na0S — Roadmap

> Na0S is a defensive SDK that companies embed in their AI products to block prompt
> injection. This roadmap tracks (1) the **detection layers** (the runtime pipeline) and
> (2) the **v1.0.0 tree-restructure refactor** we are actively executing.
>
> All numbers in this document are grep/python-verified against the live repo on
> `fix/l5-cascade-centroid-parity` (last verified 2026-06-03). Deep retrospective detail
> has been pushed to `CHANGELOG.md` and `docs/` — this file is a roadmap, not a changelog.

---

## ⚠️ NUMBERING LEGEND — read this first

The roadmap uses **two numbering systems that DO NOT correspond.**

| | Meaning | Range |
|---|---|---|
| **Layer N** (L0–L21) | a **detection layer** in the runtime pipeline (what the SDK does) | 0–21 |
| **Phase N** (1–14) | a **refactor step** in the v1.0.0 Migration Sequence (how we reorganize the source tree) | 1–14 |

They are unrelated. Example: **Phase 9** (`refactor/promote-fusion-modules`) moves **Layer 6**
(fusion / voting) code — it is NOT "Layer 9". When a section says "Phase 7", it means
migration step 7, never Layer 7. Always say **"Phase"** for refactor work and **"Layer"**
for detection work.

---

## 1. Status at a Glance

**Overall: ~79% of detection work shipped; v1.0.0 source-tree refactor in flight.**

Runtime pipeline (collapsed):

```
input → L0 sanitize/gate → L1 rules → L2 deobfuscate → L3 structural feats →
  L4 TF-IDF ML → L5 embedding ML → L6 cascade/voting → L7 LLM judge →
  L8 positive validation → verdict
  (+ L9 output scan, L10 canary, L11 supply-chain, L12 probes/taxonomy,
     L13 dataset pipeline, L14 harness/CI, L15 threat-intel, L16 multi-turn,
     L17 doc-format, L18 RAG-ingest, L19 agent/MCP, L20 taxonomy-automation,
     L21 telemetry)
```

**In flight right now:**
- **v1.0.0 tree restructure** — Phases 1–14 below. Several early phases already landed
  (`output/`, `validation/`, `structural/`, `dataset/`, `eval/` packages exist). The
  remaining work is the rename/promote/delete-shims tail.
- **Layer 13 v2** — dataset pipeline redesign (NVD/GHSA scrapers, F11/F12/F14 promotion gate).
- **Layer 17** — document-format scanners (PDF/CSV/RTF/email/SVG) at ~35%.

> Note on progress percentages below: per-layer "done/total" figures are **manual bookkeeping**,
> not code-derivable. They are kept for direction but are not independently grep-verifiable.
> The old headline "608/768" total was internally inconsistent with its own rows (rows summed to
> 592/655) and has been removed rather than reprinted.

---

## 2. Active Refactor — v1.0.0 Migration Plan

This is the live refactor we are executing. Goal: collapse **86 top-level `src/na0s/*.py`
files** down to **~9** by moving real code into sub-packages and deleting re-export shims.

**Verified current state (2026-06-03):**

| Fact | Value | How verified |
|---|---|---|
| Top-level `src/na0s/*.py` files | **86** | `ls -1 src/na0s/*.py \| wc -l` |
| — of which genuine re-export **shims** | **67** | `grep -lE 'SHIM -- do not add new code here' src/na0s/*.py` (65) ∪ `obfuscation.py` ∪ `data_schema.py` |
| — of which **real code** files | **19** | 86 − 67 |
| Python **sub-packages** in `src/na0s/` | **21** | `find … -type d ! -name __pycache__` |
| `layerN/` dirs still **not renamed** | **5** (`layer0, layer1, layer2, layer15, layer16`) | `ls -1d src/na0s/layer*` |
| Target top-level file count (post-v1.0.0) | **~9** | design goal |

> The old roadmap claimed "24 real + 61 shims = 85". That is stale. Verified split is
> **19 real + 67 shims = 86**. Resolution method for the shim count: union of files carrying
> the literal `SHIM -- do not add new code here` marker (65) plus `obfuscation.py` and
> `data_schema.py` (both genuine re-export shims with nonstandard/marker-less wording) = **67**.

### 2.1 File-destination table — real top-level files → sub-package

These 20 files are the real (non-shim) code at top level. LOC are re-verified; several drifted
from the old roadmap (corrections noted).

| File | LOC | Destination | Notes |
|---|---|---|---|
| `__init__.py` | 110 | **stays** | public API (was listed 111) |
| `__main__.py` | 6 | **stays** | CLI entry |
| `_version.py` | 1 | **stays** | |
| `_env.py` | 40 | **stays** | |
| `cli.py` | 585 | **stays** | may split to `cli/` later |
| `config.py` | 81 | **stays** | grows as layers externalize values (was listed 65) |
| `scan_result.py` | 47 | **stays** | |
| `predict.py` | 1,819 | **rename → `_pipeline.py`** (Phase 13) | was listed 1,784 |
| `cascade.py` | 1,483 | **stays** | core orchestrator |
| `rules.py` | 29 | `rules/patterns.py` (Phase 10) | shared regex constants (L1/L6/L8 import) |
| `_voting.py` | 403 | `fusion/voting.py` (Phase 9) | L6 (was listed 388) |
| `signal_boost.py` | 483 | `fusion/signal_boost.py` (Phase 9) | L6 |
| `evidence_grading.py` | 130 | `fusion/evidence_grading.py` (Phase 9) | L6 |
| `groundedness.py` | 87 | `fusion/groundedness.py` (Phase 9) | L6 |
| `compliance_evasion_rules.py` | 286 | `rules/registry/compliance_evasion.py` (Phase 10) | L1 |
| `subtle_override_rules.py` | 121 | `rules/registry/subtle_override.py` (Phase 10) | L1 |
| `multilingual_handler.py` | 288 | `detectors/multilingual_handler.py` (Phase 11) | D6 detector |
| `multilingual_intent.py` | 371 | `detectors/multilingual_intent.py` (Phase 11) | D6 semantic |
| `intent_guard.py` | 421 | `detectors/intent_guard.py` (Phase 11) | N1 category |
| `data_schema.py` | 6 (shim) | already moved → `dataset/schema.py` (311 LOC) | **already migrated**; top-level now a shim |

**Already-migrated (now shims, were listed as "real" in the old roadmap):**
`positive_validation.py` → `validation/positive.py` (537 LOC);
`structural_features.py` → `structural/features.py` (136 LOC);
`segment_grader.py` → `output/segment_grader.py`;
`obfuscation.py` → re-export shim to `layer2/` (delete at v1.0.0).

**66 top-level shim files → all delete at v1.0.0 (Phase 14).** Each re-exports from its
canonical sub-package (`canary_alert.py` → `canary.alert`, `safe_pickle.py` →
`integrity.safe_pickle`, `llm_judge.py` → `judge.llm_judge`, `embedding_classifier.py` →
`ml.embeddings.classifier`, …). v1.0.0 ships `docs/MIGRATION_v1.md` mapping every old import
to its new home, then deletes shims in one commit.

### 2.2 Migration sequence — Phases 1–14

Each phase is a `refactor/*` branch. Status reflects what is on disk now.

| Phase | Branch | What it does | Status |
|---|---|---|---|
| **1** | `refactor/create-validation-package` | `validation/` ← `positive_validation.py` (split `positive.py` + `trust_boundary.py`) + `validation_allowlist.py`. **`multi_turn_validator` → `conversation/`** (conversation-scoped), NOT `detectors/`. | `validation/` exists ✅ — see TRAP C |
| **2** | `refactor/create-output-package` | `output/` ← 4 scanners from `rag/` (output_scanner, propagation, streaming, dual_scanner) + `segment_grader.py`. | **DONE** ✅ `output/` exists; `rag/` slimmed to 3 files |
| **3** | `refactor/create-features-package` | `structural/` (L3) + absorb misplaced feature code. | **DONE** ✅ `structural/features.py` exists |
| **4** | `refactor/create-dataset-package` | `dataset/` ← `data_schema.py` + library code from `scripts/`. | **DONE** ✅ `dataset/schema.py` exists |
| **5** | `refactor/create-eval-package` | `eval/` ← `scripts/evaluate_*` + `benchmark_*` + dashboard libs. | **DONE** ✅ `eval/` + `eval/scenarios/` exist |
| **6** | `refactor/create-probes-package` | `probes/` ← `scripts/taxonomy/` lib, thin CLI wrappers in `scripts/`. | NOT STARTED |
| **7** | `refactor/create-agents-package` | `agents/` ← `detectors/mcp_tool.py` + L19 stubs. | ⚠️ **BLOCKED** — see TRAP A |
| **8** | `refactor/create-taxonomy-package` | `taxonomy/` ← `scripts/sync_taxonomy.py` etc. (L20). | NOT STARTED |
| **9** | `refactor/promote-fusion-modules` | sweep `_voting.py`, `signal_boost.py`, `evidence_grading.py`, `groundedness.py` → `fusion/` (this is **Layer 6** code). | NOT STARTED |
| **10** | `refactor/promote-rules-modules` | sweep rules extensions → `rules/registry/`. | ⚠️ ordering — see TRAP B |
| **11** | `refactor/promote-detectors-modules` | sweep intent/multilingual/etc → `detectors/`. | NOT STARTED |
| **12** | `refactor/rename-layer-packages` | `layer0/`→`input/`, `layer1/`→`rules/`, `layer2/`→`obfuscation/`, `layer15/`→`threat_intel/`, `layer16/`→`conversation/`. | ⚠️ ordering — see TRAP B |
| **13** | `refactor/rename-predict-to-pipeline` | `predict.py` → `_pipeline.py` + caller updates. | NOT STARTED |
| **14** | `refactor/delete-shims` | final purge of all 66 shims; tag `v1.0.0`. | NOT STARTED |

### 2.3 Known refactor traps — READ BEFORE EXECUTING

> **TRAP A — Phase 7 `agents/` name collision.**
> `src/na0s/agents/` **already exists on disk** and is occupied by the deploy-automation /
> OpenClaw approval system: `approval_history.py`, `approvals_sync.py`,
> `claude_gate_analyzer.py`, `deploy_approver.py`, `gate_analyzer.py`, `openclaw_bridge.py`,
> `orchestrator.py`, `quarantine_reviewer.py`, `synthetic_validator.py`. Phase 7 + Layer 19
> want `agents/` for MCP/A2A **security** code (`mcp_tool` + L19 stubs) — none of those L19
> files exist yet. **Resolve the name before running Phase 7.** Candidate: move detection code
> to `agents_security/` or `mcp/`, or relocate the deploy-automation code to `automation/`.
> Decision pending.

> **TRAP B — Phase 10 vs Phase 12 both target `rules/`.**
> Phase 10 (`promote-rules-modules`) sweeps extension modules into `rules/registry/`.
> Phase 12 (`rename-layer-packages`) renames `layer1/` → `rules/`. Both write into the same
> `rules/` dir, and Phase 10 currently runs first against a dir that does not yet exist under
> that name. **Either run Phase 12 before Phase 10** (rename first, then promote into it),
> **or** have Phase 10 create `layer1/registry/` and let Phase 12 rename it along with the rest.

> **TRAP C — Phase 1 `multi_turn_validator` routes to `conversation/`, NOT `detectors/`.**
> It is conversation-scoped. The top-level shim and any `detectors/` shim both retarget to
> `conversation/multi_turn_validator.py`. This is already the intended Phase 1 behavior —
> keep it; do not "fix" it into `detectors/`.

### 2.4 Target tree

Collapsed post-v1.0.0 top level = **~9 files** (`__init__`, `__main__`, `_version`, `_env`,
`cli`, `config`, `scan_result`, `_pipeline`, `cascade`). Full target tree (all sub-packages
+ per-layer module layout) → **`docs/TARGET_TREE_v1.md`** (move from old roadmap lines 157–298).

---

## 3. Per-Layer Status (terse)

One status line + OPEN TODOs only. Completed-item prose and per-layer target trees moved to
`CHANGELOG.md` / `docs/ARCHITECTURE.md`. Percentages are manual bookkeeping (not grep-verified).

### Layer 0 — Input Sanitization & Gating — COMPLETE
Unicode normalization, homoglyph folding, zero-width stripping, length/encoding gating.
- [ ] 7 open doc-TODOs (docstrings / inline rationale). P3.
- [ ] Char-level reassembly detector (2.5d) — see §4.

### Layer 1 — IOC / Signature Rules — ~88%
Regex signature engine: **120 rules across 68 technique IDs**, PL1–PL4 paranoia
(`RULES_PARANOIA_LEVEL`), 6 context-suppression frames, ReDoS-safe auto-compile, dual-pass
dedup, homoglyph folding. Novel detectors: summarization-extraction, authority-escalation,
constraint-negation, meta-referential, gaslighting. Detail → `CHANGELOG.md` + `rules_registry.py` docstring.
- [ ] YARA engine integration. P1.
- [ ] Phrase-DB rule mining pipeline. P1.
- [ ] 5 doc fixes. P3.
- [ ] **RCE / code-execution detection GAP (noted 2026-06-02).** L1 has `destructive_action`
  (git/shell file destruction) but **no rule** for code-exec injection: `exec()`/`eval()`,
  `os.system`/`subprocess`/`Popen`, `pickle.loads`, `__import__`, `compile()`, or remote-code
  payloads smuggled in prompts (cf. CVE-2025-6514 MCP OS-command injection, tracked separately
  by L19). Add a `code_execution` rule family + technique ID. Relates to the "SkillScanner"
  backlog item. **P1, Medium effort.**

> Verified: `RULES` has **120** entries (old roadmap said 117). Technique IDs = **68** (correct).

### Layer 2 — Obfuscation / Deobfuscation — ~98%
Peels Base64/hex/URL/ROT13/Caesar/leet/reversed/pig-latin/Morse/binary-octal-decimal-ASCII/
whitespace-stego; recursive Matryoshka unwrap (depth 4, SHA-256 cycle detection, 10× cap) with
`decoded_chain` provenance; entropy via 2-of-3 composite vote; re-feeds decoded views through
L1 + ML. Per-decoder detail → `CHANGELOG.md` v0.2.0.
- [ ] `shannon_entropy()` docstring. P3.
- [ ] Chained-obfuscation decoder (4d) — see §4.

### Layer 3 — Structural Features — ~88%
**29 numeric features** across 7 groups into a `StructuralFeatures` dataclass; `normalize_features()`
soft-caps 12 unbounded features to [0,1]; wired into the ML pipeline via `structural_scaler.pkl`
at 5 inference sites. Detail → `CHANGELOG.md` v0.2.0.
- [ ] Restore 135 deleted tests. P1.
- [ ] Benchmark feature-extraction latency. P2.
- [ ] Externalize URL/boundary/imperative patterns. P3.

### Layer 4 — TF-IDF ML Classifier — COMPLETE
Calibrated TF-IDF + linear classifier, structural features hstacked. Detail → `CHANGELOG.md`.

### Layer 5 — Embedding Classifier — COMPLETE
Centroid embedding classifier; wired into `CascadeClassifier` for scan() parity (commit `baa23e7`).

### Layer 6 — Cascade & Voting — COMPLETE
Score aggregation, complexity routing, weighted voting.
- Note: **Phase 9** moves this layer's fusion code (`_voting`, `signal_boost`,
  `evidence_grading`, `groundedness`) into `fusion/`. (Phase 9, not "Layer 9".)

### Layer 7 — LLM Judge — COMPLETE
Judge-based classification with structured verdicts.
- [ ] Delete legacy `checker.py` at v1.0.0. P2.

### Layer 8 — Positive Validation — COMPLETE
Benign-content recognition to suppress false positives.
- [ ] Consolidate `validation/` package post-Phase 1. P2.
- [ ] `multi_turn_validator` → `conversation/` (see TRAP C).

### Layer 9 — Output Scanner — COMPLETE
`OutputScanner` (now `output/scanner.py`, 848 LOC) runs 9 detector groups on LLM output —
secrets, role-break, compliance echoes, system-prompt leak, encoded data, PII, markdown/HTML
beacons, exfil URLs, egress — returning `OutputScanResult` + `redacted_text`. Companions:
`PropagationScanner`, `DualDirectionScanner`, `StreamingOutputScanner`, `RAGAttributionChecker`,
`segment_grader` (all in `output/`); `poison_detector` + `position_scanner` remain in `rag/`.
Wired into `cascade.py` only — NOT `predict.py`. Detail → `CHANGELOG.md`.
- [ ] HIGH: duplicate-redaction block in output scanner. P1.
- [ ] HIGH: threshold-bypass path in output scanner. P1.
- [ ] Wire output scan into `predict.py` (currently cascade-only). P2.

> Note: the `rag/`→`output/` migration is **DONE** (old roadmap called it future v1.0.0 work).

### Layer 10 — Canary Tokens — COMPLETE
Canary generation, embedding, and leak detection.

### Layer 11 — Supply-Chain Integrity — COMPLETE
`safe_pickle` 3-tier trust (hardcoded hash > HMAC-SHA256 sidecar > SHA-256 sidecar), atomic
writes, audit logging; hardened `safe_yaml`; env-gated provenance/encryption/rollback/dep-scan/
SBOM/chain-integrity modules. Detail → `CHANGELOG.md`.
- [ ] `safe_content.py` + `validation_allowlist.py` misfiled here — relocate. P2.

### Layer 12 — Probe Architecture & Taxonomy — COMPLETE
`ClassifierOutput` contract, `Probe` base with taxonomy auto-load, `expand()` template engine,
OWASP/AVID/LMRC tagging, 8 mutation buffs, auto-discovery. Per-category sample counts → `CHANGELOG.md`.

> **Taxonomy: 30 categories / 278 techniques** (`data/taxonomy.yaml`, verified). The old
> roadmap's "19/103+", "29/276", "103+" are all stale — use **30/278** everywhere.
> Sample-corpus figures ("~8,000+ samples / 28 probes", "1.92M unique samples") are **not
> recomputed this pass** — treat as unverified until re-counted.

### Layer 13 — Dataset Pipeline — v1 COMPLETE; v2 + F14 IN PROGRESS
Owns discover → download → quarantine → stage → train → deploy. Registry-driven sync
(`data/datasets.yaml`, **72 sources**, `datasets.lock`), 3-stage promotion gate, 6-dimension
trust scoring, SimHash/MinHash dedup, Confident-Learning audit, hard-negative mining; deployment
gated by `canary_eval.py` + `shadow_evaluate.py`. v1 pipeline detail → `CHANGELOG.md`.
- [ ] NVD-CVE + GHSA scrapers (harvest sources). P0.
- [ ] Bootstrap negatives (#58). P0.
- [ ] F11 — silent-Twitter handling. P1.
- [ ] F12 — `process_data` skip. P1.
- [ ] F14 — promotion-gate v0.1 (scenario gate). P1.
- [ ] v2 redesign task families A1–A6 / B1–B6 / F1–F14 / M1–M14 (~145 granular open tasks —
  full verbatim list in [`docs/TASK_BACKLOG.md`](docs/TASK_BACKLOG.md); includes real bugs
  e.g. F3 unhashable-raw_label TypeError, F9 quarantine file-locking, M8a runner egress
  filtering, plus the garak/PyRIT/AdvBench/HarmBench/GPTFuzz/AutoDAN harvest-source mirrors).

> **Datasets: 72 sources** (`len(datasets.yaml['sources'])`, a dict). Old roadmap cites 23, 49,
> and 52 — all stale. Use **72**.

### Layer 14 — Red-Team Harness & CI — COMPLETE
8 GitHub Actions workflows (3.9–3.12 matrix, coverage gate, CodeQL, Trusted Publishing),
17-target Makefile, probe/judge evaluators, regression dashboard, integration + Hypothesis
property tests, garak/pyrit adapters. Detail → `CHANGELOG.md`.
- Suite size figures (e.g. "4901 passed / 128 xfail") are **not re-run this pass** — stale-risk,
  treat as unverified.

### Layer 15 — Threat-Intel Sync — COMPLETE
Upstream threat-intel ingestion (ATLAS, Garak, AIID, JailbreakBench/HarmBench, OWASP,
SafetyPrompts). Detail → `CHANGELOG.md`.
- Old roadmap said "7 sources" but listed 6 — reconcile the source list when next touched.

### Layer 16 — Multi-Turn Detection — COMPLETE (basic)
Cross-turn injection and crescendo detection.
- [ ] `ConversationSecurityMonitor` (10d) — state, session backends, escalation velocity, topic
  drift, cross-turn assembly, `na0s.Session` API. See §4.
- Old roadmap cites both "50 scenarios" and "30 scenarios" — reconcile when next touched.

### Layer 17 — Document-Format Scanning — ~35%
Office parser suite present (`parsers/office/`: docx/xlsx/pptx/odf/ole + router + base, 8 modules).
`scan_document_visual` is exported; `scan_document()` is NOT yet in the public API.
- [ ] PDF scanner. [ ] CSV-formula scanner. [ ] Code-comment scanner. [ ] RTF scanner.
  [ ] Email scanner. [ ] SVG scanner. (`parsers/` top level currently has only `__init__.py`.)
- [ ] Wire `scan_document()` into the public API.

### Layer 18 — RAG Security / Ingestion — NOT STARTED (0%)
18 planned ingestion/RAG-security modules. Threat-model prose → `docs/`.

### Layer 19 — Agent / MCP Security — ~9%
- ⚠️ **See TRAP A** — `agents/` is already occupied by the deploy-automation system; the 7
  planned L19 MCP-security files (`mcp_tool_detector`, `tool_integrity`, `a2a_validator`,
  `cve_mapping`, `chain_monitor`, `etdi`, …) **do not exist**. `detectors/mcp_tool.py` exists
  (the misfiled MCP poison detector to be moved).
- [ ] Move + extend `MCPToolPoisonDetector`. [ ] Tool-integrity hashing. [ ] Param validation.
  [ ] CVE map. [ ] A2A validator. [ ] Chain monitor. [ ] ETDI. [ ] Judge routing.

### Layer 20 — Taxonomy Automation — ~25%
Automated taxonomy sync/coverage tooling over the **30-category / 278-technique** taxonomy.
- [ ] Sync pipeline. [ ] Coverage report. [ ] Source→technique mappers.
- Adjacent to Phase 8 (`taxonomy/` package) and TRAP B (`rules/` vs taxonomy naming).

### Layer 21 — Telemetry & Feedback Loop — DRAFT (0%)
Opt-in only; canary-gated capture. Hard constraints + ordered P0→P2 plan kept; telemetry
server/dir sketch → `docs/TELEMETRY.md`.

---

## 4. Cross-Layer Planned Work

Deep specs spanning multiple layers (promoted from the dated hardening logs). Recommended
order: **L0 → L2 → L16.**

- **L0 char-level reassembly** (2.5d) — reconstruct fragmented/spaced-out payloads pre-detection.
- **L2 chained-obfuscation decoder** (4d) — handle multi-encoding chains beyond current depth.
- **L16 ConversationSecurityMonitor** (10d) — session state, escalation velocity, topic drift,
  cross-turn assembly, `na0s.Session` API.
- **Open xfail backlog** — educational-context FP, D8 middle-placement, D4 chained-encode,
  C1 crescendo multi-turn. Fold into L1/L2/L16 as fixed.

---

## 5. Coverage Summary

Implementation-mapping (IM0001–IM0017): **7 YES / 7 PARTIAL / 3 NO**. Primary gaps align with
L17 (doc formats), L18 (RAG ingest), and L19 (agent/MCP). Full matrix → `docs/COVERAGE_MATRIX.md`.

---

## 6. Further Reading

Heavy reference content moved out of this roadmap (links only):

- [`docs/TASK_BACKLOG.md`](docs/TASK_BACKLOG.md) — **the full 350-item granular open-task
  backlog** (per-module docstring gaps, hardcoded-constant externalization, test-restoration,
  detector edge-case coverage, the README-overhaul phases, P0–P3 priority items). Extracted
  verbatim during this roadmap's restructure so the fine-grained tasks aren't lost; the roadmap
  body keeps only strategic/in-flight items.

- `docs/ARCHITECTURE.md` — pipeline diagram, A–F track table, per-layer descriptions,
  Implementation Reference (sprint table, reuse patterns, verification plan).
- `docs/TARGET_TREE_v1.md` — full v1.0.0 target tree + per-layer target trees.
- `docs/COVERAGE_MATRIX.md` — IM0001–IM0017 gap table.
- `docs/dataset_pipeline_v2.md`, `docs/ADVERSARIAL_LOOP_GUARDRAILS.md`,
  `docs/F14_SCENARIO_GATE.md`, `docs/HARVEST_SOURCES.md` — L13 v2 design + research corpus.
- `docs/research_sources.md`, `docs/dependencies.md`, `docs/REFERENCES.md`.
- `docs/TELEMETRY.md` — L21 design.
- `CHANGELOG.md` — all per-feature / per-fix history.
- Design-document index (pointer table) — kept in `docs/`.

> Unverifiable / marketing claims removed from the roadmap body: "industry-first" / "first-in-class"
> framing for L1/L2 detectors, "70–90% FP reduction", per-paper ASR figures presented as
> Na0S-defends facts. These are attributable to cited papers but not repo-verifiable as written;
> keep them (clearly sourced) in `docs/` only, not as roadmap fact.

---

## Archive / Session Log

Dated session logs are kept here (demoted, not deleted). Their still-open in-scope items have
already been lifted into §3/§4 above; the remainder is historical record.

- **README Overhaul** (status: Complete; mostly WONTDO) → `docs/archive/readme_overhaul.md`.
- **Research-Driven Improvement Backlog (2026-03-03)** — triaged; open in-scope items
  (NVD/GHSA scrapers, F11/F12) lifted into L13. Remainder → `docs/archive/`.
- **Test-Gap Hardening Pass (2026-03-18 → 03-31)** — triaged; open xfails (educational-context
  FP, D8, D4, C1 multi-turn) lifted into §4. Remainder → `docs/archive/`.
- **Architectural Features (2026-03-31)** — char-reassembly / chained-decoder /
  ConversationSecurityMonitor specs lifted into §4. Dated wrapper archived.
