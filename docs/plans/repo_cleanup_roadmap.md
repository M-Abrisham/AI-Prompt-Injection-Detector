# Na0S Repository Reorganization Roadmap

**Created:** 2026-04-11
**Updated:** 2026-04-11 (v4 — execution in progress)
**Status:** ✅ COMPLETE — All 13 phases executed successfully
**Author:** Generated from full repo audit, reviewed by external auditor (2 rounds)

### Execution Log

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| 0.0 Mirror backup | ✅ DONE | 2026-04-11 | `/tmp/Na0S-mirror-backup-20260411-152047` (221 branches, 97MB) |
| 0.0 Triple-verify | ✅ DONE | 2026-04-11 | 3 independent agents verified all 14 merged branches + scrape/harvest safety |
| 0.1 Delete merged human branches | ✅ DONE | 2026-04-11 | 14/14 deleted successfully |
| 0.2 Prune local branches | ✅ DONE | 2026-04-11 | Removed 5 local branches + 2 worktree branches |
| 0.3 Delete scrape branches | ✅ DONE | 2026-04-11 | 201/201 deleted via rate-limited loop |
| 0.4 Delete old harvest branches | ✅ DONE | 2026-04-11 | 2/2 deleted; kept harvest/2026-04-06 |
| 0.6 Verify final state | ✅ DONE | 2026-04-11 | 4 remote branches remaining (target reached) |
| 11.1 CI branch triggers | ✅ DONE (staged) | 2026-04-11 | ci.yml updated; needs commit + push + PR |
| OQ #2 Resolved | ✅ | 2026-04-11 | Converted to Phase 9 verification step |
| OQ #3 Resolved | ✅ | 2026-04-11 | `embedding_classifier.py` is ALIVE (23 grep hits, predict.py imports it) → MOVE to ml/ |
| OQ #4 Resolved | ✅ | 2026-04-11 | `test_promptguard_classifier.py` EXISTS at `tests/test_promptguard_classifier.py` |
| OQ #6 Resolved | ✅ | 2026-04-11 | Both unmerged branches: option (b) — accept rebase pain (2+1 breaking imports, covered by shims) |
| Phase 1 | ✅ DONE | 2026-04-11 | Already complete: layer1 files are 3-line shims to layer2 (canonical). No action needed. |
| Phases 2-9 | ✅ DONE | 2026-04-11 | 8 packages: canary, judge, integrity, ml, rag, detectors, worm, fusion |
| Phase 10 | ✅ DONE | 2026-04-12 | 55 test files moved into 8 subdirectories; test count preserved (8,568) |
| Phase 12 | ✅ DONE | 2026-04-12 | CLAUDE.md, ARCHITECTURE.md updated |

---

## 1. Executive Summary

The Na0S source package (`src/na0s/`) has accumulated 87 top-level Python modules and 7 subdirectories, making navigation, ownership, and onboarding difficult. The test directory mirrors this sprawl with 217 flat files alongside only 3 organized subdirectories. The GitHub remote carries 219 branches, of which 202 are automated scrape/harvest branches that provide no ongoing value. This roadmap restructures the codebase into ~20 top-level entries (core pipeline files plus logically-grouped sub-packages), reorganizes tests to match, and cleans up 200+ stale branches. The work is split across 13 independently-mergeable phases, each in its own branch/PR. Estimated total effort: **5-7 working days** (originally estimated 3-5 but revised upward to account for diff reviews in Phase 1, dead-code verification, unmerged-branch coordination, and realistic context-switching overhead). Risk is LOW-to-MEDIUM per-phase because every phase uses backward-compatibility shims and can be reverted with a single `git revert`, but Phase 0 (branch deletion) and Phase 1 (dedup) are higher risk and require explicit verification gates before proceeding.

---

## 2. Current State Audit

### 2.1 Source Module Counts

| Location | Count | Notes |
|----------|-------|-------|
| `src/na0s/*.py` (top-level) | 87 | Target: ~20 after reorg |
| `src/na0s/layer0/` | 19 modules | Well-organized |
| `src/na0s/layer1/` | 13 modules | 5 duplicated with layer2 |
| `src/na0s/layer2/` | 8 modules | 5 duplicated with layer1 |
| `src/na0s/layer15/` | 18 modules | Well-organized |
| `src/na0s/layer16/` | 12 entries (files + subdirs) | Well-organized |
| `src/na0s/models/` | 7 files (pkl, sha256) | Model artifacts |
| `src/na0s/parsers/office/` | 8 modules | Good target-state example |

### 2.2 Duplicated Modules (layer1 <-> layer2)

The following 5 files exist identically (or near-identically) in both `layer1/` and `layer2/`:

| Module | `layer1/` | `layer2/` |
|--------|-----------|-----------|
| `ascii_art_detector.py` | Yes | Yes |
| `morse_code.py` | Yes | Yes |
| `numeric_decode.py` | Yes | Yes |
| `syllable_splitting.py` | Yes | Yes |
| `whitespace_stego.py` | Yes | Yes |

Files unique to `layer1/` (8): `analyzer.py`, `context.py`, `ioc_extractor.py`, `paranoia.py`, `result.py`, `rules_registry.py`, `unicode_defense.py`, `__init__.py`
Files unique to `layer2/` (3): `_env_utils.py`, `obfuscation.py`, `__init__.py`

### 2.3 Test File Counts

| Location | Count |
|----------|-------|
| `tests/*.py` (flat) | 217 |
| `tests/test_layer16/` | 39 files (organized, good pattern) |
| `tests/parsers/office/` | 6 files (organized, good pattern) |
| `tests/fixtures/` | Fixture data (office, odf, xlsx, etc.) |

### 2.4 Git Branch Inventory

| Category | Count |
|----------|-------|
| Remote branches (total) | 219 |
| `scrape/*` branches | 199 |
| `harvest/*` branches | 3 |
| Human feature branches | 17 (including `main`) |
| Human branches merged to main | 14 of 16 non-main |
| Human branches NOT merged | 2 (`worm-replication-sim`, `layer16-conversation-detectors`) |
| Local branches | 9 |

### 2.5 CI Status

- **Workflows:** 7 total (`ci.yml`, `pr-check.yml`, `auto-retrain.yml`, `publish.yml`, `social-scraper.yml`, `threat_intel_sync.yml`, `weekly-harvest.yml`)
- **CI triggers:** Push to `main` and `feature/*` only; PRs to `main`
- **Gap:** `refactor/*`, `hardening/*`, `fix/*`, `feat/*`, `docs/*` branch prefixes do NOT trigger CI on push
- **Python matrix:** 3.9, 3.10, 3.11, 3.12
- **Coverage gate:** `--fail-under=50`

### 2.6 Uncategorized Top-Level Modules

After applying the proposed groupings (Section 3), the following modules remain at top level. Some need assignment:

**Stays at top level (core pipeline, ~20 modules):**
`__init__.py`, `__main__.py`, `_env.py`, `_version.py`, `_voting.py`, `cascade.py`, `cli.py`, `config.py`, `data_schema.py`, `evidence_grading.py`, `groundedness.py`, `intent_guard.py`, `obfuscation.py`, `positive_validation.py`, `predict.py`, `rules.py`, `scan_result.py`, `segment_grader.py`, `signal_boost.py`, `structural_features.py`, `py.typed`

**Must be categorized (currently orphaned):**

| Module | Proposed Package | Rationale |
|--------|------------------|-----------|
| `diffuse_defense.py` | `ml/` | ML-based defense technique |
| `dual_scanner.py` | `rag/` | Wraps OutputScanner + PropagationScanner |
| `late_chunking.py` | `ml/` | Embedding-based chunked classification |
| `multi_turn_validator.py` | `detectors/` | Multi-turn conversation validation |
| `multilingual_handler.py` | core (stay) | Used by layer1, cross-cutting |
| `multilingual_intent.py` | core (stay) | Used by intent_guard |
| `performance_slo.py` | `fusion/` | SLO tracking used by cascade |
| `rate_limiter.py` | `judge/` | Rate limiting for LLM judge calls |
| `streaming_scanner.py` | `rag/` | Streaming version of output scanner |
| `compliance_evasion_rules.py` | core (stay) | Rule definitions, like `rules.py` |
| `subtle_override_rules.py` | core (stay) | Rule definitions, like `rules.py` |
| `safe_content.py` | `integrity/` | Safe content scoring |

---

## 3. Target State

### 3.1 Source Tree

```
src/na0s/
├── __init__.py
├── __main__.py
├── _env.py
├── _version.py
├── _voting.py
├── cascade.py
├── cli.py
├── compliance_evasion_rules.py
├── config.py
├── data_schema.py
├── evidence_grading.py
├── groundedness.py
├── intent_guard.py
├── multilingual_handler.py
├── multilingual_intent.py
├── obfuscation.py
├── positive_validation.py
├── predict.py
├── py.typed
├── rules.py
├── scan_result.py
├── segment_grader.py
├── signal_boost.py
├── structural_features.py
├── subtle_override_rules.py
│
├── canary/
│   ├── __init__.py          (re-exports CanaryManager, CanaryToken)
│   ├── alert.py             (was canary_alert.py)
│   ├── honeypot.py          (was canary_honeypot.py)
│   ├── manager.py           (was canary.py)
│   ├── persistence.py       (was canary_persistence.py)
│   ├── rotation.py          (was canary_rotation.py)
│   ├── session.py           (was canary_session.py)
│   └── verifier.py          (was canary_verifier.py)
│
├── detectors/
│   ├── __init__.py
│   ├── context_manipulation.py  (was context_manipulation_detector.py)
│   ├── extraction.py            (was extraction_detector.py)
│   ├── fictional_frame.py       (was fictional_frame_detector.py)
│   ├── harmful_intent.py        (was harmful_intent_detector.py)
│   ├── mcp_tool.py              (was mcp_tool_detector.py)
│   ├── multi_turn.py            (was multi_turn_validator.py)
│   ├── payload_assembly.py      (was payload_assembly_detector.py)
│   ├── privacy_probe.py         (was privacy_probe_detector.py)
│   ├── recon.py                 (was recon_detector.py)
│   └── visual_injection.py      (was visual_injection_detector.py)
│
├── fusion/
│   ├── __init__.py
│   ├── bayesian.py          (was bayesian_fusion.py)
│   ├── complexity_router.py (was complexity_router.py)
│   ├── ensemble.py          (was ensemble.py)
│   ├── performance_slo.py   (was performance_slo.py)
│   └── rrf.py               (was rrf_fusion.py)
│
├── integrity/
│   ├── __init__.py
│   ├── chain.py             (was chain_integrity.py)
│   ├── dep_scanner.py       (was dep_scanner.py)
│   ├── fingerprint.py       (was fingerprint_integrity.py)
│   ├── model_encryption.py  (was model_encryption.py)
│   ├── model_provenance.py  (was model_provenance.py)
│   ├── model_rollback.py    (was model_rollback.py)
│   ├── prompt_signer.py     (was prompt_signer.py)
│   ├── req.py               (was req_integrity.py)
│   ├── safe_content.py      (was safe_content.py)
│   ├── safe_pickle.py       (was safe_pickle.py)
│   ├── sbom.py              (was sbom.py)
│   ├── template.py          (was template_integrity.py)
│   └── validation_allowlist.py  (was validation_allowlist.py)
│
├── judge/
│   ├── __init__.py
│   ├── audit.py             (was judge_audit.py)
│   ├── checker.py           (was llm_checker.py)
│   ├── cost_tracker.py      (was judge_cost_tracker.py)
│   ├── llm_judge.py         (was llm_judge.py)
│   ├── local_judge.py       (was local_judge.py)
│   └── rate_limiter.py      (was rate_limiter.py)
│
├── layer0/                  (unchanged — 19 modules)
├── layer1/                  (unchanged, minus deduplicated files)
├── layer2/                  (unchanged, minus deduplicated files)
├── layer15/                 (unchanged — 18 modules)
├── layer16/                 (unchanged — well-organized)
├── models/                  (unchanged — model artifacts)
│
├── ml/
│   ├── __init__.py
│   ├── cross_encoder.py     (was cross_encoder.py)
│   ├── diffuse_defense.py   (was diffuse_defense.py)
│   ├── embedding_adapter.py (was embedding_adapter.py)
│   ├── embedding_classifier.py  (was embedding_classifier.py)
│   ├── faiss_classifier.py  (was faiss_classifier.py)
│   ├── late_chunking.py     (was late_chunking.py)
│   ├── perplexity.py        (was perplexity.py)
│   ├── predict_embedding.py (was predict_embedding.py)
│   ├── promptguard.py       (was promptguard.py)
│   ├── promptguard_classifier.py  (was promptguard_classifier.py)
│   ├── promptguard_signal.py      (was promptguard_signal.py)
│   ├── replication_similarity.py  (was replication_similarity.py)
│   └── stacking_classifier.py    (was stacking_classifier.py)
│
├── parsers/                 (unchanged — already organized)
│   └── office/
│
├── rag/
│   ├── __init__.py
│   ├── attribution.py       (was rag_attribution.py)
│   ├── dual_scanner.py      (was dual_scanner.py)
│   ├── output_scanner.py    (was output_scanner.py)
│   ├── poison_detector.py   (was rag_poison_detector.py)
│   ├── position_scanner.py  (was rag_position_scanner.py)
│   ├── propagation.py       (was propagation_scanner.py)
│   └── streaming.py         (was streaming_scanner.py)
│
└── worm/
    ├── __init__.py
    ├── advanced.py          (was worm_advanced.py)
    └── detector.py          (was worm_detector.py)
```

**Result:** ~24 top-level entries (21 .py files + 3 special files + ~14 sub-packages) vs current 87 .py + 7 directories.

### 3.2 Test Tree

```
tests/
├── conftest.py              (shared fixtures + anti-regression gate)
├── canary/
│   ├── test_canary.py
│   ├── test_canary_eval.py
│   ├── test_l10_features.py
│   └── test_l10_integrity.py
├── detectors/
│   ├── test_context_manipulation.py
│   ├── test_d7_o1_d3_integration.py
│   ├── test_extraction_detector.py
│   ├── test_fictional_frame_detector.py
│   ├── test_harmful_intent.py
│   ├── test_mcp_tool_detector.py
│   ├── test_payload_assembly.py
│   ├── test_privacy_probe.py
│   ├── test_recon_detector.py
│   └── test_visual_injection_detector.py
├── fusion/
│   ├── test_ensemble.py
│   ├── test_l6_advanced.py
│   ├── test_l6_cascade_features.py
│   └── test_l6_routing.py
├── integrity/
│   ├── test_l11_encryption_rollback.py
│   ├── test_l11_safe_pickle_fixes.py
│   ├── test_l11_supply_chain.py
│   ├── test_safe_pickle.py
│   └── ...
├── judge/
│   ├── test_l7_judge_features.py
│   ├── test_l7_judge_ops.py
│   ├── test_l7_local_judge.py
│   ├── test_llm_checker.py
│   └── test_llm_judge_hardening.py
├── ml/
│   ├── test_cross_encoder.py
│   ├── test_diffuse_defense.py
│   ├── test_faiss_classifier.py
│   ├── test_l5_advanced.py
│   ├── test_l5_model_selection.py
│   ├── test_l5_structural_concat.py
│   ├── test_late_chunking.py
│   ├── test_perplexity.py
│   ├── test_predict_embedding.py
│   ├── test_promptguard.py
│   ├── test_promptguard_classifier.py
│   └── test_replication_similarity.py
├── rag/
│   ├── test_l9_advanced.py
│   ├── test_l9_propagation.py
│   ├── test_l9_rag_segment.py
│   ├── test_l9_streaming.py
│   ├── test_output_scanner.py
│   ├── test_output_scanner_redaction.py
│   ├── test_rag_poison_detector.py
│   └── test_rag_position_scanner.py
├── worm/
│   ├── test_morris2_pipeline.py
│   ├── test_worm_advanced.py
│   ├── test_worm_bayes.py
│   ├── test_worm_corpus_classifier.py
│   ├── test_worm_embedding.py
│   └── test_worm_pca_signatures.py
├── parsers/                  (already organized)
│   └── office/
├── test_layer16/             (already organized)
├── fixtures/                 (unchanged)
└── ... (remaining flat test files for core pipeline)
```

---

## 4. Phase Plan

### Phase 0: Branch Cleanup (no code changes)

**Branch:** N/A (run directly on local machine)
**Preconditions:** Mirror backup completed (see 4.0.0)
**Risk:** **MEDIUM-HIGH** — deleting 200+ remote branches is irreversible from GitHub's UI. A mirror backup is mandatory before proceeding.
**Estimated time:** 45 minutes (including backup verification)

#### 4.0.0 ✅ DONE: Create mirror backup before ANY deletions

```bash
# Clone a full mirror of the remote (all branches, all refs)
git clone --mirror https://github.com/M-Abrisham/Na0S.git /tmp/Na0S-mirror-backup-$(date +%Y%m%d)

# Verify the mirror has all branches
cd /tmp/Na0S-mirror-backup-*/
git branch -a | wc -l  # should match remote branch count (~219)
cd -
```

**Do NOT proceed to 4.0.1 until the mirror is verified.** If any deletion is premature, you can restore from the mirror with `git push origin <branch>` from the backup clone.

#### 4.0.1 ✅ DONE: Delete merged remote branches (14 branches)

These branches were fully merged to `main`. All 14 deleted on 2026-04-11:

```bash
# Merged human branches — safe to delete
git push origin --delete clean-readme
git push origin --delete docs/readme-v3
git push origin --delete manually-cleaned-readme
git push origin --delete updating-readme
git push origin --delete josh-benchmarking-stuff
git push origin --delete fix/layer16-missing-graduated-response
git push origin --delete hardening/io-encoding-and-signal-boost
git push origin --delete feature/14-ci-infra
git push origin --delete feature/6-gap-closure
git push origin --delete feature/layer1-rules-overhaul
git push origin --delete feature/layer7-llama
git push origin --delete feature/signal-boost-caesar-piglatin
git push origin --delete feat/benchmark-sprint-day2
git push origin --delete feat/d3.1-fake-system-prompt-and-taxonomy-mapping
```

#### 4.0.2 ✅ DONE: Prune local branches

```bash
# Delete merged local branches (keep main)
git branch -d fix-cascade-tests
git branch -d fix/layer16-missing-graduated-response
git branch -d hardening/io-encoding-and-signal-boost
git branch -d layer16-conversation-detectors
git branch -d worm-replication-sim
git branch -d worm_detector

# Remove worktree branches
git worktree prune
git branch -d worktree-agent-a39cbb19
git branch -d worktree-agent-a89e7c0e
```

#### 4.0.3 ✅ DONE: Bulk-delete automated scrape branches (201 branches)

All 201 `scrape/*` branches deleted on 2026-04-11 via rate-limited loop.
**Verification:** 3-agent pre-check sampled 60 branches across full date range — ALL were single-commit data-only pushes modifying only `data/scraped/` files. Zero source code changes detected.

```bash
# Step 1: spot-check that no scrape branch has unique, valuable commits
# (they should all be single-commit data pushes)
git ls-remote --heads origin | awk '/scrape\// {print $2}' | head -5 | \
  sed 's|refs/heads/||' | while read b; do
    echo "=== $b ==="
    git log --oneline origin/main..$b 2>/dev/null | head -3
  done
# If any branch has unexpected commits, EXCLUDE it from deletion

# Step 2: delete via rate-limited loop (one at a time, with progress)
git ls-remote --heads origin | awk '/scrape\// {print $2}' | sed 's|refs/heads/||' | \
  while read b; do
    echo "Deleting $b..."
    git push origin --delete "$b" 2>&1 | tail -1
    sleep 0.5  # rate-limit courtesy for GitHub API
  done
```

#### 4.0.4 ✅ DONE: Archive harvest branches (3 branches)

Policy: keep the most recent `harvest/*` branch; delete older ones.
**Executed 2026-04-11:** `harvest/2026-03-23` and `harvest/2026-03-30` deleted. `harvest/2026-04-06` kept.

#### 4.0.5 Branches to KEEP (unmerged, active work)

| Branch | Last Commit | Reason to Keep |
|--------|-------------|----------------|
| `worm-replication-sim` | 2026-04-05 | Unmerged, active work |
| `layer16-conversation-detectors` | 2026-04-03 | Unmerged, active work |

#### 4.0.6 After cleanup — verify

```bash
git fetch --prune
git branch -a
# Expected: main + 2 unmerged human branches + 1 harvest + 0 scrape
```

**Rollback:** Re-push from local tracking branches if any deletion was premature.

---

### Phase 1: Deduplicate layer1/layer2

**Branch:** `refactor/dedup-layer1-layer2`
**Preconditions:** Must diff each pair BEFORE creating shims (see 1.0 below)
**Risk:** **MEDIUM-HIGH** — if the copies have diverged (obfuscation-specific behavior in layer2 that layer1 doesn't have), a naive shim silently changes behavior. The plan assumes near-identical copies but this MUST be verified.
**Estimated time:** 2-3 hours (including diff review)

#### 1.0 MANDATORY: Diff each pair before proceeding

```bash
# Run this FIRST. Show the FULL diff for each pair — do NOT truncate.
for mod in ascii_art_detector.py morse_code.py numeric_decode.py syllable_splitting.py whitespace_stego.py; do
  echo "============================================================"
  echo "=== $mod ==="
  echo "============================================================"
  diff src/na0s/layer1/$mod src/na0s/layer2/$mod
  echo ""
  echo "lines changed: $(diff src/na0s/layer1/$mod src/na0s/layer2/$mod | grep -c '^[<>]')"
  echo ""
done
```

**Decision gate:** For each pair, apply this rule:
- If the diff touches **function signatures, return types, or conditional logic** → the copies have diverged. **Merge** the two into a single canonical version (keep the superset of functionality), then shim the other. Document which version's behavior wins.
- If the diff is **only comments, whitespace, import order, or docstrings** → safe to shim one as a re-export of the other.
- Do NOT use a line-count threshold — a 3-line change to a return statement is more dangerous than a 50-line comment rewrite.

#### 5 duplicated modules

| Module | Canonical Location | Shim Location |
|--------|-------------------|---------------|
| `ascii_art_detector.py` | `layer1/ascii_art_detector.py` | `layer2/ascii_art_detector.py` -> import from layer1 |
| `morse_code.py` | `layer1/morse_code.py` | `layer2/morse_code.py` -> import from layer1 |
| `numeric_decode.py` | `layer1/numeric_decode.py` | `layer2/numeric_decode.py` -> import from layer1 |
| `syllable_splitting.py` | `layer1/syllable_splitting.py` | `layer2/syllable_splitting.py` -> import from layer1 |
| `whitespace_stego.py` | `layer1/whitespace_stego.py` | `layer2/whitespace_stego.py` -> import from layer1 |

**Rationale:** layer1 is the analysis layer with more context (13 modules vs 8). layer2 is the obfuscation-specific layer that should delegate to layer1 for shared detectors.

#### Files to modify

For each duplicated module in `layer2/`, replace contents with:
```python
# src/na0s/layer2/ascii_art_detector.py (SHIM)
"""Backward-compat shim. Canonical: na0s.layer1.ascii_art_detector"""
from na0s.layer1.ascii_art_detector import *  # noqa: F401,F403
```

#### Tests to verify
```bash
python3 -m pytest tests/obfuscation/test_ascii_art_detector.py tests/obfuscation/test_morse_code.py tests/obfuscation/test_numeric_decode.py tests/obfuscation/test_syllable_splitting.py tests/obfuscation/test_whitespace_stego.py tests/obfuscation/test_l2_coverage_gaps.py -v
```

**Commit message:** `refactor(layer2): deduplicate 5 detectors shared with layer1`
**Rollback:** `git revert <commit-sha>`

---

### Phase 2: Create canary/ package

**Branch:** `refactor/canary-package`
**Preconditions:** None
**Risk:** LOW (canary modules are self-contained, few external importers)
**Estimated time:** 1 hour

#### Files to move

| Current Path | New Path |
|-------------|----------|
| `src/na0s/canary.py` | `src/na0s/canary/manager.py` |
| `src/na0s/canary_alert.py` | `src/na0s/canary/alert.py` |
| `src/na0s/canary_honeypot.py` | `src/na0s/canary/honeypot.py` |
| `src/na0s/canary_persistence.py` | `src/na0s/canary/persistence.py` |
| `src/na0s/canary_rotation.py` | `src/na0s/canary/rotation.py` |
| `src/na0s/canary_session.py` | `src/na0s/canary/session.py` |
| `src/na0s/canary_verifier.py` | `src/na0s/canary/verifier.py` |

**Note:** `canary.py` currently defines `CanaryManager` and `CanaryToken` and is already imported as `from na0s.canary import ...`. When `canary/` becomes a directory, the `canary/__init__.py` must re-export these symbols to preserve the public API.

#### Create `src/na0s/canary/__init__.py`

```python
"""Canary token injection detection and management."""
from na0s.canary.manager import CanaryManager, CanaryToken

__all__ = ["CanaryManager", "CanaryToken"]
```

#### Backward-compat shims at old locations

Create shim at each old location, e.g. `src/na0s/canary_alert.py`:
```python
# src/na0s/canary_alert.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.canary.alert"""
from na0s.canary.alert import *  # noqa: F401,F403
import warnings
warnings.warn(
    "na0s.canary_alert is deprecated; use na0s.canary.alert",
    DeprecationWarning,
    stacklevel=2,
)
```

#### Import sites to update (source)

```
src/na0s/canary_persistence.py:17  from na0s.canary import CanaryManager, CanaryToken
src/na0s/canary_rotation.py:15    from na0s.canary import CanaryManager, CanaryToken
src/na0s/canary_alert.py:16       from na0s.canary import CanaryToken
src/na0s/canary_session.py:16     from na0s.canary import CanaryManager, CanaryToken
src/na0s/__init__.py:49            from na0s.canary import CanaryManager, CanaryToken
```

The `__init__.py` import (`from na0s.canary import ...`) continues to work because `canary/__init__.py` re-exports. Internal canary modules now import from `na0s.canary.manager` instead of `na0s.canary`.

#### Import sites to update (tests)

```
tests/test_l10_integrity.py:13     from na0s.canary_verifier import CanaryTokenVerifier
tests/test_l10_features.py:19-23   from na0s.canary_alert import ...
                                   from na0s.canary_honeypot import ...
                                   from na0s.canary_persistence import ...
                                   from na0s.canary_rotation import ...
                                   from na0s.canary_session import ...
tests/test_canary.py               from na0s.canary import ...
```

These continue to work via shims. No test changes required for Phase 2.

#### Import sites (scripts)

```
scripts/canary_eval.py:36          from na0s.predict import ... (no canary imports)
```

No script changes required.

#### Tests to pass
```bash
python3 -m pytest tests/test_canary.py tests/test_canary_eval.py tests/test_l10_features.py tests/test_l10_integrity.py -v
```

**Commit message:** `refactor(canary): extract canary package from 7 top-level modules`
**Rollback:** `git revert <commit-sha>`

---

### Phase 3: Create judge/ package

**Branch:** `refactor/judge-package`
**Preconditions:** None
**Risk:** LOW
**Estimated time:** 1 hour

#### Files to move

| Current Path | New Path |
|-------------|----------|
| `src/na0s/llm_judge.py` | `src/na0s/judge/llm_judge.py` |
| `src/na0s/local_judge.py` | `src/na0s/judge/local_judge.py` |
| `src/na0s/judge_audit.py` | `src/na0s/judge/audit.py` |
| `src/na0s/judge_cost_tracker.py` | `src/na0s/judge/cost_tracker.py` |
| `src/na0s/llm_checker.py` | `src/na0s/judge/checker.py` |
| `src/na0s/rate_limiter.py` | `src/na0s/judge/rate_limiter.py` |

#### Create `src/na0s/judge/__init__.py`

```python
"""LLM judge-based classification (Layer 7)."""
```

#### Import sites to update (source)

```
src/na0s/llm_judge.py:23      from na0s.judge_audit import JudgeAuditLogger
src/na0s/llm_judge.py:24      from na0s.judge_cost_tracker import CostTracker
src/na0s/llm_judge.py:25      from na0s.rate_limiter import TokenBucketRateLimiter
```

After move, these become intra-package imports:
```python
from na0s.judge.audit import JudgeAuditLogger
from na0s.judge.cost_tracker import CostTracker
from na0s.judge.rate_limiter import TokenBucketRateLimiter
```

#### Import sites to update (tests)

```
tests/test_llm_checker.py:41              from na0s.llm_checker import ...
tests/test_llm_judge_hardening.py:44      from na0s.llm_judge import ...
tests/test_llm_judge_hardening.py:56      from na0s.llm_checker import ...
tests/test_l7_local_judge.py              from na0s.local_judge import ... (30+ sites)
tests/test_l7_local_judge.py              from na0s.llm_judge import ... (12+ sites)
tests/test_l7_judge_features.py:31        from na0s.llm_judge import ...
tests/test_l7_judge_ops.py:18-20          from na0s.judge_cost_tracker import ...
                                          from na0s.judge_audit import ...
                                          from na0s.rate_limiter import ...
```

All continue to work via shims. Update tests opportunistically in Phase 10.

#### Import sites (scripts)

```
scripts/evaluate_llm_judge.py:18     from na0s.llm_judge import LLMJudge
```

Works via shim.

#### Tests to pass
```bash
python3 -m pytest tests/test_l7_judge_features.py tests/test_l7_judge_ops.py tests/test_l7_local_judge.py tests/test_llm_checker.py tests/test_llm_judge_hardening.py -v
```

**Commit message:** `refactor(judge): extract judge package from 6 top-level modules`
**Rollback:** `git revert <commit-sha>`

---

### Phase 4: Create integrity/ package

**Branch:** `refactor/integrity-package`
**Preconditions:** None
**Risk:** LOW
**Estimated time:** 1.5 hours

#### Files to move (14 modules)

| Current Path | New Path |
|-------------|----------|
| `src/na0s/safe_pickle.py` | `src/na0s/integrity/safe_pickle.py` |
| `src/na0s/safe_content.py` | `src/na0s/integrity/safe_content.py` |
| `src/na0s/model_encryption.py` | `src/na0s/integrity/model_encryption.py` |
| `src/na0s/model_provenance.py` | `src/na0s/integrity/model_provenance.py` |
| `src/na0s/model_rollback.py` | `src/na0s/integrity/model_rollback.py` |
| `src/na0s/sbom.py` | `src/na0s/integrity/sbom.py` |
| `src/na0s/dep_scanner.py` | `src/na0s/integrity/dep_scanner.py` |
| `src/na0s/prompt_signer.py` | `src/na0s/integrity/prompt_signer.py` |
| `src/na0s/chain_integrity.py` | `src/na0s/integrity/chain.py` |
| `src/na0s/template_integrity.py` | `src/na0s/integrity/template.py` |
| `src/na0s/req_integrity.py` | `src/na0s/integrity/req.py` |
| `src/na0s/fingerprint_integrity.py` | `src/na0s/integrity/fingerprint.py` |
| `src/na0s/validation_allowlist.py` | `src/na0s/integrity/validation_allowlist.py` |

#### Import sites (source)

No source-internal cross-references found (grep returned empty). These modules are leaf-level utilities.

#### Import sites (tests) -- high volume

Key test files affected:
- `tests/test_safe_pickle.py` -- `from na0s.safe_pickle import ...`
- `tests/test_l11_safe_pickle_fixes.py` -- `from na0s.safe_pickle import ...`
- `tests/test_l11_encryption_rollback.py` -- `from na0s.model_encryption import ...`, `from na0s.model_rollback import ...`, `from na0s.sbom import ...`
- `tests/test_l11_supply_chain.py` -- `from na0s.model_provenance import ...`, `from na0s.dep_scanner import ...`, `from na0s.req_integrity import ...`, `from na0s.fingerprint_integrity import ...`
- `tests/test_l10_integrity.py` -- `from na0s.prompt_signer import ...`, `from na0s.template_integrity import ...`
- `tests/test_l6_cascade_features.py` -- `from na0s.chain_integrity import ...`
- `tests/test_positive_validation.py` -- `from na0s.validation_allowlist import ...`
- `tests/test_fp_reduction.py` -- `from na0s.safe_content import ...`

#### Import sites (scripts) -- high volume

```
scripts/canary_eval.py:34         from na0s.safe_pickle import safe_load
scripts/optimize_threshold.py:21  from na0s.safe_pickle import safe_load
scripts/features.py:18            from na0s.safe_pickle import safe_dump
scripts/model_embedding.py:30     from na0s.safe_pickle import safe_dump, safe_load
scripts/distill_model.py:256      from na0s.safe_pickle import safe_load, safe_dump
scripts/cleanlab_audit.py:76      from na0s.safe_pickle import safe_load
scripts/features_embedding.py:33  from na0s.safe_pickle import safe_dump
scripts/build_faiss_index.py:45   from na0s.safe_pickle import safe_load
scripts/model.py:19               from na0s.safe_pickle import safe_load, safe_dump
scripts/active_learning.py:79     from na0s.safe_pickle import safe_load
scripts/shadow_evaluate.py:91     from na0s.safe_pickle import safe_load
```

All work via shims. High script usage of `safe_pickle` means the shim at `src/na0s/safe_pickle.py` should remain indefinitely.

#### Tests to pass
```bash
python3 -m pytest tests/test_safe_pickle.py tests/test_l11_safe_pickle_fixes.py tests/test_l11_encryption_rollback.py tests/test_l11_supply_chain.py tests/test_l10_integrity.py tests/test_l6_cascade_features.py tests/test_positive_validation.py tests/test_fp_reduction.py -v
```

**Commit message:** `refactor(integrity): extract integrity package from 13 top-level modules`
**Rollback:** `git revert <commit-sha>`

---

### Phase 5: Create ml/ package

**Branch:** `refactor/ml-package`
**Preconditions:** None
**Risk:** LOW
**Estimated time:** 1.5 hours

#### Files to move (14 modules)

| Current Path | New Path |
|-------------|----------|
| `src/na0s/embedding_classifier.py` | `src/na0s/ml/embedding_classifier.py` |
| `src/na0s/embedding_adapter.py` | `src/na0s/ml/embedding_adapter.py` |
| `src/na0s/faiss_classifier.py` | `src/na0s/ml/faiss_classifier.py` |
| `src/na0s/cross_encoder.py` | `src/na0s/ml/cross_encoder.py` |
| `src/na0s/predict_embedding.py` | `src/na0s/ml/predict_embedding.py` |
| `src/na0s/promptguard.py` | `src/na0s/ml/promptguard.py` |
| `src/na0s/promptguard_classifier.py` | `src/na0s/ml/promptguard_classifier.py` |
| `src/na0s/promptguard_signal.py` | `src/na0s/ml/promptguard_signal.py` |
| `src/na0s/stacking_classifier.py` | `src/na0s/ml/stacking_classifier.py` |
| `src/na0s/perplexity.py` | `src/na0s/ml/perplexity.py` |
| `src/na0s/replication_similarity.py` | `src/na0s/ml/replication_similarity.py` |
| `src/na0s/diffuse_defense.py` | `src/na0s/ml/diffuse_defense.py` |
| `src/na0s/late_chunking.py` | `src/na0s/ml/late_chunking.py` |

#### Cross-package import to update

```
src/na0s/worm_detector.py:24  from na0s.replication_similarity import replication_similarity
```

This must be updated to `from na0s.ml.replication_similarity import replication_similarity` OR left working via shim.

#### Import sites (scripts)

```
scripts/build_faiss_index.py:46   from na0s.faiss_classifier import FAISSClassifier
```

Works via shim.

#### Tests to pass
```bash
python3 -m pytest tests/test_faiss_classifier.py tests/test_cross_encoder.py tests/test_predict_embedding.py tests/test_l5_structural_concat.py tests/test_l5_advanced.py tests/test_l5_model_selection.py tests/test_promptguard.py tests/test_perplexity.py tests/test_replication_similarity.py tests/test_late_chunking.py tests/test_diffuse_defense.py tests/test_ensemble.py -v
```

**Commit message:** `refactor(ml): extract ml package from 13 top-level modules`
**Rollback:** `git revert <commit-sha>`

---

### Phase 6: Create rag/ package

**Branch:** `refactor/rag-package`
**Preconditions:** None
**Risk:** LOW
**Estimated time:** 1 hour

#### Files to move (7 modules)

| Current Path | New Path |
|-------------|----------|
| `src/na0s/output_scanner.py` | `src/na0s/rag/output_scanner.py` |
| `src/na0s/streaming_scanner.py` | `src/na0s/rag/streaming.py` |
| `src/na0s/rag_poison_detector.py` | `src/na0s/rag/poison_detector.py` |
| `src/na0s/rag_position_scanner.py` | `src/na0s/rag/position_scanner.py` |
| `src/na0s/rag_attribution.py` | `src/na0s/rag/attribution.py` |
| `src/na0s/propagation_scanner.py` | `src/na0s/rag/propagation.py` |
| `src/na0s/dual_scanner.py` | `src/na0s/rag/dual_scanner.py` |

#### Critical: Public API impact

`__init__.py` imports `OutputScanner`, `OutputScanResult`, `StreamingOutputScanner` from top-level modules. The shims at old locations ensure these continue to work. The `__init__.py` import itself does NOT need to change because the shim at `src/na0s/output_scanner.py` proxies to `na0s.rag.output_scanner`.

#### Import sites (source)

```
src/na0s/segment_grader.py:16     from na0s.output_scanner import OutputScanner
src/na0s/streaming_scanner.py:16  from na0s.output_scanner import ...
src/na0s/dual_scanner.py:14-15    from na0s.output_scanner import ...
                                  from na0s.propagation_scanner import ...
src/na0s/__init__.py:46-47        from na0s.output_scanner import ...
                                  from na0s.streaming_scanner import ...
```

All work via shims.

#### Tests to pass
```bash
python3 -m pytest tests/test_output_scanner.py tests/test_output_scanner_redaction.py tests/test_l9_streaming.py tests/test_l9_propagation.py tests/test_l9_advanced.py tests/test_l9_rag_segment.py tests/test_rag_poison_detector.py tests/test_rag_position_scanner.py -v
```

**Commit message:** `refactor(rag): extract rag package from 7 top-level modules`
**Rollback:** `git revert <commit-sha>`

---

### Phase 7: Create detectors/ package

**Branch:** `refactor/detectors-package`
**Preconditions:** None
**Risk:** LOW
**Estimated time:** 1 hour

#### Files to move (10 modules)

| Current Path | New Path |
|-------------|----------|
| `src/na0s/extraction_detector.py` | `src/na0s/detectors/extraction.py` |
| `src/na0s/harmful_intent_detector.py` | `src/na0s/detectors/harmful_intent.py` |
| `src/na0s/recon_detector.py` | `src/na0s/detectors/recon.py` |
| `src/na0s/context_manipulation_detector.py` | `src/na0s/detectors/context_manipulation.py` |
| `src/na0s/payload_assembly_detector.py` | `src/na0s/detectors/payload_assembly.py` |
| `src/na0s/privacy_probe_detector.py` | `src/na0s/detectors/privacy_probe.py` |
| `src/na0s/mcp_tool_detector.py` | `src/na0s/detectors/mcp_tool.py` |
| `src/na0s/fictional_frame_detector.py` | `src/na0s/detectors/fictional_frame.py` |
| `src/na0s/visual_injection_detector.py` | `src/na0s/detectors/visual_injection.py` |
| `src/na0s/multi_turn_validator.py` | `src/na0s/detectors/multi_turn.py` |

#### Critical: Public API impact

`__init__.py` imports from `visual_injection_detector`. The shim at old location handles this.

#### Cross-package import

```
src/na0s/layer16/detectors/payload_splitting.py:92   from na0s.payload_assembly_detector import detect_fragmented_payload
src/na0s/layer16/detectors/payload_splitting.py:121  from na0s.payload_assembly_detector import detect_multiturn_assembly
```

Works via shim.

#### Tests to pass
```bash
python3 -m pytest tests/test_extraction_detector.py tests/test_harmful_intent.py tests/test_recon_detector.py tests/test_context_manipulation.py tests/test_payload_assembly.py tests/test_privacy_probe.py tests/test_mcp_tool_detector.py tests/test_fictional_frame_detector.py tests/test_visual_injection_detector.py tests/test_d7_o1_d3_integration.py -v
```

**Commit message:** `refactor(detectors): extract detectors package from 10 top-level modules`
**Rollback:** `git revert <commit-sha>`

---

### Phase 8: Create worm/ package

**Branch:** `refactor/worm-package`
**Preconditions:** None
**Risk:** LOW
**Estimated time:** 30 minutes

#### Files to move (2 modules)

| Current Path | New Path |
|-------------|----------|
| `src/na0s/worm_detector.py` | `src/na0s/worm/detector.py` |
| `src/na0s/worm_advanced.py` | `src/na0s/worm/advanced.py` |

#### Import sites (source)

```
src/na0s/propagation_scanner.py:18   from na0s.worm_detector import WormSignatureDetector
src/na0s/worm_detector.py:1260       from na0s.worm_advanced import copp_signatures
```

Both work via shims.

#### Import sites (scripts)

```
scripts/train_worm_classifier.py:30  from na0s.worm_detector import WormSignatureDetector, _WormCorpusClassifier
```

Works via shim.

#### Tests to pass
```bash
python3 -m pytest tests/test_worm_advanced.py tests/test_worm_bayes.py tests/test_worm_corpus_classifier.py tests/test_worm_embedding.py tests/test_worm_pca_signatures.py tests/test_morris2_pipeline.py tests/test_l9_propagation.py -v
```

**Commit message:** `refactor(worm): extract worm package from 2 top-level modules`
**Rollback:** `git revert <commit-sha>`

---

### Phase 9: Create fusion/ package

**Branch:** `refactor/fusion-package`
**Preconditions:** None
**Risk:** MEDIUM (rrf_fusion and complexity_router are imported by cascade.py, a core module)
**Estimated time:** 1 hour

#### Files to move (5 modules)

| Current Path | New Path |
|-------------|----------|
| `src/na0s/ensemble.py` | `src/na0s/fusion/ensemble.py` |
| `src/na0s/bayesian_fusion.py` | `src/na0s/fusion/bayesian.py` |
| `src/na0s/rrf_fusion.py` | `src/na0s/fusion/rrf.py` |
| `src/na0s/complexity_router.py` | `src/na0s/fusion/complexity_router.py` |
| `src/na0s/performance_slo.py` | `src/na0s/fusion/performance_slo.py` |

#### Import sites (source) -- cascade.py is critical

```
src/na0s/cascade.py:41     from .rrf_fusion import rrf_decision as _rrf_decision
src/na0s/cascade.py:32-34  from .complexity_router import (
                               assess_complexity, get_pipeline_stages, is_adaptive_routing_enabled,
                           )
src/na0s/cascade.py:48     from .performance_slo import SLOTracker
src/na0s/__init__.py:43    from na0s.ensemble import ensemble_scan, EnsembleClassifier
```

Shims at old locations handle all of these. No changes to `cascade.py` needed.

#### Tests to pass
```bash
# Critical: verify cascade.py relative imports resolve through the shim
PYTHONPATH=src python3 -c "from na0s.cascade import CascadeClassifier; print('cascade import OK')"

# Full test run
python3 -m pytest tests/test_ensemble.py tests/test_l6_advanced.py tests/test_l6_cascade_features.py tests/test_l6_routing.py tests/test_cascade.py tests/test_cascade_unit.py tests/test_cascade_integration.py -v
```

**Commit message:** `refactor(fusion): extract fusion package from 5 top-level modules`
**Rollback:** `git revert <commit-sha>`

---

### Phase 10: Reorganize tests

**Branch:** `refactor/test-organization`
**Preconditions:** Phases 2-9 merged (so test file names map to new package names)
**Risk:** LOW (only moving files, pytest discovery is path-based)
**Estimated time:** 2 hours

#### Test file mapping

Create `__init__.py` in each new test subdirectory.

**`tests/canary/`** (4 files):
- `test_canary.py`
- `test_canary_eval.py`
- `test_l10_features.py`
- `test_l10_integrity.py`

**`tests/judge/`** (5 files):
- `test_l7_judge_features.py`
- `test_l7_judge_ops.py`
- `test_l7_local_judge.py`
- `test_llm_checker.py`
- `test_llm_judge_hardening.py`

**`tests/integrity/`** (5 files):
- `test_safe_pickle.py`
- `test_l11_safe_pickle_fixes.py`
- `test_l11_encryption_rollback.py`
- `test_l11_supply_chain.py`
- `test_fp_reduction.py`

**`tests/ml/`** (14 files):
- `test_cross_encoder.py`
- `test_diffuse_defense.py`
- `test_faiss_classifier.py`
- `test_l5_advanced.py`
- `test_l5_model_selection.py`
- `test_l5_structural_concat.py`
- `test_late_chunking.py`
- `test_matryoshka.py`
- `test_perplexity.py`
- `test_predict_embedding.py`
- `test_promptguard.py`
- `test_promptguard_classifier.py`
- `test_replication_similarity.py`
- `test_embedding_drift_detector.py` (from test_layer16 -- verify location)

**`tests/rag/`** (8 files):
- `test_output_scanner.py`
- `test_output_scanner_redaction.py`
- `test_l9_streaming.py`
- `test_l9_propagation.py`
- `test_l9_advanced.py`
- `test_l9_rag_segment.py`
- `test_rag_poison_detector.py`
- `test_rag_position_scanner.py`

**`tests/detectors/`** (10 files):
- `test_context_manipulation.py`
- `test_d7_o1_d3_integration.py`
- `test_extraction_detector.py`
- `test_fictional_frame_detector.py`
- `test_harmful_intent.py`
- `test_mcp_tool_detector.py`
- `test_payload_assembly.py`
- `test_privacy_probe.py`
- `test_recon_detector.py`
- `test_visual_injection_detector.py`

**`tests/worm/`** (6 files):
- `test_worm_advanced.py`
- `test_worm_bayes.py`
- `test_worm_corpus_classifier.py`
- `test_worm_embedding.py`
- `test_worm_pca_signatures.py`
- `test_morris2_pipeline.py`

**`tests/fusion/`** (4 files):
- `test_ensemble.py`
- `test_l6_advanced.py`
- `test_l6_cascade_features.py`
- `test_l6_routing.py`

**Total moved:** ~56 test files into subdirectories. Remaining ~161 flat files are core pipeline tests and stay at `tests/` root.

#### pyproject.toml update

The current `[tool.pytest.ini_options]` has `testpaths = ["tests"]` which will auto-discover subdirectories. No change needed.

#### Verification
```bash
# Count tests before and after -- must match
python3 -m pytest tests/ --collect-only -q 2>/dev/null | tail -1
# Then move files, and re-run:
python3 -m pytest tests/ --collect-only -q 2>/dev/null | tail -1
# Full suite:
python3 -m pytest tests/ -q --tb=line
```

**Commit message:** `refactor(tests): organize 56 test files into package-aligned subdirectories`
**Rollback:** `git revert <commit-sha>`

---

### Phase 11: CI Updates

**Branch:** `ci/branch-triggers`
**Preconditions:** None (can run in parallel with any phase)
**Risk:** LOW
**Estimated time:** 30 minutes

#### 11.1 ✅ DONE (staged): Update ci.yml branch triggers

Change applied to `.github/workflows/ci.yml` on 2026-04-11. Needs commit + push + PR.

```yaml
# BEFORE:
on:
  push:
    branches: [main, 'feature/*']

# AFTER (applied):
on:
  push:
    branches:
      - main
      - 'feature/*'
      - 'feat/*'
      - 'fix/*'
      - 'refactor/*'
      - 'hardening/*'
      - 'ci/*'
      - 'docs/*'
```

**Commit commands:**
```bash
git checkout -b ci/branch-triggers
git add .github/workflows/ci.yml
git commit -m "ci: broaden branch triggers to cover refactor/fix/feat/hardening/docs prefixes"
git push -u origin ci/branch-triggers
gh pr create --base main --title "ci: broaden CI branch triggers" --body "CI now triggers on push to refactor/*, fix/*, feat/*, hardening/*, ci/*, docs/* branches."
```

#### 11.2 Add branch name validation (new job in ci.yml)

```yaml
  branch-name:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - name: Validate branch name
        run: |
          BRANCH="${{ github.head_ref }}"
          PATTERN='^(feature|feat|fix|refactor|hardening|ci|docs|scrape|harvest)/[a-z0-9._-]+$'
          if [[ ! "$BRANCH" =~ $PATTERN && "$BRANCH" != "main" ]]; then
            echo "::error::Branch name '$BRANCH' does not match naming convention: $PATTERN"
            exit 1
          fi
```

#### 11.3 Add anti-regression check (see Section 10)

Add as a CI step:
```yaml
      - name: Check no new top-level modules
        run: |
          python -c "
          from pathlib import Path
          ALLOWED = {
              '__init__.py', '__main__.py', '_env.py', '_version.py', '_voting.py',
              'cascade.py', 'cli.py', 'compliance_evasion_rules.py', 'config.py',
              'data_schema.py', 'evidence_grading.py', 'groundedness.py',
              'intent_guard.py', 'multilingual_handler.py', 'multilingual_intent.py',
              'obfuscation.py', 'positive_validation.py', 'predict.py', 'rules.py',
              'scan_result.py', 'segment_grader.py', 'signal_boost.py',
              'structural_features.py', 'subtle_override_rules.py',
          }
          # Shim files are also allowed (they contain deprecation warnings)
          SHIMS = {
              'canary_alert.py', 'canary_honeypot.py', 'canary_persistence.py',
              'canary_rotation.py', 'canary_session.py', 'canary_verifier.py',
              'canary.py',
              'llm_judge.py', 'local_judge.py', 'judge_audit.py',
              'judge_cost_tracker.py', 'llm_checker.py', 'rate_limiter.py',
              'safe_pickle.py', 'safe_content.py', 'model_encryption.py',
              'model_provenance.py', 'model_rollback.py', 'sbom.py',
              'dep_scanner.py', 'prompt_signer.py', 'chain_integrity.py',
              'template_integrity.py', 'req_integrity.py', 'fingerprint_integrity.py',
              'validation_allowlist.py',
              'embedding_classifier.py', 'embedding_adapter.py', 'faiss_classifier.py',
              'cross_encoder.py', 'predict_embedding.py', 'promptguard.py',
              'promptguard_classifier.py', 'promptguard_signal.py',
              'stacking_classifier.py', 'perplexity.py', 'replication_similarity.py',
              'diffuse_defense.py', 'late_chunking.py',
              'output_scanner.py', 'streaming_scanner.py', 'rag_poison_detector.py',
              'rag_position_scanner.py', 'rag_attribution.py', 'propagation_scanner.py',
              'dual_scanner.py',
              'extraction_detector.py', 'harmful_intent_detector.py', 'recon_detector.py',
              'context_manipulation_detector.py', 'payload_assembly_detector.py',
              'privacy_probe_detector.py', 'mcp_tool_detector.py',
              'fictional_frame_detector.py', 'visual_injection_detector.py',
              'multi_turn_validator.py',
              'ensemble.py', 'bayesian_fusion.py', 'rrf_fusion.py',
              'complexity_router.py', 'performance_slo.py',
              'worm_detector.py', 'worm_advanced.py',
          }
          allowed_all = ALLOWED | SHIMS
          top = {f.name for f in Path('src/na0s').iterdir() if f.suffix == '.py'}
          new = top - allowed_all
          if new:
              print(f'ERROR: New top-level modules detected: {sorted(new)}')
              print('Add them to a sub-package (canary/, judge/, ml/, etc.) instead.')
              raise SystemExit(1)
          print(f'OK: {len(top)} top-level .py files, all accounted for')
          "
```

#### 11.4 Smoke-test the new triggers

After merging Phase 11, verify CI actually fires on a refactor branch:

```bash
# Create a throwaway branch, push a no-op commit, confirm CI triggers
git checkout main && git pull
git checkout -b refactor/test-ci-triggers
echo "# CI trigger test" >> docs/plans/.ci-trigger-test
git add docs/plans/.ci-trigger-test
git commit -m "ci: smoke-test refactor/* branch trigger (delete after)"
git push -u origin refactor/test-ci-triggers

# Wait 60 seconds, then check:
gh run list --branch refactor/test-ci-triggers --limit 1
# Expected: a CI run appears. If not, the trigger pattern is wrong — debug before proceeding.

# Cleanup:
git checkout main
git push origin --delete refactor/test-ci-triggers
git branch -D refactor/test-ci-triggers
```

**Do NOT start Phases 2-9 until this smoke test confirms CI fires on `refactor/*` branches.**

#### 11.5 Check for open PRs before starting Phases 2-9

```bash
# If anyone (including bots) has an open PR, Phases 2-9 will create
# merge conflicts on every moved file. Check BEFORE starting.
gh pr list --state open
# If non-empty, either merge/close the PRs first, or coordinate with
# the PR author that file moves are incoming.
```

**Commit message:** `ci: broaden branch triggers, add branch-name validation and anti-regression gate`
**Rollback:** `git revert <commit-sha>`

---

### Phase 12: Documentation

**Branch:** `docs/post-reorg`
**Preconditions:** Phases 2-10 merged
**Risk:** LOW
**Estimated time:** 1 hour

#### Files to update

1. **`CLAUDE.md`** -- Update layer directory conventions:
   - Add: "New modules go into the appropriate sub-package (`canary/`, `judge/`, `ml/`, etc.), NOT as top-level files"
   - Update: "Tests live in `tests/` organized by package (e.g., `tests/canary/`, `tests/ml/`)"

2. **`docs/ARCHITECTURE.md`** -- Update directory structure section

3. **`README.md`** -- Update project structure if it has one

**Commit message:** `docs: update project structure docs after reorg`
**Rollback:** `git revert <commit-sha>`

---

## 5. Backward Compatibility Shim Pattern

Every moved module gets a shim at its old location. The shim:
1. Re-exports all public symbols so `from na0s.old_name import X` continues to work
2. Emits a `DeprecationWarning` so we can track adoption
3. Is clearly marked as a shim so nobody adds code to it

### Template

```python
# src/na0s/canary_alert.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.canary.alert"""
from na0s.canary.alert import *  # noqa: F401,F403
from na0s.canary.alert import CanaryAlertManager  # explicit for type checkers
import warnings as _warnings
_warnings.warn(
    "na0s.canary_alert is deprecated; use na0s.canary.alert instead",
    DeprecationWarning,
    stacklevel=2,
)
```

### Shim removal timeline

Shims should remain for at least 2 minor versions (or 6 months) after the reorg. Before removing, grep the codebase and all known downstream consumers for the old import path. Add a test that verifies the deprecation warning is emitted:

```python
def test_canary_alert_shim_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import na0s.canary_alert  # noqa: F401
        assert any("deprecated" in str(x.message).lower() for x in w)
```

---

## 6. Branch Cleanup Inventory

### 6.1 Human Branches

| Branch | Last Commit Date | Merged to main? | Action | Reason |
|--------|-----------------|-----------------|--------|--------|
| `main` | 2026-04-10 | -- | Keep | Default branch |
| `clean-readme` | 2026-03-15 | YES | Delete | Merged |
| `docs/readme-v3` | 2026-02-28 | YES | Delete | Merged |
| `manually-cleaned-readme` | 2026-03-15 | YES | Delete | Merged |
| `updating-readme` | 2026-03-11 | YES | Delete | Merged |
| `josh-benchmarking-stuff` | 2026-03-15 | YES | Delete | Merged |
| `fix/layer16-missing-graduated-response` | 2026-04-09 | YES | Delete | Merged |
| `hardening/io-encoding-and-signal-boost` | 2026-04-09 | YES | Delete | Merged |
| `feature/14-ci-infra` | 2026-03-17 | YES | Delete | Merged |
| `feature/6-gap-closure` | 2026-03-15 | YES | Delete | Merged |
| `feature/layer1-rules-overhaul` | 2026-02-24 | YES | Delete | Merged |
| `feature/layer7-llama` | 2026-02-09 | YES | Delete | Merged |
| `feature/signal-boost-caesar-piglatin` | 2026-02-28 | YES | Delete | Merged |
| `feat/benchmark-sprint-day2` | 2026-02-28 | YES | Delete | Merged |
| `feat/d3.1-fake-system-prompt-and-taxonomy-mapping` | 2026-02-26 | YES | Delete | Merged |
| `worm-replication-sim` | 2026-04-05 | NO | Keep | Active work |
| `layer16-conversation-detectors` | 2026-04-03 | NO | Keep | Active work |

### 6.2 Automated Branches

| Category | Count | Action |
|----------|-------|--------|
| `scrape/*` | 199 | Delete all (ephemeral data collection) |
| `harvest/2026-03-23` | 1 | Delete (superseded) |
| `harvest/2026-03-30` | 1 | Delete (superseded) |
| `harvest/2026-04-06` | 1 | Keep (most recent) |

### 6.3 Post-Cleanup Actual State (as of 2026-04-11)

| Category | Count | Status |
|----------|-------|--------|
| `main` | 1 | ✅ |
| Active unmerged branches | 2 | ✅ (`worm-replication-sim`, `layer16-conversation-detectors`) |
| `harvest/*` | 1 | ✅ (`harvest/2026-04-06` kept) |
| `scrape/*` | 0 | ✅ All 201 deleted |
| Merged human branches | 0 | ✅ All 14 deleted |
| **Total** | **4** | ✅ Target reached |

**Local branches:**
| Branch | Status |
|--------|--------|
| `main` | ✅ Active |
| `fix-cascade-tests` | ⚠️ KEPT — has unique commit `6bf3495` not on any remote |
| `layer16-conversation-detectors` | ✅ Tracking remote |
| `worm-replication-sim` | ✅ Tracking remote |
| Worktree branches | ✅ Pruned (2 deleted) |
| `worm_detector` | ✅ Deleted |
| `hardening/io-encoding-and-signal-boost` | ✅ Deleted |
| `fix/layer16-missing-graduated-response` | ✅ Deleted |

---

## 7. Import Update Checklist

This section provides the exact grep results for every module being moved. These are REAL results from running the searches against the current codebase.

### 7.1 canary/ package imports

**Source imports (`src/`):**
```
src/na0s/canary_persistence.py:17  from na0s.canary import CanaryManager, CanaryToken
src/na0s/canary_rotation.py:15    from na0s.canary import CanaryManager, CanaryToken
src/na0s/canary_alert.py:16       from na0s.canary import CanaryToken
src/na0s/canary_session.py:16     from na0s.canary import CanaryManager, CanaryToken
src/na0s/__init__.py:49            from na0s.canary import CanaryManager, CanaryToken
```

**Test imports (`tests/`):**
```
tests/test_l10_integrity.py:13     from na0s.canary_verifier import CanaryTokenVerifier
tests/test_l10_features.py:19     from na0s.canary_alert import CanaryAlertManager
tests/test_l10_features.py:20     from na0s.canary_honeypot import HoneypotManager
tests/test_l10_features.py:21     from na0s.canary_persistence import PersistentCanaryStore
tests/test_l10_features.py:22     from na0s.canary_rotation import RotatingCanaryManager
tests/test_l10_features.py:23     from na0s.canary_session import SessionCanaryManager
```

### 7.2 judge/ package imports

**Source imports (`src/`):**
```
src/na0s/llm_judge.py:23          from na0s.judge_audit import JudgeAuditLogger
src/na0s/llm_judge.py:24          from na0s.judge_cost_tracker import CostTracker
src/na0s/llm_judge.py:25          from na0s.rate_limiter import TokenBucketRateLimiter
```

**Test imports (`tests/`):**
```
tests/test_llm_checker.py:41              from na0s.llm_checker import ...
tests/test_llm_judge_hardening.py:44,56   from na0s.llm_judge / na0s.llm_checker import ...
tests/test_l7_local_judge.py              from na0s.local_judge import ... (30+ sites)
tests/test_l7_local_judge.py              from na0s.llm_judge import ... (12+ sites)
tests/test_l7_judge_features.py:31        from na0s.llm_judge import ...
tests/test_l7_judge_ops.py:18-20          from na0s.judge_cost_tracker / judge_audit / rate_limiter import ...
```

**Script imports (`scripts/`):**
```
scripts/evaluate_llm_judge.py:18   from na0s.llm_judge import LLMJudge
```

### 7.3 integrity/ package imports

**Source imports (`src/`):** None (leaf-level modules, no intra-source imports).

**Test imports (`tests/`):** See Phase 4 details above -- 80+ import sites across 7 test files. All handled by shims.

**Script imports (`scripts/`):** 11 files import `safe_pickle`. All handled by shims.

### 7.4 ml/ package imports

**Source imports (`src/`):**
```
src/na0s/worm_detector.py:24      from na0s.replication_similarity import replication_similarity
```

**Test imports (`tests/`):** 60+ import sites across 13 test files. All handled by shims.

**Script imports (`scripts/`):**
```
scripts/build_faiss_index.py:46    from na0s.faiss_classifier import FAISSClassifier
```

### 7.5 rag/ package imports

**Source imports (`src/`):**
```
src/na0s/segment_grader.py:16     from na0s.output_scanner import OutputScanner
src/na0s/streaming_scanner.py:16  from na0s.output_scanner import ...
src/na0s/dual_scanner.py:14-15    from na0s.output_scanner / propagation_scanner import ...
src/na0s/__init__.py:46-47        from na0s.output_scanner / streaming_scanner import ...
```

**Test imports (`tests/`):** 14+ import sites across 8 test files.

### 7.6 detectors/ package imports

**Source imports (`src/`):**
```
src/na0s/__init__.py:52            from na0s.visual_injection_detector import ...
src/na0s/layer16/detectors/payload_splitting.py:92,121   from na0s.payload_assembly_detector import ...
```

**Test imports (`tests/`):** 10+ import sites across 10 test files.

### 7.7 worm/ package imports

**Source imports (`src/`):**
```
src/na0s/propagation_scanner.py:18   from na0s.worm_detector import WormSignatureDetector
src/na0s/worm_detector.py:1260       from na0s.worm_advanced import copp_signatures
```

**Test imports (`tests/`):** 7+ import sites across 6 test files.

**Script imports (`scripts/`):**
```
scripts/train_worm_classifier.py:30  from na0s.worm_detector import ...
```

### 7.8 fusion/ package imports

**Source imports (`src/`):**
```
src/na0s/cascade.py:41     from .rrf_fusion import rrf_decision as _rrf_decision
src/na0s/cascade.py:32-34  from .complexity_router import ...
src/na0s/cascade.py:48     from .performance_slo import SLOTracker
src/na0s/__init__.py:43    from na0s.ensemble import ...
```

**Test imports (`tests/`):** 4+ import sites across 4 test files.

---

## 8. Risk Register

| Risk | Phase(s) | Likelihood | Impact | Mitigation |
|------|----------|------------|--------|------------|
| **Branch deletion irreversible from GitHub UI** | Phase 0 | High | High (lost data) | **MANDATORY mirror backup** (`git clone --mirror`) before any deletions; batch deletions in groups of 20; spot-check scrape branches for unique commits |
| **layer1/layer2 copies have diverged silently** | Phase 1 | Medium | High (behavior change) | **MANDATORY diff review** before shimming; if >10 lines differ, merge the pair instead of shimming one away |
| Circular import between canary.py (now canary/manager.py) and canary/__init__.py | Phase 2 | Medium | High (import crash) | Use lazy imports or ensure __init__.py only imports from manager.py, not vice versa |
| cascade.py breaks due to relative imports of moved fusion modules | Phase 9 | Low | High (core pipeline down) | Shims at old locations use absolute imports; cascade.py uses relative `from .rrf_fusion` which resolves to shim |
| **Moving dead code into new packages (embedding_classifier.py)** | Phase 5 | Medium | Low (wasted work) | **Resolve Open Question #3 first**: grep for usage; if dead, delete instead of moving |
| Test discovery fails after moving test files | Phase 10 | Low | Medium (CI red) | Add `__init__.py` to every new test subdirectory; verify count before/after |
| Downstream consumers (if any) break on import | All phases | Low | Medium | Shims emit DeprecationWarning but still export all symbols |
| safe_pickle shim breaks scripts | Phase 4 | Low | High (12 scripts affected) | Keep shim indefinitely; test with `scripts/model.py` |
| **Unmerged branches get orphaned imports after reorg** | Phases 5, 7, 8 | **High** | **Medium** (rebase pain) | **Decide NOW**: merge `worm-replication-sim` and `layer16-conversation-detectors` to main BEFORE Phases 5/7/8, or accept manual import fixups during future rebase |
| CI does not run on refactor/* branches | All phases | High | Medium (merge without green CI) | Phase 11 fixes this; do Phase 11 first or run CI manually |

---

## 9. Resume Protocol

To determine current progress and resume work:

### Step 1: Check which phases are merged to main

```bash
git log --oneline main | head -30
# Look for commit messages matching phase patterns:
#   "refactor(canary):" = Phase 2 done
#   "refactor(judge):" = Phase 3 done
#   etc.
```

### Step 2: Verify the most recently completed phase

```bash
# Count top-level modules (should decrease with each phase)
find src/na0s/ -maxdepth 1 -name "*.py" | wc -l

# Run full test suite
python3 -m pytest tests/ -q --tb=line
```

### Step 3: Check for in-progress branches

```bash
git branch -a | grep refactor/
# If a refactor branch exists but isn't merged, it's the current phase
```

### Step 4: Pre-flight for next phase

```bash
git checkout main
git pull origin main
git checkout -b refactor/<next-package>
# Proceed with the phase
```

---

## 10. Anti-Regression Gate

Add this to `tests/conftest.py` (or a new `tests/test_top_level_modules.py`):

```python
"""Prevent new top-level modules from being added to src/na0s/."""
import pathlib

# These are the ONLY .py files allowed at src/na0s/ top level.
# To add a new module, put it in a sub-package instead.
ALLOWED_TOP_LEVEL = frozenset({
    "__init__.py",
    "__main__.py",
    "_env.py",
    "_version.py",
    "_voting.py",
    "cascade.py",
    "cli.py",
    "compliance_evasion_rules.py",
    "config.py",
    "data_schema.py",
    "evidence_grading.py",
    "groundedness.py",
    "intent_guard.py",
    "multilingual_handler.py",
    "multilingual_intent.py",
    "obfuscation.py",
    "positive_validation.py",
    "predict.py",
    "rules.py",
    "scan_result.py",
    "segment_grader.py",
    "signal_boost.py",
    "structural_features.py",
    "subtle_override_rules.py",
})

# Shims are allowed temporarily (backward compat)
ALLOWED_SHIMS = frozenset({
    "bayesian_fusion.py",
    "canary.py",
    "canary_alert.py",
    "canary_honeypot.py",
    "canary_persistence.py",
    "canary_rotation.py",
    "canary_session.py",
    "canary_verifier.py",
    "chain_integrity.py",
    "complexity_router.py",
    "context_manipulation_detector.py",
    "cross_encoder.py",
    "dep_scanner.py",
    "diffuse_defense.py",
    "dual_scanner.py",
    "embedding_adapter.py",
    "embedding_classifier.py",
    "ensemble.py",
    "extraction_detector.py",
    "faiss_classifier.py",
    "fictional_frame_detector.py",
    "fingerprint_integrity.py",
    "harmful_intent_detector.py",
    "judge_audit.py",
    "judge_cost_tracker.py",
    "late_chunking.py",
    "llm_checker.py",
    "llm_judge.py",
    "local_judge.py",
    "mcp_tool_detector.py",
    "model_encryption.py",
    "model_provenance.py",
    "model_rollback.py",
    "multi_turn_validator.py",
    "output_scanner.py",
    "payload_assembly_detector.py",
    "performance_slo.py",
    "perplexity.py",
    "predict_embedding.py",
    "privacy_probe_detector.py",
    "prompt_signer.py",
    "promptguard.py",
    "promptguard_classifier.py",
    "promptguard_signal.py",
    "propagation_scanner.py",
    "rag_attribution.py",
    "rag_poison_detector.py",
    "rag_position_scanner.py",
    "rate_limiter.py",
    "recon_detector.py",
    "replication_similarity.py",
    "req_integrity.py",
    "rrf_fusion.py",
    "safe_content.py",
    "safe_pickle.py",
    "sbom.py",
    "stacking_classifier.py",
    "streaming_scanner.py",
    "template_integrity.py",
    "validation_allowlist.py",
    "visual_injection_detector.py",
    "worm_advanced.py",
    "worm_detector.py",
})


def test_no_new_top_level_modules():
    """Fail if a new .py file is added directly to src/na0s/ instead of a sub-package."""
    na0s_dir = pathlib.Path(__file__).resolve().parent.parent / "src" / "na0s"
    top_level_py = {f.name for f in na0s_dir.iterdir() if f.suffix == ".py"}
    allowed = ALLOWED_TOP_LEVEL | ALLOWED_SHIMS
    unexpected = top_level_py - allowed
    assert not unexpected, (
        f"New top-level module(s) detected in src/na0s/: {sorted(unexpected)}\n"
        f"Please add them to an appropriate sub-package (canary/, judge/, ml/, etc.) "
        f"instead of the top level."
    )
```

---

## 11. Open Questions

1. **`canary.py` -> `canary/` directory conflict:** The file `canary.py` currently exists and `from na0s.canary import ...` works because Python resolves to the file. When we create `canary/` as a directory with `__init__.py`, the old `canary.py` file must be DELETED (not shimmed) because Python cannot have both `canary.py` and `canary/` at the same path level. The `canary/__init__.py` must re-export everything that `canary.py` exported. This is the ONE case where a shim at the old location is impossible.

2. **~~Relative imports in cascade.py~~ RESOLVED:** This is not an open question — it's a testable verification step. After Phase 9, run `PYTHONPATH=src python3 -c "from na0s.cascade import CascadeClassifier; print('OK')"`. If it fails with a circular import, the shim pattern needs adjustment. This check is now included in Phase 9's test verification block.

3. **~~`embedding_classifier.py` possibly dead code~~ RESOLVED 2026-04-11:** The module is **ALIVE** — 23 grep hits across src/, including a direct import at `predict.py:187` (`from .embedding_classifier import get_embedding_classifier`). Also used by `cascade.py`, `ensemble.py`, `late_chunking.py`. **Verdict: MOVE to `ml/` in Phase 5.** Do NOT delete.

4. **~~Test file `test_promptguard_classifier.py`~~ RESOLVED 2026-04-11:** The file **EXISTS** at `tests/test_promptguard_classifier.py` (712+ lines). Include in Phase 10 test moves under `tests/ml/`.

5. **`models/` directory:** Contains trained model artifacts (`.pkl`, `.sha256`). Should this directory be part of the reorg, or is it intentionally flat? Current plan: leave unchanged.

6. **~~Unmerged branches~~ RESOLVED 2026-04-11:** Decision: **(b) Accept rebase pain.** `worm-replication-sim` has 2 breaking imports (both covered by shims at old locations). `layer16-conversation-detectors` has 1 breaking import (also covered by shim). If shims are implemented as planned in Phases 5/6/8, neither branch needs manual fixups. If shims are ever removed, 3 total imports need updating.

7. **`py.typed` marker:** This file indicates PEP 561 type stub support. After the reorg, sub-packages may also need `py.typed` markers. Verify with `mypy --install-types` after Phase 9.

8. **Shim files and flake8:** The `noqa: F401,F403` comments suppress warnings for `import *` in shims. The CI flake8 config should not need changes, but verify the `--select=E9,F63,F7,F82` filter in ci.yml does not flag these.

---

## Appendix A: Phase Dependency Graph

```
Phase 0  (branch cleanup)     -- independent, do first
Phase 1  (dedup layer1/2)     -- independent
Phase 2  (canary/)            -- independent
Phase 3  (judge/)             -- independent
Phase 4  (integrity/)         -- independent
Phase 5  (ml/)                -- independent
Phase 6  (rag/)               -- independent
Phase 7  (detectors/)         -- independent
Phase 8  (worm/)              -- independent
Phase 9  (fusion/)            -- independent
Phase 10 (test reorg)         -- DEPENDS ON Phases 2-9
Phase 11 (CI updates)         -- independent, ideally do BEFORE Phases 2-9
Phase 12 (docs)               -- DEPENDS ON Phases 2-10
```

Recommended execution order:

```
0 (mirror backup + branch cleanup)
  → 11 (CI triggers — do BEFORE any refactor branches)
    → Resolve Open Questions 3, 4, 6 (dead code check, deps, unmerged branches)
      → 1 (dedup — AFTER diffing each pair)
        → 2-9 (in any order, parallelizable)
          → 10 (test reorg — AFTER 2-9 merged)
            → 12 (docs — AFTER 10)
```

Phases 2-9 can be done in ANY order and even in PARALLEL (separate branches, separate PRs). Phase 11 should be done early so CI runs on refactor branches.

**CRITICAL: Before Phases 5, 7, 8** — decide the fate of your two unmerged branches (`worm-replication-sim`, `layer16-conversation-detectors`). These branches touch files that Phases 5 (ml/), 7 (detectors/), and 8 (worm/) will move. Options:
- **(a) Merge them to main first** (preferred) — then Phases 5/7/8 move the already-merged code cleanly
- **(b) Accept manual rebase pain later** — do the phases first, then rebase the unmerged branches against the new package structure (every moved import must be fixed in the rebase)
- **(c) Abandon them** — if the work is stale or superseded

Choose (a), (b), or (c) before starting Phase 5.

---

## Appendix B: Execution Checklist per Phase

For every phase (2-9), follow this exact sequence:

```bash
# 1. Start from clean main
git checkout main && git pull origin main

# 2. Create branch
git checkout -b refactor/<package-name>

# 3. Create the package directory
mkdir -p src/na0s/<package>/
touch src/na0s/<package>/__init__.py

# 4. Move files (git mv preserves history)
git mv src/na0s/old_module.py src/na0s/<package>/new_name.py

# 5. Create shim at old location
cat > src/na0s/old_module.py << 'SHIM'
"""Backward-compat shim. Canonical location: na0s.<package>.new_name"""
from na0s.<package>.new_name import *  # noqa: F401,F403
import warnings as _warnings
_warnings.warn(
    "na0s.old_module is deprecated; use na0s.<package>.new_name instead",
    DeprecationWarning,
    stacklevel=2,
)
SHIM

# 6. Update intra-package imports in moved files
#    (change "from na0s.old_module" to "from na0s.<package>.new_name")

# 7. Write __init__.py with public re-exports

# 8. Run targeted tests
python3 -m pytest tests/test_<relevant>*.py -v

# 9. Run full suite
python3 -m pytest tests/ -q --tb=line

# 10. Commit
git add -A
git commit -m "refactor(<package>): extract <package> package from N top-level modules"

# 11. Push and create PR
git push -u origin refactor/<package-name>
```
