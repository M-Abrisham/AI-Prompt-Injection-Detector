---
item: R7
title: Unified HMAC-signed self-derived manifest — one signed sidecar carrying BOTH the #4 per-file allowlist AND the #17/DI-5 decontam-LODO attestation
classification: Supply-chain / eval-integrity / governance infrastructure (NOT a prompt-injection attack class) — L11 integrity surface
dedup_status: NEW — re-verified against THIS worktree (`research/local-items` @ 7376bb7, base advanced via #476 / 81db256). NO R7 / "unified manifest" / "signed manifest" roadmap line exists (`grep -n "R7\|unified.*manifest\|signed.*manifest" ROADMAP_V2.md` → only M14:1797 content-addressed corpus manifest + R7-research-finding mentions at :1819/:1865, neither is THIS item). #476 added NO R7 cross-link. R7 is the successor of Item 4 (allowlist `find_class`, `specs/hardening/04-toctou-allowlist-unpickler.md`) + Item 17/DI-5 (decontam attestation + `ModelProvenance` `.meta.json`, `specs/hardening/17-di-decontam-retrain.md`). Lands AFTER #4 + #17 (+ #7 sidecar-resolution, #9 digest-validation — the load-path tiers R7 layers above).
scope: NEW DESIGN SPEC. One HMAC-signed sidecar (`<model>.manifest` / a bundle-level `models/MANIFEST`) that carries, in one signed-once document: (a) the #4 per-class allowlist of pickle globals the bundled artifacts legitimately need, and (b) the #17/DI-5 decontam-LODO attestation (sealed-corpus hash, leave-one-dataset-out provenance, sample counts). Signed via the EXISTING `safe_pickle` HMAC tier (`write_digest_sidecar` HMAC branch, `src/na0s/integrity/safe_pickle.py:701-707`). RESOLVES the design Q: `KNOWN_HASHES`/`.sha256` is plain SHA-256, not HMAC → **layer the manifest ABOVE the per-file hash chain** (manifest is the HMAC-signed root; per-file `KNOWN_HASHES`/`.sha256`/`.hmac` tiers stay as the unchanged hash-chain leaves the manifest references). Do NOT promote `.sha256` itself to an HMAC tier (that breaks keyless deployments — the documented anti-DoS contract at `safe_pickle.py:30-33`).
applicable_steps: [1, 2, 3, 4, 6, 7, 8, 9]   # explore, refs, root-cause, implement+wire, tests, cleanup, roadmap, README/benchmark
na_steps: [5, 10, 11]                         # harvester (no attack dataset), taxonomy/coverage (no attack class), GitHub (LOCAL only)
skills: [na0s-review-checklist, subsystem-context-pack]
depends_on: [Item 4, Item 17, Item 7, Item 9]   # HARD: 4 (allowlist payload), 17/DI-5 (attestation payload + ModelProvenance). SOFT: 7 (sidecar-resolution tiers R7 layers above), 9 (64-hex digest validation R7 reuses)
local_only: true   # NO GitHub — no push / PR at any execution step
---

# R7 — Unified HMAC-signed self-derived manifest

## 0. One-sentence framing
R7 collapses three TODAY-fragmented, mostly-UNSIGNED integrity artifacts — the per-file
`KNOWN_HASHES`/`.sha256` hash chain, the #4 pickle-globals allowlist, and the #17/DI-5
decontam attestation + `ModelProvenance` `.meta.json` — into ONE manifest signed exactly once
by the existing `safe_pickle` HMAC tier, so a single `hmac.compare_digest` over the manifest
authenticates the entire model bundle's allowlist + provenance, with the per-file digests
demoted to hash-chain leaves the signed manifest references (not independent trust roots).

## Dedup re-verification (against THIS worktree)
- **No standalone R7 row.** `grep -n "R7\|R8\|unified.*manifest\|signed.*manifest" ROADMAP_V2.md`
  returns: `1819` (R1-recon finding, F14), `1865` ("per R7 research — more gates = more false
  rejections", gate-type separation) — these are *research-note* mentions, NOT a tracked R7
  manifest item. `M14` (ROADMAP_V2.md:**1797**) is the *content-addressed training-corpus*
  SHA-256 manifest (a DIFFERENT artifact: corpus snapshot, not model-bundle attestation) — R7
  CROSS-LINKS it (the corpus manifest is one *input* to R7's decontam-LODO attestation) but does
  not duplicate it.
- **#476 (81db256) check:** present in this worktree's `git log` (`81db256 docs(roadmap): record
  hardening campaign (8 PRs) …`); it recorded the L11 campaign but added **no** R7 line and **no**
  R7 cross-link into Item 4 / Item 17. Dedup unchanged → R7 is genuinely NEW.
- **The two predecessor payloads exist as specs, not yet as a signed bundle:**
  - **#4 allowlist** (`specs/hardening/04-toctou-allowlist-unpickler.md`): adds a restricted
    `find_class` allowlist of the sklearn/numpy globals the bundled `.pkl`s need. Today that
    allowlist (once #4 lands) is a HARD-CODED Python set inside `safe_pickle.py` — NOT a signed,
    per-bundle, self-derived document. R7's payload (a) = that allowlist, *derived from the actual
    shipped artifacts* and *signed*, so it can travel with a user-retrained bundle.
  - **#17/DI-5 attestation** (`specs/hardening/17-di-decontam-retrain.md` §1, §3 step 6;
    `src/na0s/integrity/model_provenance.py`): `ModelProvenance.write()` writes a `.meta.json`
    sidecar (`model_provenance.py:58` `_meta_path` = `<path>.meta.json`; `:61` `write`; `:86`
    `verify`) gated on `NA0S_MODEL_PROVENANCE=1` (`:19-20`), carrying `sha256` + training metadata.
    Item 17's decontam attestation is captured TODAY only in the **PR body** (Item 17 §3 step 3:
    "Capture both outputs into the PR body as the decontamination attestation") — i.e. NOT a
    machine-verifiable on-disk artifact, and the `.meta.json` it does write is **UNSIGNED** (plain
    SHA-256, no HMAC). R7's payload (b) = that attestation, promoted into the signed manifest.
- **An existing UNSIGNED multi-file manifest already ships:** `SBOMGenerator.generate()`
  (`src/na0s/integrity/sbom.py:97`) emits a CycloneDX-lite dict with per-model `{"sha256": …}`
  (`sbom.py:77`) over `_discover_models()` (`:65`). This is the closest existing precedent to R7
  but is (i) unsigned, (ii) carries no allowlist, (iii) carries no decontam attestation. R7 = the
  SIGNED superset. **Do NOT create a second manifest format in parallel — extend/wrap the SBOM
  shape so there is one manifest, not two** (na0s-review-checklist §11 dead-code/duplicate trap).

---

## Applicable steps

### Step 1 — Explore current rules around R7 (the integrity surface as it is TODAY)
Pack the subsystem first: `subsystem-context-pack` over
`src/na0s/integrity/**,src/na0s/models/**,tests/test_safe_pickle*.py,tests/integrity/**`
(`--compress`, markdown). Then map every trust artifact the model bundle relies on (all
line-verified in THIS worktree):

| Artifact | Where | Signed? | Self-derived? | R7 role |
|---|---|---|---|---|
| `KNOWN_HASHES` (4 entries) | `src/na0s/models/__init__.py:26-31` | source-signed (wheel) | yes (per build) | hash-chain leaf the manifest references |
| `.sha256` sidecars (only `model.pkl.sha256`, `tfidf_vectorizer.pkl.sha256` ship) | `src/na0s/models/` | NO (attacker-recomputable, `safe_pickle.py:20-23`) | yes | leaf; **NOT promoted to HMAC** (keyless contract) |
| `.hmac` sidecars | written when `NA0S_PICKLE_KEY` set (`safe_pickle.py:701-707`) | YES (HMAC-SHA256) | yes | the EXISTING signing primitive R7 reuses for the manifest |
| pickle-globals allowlist | (lands with #4) hard-coded set in `safe_pickle.py` `find_class` | source-signed only | NO (static set) | R7 payload (a) — self-derive + sign |
| `.meta.json` provenance | `model_provenance.py:58` `_meta_path`; gated `NA0S_MODEL_PROVENANCE=1` | NO (plain SHA-256) | partly | R7 payload (b) — fold into signed manifest |
| decontam-LODO attestation | Item 17 §3 step 3 → **PR body only** | NO | yes | R7 payload (b) — promote to on-disk signed artifact |
| CycloneDX-lite SBOM | `sbom.py:97` `generate()`; per-model `sha256` `:77` | NO | yes | the manifest SHAPE R7 extends (don't fork it) |

- **Sidecar format primitive R7 reuses** (no new format invented): `_format_sidecar(algo, digest)`
  → `"v1:{algo}:{digest}"` (`safe_pickle.py:347-349`); parsed/shape-validated by
  `_parse_sidecar_typed` (`:404`) against `_HEX64_RE` (`:70`). The manifest's OWN signature sidecar
  reuses this exact `v1:hmac-sha256:<64hex>` shape via `write_digest_sidecar` — so R7 introduces
  **zero new on-disk signature format**, only a new *payload document* (the manifest body) that
  gets the standard sidecar treatment.
- **Signing entry point R7 reuses** (verified, not hallucinated): `write_digest_sidecar(path, …)`
  (`safe_pickle.py:680`) hashes the bytes already on disk and writes `<path>.hmac` (HMAC tier when
  `NA0S_PICKLE_KEY` set, `:701-707`) or `<path>.sha256` (keyless, with a `UserWarning`, `:708-719`).
  `verify_file_digest(path)` (`:736`) is the format-agnostic verifier (it does NOT unpickle — safe
  for a JSON manifest). **R7 writes the manifest as a plain JSON file, then calls
  `write_digest_sidecar(manifest_path)` to sign it, and `verify_file_digest(manifest_path)` to
  verify it.** Both already exist; R7 adds the manifest builder/reader on top.

### Step 2 — Roadmap / Taxonomy / README / Coverage / benchmark for the gap
- **Roadmap:** L11 description ROADMAP_V2.md:**1289** + :**1354** (3-tier trust hierarchy,
  `ModelProvenance`, `SBOMGenerator`, `PromptSigner`). **NOTE the drift:** the task's
  `~ROADMAP_V2.md:1224` ref is STALE — line 1224 is L9 OUTPUT-scanner content (image-host
  allowlist), NOT the HMAC gate. The real L11/HMAC-gate prose is :1289/:1354; the L11 *open-items*
  block (Items 1/4/7/8/9/10/14/15/16/17) is :1379-1397. Step 8 adds the R7 line into that block.
- **Taxonomy / Coverage Matrix:** N/A reference — R7 is governance infra, no attack class (Step 10).
- **README / Benchmark:** README's security/model-integrity section + `docs/MODEL_PROVENANCE.md`
  (DI-5 owner) are the only doc surfaces; `SECURITY.md` lists the supply-chain controls. Step 9
  updates them to say "model bundles ship a single HMAC-signed manifest binding the per-file
  hashes + the unpickle allowlist + the decontam/provenance attestation". No benchmark *number*
  changes (R7 verifies bytes; it does not score prompts).
- **Cross-link target:** M14 (ROADMAP_V2.md:**1797**) corpus SHA-256 manifest — R7's decontam-LODO
  attestation should reference the M14 corpus-manifest digest as its `corpus_snapshot` field, so
  the two manifests chain (corpus → model). Record this as a cross-link, do not absorb M14.

### Step 3 — Root-cause implementation plan (the design, with the design-Q resolved)

**Gap vs ideal (root cause):** today, three integrity facts about a model bundle live in three
places with three trust levels — `KNOWN_HASHES` (source-signed), the (post-#4) allowlist
(source-signed, static), and the decontam attestation (PR-body prose, unsigned). A user who
*retrains* (Item 17 path) gets fresh `.sha256` sidecars but NO signed allowlist and NO signed
attestation — the strongest trust statement (HMAC) covers only individual file bytes, never the
*bundle-level* facts (which globals are legitimate, on what sealed corpus it was trained). Ideal:
one document carrying all three, signed once, so verifying the manifest's HMAC authenticates the
whole bundle's integrity story.

**Design-Q resolution (the load-bearing decision):**
`KNOWN_HASHES`/`.sha256` is plain SHA-256, NOT HMAC. Two candidate resolutions:
- (A) *Promote `.sha256` to an HMAC tier.* **REJECTED** — it breaks the documented keyless contract
  (`safe_pickle.py:30-33`: in keyless mode the `.sha256` is the verifiable artifact and a
  present-but-unverifiable `.hmac` must never veto it). Forcing HMAC everywhere makes keyless
  deployments unverifiable and re-introduces the dropped-file DoS that #7 closed.
- (B) *Layer the manifest ABOVE the per-file hash chain.* **CHOSEN.** The manifest BODY lists, per
  bundled file, its `KNOWN_HASHES`/sidecar digest (the existing leaf) PLUS the allowlist + the
  attestation. The MANIFEST FILE then gets ONE `write_digest_sidecar` signature — HMAC when a key
  is set (strong, self-contained bundle authentication), `.sha256` when keyless (no worse than
  today's per-file `.sha256`, and STILL strictly better because it binds the *set* of files +
  allowlist + attestation into one tamper-evident document instead of N independently-swappable
  sidecars). Per-file tiers are UNCHANGED; the manifest is a NEW outer envelope, not a replacement.
  This is the only resolution that (i) keeps keyless deployments working, (ii) gives HMAC users a
  single bundle-level signature, (iii) touches zero existing load-path tier code.

**Numbered LOCAL implementation plan:**
1. **New module `src/na0s/integrity/manifest.py`** (sub-package per CLAUDE.md; NOT a top-level
   file). Public API (all NEW, none hallucinated — they don't exist yet, this spec creates them):
   - `build_manifest(models_dir, *, allowlist, attestation) -> dict` — self-derives the per-file
     section by hashing each artifact under `models_dir` (reuse `sbom._discover_models` / the
     `_sha256` chunked hasher, do NOT re-implement), cross-checks each digest against
     `KNOWN_HASHES` (mismatch → `ValueError`, fail closed), and embeds the allowlist + attestation.
     Returns a plain dict with a fixed schema version `"na0s-manifest/v1"`.
   - `write_manifest(models_dir, manifest, path) -> str` — `json.dump` (sorted keys, no pickle) to
     `path`, then `safe_pickle.write_digest_sidecar(path)` to sign it. Returns the sidecar path.
   - `verify_manifest(path) -> dict` — `safe_pickle.verify_file_digest(path)` FIRST (constant-time,
     no deserialize of executable code — it's JSON), THEN `json.load`, THEN re-derive each file's
     digest and `hmac.compare_digest` against the manifest's recorded digest (fail closed on any
     mismatch / missing file / extra file). Returns the validated dict.
   - `load_allowlist(path) -> frozenset[str]` / `load_attestation(path) -> dict` — convenience
     readers that call `verify_manifest` then pull the sub-section (so a caller can NEVER read an
     unverified allowlist).
2. **Self-derivation, not hand-authoring (the "self-derived" in the title):** the per-file hash
   section is computed FROM the artifacts on disk at build time, never typed by hand; the allowlist
   is computed by the #4 build step that already enumerates the globals the real `.pkl`s reference
   (R7 consumes #4's enumeration output rather than re-deriving — DEPENDS-ON #4). The attestation is
   read from Item 17's `data/processed/training_metrics.json` + the `ModelProvenance` record + the
   M14 corpus-snapshot digest (DEPENDS-ON #17).
3. **No new threshold / similarity cutoff.** R7 does byte-exact `hmac.compare_digest` only. There is
   NO similarity score, NO short-injection-string cutoff, NO probabilistic match — so the
   "similarity-cutoffs must be locally re-calibrated for short injection strings" caveat is
   **inapplicable** (R7 introduces no such number). The ONLY numeric constants are reused, named,
   already-justified ones: `_HEX64_RE` 64-char invariant (`safe_pickle.py:63-70`) and the chunk
   size `INTEGRITY_HASH_CHUNK_BYTES` (`config.py`). Add NO magic numbers.
4. **Fail-closed everywhere** (na0s-review-checklist §6): missing manifest at load → caller's
   existing per-file path still works (manifest is additive, env-gated `NA0S_BUNDLE_MANIFEST=1` for
   staged rollout, mirroring `NA0S_MODEL_PROVENANCE`), but if the manifest IS present and its HMAC
   fails / a referenced file's digest drifts / a file is missing-or-extra → `ValueError`, never a
   silent skip. Keyless + bundled-but-unverifiable manifest → refuse (same logic as the lone-`.hmac`
   refusal `safe_pickle.py:765-772`).

### Step 4 — Implement & wire into the pipeline
- **Wiring point is the model LOAD path, NOT `predict.py`/`cascade.py` scan dispatch.** R7 verifies
  model-bundle integrity at load, before any `safe_load`. Concretely: the cached model loaders
  (`predict.py` `_load_model_and_vectorizer` / `predict_embedding.py` loaders — the ~6 fixed model
  pkls named at `safe_pickle.py:104-108`) gain an OPTIONAL pre-load `verify_manifest(models_dir /
  "MANIFEST")` call behind `_HAS_BUNDLE_MANIFEST` + `NA0S_BUNDLE_MANIFEST=1`, with `try/except
  ImportError` graceful-fallback (mirror the `_HAS_*` registration idiom CLAUDE.md mandates).
  - **Parity note:** predict.py and cascade.py load the SAME bundle via the same loader helpers, so
    wiring the verify into the shared loader gives both paths parity automatically — there is no
    separate cascade load site to mirror (verify by grep: both import the model via
    `na0s.predict`/`na0s.predict_embedding`, not a private copy). This is the §Q8 answer: neither
    `predict.py` nor `cascade.py` references an *attack class* called "manifest" (R7 is not an attack
    class), but BOTH consume the manifest indirectly through the shared loader.
- **Deploy/retrain wiring (Item 17 hand-off):** `scripts/deploy_model.py` (which already rewrites
  `KNOWN_HASHES` and refreshes sidecars per Item 17 §3 step 6) gains a final step
  `python -m na0s.integrity.manifest --build src/na0s/models/` that self-derives + signs the bundle
  manifest from the freshly-deployed bytes. This is the ONLY producer; the SDK is a pure consumer.
- **Assign agents:** `layer-9-11-auditor` (owns the L11 manifest module + the trust-tier layering
  decision), `l3-l5-code-auditor` (owns the loader-side wiring in `predict.py`/`predict_embedding.py`
  + deploy_model producer step), `silent-failure-hunter` (prove every fail-closed branch raises and
  the env-gate cannot silently disable verification once the manifest is present).

### Step 6 — Test plan (Code + behavior; NO hollow tests)
New `tests/integrity/test_manifest.py` (mirrors source per CLAUDE.md test-org). Every test must FAIL
if its target is broken (na0s-review-checklist §4 — "comment out the code-under-test, confirm red"):
- **Round-trip (code):** `build_manifest` → `write_manifest` → `verify_manifest` over a tmp models
  dir returns the same dict; assert allowlist + attestation sub-sections survive byte-for-byte.
- **HMAC signature (behavior, KEY set):** with `NA0S_PICKLE_KEY` set, the manifest sidecar is
  `<path>.hmac` and `verify_manifest` passes; flip one byte of the manifest JSON → `verify_manifest`
  raises `ValueError` (the HMAC catches it). This proves the signature is load-bearing, not decorative.
- **Tamper a referenced file (behavior):** sign the manifest, then mutate a `.pkl` it references →
  `verify_manifest` raises (the per-file digest re-derivation catches drift even if the manifest
  HMAC still verifies — defense in depth).
- **Missing / extra file (behavior):** delete a referenced artifact / add an unlisted `.pkl` →
  `verify_manifest` raises (closed-world check).
- **Keyless mode (behavior):** unset `NA0S_PICKLE_KEY` → manifest is signed with `.sha256` + the
  documented `UserWarning`; `verify_manifest` still passes; a `.sha256` is NOT silently treated as
  HMAC (assert no privilege escalation; mirror `safe_pickle.py:30-33`).
- **Design-Q regression (behavior):** assert the manifest is layered ABOVE, not replacing, the
  per-file chain — i.e. removing the manifest entirely still lets the existing per-file `safe_load`
  succeed (manifest is additive), AND a present manifest with a key gives a single bundle signature.
- **Allowlist binding (cross-#4 behavior):** `load_allowlist` refuses to return the allowlist if the
  manifest HMAC fails — you can NEVER read an unsigned allowlist. (Guards the #4 → R7 trust edge.)
- **CLI/suite smoke (na0s-review-checklist §4/§11 — the real-load gate, NOT a mock):**
  1. `NA0S_PICKLE_KEY=$(python3 -c "import secrets;print(secrets.token_hex(32))") python3 -m na0s.integrity.manifest --build src/na0s/models/` runs end-to-end against the REAL bundled `.pkl`s.
  2. `python3 -c "from na0s.integrity.manifest import verify_manifest; verify_manifest('src/na0s/models/MANIFEST'); print('ok')"` against the real signed manifest.
  3. `python3 -m pytest tests/integrity/test_manifest.py tests/test_safe_pickle.py -q --tb=line`.
  4. Full suite: `python3 -m pytest tests/ -q --tb=line` (CLAUDE.md gate — zero regressions).

### Step 7 — Cleanup / refactor
- **No duplicate manifest format.** Either extend `SBOMGenerator.generate()` (`sbom.py:97`) to be
  the manifest body, or have `manifest.py` import the SBOM model-section helper — do NOT ship two
  parallel `{sha256:…}` model lists (na0s-review-checklist §11). Decide at execution; record which.
- **`integrity/__init__.py` is a bare docstring (no `__all__`)** — if R7 adds the first public
  export, add `manifest` symbols to a new `__all__` consistently (do not leave a half-exported pkg).
- **No top-level shim.** New code goes ONLY in `integrity/manifest.py`; do NOT add a
  `src/na0s/manifest.py` top-level file or any shim (CLAUDE.md code-org standard).
- **`ModelProvenance` reconciliation:** once the attestation lives in the signed manifest, the
  unsigned `.meta.json` becomes redundant. Do NOT silently delete it (Item 17/DI-5 owns it);
  instead, have `ModelProvenance.write` ALSO feed the manifest, and document `.meta.json` as the
  human-readable mirror of the signed `attestation` block. Coordinate the decision with Item 17.

### Step 8 — Roadmap update (LOCAL)
- Add ONE checkbox into the L11 open-items block (ROADMAP_V2.md:1379-1397), after Item 17:
  `- [ ] **Item R7 — unified HMAC-signed self-derived bundle manifest (allowlist + decontam/DI-5
  attestation)** (P2 — supply-chain governance). depends-on: 4, 17 (HARD), 7, 9 (SOFT). Spec:
  specs/research/R7-unified-signed-manifest.md.`
- Cross-link M14 (:1797): append "(R7's model manifest references this corpus manifest's digest as
  its `corpus_snapshot`)". Cite the local commit SHA when any code lands. NO roadmap edit until
  execution; this spec is PLAN-ONLY.

### Step 9 — README / Benchmark
- **README / SECURITY.md / `docs/MODEL_PROVENANCE.md`:** add the one-line guarantee
  ("model bundles ship a single HMAC-signed manifest binding per-file hashes + the unpickle
  allowlist + the sealed-corpus decontam attestation; verified before load when
  `NA0S_BUNDLE_MANIFEST=1`"). Keep it gated/optional so the keyless default story is unchanged.
- **Benchmark:** NO number changes — R7 verifies bytes, it scores no prompts; there is no recall/FPR
  delta. The docs-drift CI gate (`docs/facts.yaml`) only needs a touch if a new env var / module is
  enumerated there; re-run `scripts/extract_facts.py` if so (verify the gate stays green).

---

## N/A steps (honest justifications)
- **Step 5 — Harvester / dataset audit.** N/A — R7 signs the model's OWN integrity artifacts
  (hashes + allowlist + provenance); there is no HARVESTED ATTACK DATASET, no HuggingFace/arXiv/
  GitHub intel, no F14 scenario, no decontam *of prompts* involved. (The decontam *attestation* R7
  carries is Item 17's *output*, not a harvest input.)
- **Step 10 — Taxonomy + Coverage Matrix + per-feature thresholds.** N/A — a bundle-signing
  mechanism is not a detectable attack class: no taxonomy code, no Coverage-Matrix row, no scored
  threshold, no similarity cutoff. (R7's only constants are reused, already-justified hash
  invariants — `_HEX64_RE` 64-char, the config chunk size — not new thresholds.)
- **Step 11 — PR / GitHub.** N/A — LOCAL ONLY per directive. R7 code lands in a dedicated worktree
  branch (`hardening/unified-signed-manifest`); no push / PR / GitHub at any execution step until
  the user explicitly authorizes merge-to-main.

---

## Q&A self-check
- **Q1 — Can Na0S handle R7 (run scan/suite)?** Not a detector — R7 is bundle signing, not prompt
  detection. The relevant gate is the §6 manifest round-trip/tamper suite + the full
  `pytest tests/ -q` regression run (zero regressions required), plus the real-bundle CLI smoke.
- **Q2 — Cleanup done / clutter?** Yes — one new module (`integrity/manifest.py`), one new test file
  (`tests/integrity/test_manifest.py`), no top-level shim, no duplicate manifest format (extend/reuse
  SBOM), `integrity/__init__` exports kept consistent.
- **Q3 — Pipeline wiring correct?** Yes — verify hooks into the SHARED model loader (so predict +
  cascade get parity for free) behind `_HAS_BUNDLE_MANIFEST` + `NA0S_BUNDLE_MANIFEST=1`, fail-closed;
  the producer is `deploy_model.py` only.
- **Q4 — Tested for code AND use-case?** Code: round-trip, signature, keyless, missing/extra-file,
  allowlist-binding unit tests. Use-case: real-bundle CLI build+verify smoke against the shipped
  `.pkl`s (not a mock — na0s-review-checklist §4/§11).
- **Q5 — Harvester audit.** N/A — model-integrity artifacts, not harvested intel (Step 5).
- **Q6 — Taxonomy + coverage (no dups)?** N/A — no attack class; the dedup audit above confirms no
  duplicate R7 row exists (M14 is a *different*, cross-linked artifact, not a dup).
- **Q7 — Does the scorer score R7 right?** N/A — R7 introduces no scorer/threshold; it is byte-exact
  `hmac.compare_digest` (verdict = match/no-match, no probability).
- **Q8 — Do predict.py / cascade.py reference R7?** Indirectly — neither names a "manifest" attack
  class (R7 is not one), but BOTH consume the bundle through the SHARED loader where the verify hook
  lives, giving parity. No per-path manifest copy exists (verify by grep at execution).
- **Q9 — Harvester agent harvest this type?** N/A — not a harvestable intel type.
- **Q10 — Other correctness check.** (i) Design-Q resolved as "layer above the hash chain," NOT
  "promote `.sha256` to HMAC" (the latter breaks keyless — `safe_pickle.py:30-33`). (ii) Manifest is
  ADDITIVE/env-gated → never breaks the existing per-file load path. (iii) Self-derived: per-file
  digests computed from disk + cross-checked vs `KNOWN_HASHES`, never hand-typed. (iv) Reuse SBOM +
  `write_digest_sidecar`/`verify_file_digest`/`_format_sidecar` — invent no new format. (v) Re-grep
  every cited line at execution (Item 4/17 refs already drifted; the task's `:1224` ref is stale →
  use :1289/:1354/:1797).

---

## Execution preconditions / dependencies
- **DEPENDS-ON Item 4 (HARD)** — R7's allowlist payload IS #4's `find_class` global enumeration.
  Until #4 lands and produces a concrete allowlist of the globals the shipped `.pkl`s use, R7's
  payload (a) is empty. Do NOT open R7's branch before #4 merges (or co-develop on a stacked branch).
- **DEPENDS-ON Item 17 / DI-5 (HARD)** — R7's attestation payload IS Item 17's decontam-LODO output
  + the `ModelProvenance` record. Until the sealed-corpus retrain runs (Item 17 needs `dvc pull` /
  CI — it "cannot run on this laptop", Item 17 §1), the attestation block is a placeholder. R7 must
  consume Item 17's `training_metrics.json` + provenance, not invent a parallel mechanism.
- **SOFT-depends #7 (sidecar resolution)** + **#9 (64-hex digest validation)** — R7 layers ABOVE the
  load-path tiers #7 reworks and reuses the digest validation #9 hardens; merge-collision avoidance
  on `safe_pickle.py` (R7 ideally imports its primitives, edits nothing in the tier code).
- **Env:** `NA0S_PICKLE_KEY` (HMAC tier) optional — keyless path MUST work (`.sha256`-signed
  manifest). New gate `NA0S_BUNDLE_MANIFEST=1` for staged rollout (mirror `NA0S_MODEL_PROVENANCE`).
  No new third-party dep (`hashlib`/`hmac`/`json` are stdlib; SBOM reuse adds nothing).
- **Worktree discipline:** verify symbols against THIS worktree / MAIN (`PYTHONPATH=<worktree>/src`),
  NOT the stale editable install pointing at the d8 primary checkout. Work in a dedicated git
  worktree; never branch-switch the primary checkout (Multi-Agent Worktree Discipline).
- **Keyless-correctness:** the user has only a Claude SUBSCRIPTION, no API key — R7 is fully local /
  keyless (HMAC is symmetric, no cloud call). No LLM-judge dependency.
- **LOCAL ONLY** — no push / PR / GitHub at any step.

## Definition of done
- [ ] #4 + #17 landed (HARD deps); their allowlist + attestation outputs are available as R7 inputs.
- [ ] `src/na0s/integrity/manifest.py` ships `build_manifest` / `write_manifest` / `verify_manifest`
      / `load_allowlist` / `load_attestation`; reuses `write_digest_sidecar` + `verify_file_digest`
      + `_format_sidecar` (no new on-disk signature format) and the SBOM model-section helper
      (no duplicate manifest format).
- [ ] Design-Q RESOLVED in code + comment: manifest is layered ABOVE the per-file hash chain; the
      `.sha256` tier is NOT promoted to HMAC (keyless contract `safe_pickle.py:30-33` preserved).
- [ ] Self-derived: per-file digests computed from disk and cross-checked vs `KNOWN_HASHES`
      (mismatch → fail closed); allowlist from #4; attestation from #17 + M14 corpus-snapshot.
- [ ] Verify hook wired into the SHARED model loader behind `_HAS_BUNDLE_MANIFEST` +
      `NA0S_BUNDLE_MANIFEST=1`, fail-closed; producer step added to `deploy_model.py`;
      predict/cascade parity via the shared loader (verified by grep).
- [ ] `tests/integrity/test_manifest.py`: round-trip, HMAC-tamper-byte, tamper-referenced-file,
      missing/extra-file, keyless-no-escalation, design-Q-additive, allowlist-binding — each proven
      to go red when its target is broken (na0s-review-checklist §4).
- [ ] CLI smoke run against the REAL bundled `.pkl`s (build + verify) pasted; full
      `pytest tests/ -q --tb=line` green, zero regressions.
- [ ] No magic threshold / similarity cutoff introduced (only reused, named hash invariants).
- [ ] No top-level shim; `integrity/__init__` exports consistent; SBOM not forked;
      `.meta.json`/`ModelProvenance` reconciliation coordinated with Item 17.
- [ ] Roadmap: R7 line added to the L11 open-items block (:1379-1397), M14 (:1797) cross-linked;
      README/SECURITY.md/MODEL_PROVENANCE.md updated; docs-drift gate green (facts re-extracted).
- [ ] LOCAL-only throughout — no GitHub until the user authorizes merge.

## Skills to reload at execution (Step 1)
- `na0s-review-checklist` — inject §1 (hallucinated APIs — every `manifest.py` symbol is NEW, verify
  the reused `safe_pickle`/`sbom` calls), §2 (imports), §4 (hollow tests — prove the manifest tamper
  tests go red; real-bundle CLI smoke, not a mock), §6 (fail-closed error handling), §7 (confirm R7
  introduces NO new threshold), §11 (smoke-first-wire-second; no duplicate manifest format) into
  every subagent.
- `subsystem-context-pack` — pack `src/na0s/integrity/**` + `src/na0s/models/**` + `tests/integrity/**`
  + `tests/test_safe_pickle*.py` (`--compress`, markdown) for each auditor's bounded context.
- N/A skills: `data-harvesting`, `eval-scenario-curation`, `incident-to-scenario`,
  `detector-authoring`, `eval-harness`, `cron-scheduling` (no harvest / scenario / detector / eval /
  cron surface — R7 is bundle signing).

## Agents to assign
- `layer-9-11-auditor` — owns `integrity/manifest.py`, the trust-tier layering decision, and the
  SBOM/`ModelProvenance` reconciliation (L11 supply-chain is R7's home).
- `l3-l5-code-auditor` — owns the shared-loader verify wiring (`predict.py`/`predict_embedding.py`)
  and the `deploy_model.py` producer step.
- `silent-failure-hunter` — prove every fail-closed branch raises (HMAC fail, digest drift,
  missing/extra file, keyless-lone-manifest) and the env gate can't silently disable verification
  once a manifest is present.
- (No `github-pr-*` agent — LOCAL only; R7 ships on its own worktree branch, merge gated on the user.)
