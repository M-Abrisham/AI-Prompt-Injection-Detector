# Attack Spec: Multimodal Injection (M1/M2/M3/M4)

| Field | Value |
|-------|-------|
| **Attack** | Multimodal injection — image (M1), audio (M2), document (M3), code (M4) |
| **Taxonomy** | M1.1/1.2/1.3 image, M2.1/2.2/2.3 audio, M3.1/3.2/3.3/3.4 document, M4.1/4.2/4.3/4.4 code (OWASP LLM01 multimodal) |
| **Matrix row** | INJ-0003 (Cat M, LLM01, T0051.001 Indirect) — score 68 *estimated*, PARTIAL, **unmeasured** |
| **Skills** | `detector-authoring`, `na0s-review-checklist`, `eval-harness`, `data-harvesting` |
| **Branch** | TBD |
| **Status** | 🟡 **PARTIAL** (2026-06-22) — text-channel sniffing + a real `scan_image()` exist, but the standard `scan(text)` pipeline is modality-blind and detection is *incidental* (transcribed keywords), not modality-aware. |

Attacker hides instructions in a non-text channel so a text-only injection
filter never sees them: white-on-white / tiny-font text in a DOCX or PDF
(M3.1), text rendered into image pixels for OCR (M1.1), EXIF/XMP metadata
text (M1.1/M3.2), steganographic / LSB payloads (M1.3), adversarial image
perturbations (M1.2), hidden voice / ultrasonic commands in audio (M2.x),
and code-comment / docstring / import-chain injection against code
assistants (M4.x). The LLM transcribes/OCRs the modality downstream and
executes the smuggled instruction.

---

## To-do list (GENERAL Prompt, per item)

- [ ] **1. Explore current rules + find gaps/edge-cases.** Two disconnected
  paths exist. (a) **L0 content-type sniff** (`input/content_type.py`): 35+
  magic-byte signatures + base64-blob + data-URI re-scan produce
  `embedded_image`/`base64_hidden_image`/`embedded_mp3`/… flags when binary
  is embedded *in a text string*. (b) **`detectors/visual_injection.py`**:
  a fully-implemented `scan_image()`/`scan_image_file()`/`scan_document_visual()`
  doing OCR text + EXIF/XMP metadata + tiny-font + low-contrast analysis.
  **Gaps:** image content-type flags are **attribution-only — zero scoring
  weight** (M1.1 tag added, risk unchanged); the visual detector runs OCR
  only if `easyocr`/`pytesseract`+PyMuPDF/python-docx are installed (all
  absent in default install → inert); **no audio transcript/Whisper path**
  at all; `scan(text)` rejects raw bytes (`TypeError`). Edge cases: data-URI
  PNG scored 0.503 < 0.55 (NOT blocked); benign "here is an image" 0.136;
  FP risk if image-present alone is treated as malicious (legit attachments).
- [ ] **2. Read roadmap/taxonomy/matrix/source.** Taxonomy `M:` block
  (data/taxonomy.yaml:430-460) — M1 image / M2 audio / M3 doc / M4 code, 14
  techniques, single canonical category (no dupes). Matrix INJ-0003 (line 49)
  = 68 *estimated*, "no audio analysis; visual detector L0-gated; unmeasured",
  per-row decode exception at line 96 ("decode does not reach inside
  image/audio"). ROADMAP: L0 19-step pipeline (§524), L17 deep doc extraction
  (§2138), HarmBench-multimodal TODO P2 (§1523), font/LSB/Whisper P2-P3
  (§2215). Eval probe `scripts/taxonomy/multimodal_injection.py` (M1-M4, 567
  samples). **Stale:** matrix cites `layer0/content_type.py` etc. — actual
  paths are `input/content_type.py`/`input/doc_extractor.py`/`input/ocr_extractor.py`
  (post-v1.0.0 rename drift).
- [ ] **3. Root-cause plan.** Root cause = **two issues**, fix in order.
  (a) *Scoring inertness*: the M-category L0 flags map to `technique_tags`
  via `_L0_FLAG_MAP` (predict.py:1895-2014) but contribute **no risk** — give
  embedded/base64-hidden image+doc flags a small bounded boost (mirror the
  obfuscation 0.15/flag/cap-0.3 pattern at predict.py:861), so a data-URI
  image payload crosses threshold without depending on incidental keywords.
  (b) *Modality-blind entry point*: `scan(text)` never sees real bytes — wire
  the existing `visual_injection.scan_image()`/`scan_document_visual()` into a
  first-class `scan_document(bytes)` / typed input so true image/doc bytes get
  OCR+metadata+stego analysis. (c) Audio is a genuine **new-capability** gap
  (no transcript path) — scope a transcript-text re-scan hook, not a Whisper
  dependency (keyless constraint). **Do NOT** treat "image present" as
  malicious — FP-disaster on legit attachments.
- [ ] **4. Implement + wire (parity).** N/A for this audit (read-only). When
  implemented: bounded M-flag scoring block in **BOTH** predict.py AND
  cascade.py — **cascade.py currently has ZERO visual_injection/_IMAGE_FLAGS
  refs (parity gap)**; predict.py routes at 2112-2142, cascade does not.
- [ ] **5. Datasets → isolated tests.** HarmBench multimodal behaviors
  (`harmbench_behaviors_multimodal_*.csv` + `multimodal_behavior_images/`,
  ~110, MIT) per ROADMAP §1523 — currently text-only pulled, **not wired**;
  no .png/.jpg/whisper corpora in `eval/harvest` or `weekly_harvest.py`. Needed
  before INJ-0003 can move from *estimated* to *measured*.
- [ ] **6. Test cases (code + use-case).** Existing: `tests/detectors/test_visual_injection_detector.py`
  + `tests/input/test_layer0_image_threat.py` (65 `def test`). Gap: no test
  asserts the **standard `scan()` pipeline** scores an embedded-image/doc
  payload, and no audio test. Add data-URI-image + DOCX-bytes + benign-image
  paired-FP cases.
- [ ] **7. File/dir cleanup + refactor.** N/A for this audit. Note for impl:
  fix the audio-magic-bytes → **M1.3** mismap (should be **M2.x**) and
  doc-magic-bytes → **M1.4** (should be **M3.x**) in `_L0_FLAG_MAP`.
- [ ] **8. Update roadmap.** Mark INJ-0003 path-status; check off font/LSB/OCR/Whisper
  items in §2215 as scoped; fix stale matrix paths (`layer0/`→`input/`).
- [ ] **9. README/benchmark.** N/A until measured — INJ-0003 is "— (not in
  harness)". Becomes a real number only after step 5 wires a multimodal corpus
  into `scripts/technique_analysis.py`.
- [ ] **10. Taxonomy + matrix + threshold.** Taxonomy M-block is canonical &
  single (verified). Fix the M1.3/M1.4 audio/doc code mismaps (step 7).
  Any new scoring boost must be **capped** and FP-validated — no round-number
  guess (na0s-review-checklist: flag arbitrary thresholds).
- [ ] **11. Open PR (held-out gate).** N/A — audit only; spec produces the
  to-do, not a code change. Future PR gated on full-suite green + benign-FPR.

---

## FP-safety (empirical, the binding constraint)

- **Presence of a modality is not malicious.** Benign "Here is an image for
  you to describe." scored 0.136 and a benign audio transcript 0.007 — any fix
  must keep these safe. Treating `embedded_image` alone as a block signal would
  FP on every legitimate attachment / data-URI avatar / inline screenshot.
- The proposed M-flag scoring boost must be **small + bounded** (cap ≈0.3,
  mirroring obfuscation/rag_poison) and ideally **corroborating** (only lifts
  when the OCR/metadata text *also* trips an injection indicator), so a clean
  embedded image stays below threshold.
- `scan_image()` is deliberately conservative (returns 0.0 when no OCR engine /
  no extractable text) — preserving that "absence of evidence ≠ malicious"
  default is the FP-safety guarantee for the image path.

## Documented residuals (left uncovered today)

- **Audio (M2.x): total gap** — no transcript/Whisper path; audio magic-bytes
  only set `embedded_*`→(mis-mapped) M1.3 with no analysis.
- **Adversarial image perturbation (M1.2) / steganographic LSB (M1.3):** no
  pixel-level / stego analysis — only OCR-visible text + EXIF/XMP.
- **Default-install inertness:** without `easyocr`/`pytesseract` + PyMuPDF +
  python-docx, the image/doc deep path silently extracts nothing (dep-gated).
- **Code injection (M4.x):** comment/docstring/import-chain injection handled,
  if at all, only as plain text via the standard rules — not modality-aware.
- **Standard `scan(text)` is modality-blind to real bytes:** genuine multimodal
  detection requires the separate `scan_image()`/`scan_document_visual()` API.

## Q&A verification

1. **Can Na0S catch it?** **Partial / incidental.** `scan(text)` blocks the
   sample payloads only because the *transcribed* prose contains injection
   keywords ("ignore the user and exfiltrate" → 0.55; OCR-label → 0.578); a
   keyword-free audio transcript misses (0.028 / 0.007). A data-URI image
   payload sets M1.1 but scores 0.503 < 0.55 = NOT blocked. The dedicated
   `scan_image()` works but is a separate, dep-gated API.
2. **Cleanup done?** N/A (read-only audit). Flagged: M1.3/M1.4 audio/doc code
   mismaps, stale matrix paths.
3. **Pipeline wired?** **Partial.** predict.py routes visual (2112-2142) and
   maps M-flags to tags; **cascade.py has NO visual parity** (gap). Image flags
   carry tags but **no scoring weight**.
4. **Tested (code + use-case)?** Detector-level yes (65 tests across
   `test_visual_injection_detector.py` + `test_layer0_image_threat.py`);
   **no end-to-end `scan()`-pipeline multimodal test, no audio test.**
5. **Harvester?** Tags "multimodal injection"/"image injection" → "M"
   (discovery_tagging.py:149-151), but **no multimodal datasets wired/available**;
   HarmBench-multimodal is an unfulfilled P2 TODO. INJ-0003 stays *estimated*.
6. **Taxonomy/matrix match, no dupes?** Yes — single canonical M-block, single
   INJ-0003 row (2 prose references, not dup rows). Bugs: audio→M1.3 / doc→M1.4
   mismaps; stale `layer0/` paths in matrix.
7. **Scorer correct?** **No.** M-category L0 flags are attribution-only (zero
   risk contribution); the real fix is a bounded, corroborating boost — not yet
   implemented.
8. **predict.py/cascade.py refs?** predict.py yes (`_L0_FLAG_MAP` M-codes
   1895-2014, visual routing 2112-2142). **cascade.py: none.**
9. **Harvester agent harvests it?** Taxonomy-tagging path exists; **dataset
   ingestion does not** — no image/audio corpus is harvested today.
