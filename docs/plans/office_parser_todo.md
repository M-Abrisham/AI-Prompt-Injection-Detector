# Na0S Office Parser -- Multi-Day TODO Plan

> **Status**: Days 1-7 COMPLETE. Days 8-10 remain.
> **Open Questions**: See [Section 11](#11-open-questions) -- 4 items require decisions before Day 8 begins.
> **Last updated**: 2026-04-11

---

## 1. Context Snapshot

### Files Read

| File | Lines | Summary |
|------|------:|---------|
| `src/na0s/predict.py` | 1784 | Unified `scan()` entry point; 21-layer classification pipeline; returns `ScanResult` |
| `src/na0s/cascade.py` | 1483 | `CascadeClassifier.scan()` -- whitelist + weighted voting + LLM judge; returns `ScanResult` |
| `src/na0s/layer1/rules_registry.py` | ~800 | 117 pre-compiled regex rules; `classify_prompt()` delegates to `rule_score_detailed()` |
| `src/na0s/scan_result.py` | 48 | `ScanResult` dataclass -- label, risk_score, technique_tags, rule_hits, anomaly_flags, etc. |
| `src/na0s/__init__.py` | 50 | Public API: `scan()`, `CascadeClassifier`, `scan_output()`, `scan_image()` |
| `src/na0s/layer0/doc_extractor.py` | 673 | Naive doc extraction (PDF/DOCX/XLSX/PPTX/RTF) via 3rd-party libs; returns `DocResult` with flat text blob |
| `src/na0s/parsers/__init__.py` | -- | Empty package marker |
| `src/na0s/parsers/office/__init__.py` | 32 | Re-exports `extract()`, `detect_format()`, `ExtractedArtifact`, `UnsupportedDocumentError` |
| `src/na0s/parsers/office/base.py` | 281 | `ExtractedArtifact` dataclass, `OfficeExtractor` ABC, `detect_format()`, zip-bomb guards |
| `src/na0s/parsers/office/router.py` | 106 | `extract(data: bytes) -> list[ExtractedArtifact]` -- magic-byte dispatch to format extractors |
| `src/na0s/parsers/office/docx_extractor.py` | 519 | 19 hiding spots: body, comments, tracked changes, hidden text, headers/footers, custom XML, etc. |
| `src/na0s/parsers/office/xlsx_extractor.py` | 679 | Shared strings, inline strings, comments, hidden/veryHidden sheets, defined names, formulas, hyperlinks |
| `src/na0s/parsers/office/pptx_extractor.py` | 611 | Slides, speaker notes, alt text, legacy+modern comments, masters, layouts, tags, custom properties |
| `src/na0s/parsers/office/odf_extractor.py` | 860 | ODT/ODS/ODP: body, tracked changes, annotations, hidden text/sections/sheets/slides, metadata, scripts |
| `src/na0s/parsers/office/ole_extractor.py` | 535 | Legacy .doc/.xls/.ppt: 3-tier (summary metadata, VBA macros, raw string fallback) |
| `tests/parsers/office/test_docx.py` | 129 | 4 fixture-based test classes (comment, tracked change, custom property, clean) |
| `tests/parsers/office/test_xlsx.py` | 166 | 4 fixture-based test classes (comment, defined name, hidden sheet, clean) |
| `tests/parsers/office/test_pptx.py` | 139 | 4 fixture-based test classes (notes, alt text, hidden slide, clean) |
| `tests/parsers/office/test_odf.py` | 139 | 4 fixture-based test classes (annotation, hidden text, metadata, clean) |
| `tests/parsers/office/test_router.py` | 122 | Format detection + round-trip routing for all fixtures; `.pages` rejection test |
| `tests/fixtures/office/` | -- | 16 binary fixtures (4 DOCX, 4 XLSX, 4 PPTX, 4 ODF) + 4 builder scripts |
| `docs/research/hiding_spots_docx.md` | -- | 19 DOCX hiding spots inventory |
| `docs/research/hiding_spots_xlsx.md` | -- | 17 XLSX hiding spots inventory |
| `docs/research/hiding_spots_pptx.md` | -- | 18 PPTX hiding spots inventory |
| `docs/research/hiding_spots_odf.md` | -- | 21 ODF hiding spots inventory |
| `docs/research/hiding_spots_ole.md` | -- | OLE legacy format hiding spots inventory |
| `.github/workflows/ci.yml` | 97 | CI: Python 3.9-3.12 matrix, lint, compile check, pytest + coverage (>=50%), bench |
| `.github/workflows/pr-check.yml` | 67 | PR gate: syntax, lint, full test suite + coverage |
| `pyproject.toml` | 128 | Python >=3.9, deps: scikit-learn, numpy, tiktoken, chardet, ftfy; `[docs]` extra has pymupdf/python-docx/openpyxl/python-pptx |
| `README.md` | 332 | Project overview, 17-layer pipeline, taxonomy, CLI usage |
| `docs/ARCHITECTURE.md` | ~80+ | Mermaid flowchart of 15-layer pipeline |
| `CLAUDE.md` | -- | Agent conventions: max 4 parallel agents, always run full test suite, never weaken assertions |

### Current State

- **Tests currently failing**: All 5 test files in `tests/parsers/office/` fail with `ModuleNotFoundError: No module named 'na0s'` -- the package is not installed in the active Python 3.9 environment. This is an environment issue, not a code bug. When installed (`pip install -e .`), the tests should pass.
- **Implemented**: scaffold, base.py, router.py, all 5 extractors (DOCX/XLSX/PPTX/ODF/OLE), 5 research inventories, 5 test suites + router tests, 16 binary fixtures + 4 builders
- **Not implemented**: pipeline integration (`predict.scan()` / `cascade.scan()`), hardening edge cases, OLE test fixtures, end-to-end integration tests, documentation updates, coverage report, CI verification

---

## 2. Goal and Definition of Done

The Na0S Office Parser extracts text from every user-controllable surface in Office documents (DOCX, XLSX, PPTX, ODT/ODS/ODP, legacy OLE) and feeds each extracted artifact through the Na0S prompt injection detection pipeline. The goal is to close the "document smuggling" attack vector where adversaries hide prompt injections in comments, tracked changes, hidden sheets, speaker notes, metadata, and other surfaces that naive text extractors miss.

### Acceptance Criteria

1. `python3 -m pytest tests/parsers/office/ -q` passes with >= 66 tests and zero failures (current count preserved or increased)
2. A new function `scan_document(data: bytes) -> DocumentScanResult` exists in `src/na0s/parsers/office/integration.py` that accepts raw document bytes, extracts all artifacts via `router.extract()`, feeds each artifact's `.text` through `predict.scan()`, and returns an aggregated result
3. `DocumentScanResult` contains: `artifacts: list[ArtifactScanResult]`, `is_malicious: bool` (True if any artifact is malicious), `risk_score: float` (max artifact risk), `format: str`, `artifact_count: int`, `malicious_artifact_count: int`
4. `ArtifactScanResult` contains: `location: str` (from `ExtractedArtifact.location`), `scan_result: ScanResult`, and the user can see WHERE in the document the injection was found
5. At least 4 end-to-end integration tests exist in `tests/parsers/office/test_integration.py` that build a document with a known injection payload, run it through `scan_document()`, and verify `is_malicious=True` with the correct `location` tag
6. Zip-bomb edge case: a fixture with >10,000 ZIP entries returns zero artifacts (not a crash)
7. Malformed XML edge case: a DOCX with corrupt `word/document.xml` returns empty artifacts for that part (not a crash), other parts still extracted
8. Encrypted OLE edge case: `UnsupportedDocumentError` is raised with a clear message
9. `python3 -m pytest tests/ -q --tb=line` passes with zero regressions (full suite)
10. Coverage for `src/na0s/parsers/office/` is >= 80% (measured by `coverage run -m pytest tests/parsers/office/ && coverage report`)
11. `pyproject.toml` `[docs]` extra includes `olefile` for OLE support (currently missing)
12. CI (`ci.yml`) runs office parser tests without failure on Python 3.9-3.12

---

## 3. Architecture Decision Record

### 3.1 Directory Layout (DECIDED)

```
src/na0s/parsers/
    __init__.py                     # empty package
    office/
        __init__.py                 # re-exports extract(), detect_format(), etc.
        base.py                     # ExtractedArtifact, OfficeExtractor ABC, detect_format()
        router.py                   # extract(data) -> list[ExtractedArtifact]
        docx_extractor.py           # OOXML WordprocessingML
        xlsx_extractor.py           # OOXML SpreadsheetML
        pptx_extractor.py           # OOXML PresentationML
        odf_extractor.py            # ODF (ODT, ODS, ODP)
        ole_extractor.py            # OLE Compound Binary (legacy .doc/.xls/.ppt)
        integration.py              # NEW: scan_document() pipeline wiring
```

Tests mirror this structure:

```
tests/parsers/office/
    __init__.py
    test_docx.py                    # 129 lines, fixture-based
    test_xlsx.py                    # 166 lines, fixture-based
    test_pptx.py                    # 139 lines, fixture-based
    test_odf.py                     # 139 lines, fixture-based
    test_router.py                  # 122 lines, format detection + routing
    test_integration.py             # NEW: end-to-end pipeline tests
```

Fixtures:

```
tests/fixtures/office/
    _builders/
        build_docx.py, build_xlsx.py, build_pptx.py, build_odf.py
    docx/ (4 files), xlsx/ (4 files), pptx/ (4 files), odf/ (4 files)
```

### 3.2 ExtractedArtifact Schema (DECIDED)

```python
@dataclass
class ExtractedArtifact:
    location: str               # e.g. "docx:comments/comment[3]"
    text: str                   # extracted text content
    metadata: Dict[str, str]    # optional key-value pairs (author, date, etc.)
```

Location tag format: `{format}:{part}/{element}[{index}]`
- DOCX: `docx:body`, `docx:comments/comment[0]`, `docx:document/hidden-text`
- XLSX: `xlsx:sheet1/A1`, `xlsx:sheet2[veryHidden]/B3`, `xlsx:definedNames/_secret`
- PPTX: `pptx:slide1/notes`, `pptx:slide2[hidden]/text`, `pptx:modernComment[1]`
- ODF:  `odt:annotation[1]`, `ods:Sheet1[hidden]/A1`, `odp:slide1/notes`
- OLE:  `ole:summary/title`, `ole:vba/Module1`, `ole:ppt/slide_text[1]`

### 3.3 Magic-Byte Detection Table (DECIDED)

| Magic Bytes | Format | Extractor Module |
|-------------|--------|-----------------|
| `\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1` | `ole` | `ole_extractor` |
| `PK\x03\x04` + `word/` in ZIP | `docx` | `docx_extractor` |
| `PK\x03\x04` + `xl/` in ZIP | `xlsx` | `xlsx_extractor` |
| `PK\x03\x04` + `ppt/` in ZIP | `pptx` | `pptx_extractor` |
| `PK\x03\x04` + `mimetype` = `...opendocument.text` | `odt` | `odf_extractor` |
| `PK\x03\x04` + `mimetype` = `...opendocument.spreadsheet` | `ods` | `odf_extractor` |
| `PK\x03\x04` + `mimetype` = `...opendocument.presentation` | `odp` | `odf_extractor` |
| `PK\x03\x04` + `Index/` paths | `.pages` | REJECTED: `UnsupportedDocumentError` |

### 3.4 Error Taxonomy

| Error | When | Behavior |
|-------|------|----------|
| `UnsupportedDocumentError` | `.pages` file, encrypted OLE | Raised to caller; caller decides to block or warn |
| `ValueError` | Unknown magic bytes | Raised by `router.extract()` |
| `ImportError` | `olefile` not installed for OLE parsing | Raised by `ole_extractor.Extractor.extract()` |
| `zipfile.BadZipFile` | Corrupt ZIP | Caught inside extractor; returns `[]` |
| `ET.ParseError` | Malformed XML inside ZIP | Caught inside extractor per-entry; returns `[]` for that entry, continues others |
| Zip-bomb detection | >10,000 entries or >200MB decompressed | Caught by `validate_zip_safety()`; returns `[]` with warning log |

### 3.5 Integration Point: How Office Parser Connects to the Pipeline

The office parser lives OUTSIDE the existing 21-layer pipeline. It does NOT modify `predict.scan()` or `cascade.scan()`. Instead, it provides a standalone `scan_document()` function that:

1. Accepts raw document bytes
2. Calls `router.extract(data)` to get `list[ExtractedArtifact]`
3. For each artifact, calls `predict.scan(artifact.text)` to get a `ScanResult`
4. Wraps each `(artifact, ScanResult)` pair into an `ArtifactScanResult`
5. Aggregates all artifact-level results into a `DocumentScanResult`

**Data flow**:

```
raw bytes
    |
    v
router.extract(data) --> list[ExtractedArtifact]
    |
    v  (for each artifact)
predict.scan(artifact.text) --> ScanResult
    |
    v
ArtifactScanResult(location=artifact.location, scan_result=ScanResult)
    |
    v
DocumentScanResult(
    artifacts=[ArtifactScanResult, ...],
    is_malicious=any(a.scan_result.is_malicious for a in artifacts),
    risk_score=max(a.scan_result.risk_score for a in artifacts),
    format="docx",
    artifact_count=len(artifacts),
    malicious_artifact_count=sum(1 for a if malicious),
)
```

**Why not modify `predict.scan()`?**

- `predict.scan()` accepts `text: str`. Office documents are `bytes`. Mixing binary dispatch into the text pipeline adds complexity and regression risk to the 21-layer system.
- The office parser produces MULTIPLE artifacts from a single document. The existing `scan()` returns a single `ScanResult`. Forcing multi-artifact semantics into the existing return type would break every downstream consumer.
- The office parser is a PRE-PROCESSING step. It decomposes a document into text surfaces, then feeds each surface into the existing pipeline unchanged.

**Integration surface for consumers**:

```python
from na0s.parsers.office.integration import scan_document

result = scan_document(open("suspect.docx", "rb").read())
if result.is_malicious:
    for artifact in result.artifacts:
        if artifact.scan_result.is_malicious:
            print(f"Injection at {artifact.location}: {artifact.scan_result.rule_hits}")
```

**Relationship to `layer0/doc_extractor.py`**:

The existing `doc_extractor.py` is a NAIVE extractor -- it uses `python-docx`, `openpyxl`, `python-pptx` to extract visible body text only. It does NOT extract comments, tracked changes, hidden sheets, speaker notes, metadata, or any of the other ~70 hiding spots covered by the office parser. The office parser is a DEEP extractor that replaces `doc_extractor.py` for security-sensitive use cases. The two systems coexist:

- `doc_extractor.extract_text_from_document()` -- fast, simple, visible text only. Used by L0 for basic document text extraction.
- `scan_document()` -- thorough, extracts ALL hiding spots. Used when the operator wants full document security scanning.

---

## 4. Day-by-Day Breakdown

### Days 1-2: Research and Base Classes [x] COMPLETE

- [x] Research hiding spots for all 5 format families
- [x] Write `docs/research/hiding_spots_docx.md` (19 spots)
- [x] Write `docs/research/hiding_spots_xlsx.md` (17 spots)
- [x] Write `docs/research/hiding_spots_pptx.md` (18 spots)
- [x] Write `docs/research/hiding_spots_odf.md` (21 spots)
- [x] Write `docs/research/hiding_spots_ole.md`
- [x] Design `ExtractedArtifact` dataclass
- [x] Design `OfficeExtractor` ABC
- [x] Implement `detect_format()` with magic bytes
- [x] Implement zip-bomb safety utilities (`validate_zip_safety`, `safe_read_zip_entry`)
- [x] Implement `UnsupportedDocumentError` for `.pages` rejection

**Delivered**: `base.py` (281 lines), 5 research inventories

### Days 3-4: OOXML Extractors [x] COMPLETE

- [x] DOCX extractor: body, comments, tracked changes, hidden text, headers/footers, footnotes/endnotes, core/custom/app properties, custom XML, hyperlinks, field codes, SDTs, smart tags, white/tiny text
- [x] XLSX extractor: shared strings, inline strings, cell formulas, hidden/veryHidden sheets, comments, defined names, data validation, header/footer strings, hyperlinks, core/custom properties
- [x] PPTX extractor: slide text, speaker notes, alt text, legacy+modern comments, hidden slides, slide masters/layouts, core/custom properties, user-defined tags, hyperlink tooltips

**Delivered**: `docx_extractor.py` (519 lines), `xlsx_extractor.py` (679 lines), `pptx_extractor.py` (611 lines)

### Days 5-6: ODF and OLE Extractors [x] COMPLETE

- [x] ODF extractor: ODT (body, tracked changes, annotations, hidden text/sections/paragraphs, text boxes, footnotes/endnotes, headers/footers), ODS (cells, hidden sheets, annotations, named ranges), ODP (slides, hidden slides, speaker notes), shared (metadata, scripts)
- [x] OLE extractor: 3-tier (summary metadata, VBA macros via oletools, raw string fallback); encryption detection; PPT slide text via binary record parsing; XLS sheet names via BoundSheet8 records; xlrd cell values

**Delivered**: `odf_extractor.py` (860 lines), `ole_extractor.py` (535 lines)

### Day 7: Router, Tests, Fixtures [x] COMPLETE

- [x] Router with lazy imports and format dispatch
- [x] 4 fixture builders (build_docx.py, build_xlsx.py, build_pptx.py, build_odf.py)
- [x] 16 binary fixtures (4 per format: clean + 3 injection variants)
- [x] Test suites: test_docx.py, test_xlsx.py, test_pptx.py, test_odf.py, test_router.py
- [x] `__init__.py` re-exports for clean public API

**Delivered**: `router.py` (106 lines), 695 lines of tests, 16 fixtures, 4 builders

**Gap found**: Tests fail in system Python 3.9 due to missing `pip install -e .` -- environment issue, not code bug. Must verify in CI.

---

### Day 8: Pipeline Integration and Wiring

**Goal**: Create `scan_document()` and `DocumentScanResult` that wire the office parser output into the Na0S detection pipeline, with end-to-end tests proving injections are detected.

**Preconditions**: All extractor tests pass (`pip install -e ".[dev]"` first).

**Tasks**:

- [ ] Create `src/na0s/parsers/office/integration.py`:
  - [ ] Define `ArtifactScanResult` dataclass: `location: str`, `text_preview: str` (first 200 chars), `scan_result: ScanResult`
  - [ ] Define `DocumentScanResult` dataclass: `format: str`, `artifact_count: int`, `malicious_artifact_count: int`, `is_malicious: bool`, `risk_score: float`, `artifacts: list[ArtifactScanResult]`, `errors: list[str]`
  - [ ] Implement `scan_document(data: bytes, threshold: float = None) -> DocumentScanResult`
  - [ ] Handle `UnsupportedDocumentError` -- return `DocumentScanResult` with `errors` populated
  - [ ] Handle `ValueError` (unknown format) -- return `DocumentScanResult` with `errors` populated
  - [ ] Add `max_artifacts` guard (default 500) to prevent resource exhaustion on maliciously crafted documents with thousands of artifacts
  - [ ] Add logging: format detected, artifact count, malicious count, elapsed time

- [ ] Update `src/na0s/parsers/office/__init__.py`:
  - [ ] Re-export `scan_document`, `DocumentScanResult`, `ArtifactScanResult`

- [ ] Create `tests/parsers/office/test_integration.py`:
  - [ ] `test_docx_comment_injection_detected`: build DOCX with injection in comment, verify `is_malicious=True` and location contains `comments`
  - [ ] `test_xlsx_hidden_sheet_injection_detected`: build XLSX with injection in hidden sheet, verify detection and location contains `hidden`
  - [ ] `test_pptx_notes_injection_detected`: build PPTX with injection in speaker notes, verify detection
  - [ ] `test_odf_annotation_injection_detected`: build ODT with injection in annotation, verify detection
  - [ ] `test_clean_document_safe`: build clean DOCX, verify `is_malicious=False`
  - [ ] `test_unknown_format_returns_error`: pass random bytes, verify `errors` list is non-empty
  - [ ] `test_unsupported_format_returns_error`: pass `.pages`-like bytes, verify `errors` list mentions `.pages`
  - [ ] `test_max_artifacts_guard`: mock `router.extract()` to return 1000 artifacts, verify only `max_artifacts` are scanned
  - [ ] `test_artifact_location_preserved`: verify the `location` field from the extractor appears in `ArtifactScanResult.location`

- [ ] Verify: `python3 -m pytest tests/parsers/office/ -v` passes (all old + new tests)

**Files created/modified**:
- CREATE: `src/na0s/parsers/office/integration.py`
- MODIFY: `src/na0s/parsers/office/__init__.py`
- CREATE: `tests/parsers/office/test_integration.py`

**Tests that must pass**:
- `tests/parsers/office/test_integration.py::test_docx_comment_injection_detected`
- `tests/parsers/office/test_integration.py::test_xlsx_hidden_sheet_injection_detected`
- `tests/parsers/office/test_integration.py::test_clean_document_safe`
- `tests/parsers/office/test_integration.py::test_unknown_format_returns_error`
- All 5 existing test files continue to pass

**Commit checkpoint**: `feat(office): add scan_document() integration wiring with DocumentScanResult`

**Rollback**: `git checkout HEAD -- src/na0s/parsers/office/__init__.py && git rm src/na0s/parsers/office/integration.py tests/parsers/office/test_integration.py`

---

### Day 9: Hardening Pass

**Goal**: Add edge-case tests and fix any bugs found for zip bombs, malformed XML, encrypted OLE, and defensive coding gaps.

**Preconditions**: Day 8 complete. `scan_document()` exists and works for happy path.

**Tasks**:

- [ ] Create OLE test fixtures and tests:
  - [ ] Create `tests/fixtures/office/_builders/build_ole.py` that generates a minimal OLE file with metadata (requires `olefile`)
  - [ ] Create `tests/parsers/office/test_ole.py`:
    - [ ] `test_ole_summary_extraction`: verify summary metadata is extracted
    - [ ] `test_ole_encrypted_raises`: verify `UnsupportedDocumentError` for encrypted OLE
    - [ ] `test_ole_missing_olefile_raises`: mock `_HAS_OLEFILE = False`, verify `UnsupportedDocumentError`
    - [ ] `test_ole_vba_detection`: verify VBA storage detection (without oletools)
    - [ ] `test_ole_stream_listing`: verify stream names are captured

- [ ] Zip-bomb hardening tests (in `tests/parsers/office/test_hardening.py`):
  - [ ] `test_zipbomb_many_entries`: create ZIP with >10,000 empty entries, verify `extract()` returns `[]` without crashing
  - [ ] `test_zipbomb_large_decompressed`: create ZIP with entries totaling >200MB decompressed size (via spoofed `file_size`), verify `[]` returned
  - [ ] `test_corrupt_zip_returns_empty`: pass truncated ZIP bytes, verify `[]` returned
  - [ ] `test_single_entry_exceeds_max_xml_bytes`: create ZIP with one entry >50MB, verify `safe_read_zip_entry()` returns `None`

- [ ] Malformed XML hardening tests (in `test_hardening.py`):
  - [ ] `test_docx_corrupt_document_xml`: create valid ZIP with `word/document.xml` containing invalid XML, verify body extraction fails gracefully but comments still work
  - [ ] `test_xlsx_corrupt_shared_strings`: create valid XLSX ZIP with corrupt `xl/sharedStrings.xml`, verify extraction continues
  - [ ] `test_billion_laughs_safe`: verify XML parsing does not expand entity references (stdlib `ET.fromstring` is safe by default)

- [ ] Encrypted document tests (in `test_hardening.py`):
  - [ ] `test_encrypted_ole_raises_clear_error`: verify `UnsupportedDocumentError` message includes "decrypt before scanning"
  - [ ] `test_password_protected_ooxml`: verify that a password-protected OOXML file (which is actually an OLE container) is detected as encrypted

- [ ] Review all extractors for:
  - [ ] Every `_parse_xml` / `_safe_parse_xml` call has a `None` check
  - [ ] Every `safe_read_zip_entry` call handles `None` return
  - [ ] No unbounded loops over ZIP entries (all are bounded by `MAX_ZIP_ENTRIES` check)

**Files created/modified**:
- CREATE: `tests/parsers/office/test_ole.py`
- CREATE: `tests/parsers/office/test_hardening.py`
- CREATE: `tests/fixtures/office/_builders/build_ole.py` (if olefile available, else skip)

**Tests that must pass**:
- All of `tests/parsers/office/` including new hardening and OLE tests
- `python3 -m pytest tests/ -q --tb=line` -- full suite, zero regressions

**Commit checkpoint**: `test(office): add hardening tests for zip bombs, malformed XML, encrypted OLE`

**Rollback**: `git rm tests/parsers/office/test_hardening.py tests/parsers/office/test_ole.py`

---

### Day 10: Documentation, Coverage, Closure

**Goal**: Update pyproject.toml, verify CI, measure coverage, update ARCHITECTURE.md reference, and close out IM0003.

**Preconditions**: Days 8-9 complete. All tests pass.

**Tasks**:

- [ ] Update `pyproject.toml`:
  - [ ] Add `olefile>=0.46,<1` to `[project.optional-dependencies] docs`
  - [ ] Add `oletools>=0.60,<1` to `[project.optional-dependencies] docs` (optional, for VBA extraction)
  - [ ] Add `xlrd>=2.0,<3` to `[project.optional-dependencies] docs` (optional, for legacy XLS cell values)

- [ ] Verify CI compatibility:
  - [ ] Run `python3 -m pytest tests/ -q --tb=line` locally on Python 3.9
  - [ ] Confirm tests handle missing optional deps gracefully (olefile, oletools, xlrd)
  - [ ] Ensure binary fixtures in `tests/fixtures/office/` are committed to git (not gitignored)
  - [ ] Check that `.github/workflows/ci.yml` `pip install -e ".[dev]"` pulls in `[docs]` deps (it does -- `dev` includes `all` which includes `docs`)

- [ ] Coverage measurement:
  - [ ] Run `coverage run -m pytest tests/parsers/office/ && coverage report --include="src/na0s/parsers/office/*"`
  - [ ] Target >= 80% for all files in `src/na0s/parsers/office/`
  - [ ] Identify any uncovered branches and add targeted tests if below 80%

- [ ] Documentation:
  - [ ] Add a brief section to `README.md` under "How It Works" or a new "Document Scanning" section showing `scan_document()` usage
  - [ ] Add `scan_document` to `src/na0s/__init__.py` public API (conditional import with `try/except ImportError`)

- [ ] Smoke test the full pipeline:
  - [ ] Build a DOCX with "Ignore all previous instructions" in a comment
  - [ ] Run `scan_document()` on it
  - [ ] Verify `is_malicious=True`, `location` contains `comments`, `rule_hits` contains `override`

- [ ] Final full test suite run:
  - [ ] `python3 -m pytest tests/ -q --tb=line` -- zero failures

**Files created/modified**:
- MODIFY: `pyproject.toml`
- MODIFY: `README.md` (add document scanning section)
- MODIFY: `src/na0s/__init__.py` (conditional re-export of `scan_document`)

**Commit checkpoint**: `docs(office): update pyproject deps, README usage, public API export`

**Rollback**: `git checkout HEAD -- pyproject.toml README.md src/na0s/__init__.py`

---

## 5. Task Dependency Graph

```
Days 1-7 [DONE] ──────────────────────────────────────────────┐
  base.py, router.py, 5 extractors, 5 test suites,           |
  16 fixtures, 4 builders, 5 research inventories             |
                                                              v
Day 8: Integration Wiring ────────────────────────────────────┐
  integration.py (scan_document, DocumentScanResult)          |
  test_integration.py (9 tests)                               |
  __init__.py update                                          |
                                                              v
Day 9: Hardening Pass ────────────────────────────────────────┐
  test_hardening.py (zip bomb, malformed XML, encrypted)      |
  test_ole.py (5+ tests)                                      |
  build_ole.py fixture builder                                |
  Code review of all extractors                               |
                                                              v
Day 10: Closure ──────────────────────────────────────────────┘
  pyproject.toml deps
  README.md docs
  __init__.py public API
  Coverage >= 80%
  Full suite green
```

**Gate conditions**:
- Day 8 requires: `pip install -e ".[dev]"` succeeds, all Day 7 tests pass
- Day 9 requires: `scan_document()` exists and passes happy-path tests
- Day 10 requires: All hardening tests pass, no regressions in full suite

---

## 6. Agent / Team Assignments

### Agent A: Integration Engineer (Day 8)

**Scope**: Pipeline wiring only. Does NOT modify any extractor.

**Files touched**:
- `src/na0s/parsers/office/integration.py` (CREATE)
- `src/na0s/parsers/office/__init__.py` (MODIFY -- add re-exports)
- `tests/parsers/office/test_integration.py` (CREATE)

**Must NOT touch**: `predict.py`, `cascade.py`, `rules_registry.py`, any existing extractor file.

### Agent B: Hardening Engineer (Day 9)

**Scope**: Edge-case tests and OLE test coverage. May fix bugs in extractors if tests reveal them.

**Files touched**:
- `tests/parsers/office/test_hardening.py` (CREATE)
- `tests/parsers/office/test_ole.py` (CREATE)
- `tests/fixtures/office/_builders/build_ole.py` (CREATE)
- `src/na0s/parsers/office/ole_extractor.py` (MODIFY -- only if bugs found)
- `src/na0s/parsers/office/base.py` (MODIFY -- only if bugs found)

**Must NOT touch**: `integration.py`, `predict.py`, `cascade.py`.

### Agent C: Documentation and Closure (Day 10)

**Scope**: Config, docs, public API, coverage.

**Files touched**:
- `pyproject.toml` (MODIFY)
- `README.md` (MODIFY)
- `src/na0s/__init__.py` (MODIFY)

**Must NOT touch**: Any extractor, any test, `predict.py`, `cascade.py`.

---

## 7. Fixture Plan

### Existing Fixtures [x]

| # | Path | Format | Injection Type | Builder |
|---|------|--------|---------------|---------|
| 1 | `tests/fixtures/office/docx/clean.docx` | DOCX | None (baseline) | `build_docx.py` |
| 2 | `tests/fixtures/office/docx/comment_injection.docx` | DOCX | Comment with payload | `build_docx.py` |
| 3 | `tests/fixtures/office/docx/custom_property_injection.docx` | DOCX | Custom property with payload | `build_docx.py` |
| 4 | `tests/fixtures/office/docx/tracked_change_injection.docx` | DOCX | Tracked change with payload | `build_docx.py` |
| 5 | `tests/fixtures/office/xlsx/clean.xlsx` | XLSX | None (baseline) | `build_xlsx.py` |
| 6 | `tests/fixtures/office/xlsx/comment_injection.xlsx` | XLSX | Cell comment with payload | `build_xlsx.py` |
| 7 | `tests/fixtures/office/xlsx/defined_name_injection.xlsx` | XLSX | Defined name with payload | `build_xlsx.py` |
| 8 | `tests/fixtures/office/xlsx/hidden_sheet_injection.xlsx` | XLSX | Hidden sheet with payload | `build_xlsx.py` |
| 9 | `tests/fixtures/office/pptx/clean.pptx` | PPTX | None (baseline) | `build_pptx.py` |
| 10 | `tests/fixtures/office/pptx/notes_injection.pptx` | PPTX | Speaker notes with payload | `build_pptx.py` |
| 11 | `tests/fixtures/office/pptx/alt_text_injection.pptx` | PPTX | Alt text with payload | `build_pptx.py` |
| 12 | `tests/fixtures/office/pptx/hidden_slide_injection.pptx` | PPTX | Hidden slide with payload | `build_pptx.py` |
| 13 | `tests/fixtures/office/odf/clean.odt` | ODT | None (baseline) | `build_odf.py` |
| 14 | `tests/fixtures/office/odf/annotation_injection.odt` | ODT | Annotation with payload | `build_odf.py` |
| 15 | `tests/fixtures/office/odf/hidden_text_injection.odt` | ODT | Hidden text with payload | `build_odf.py` |
| 16 | `tests/fixtures/office/odf/metadata_injection.odt` | ODT | User-defined metadata with payload | `build_odf.py` |

### Planned Fixtures [ ]

| # | Path | Format | Purpose | Day |
|---|------|--------|---------|-----|
| 17 | (in-memory) | ZIP | Zip-bomb: >10,000 entries | 9 |
| 18 | (in-memory) | ZIP | Zip-bomb: >200MB decompressed | 9 |
| 19 | (in-memory) | DOCX | Corrupt `word/document.xml` | 9 |
| 20 | (in-memory) | XLSX | Corrupt `xl/sharedStrings.xml` | 9 |
| 21 | (in-memory) | OLE | Encrypted (DataSpaces stream) | 9 |
| 22 | (in-memory, via builder) | OLE | Clean with summary metadata | 9 |

Note: Hardening fixtures are generated in-memory in tests (not persisted as files) to avoid bloating the repo with binary artifacts. The OLE builder (`build_ole.py`) generates fixtures only if `olefile` is installed; tests skip otherwise.

---

## 8. Risk Register

| # | Risk | Likelihood | Impact | Mitigation | Owner-Day |
|---|------|-----------|--------|------------|-----------|
| R1 | Zip bomb crashes CI | Low | High | `validate_zip_safety()` already implemented; Day 9 adds explicit tests | B-9 |
| R2 | Encrypted OLE file causes unhandled exception | Medium | Medium | `_check_encryption()` already implemented; Day 9 adds test | B-9 |
| R3 | Malformed XML in OOXML causes `ET.ParseError` crash | Low | Medium | All extractors use `_parse_xml()` / `_safe_parse_xml()` wrappers that return `None`; Day 9 verifies | B-9 |
| R4 | `.pages` file bypasses rejection | Low | Low | `detect_format()` checks for `Index/` paths; test exists in `test_router.py` | -- (done) |
| R5 | CI binary fixture handling -- large fixtures bloat repo | Low | Low | Fixtures are 5-20KB each (16 files = ~200KB total); within acceptable limits | -- (done) |
| R6 | Integration `scan_document()` breaks existing pipeline | Low | High | `scan_document()` is additive -- it calls `predict.scan()` but does not modify it. No existing code paths change. | A-8 |
| R7 | `olefile` / `oletools` / `xlrd` not installed in CI | Medium | Medium | OLE tests must `pytest.importorskip("olefile")` or mock the dependency; `[dev]` includes `[docs]` but `[docs]` does NOT include `olefile` yet | B-9, C-10 |
| R8 | `predict.scan()` is slow -- calling it per-artifact on documents with 500+ artifacts causes timeout | Medium | High | `max_artifacts` guard (default 500) in `scan_document()`; consider pre-filtering empty artifacts before scanning | A-8 |
| R9 | Python 3.9 compatibility -- `list[str]` type hints in base.py | Low | Medium | `from __future__ import annotations` already present in all files | -- (done) |
| R10 | Full test suite regression from new tests | Low | High | Day 10 runs `pytest tests/ -q --tb=line` to confirm zero regressions | C-10 |

---

## 9. Resume Protocol

If context is lost between sessions, run these commands to get oriented:

```bash
# 1. Check current branch and recent commits
cd /Users/mehrnoosh/Na0S
git log --oneline -10
git status

# 2. Install the package (required for tests)
pip install -e ".[dev]"

# 3. Run office parser tests to see current state
python3 -m pytest tests/parsers/office/ -v --tb=short

# 4. Check which files exist in the integration layer
ls -la src/na0s/parsers/office/integration.py 2>/dev/null && echo "EXISTS" || echo "NOT YET CREATED"

# 5. Check test count
python3 -m pytest tests/parsers/office/ -q --tb=no 2>&1 | tail -1

# 6. Read this plan
cat docs/plans/office_parser_todo.md
```

**Key files to re-read for context**:
- This plan: `docs/plans/office_parser_todo.md`
- Public API: `src/na0s/parsers/office/__init__.py`
- Router: `src/na0s/parsers/office/router.py`
- Base classes: `src/na0s/parsers/office/base.py`
- Integration (if created): `src/na0s/parsers/office/integration.py`

**Decision log** (append here during execution):
- (none yet)

---

## 10. Out of Scope

The following are explicitly excluded from this plan:

1. **Apple .pages format** -- detected and rejected with `UnsupportedDocumentError`. No parser will be written.
2. **RTF (.rtf)** -- already handled by `layer0/doc_extractor.py` via `striprtf`. No deep extraction needed (RTF has no hidden surfaces comparable to OOXML/ODF).
3. **Email formats (.eml, .msg)** -- different attack surface; separate feature.
4. **Real-time API / streaming** -- `scan_document()` is synchronous batch processing.
5. **UI / web interface** -- API only.
6. **Changes to existing 21 detection layers** -- `predict.scan()`, `cascade.scan()`, `rules_registry.py`, and all L0-L16 modules are untouched. The office parser calls INTO the pipeline; it does not modify it.
7. **PDF** -- already handled by `layer0/doc_extractor.py` with pymupdf/pdfplumber/PyPDF2. PDF has its own JavaScript/action detection. Not part of this feature.
8. **Image-based attacks in documents** -- embedded images inside Office documents could contain visual prompt injections. This is covered by the existing `scan_image()` / visual injection detector and is a separate integration concern.
9. **Macro execution / sandbox** -- VBA macros are EXTRACTED as text and scanned for injection patterns. They are NOT executed.
10. **Performance optimization** -- no parallelism or caching for artifact scanning. `scan_document()` calls `predict.scan()` sequentially per artifact. Optimization is a future concern.

---

## 11. Open Questions

> **FLAG**: 4 open questions require decisions before Day 8 implementation begins.

1. **Should `scan_document()` be exposed via the CLI (`na0s scan -f document.docx`)?**
   Currently the CLI's `-f` flag reads the file as text. Adding binary document support to the CLI is a separate concern but would be a natural extension. Decision needed: Day 8 (integration) or defer to a follow-up.

2. **Should `DocumentScanResult` include the raw `list[ExtractedArtifact]` for debugging?**
   This would let operators inspect what was extracted even when no injection was found. Tradeoff: memory usage for large documents with many artifacts. Proposal: include a `debug: bool` parameter on `scan_document()` that attaches raw artifacts when True.

3. **Should `scan_document()` pre-filter artifacts with empty `.text` before calling `predict.scan()`?**
   Some extractors produce artifacts with empty text (e.g., a comment field that exists but has no content). Scanning empty strings wastes cycles and returns `ScanResult(label="safe")` for empty input. Proposal: skip artifacts where `artifact.text.strip() == ""`.

4. **Should `olefile` be a hard dependency of `[docs]` or remain fully optional?**
   Currently `ole_extractor.py` guards with `_HAS_OLEFILE`. If `olefile` is not in `[docs]`, CI tests for OLE will always skip. Proposal: add to `[docs]` with appropriate version bounds (`olefile>=0.46,<1`).
