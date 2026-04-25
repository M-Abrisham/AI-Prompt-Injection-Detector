# README Outline — structural skeleton

Skeleton only. No prose. Each section specifies:
**Purpose** (1-2 sentences) · **Facts pulled** (from `docs/facts.yaml`) ·
**Code example** (yes/no + what it shows) · **Audience** (primary).

Sections marked **[NEW EXTRACTOR]** need facts that aren't in
`facts.yaml` yet — flagged inline. Order optimized per comparable analysis:
hero → quickstart → install → conceptual → API → taxonomy → benchmarks →
ops sections → footer.

---

## 1. Hero / tagline

- **Purpose**: One-sentence description + one analogy line. No stat-stack, no badges-as-headline. Hook the reader in <10 seconds.
- **Facts pulled**: none — this section is positioning, not data.
- **Code example**: no.
- **Audience**: security engineer evaluating in 30s; recruiter scanning.

## 2. Badges

- **Purpose**: License + Python + CI + PyPI + downloads. 5-7 max.
- **Facts pulled**: none from `facts.yaml`. **[NEW EXTRACTOR — pypi_metadata]** if we want PyPI version / download count surfaced; otherwise hand-add badge URLs that auto-fetch.
- **Code example**: no.
- **Audience**: maintainers, contributors, package-evaluators.

## 3. Quickstart (the 5-line example)

- **Purpose**: Working code in the first scroll. Demonstrates the smallest path to a verdict. Comparable peers (garak, NeMo) place this near the top; we follow that.
- **Facts pulled**: `public_exports[scan]`, `constants.DECISION_THRESHOLD`.
- **Code example**: **yes** — 3-5 lines: `from na0s import scan; r = scan("..."); print(r.is_malicious, r.risk_score, r.label)`. Use the example we verified at runtime in audit prompt #2.
- **Audience**: security engineer trying it in 60s.

## 4. Install

- **Purpose**: pip-install path + extras matrix.
- **Facts pulled**: none currently in `facts.yaml`. **[NEW EXTRACTOR — pyproject_extras]** to enumerate `[project.optional-dependencies]` keys (currently we know they exist: `all, audit, data, dev, docs, embedding, lang, llm, ocr, threatintel`). Without an extractor, this section will rot the same way the current README rots.
- **Code example**: shell only (`pip install -e ".[embedding]"`).
- **Audience**: anyone installing.

## 5. What it detects

- **Purpose**: 4-6 bullet conceptual list of attack families covered. Explicitly *not* the 29-category taxonomy — that goes in section 9.
- **Facts pulled**: `taxonomy.techniques_by_category` (used to motivate which families exist; we name 4-6 high-level groupings, not all 29). `detection_signals_in_scan.calls` (to anchor the list to actual code paths).
- **Code example**: no — conceptual section.
- **Audience**: security engineer deciding "does this address my threat model?"

## 6. How it works (architecture)

- **Purpose**: One diagram or one short sequential list (Rebuff's 4-layer model is the reference). Honest about optional/conditional layers.
- **Facts pulled**:
  - `rule_count.total_ast` (motivate L1 size).
  - `constants.DECISION_THRESHOLD`, `constants.MAX_CHUNKS`, `constants._CHUNK_WORD_THRESHOLD` (concrete pipeline constants).
  - `detection_signals_in_scan` (full list of 36 calls — link to it as appendix; do not inline).
- **Facts NOT yet extracted**:
  - Layer-stage → file mapping (which `_HAS_*` flag gates which detector). **[NEW EXTRACTOR — layer_map]** that AST-walks `predict.py` for every `_HAS_*` flag and the `try/except ImportError` block that defines it, producing `{flag_name: source_module, default_enabled: bool}`. The current README's "L0 | rules.py" table is wrong because it isn't sourced from code.
- **Code example**: no.
- **Audience**: security engineer evaluating depth; OSS contributor mapping the codebase.

## 7. Library API / Integration

- **Purpose**: Reference of the public surface — what to import, what each thing does, signature of the main calls. NOT comprehensive — that's autodoc territory. Just the 5-7 things 95% of users touch.
- **Facts pulled**: `public_exports` (18 names — pick the 5-7 most-used, link the rest).
- **Facts NOT yet extracted**:
  - Function/class signatures for each exported name. **[NEW EXTRACTOR — public_api_signatures]** that AST-extracts `ast.FunctionDef.args` and `ast.ClassDef.body[0]` (`__init__` signature) for every name in `__all__`, producing `{name: {signature, docstring_first_line}}`. Without this, the README will drift from the actual signatures.
- **Code example**: **yes** — 2-3 short snippets:
  - `scan()` minimal call (already in section 3 — link, don't repeat).
  - `CascadeClassifier().classify(text)` showing the 4-tuple return.
  - `OutputScanner().scan(output_text=...)` for output-side scanning.
- **Audience**: developer integrating the library into an existing app.

## 8. CLI

- **Purpose**: One-screen reference of `na0s scan` — modes (inline, file, stdin, JSONL batch) + exit codes + output formats.
- **Facts pulled**: none currently in `facts.yaml`. **[NEW EXTRACTOR — cli_surface]** that imports `na0s.cli` and walks the argparse parser tree, producing `{commands: [{name, args, choices, help}], exit_codes: {EXIT_*: value}}`. Audit-prompt #2 confirmed exit codes 0/1/2/3 and `--output-format ∈ {json,text,csv}`; without an extractor, those numbers will silently drift.
- **Code example**: **yes** — shell snippets only (4-6 lines).
- **Audience**: anyone using the CLI / wiring it into CI/CD.

## 9. Taxonomy overview

- **Purpose**: Pointer to the full taxonomy doc + a summary stat. Per comparable analysis: do NOT inline the 29-row table.
- **Facts pulled**: `taxonomy.category_count`, `taxonomy.technique_count_total`. Optionally name the top 3-5 largest categories from `taxonomy.techniques_by_category`.
- **Code example**: no.
- **Audience**: security engineer comparing coverage; researcher.

## 10. Benchmarks

- **Purpose**: Reproducible numbers from the project's own harness. Comparable peers (LLM-Guard, Rebuff, garak, NeMo) all omit numerical benchmarks. We diverge only if numbers come with a runnable harness — otherwise this section becomes the next maintenance debt.
- **Facts pulled**: nothing currently in `facts.yaml`. **[NEW EXTRACTOR — benchmark_results]** REQUIRED before this section is allowed to ship. Must:
  - Run a fixed harness against pinned datasets (e.g., `data/benchmark/*.jsonl`).
  - Emit `docs/benchmarks.yaml` with `{dataset, n, precision, recall, f1, p50_ms, p95_ms, model_version, git_sha}`.
  - Be runnable via `make bench` or `python scripts/extract_benchmarks.py`.
  - Pin a commit hash + dataset SHA so numbers are reproducible.
- **Code example**: no — link to the harness command instead.
- **Audience**: security engineer doing rigorous comparison; reviewer who will not trust uncited numbers.
- **Status**: **HOLD this section until the extractor exists.** Without it, omit entirely (peers do).

## 11. Threat-intel sync

- **Purpose**: Describe Layer 15's role — fetching attack-pattern feeds, fingerprint sync. One paragraph + one CLI snippet.
- **Facts pulled**: nothing currently in `facts.yaml`. **[NEW EXTRACTOR — l15_sources]** that walks `src/na0s/layer15/` for any `SOURCES = [...]` / `FEEDS = [...]` / `URL = "..."` constants and the public functions in `__init__.py`, producing `{feeds: [{name, url, schedule, format}], cli_commands: [...]}`. Currently there is nothing in `facts.yaml` for L15.
- **Code example**: **yes if a sync command exists** — one CLI line. Otherwise omit code.
- **Audience**: ops engineer deploying Na0S in production; security engineer asking "where do detections come from?"

## 12. Multi-turn / conversation detection (Layer 16)

- **Purpose**: Brief callout that Na0S has session-level detectors for slow-burn attacks — fabricated history, escalation, payload splitting, etc.
- **Facts pulled**: `L16_detectors` (8 detector class names from `facts.yaml`). Render as one line per detector or a 2-column list.
- **Code example**: **yes** — 4-5 lines showing `scan(text, session_id="user-42")` and inspecting `result.multi_turn_alerts`.
- **Audience**: security engineer building chat agents.

## 13. Project structure

- **Purpose**: Compact tree of the source layout — supports OSS contributors finding the right module fast. Small (15-20 lines max).
- **Facts pulled**: nothing currently in `facts.yaml`. **[NEW EXTRACTOR — package_layout]** that lists each subpackage under `src/na0s/` with module count (excluding `__init__.py`). Audit-prompt #2 caught the current README claiming `rag/` has 7 modules when it has 2 — exactly the kind of error a code-sourced count prevents.
- **Code example**: no.
- **Audience**: OSS contributor.

## 14. Testing

- **Purpose**: How to run tests + headline counts.
- **Facts pulled**: `test_count.count`, `test_count.collection_errors`, `test_files`. (Surface the collection-error count honestly — currently 8.)
- **Code example**: shell only (`make test-fast`, `pytest tests/canary/`).
- **Audience**: contributor; reviewer evaluating code quality.

## 15. Contributing

- **Purpose**: Branch naming, commit style, where to add new layers/probes/tests, how to run pre-commit.
- **Facts pulled**: none — this is process, not data.
- **Code example**: yes — one shell snippet showing the standard PR loop.
- **Audience**: OSS contributor.

## 16. License

- **Purpose**: One line: MIT, with link.
- **Facts pulled**: none.
- **Code example**: no.
- **Audience**: legal / compliance reviewer; recruiter.

---

## Summary of [NEW EXTRACTOR] flags

Sections that cannot be honestly written without adding fact extractors:

| # | Section | Extractor needed | Output addition to `facts.yaml` |
|---|---|---|---|
| 4 | Install | `pyproject_extras` | `extras: [name, …]` from `[project.optional-dependencies]` |
| 6 | How it works | `layer_map` | `layer_map: {flag: {module, default_enabled}}` from `_HAS_*` blocks in `predict.py` |
| 7 | Library API | `public_api_signatures` | `public_api: {name: {signature, summary}}` for each `__all__` entry |
| 8 | CLI | `cli_surface` | `cli: {commands, args, choices, exit_codes}` from argparse walk |
| 10 | Benchmarks | `benchmark_results` (separate `docs/benchmarks.yaml`) | reproducible F1/latency per dataset; required before this section ships |
| 11 | Threat-intel sync | `l15_sources` | `l15: {feeds, cli_commands}` |
| 13 | Project structure | `package_layout` | `package_layout: {subpkg: {module_count, modules: […]}}` |

## Sections that map directly to existing facts (no new extractor)

- Section 3 (Quickstart) — `public_exports`, `constants`
- Section 5 (What it detects) — `taxonomy`, `detection_signals_in_scan`
- Section 9 (Taxonomy overview) — `taxonomy`
- Section 12 (Multi-turn) — `L16_detectors`
- Section 14 (Testing) — `test_count`, `test_files`

## Sections requiring no facts (positioning / process / legal)

- Section 1 (Hero), Section 2 (Badges, modulo a pypi extractor), Section 15 (Contributing), Section 16 (License)

---

## Comparable-analysis directives applied

- **No stat-stack header** (drops the current "21-Layer Defense | 117+ Techniques | 8,500+ Tests" line). Section 1 is one tagline + one analogy.
- **No inline 29-row taxonomy table** in the README itself (Section 9 is a pointer; full table lives in `docs/TAXONOMY.md`).
- **At most one diagram or one screenshot in Section 6** — no decorative typing-SVG, no waving footer image.
- **Code example up top** (Section 3) per garak/Rebuff pattern, kept to ~5 lines.
- **Architecture as a small honest list (4-6 items)** with `_HAS_*` flags surfaced as "optional", not 21 hand-counted layers.
- **Benchmarks omitted by default** unless backed by reproducible harness output (Section 10 status: HOLD).
- **Badges count: 5-7** — drops the typing-svg and capsule-render images, adds CI / PyPI / downloads.
