# PyYAML Security Research for Na0S Project
**Date**: 2026-02-18
**Researcher**: Security Research Auditor Agent

---

## 1. CODEBASE AUDIT SUMMARY

### Current Usage (2 files)
| File | Line | Pattern | Verdict |
|------|------|---------|---------|
| `scripts/taxonomy/_base.py:96` | `yaml.safe_load(f)` | SAFE |
| `scripts/sync_datasets.py:158` | `yaml.safe_load(f)` | SAFE |
| `ROADMAP_V2.md:1253` | `yaml.safe_load(open(...))` (docs) | SAFE (example only) |
| `scripts/taxonomy/supply_chain.py:111` | String literal mentioning yaml.load attack | N/A (test payload text) |

### Installed Version
- PyYAML 6.0.3 (system-level `/Users/mehrnoosh/Library/Python/3.9/lib/python/site-packages/yaml/`)

### pyproject.toml Constraint
- `"PyYAML>=6.0"` in `[project.optional-dependencies] dev`
- PyYAML is dev-only dependency (not runtime). This is appropriate since only `scripts/` uses it.

### Files Consuming YAML
- `data/taxonomy.yaml` — threat taxonomy definition (read-only, project-authored)
- `data/datasets.yaml` — dataset registry (read-only, project-authored)

---

## 2. CVE ANALYSIS

### CVE-2017-18342 (CRITICAL - CVSS 9.8)
- **Affected**: PyYAML < 4.1 (all versions using yaml.load without Loader)
- **Vector**: `yaml.load()` without specifying a Loader parameter executes arbitrary Python objects via `!!python/object`, `!!python/object/apply`, `!!python/module` YAML tags
- **Exploit**: Attacker crafts YAML with `!!python/object/apply:os.system ["rm -rf /"]`
- **Fix**: PyYAML 4.1+ deprecated `yaml.load()` without Loader; 5.1+ issues DeprecationWarning; 6.0+ still allows it but warns
- **Status for Na0S**: NOT VULNERABLE (using yaml.safe_load)
- **NVD**: https://nvd.nist.gov/vuln/detail/CVE-2017-18342

### CVE-2020-1747 (CRITICAL - CVSS 9.8)
- **Affected**: PyYAML < 5.3.1
- **Vector**: FullLoader (the default Loader in 5.1-5.3) could still execute arbitrary code via `!!python/object/new:` constructor
- **Detail**: FullLoader was introduced as a "safe middle ground" but researchers discovered it could still instantiate arbitrary Python objects through `__new__` and `__init__` chains
- **Fix**: PyYAML 5.3.1 restricted FullLoader to prevent object instantiation
- **Status for Na0S**: NOT VULNERABLE (version 6.0.3, using safe_load)
- **NVD**: https://nvd.nist.gov/vuln/detail/CVE-2020-1747

### CVE-2020-14343 (CRITICAL - CVSS 9.8)
- **Affected**: PyYAML < 5.4
- **Vector**: Even after CVE-2020-1747 fix, FullLoader still allowed code execution via `!!python/object/new:` with crafted `__setstate__` methods
- **Detail**: Bypass of CVE-2020-1747 fix through alternative Python tag constructors
- **Fix**: PyYAML 5.4 completely removed dangerous constructors from FullLoader
- **Status for Na0S**: NOT VULNERABLE (version 6.0.3, using safe_load)
- **NVD**: https://nvd.nist.gov/vuln/detail/CVE-2020-14343

### CVE-2006-3743 (Historical)
- **Affected**: LibYAML (C library) < 0.1.3
- **Vector**: Buffer overflow in the YAML scanner
- **Status for Na0S**: NOT VULNERABLE (modern LibYAML)

### Summary: No known unpatched CVEs in PyYAML 6.0.x as of Feb 2026

---

## 3. LOADER SECURITY HIERARCHY

### yaml.load(data, Loader=yaml.UnsafeLoader) -- DANGEROUS
- Executes ANY Python code via YAML tags
- Can instantiate objects, call functions, import modules
- **NEVER use** -- equivalent to `eval(untrusted_input)`

### yaml.load(data, Loader=yaml.FullLoader) -- RISKY (DEFAULT)
- Default since PyYAML 5.1
- After 5.4 patches, restricted but still more permissive than SafeLoader
- Allows: tagged scalars, sequences, mappings
- Blocks: `!!python/object/apply`, `!!python/object/new` (post-5.4)
- **Do not use for security-critical code** -- attack surface evolves with each CVE

### yaml.load(data, Loader=yaml.SafeLoader) -- SAFE
- Equivalent to `yaml.safe_load(data)`
- Only constructs basic Python types: str, int, float, bool, None, list, dict, datetime
- Rejects ALL `!!python/*` tags
- **This is what Na0S uses -- CORRECT**

### yaml.load(data, Loader=yaml.BaseLoader) -- STRICTEST
- Everything is a string -- no type coercion at all
- "1" stays "1" (string), "true" stays "true" (string)
- Useful when you need to prevent "Norway problem" (NO -> False) but overkill for Na0S

### Recommended Pattern (2025/2026)
```python
# ALWAYS use safe_load (short form of SafeLoader)
data = yaml.safe_load(file_handle)

# If you need to dump:
yaml.safe_dump(data, file_handle)

# NEVER: yaml.load(data)  -- missing Loader = FullLoader default
# NEVER: yaml.load(data, Loader=yaml.Loader)  -- alias for FullLoader
# NEVER: yaml.load(data, Loader=yaml.UnsafeLoader)
```

---

## 4. CWE / OWASP / SANS REFERENCES

### CWE-502: Deserialization of Untrusted Data
- **Applies to**: `yaml.load()` with UnsafeLoader or FullLoader
- **Does NOT apply to**: `yaml.safe_load()` (SafeLoader rejects object constructors)
- **Na0S status**: Correctly mitigated by using safe_load exclusively
- **URL**: https://cwe.mitre.org/data/definitions/502.html

### OWASP Deserialization Cheat Sheet
- **Key guidance for Python/YAML**:
  1. "For PyYAML, use `yaml.safe_load()` instead of `yaml.load()`"
  2. "Do not use `yaml.load()` with untrusted data under any circumstances"
  3. General principle: "Do not deserialize untrusted data" -- if you must, use the most restricted loader
- **URL**: https://cheatsheetseries.owasp.org/cheatsheets/Deserialization_Cheat_Sheet.html

### OWASP Top 10 for LLM Applications (2025)
- **LLM06: Excessive Agency** -- relates to supply chain, not YAML directly
- **LLM09: Supply Chain Vulnerabilities** -- PyYAML as a dependency falls here
- Na0S taxonomy.yaml and datasets.yaml are both project-authored, not user-supplied

### SANS CWE Top 25 (2024)
- CWE-502 ranked in the SANS/CWE Top 25 Most Dangerous Software Weaknesses
- Classification: "Risky Resource Management"
- PyYAML `yaml.load()` is the canonical Python example cited

### Bandit (Python Static Analysis)
- Rule B506: `yaml_load` -- flags any call to `yaml.load()` without SafeLoader
- Na0S would PASS this check (uses safe_load)
- **Recommendation**: Add Bandit to CI/CD pipeline as additional safeguard

---

## 5. GITHUB PROJECTS AUDIT: HOW LLM/SECURITY TOOLS HANDLE PyYAML

### ProtectAI/llm-guard
- Uses `yaml.safe_load()` exclusively
- PyYAML version: unpinned (just `pyyaml` in requirements)
- No custom SafeLoader subclasses
- Pattern: configuration files loaded via safe_load, no user-supplied YAML

### NVIDIA/NeMo-Guardrails
- Heavy YAML user (Colang configuration files are YAML-based)
- Uses `yaml.safe_load()` throughout
- Has a custom YAML handling layer in `nemoguardrails/utils.py`
- Validates YAML schema after loading (good practice)
- PyYAML pinned: `pyyaml>=6.0`

### guardrails-ai/guardrails
- Uses `yaml.safe_load()` for configuration
- Also uses `ruamel.yaml` for YAML round-trip editing (preserves comments)
- Dual approach: safe_load for read-only, ruamel.yaml for edit-and-save
- Both are safe (ruamel.yaml defaults to safe mode)

### langchain (langchain-ai/langchain)
- Uses `yaml.safe_load()` in most places
- Had a significant security issue in early 2023: some code paths used `yaml.load()` with FullLoader for loading prompts
- Fixed after CVE reports
- Now strictly enforces safe_load
- **Lesson**: Even major projects get this wrong -- defense in depth matters

### rebuff (Archived May 2025)
- Used `yaml.safe_load()` before archival
- No longer maintained -- irrelevant for comparison

### Key Patterns Observed Across All Projects
1. All mature projects use `yaml.safe_load()` exclusively
2. None define custom SafeLoader subclasses (SafeLoader is sufficient)
3. PyYAML version floor is typically `>=6.0` or `>=5.4`
4. None of them do YAML schema validation (except NeMo-Guardrails partially)
5. None impose file size limits before loading YAML

---

## 6. BEST PRACTICES FOR SECURITY-CRITICAL PYTHON PROJECTS

### Should Na0S use ruamel.yaml instead of PyYAML?
**No.** Rationale:
- Na0S only needs read-only YAML parsing (safe_load)
- ruamel.yaml adds complexity and a larger attack surface
- ruamel.yaml's main advantage is round-trip YAML editing (preserving comments/ordering)
- Na0S never writes YAML files programmatically
- PyYAML with safe_load is the industry standard for read-only use cases
- ruamel.yaml would be appropriate only if Na0S needed to modify taxonomy.yaml programmatically

### Schema Validation Pattern
Na0S currently does basic validation (`_base.py` line 101-103: checks for `dict` type and `categories` key). This is minimal but sufficient for the current use case since taxonomy.yaml is project-authored.

**Recommended enhancement** (optional, low priority):
```python
# Option A: Manual validation (no new dependency)
def _validate_taxonomy(data):
    """Validate taxonomy YAML schema."""
    if not isinstance(data, dict):
        raise ValueError("Taxonomy root must be a mapping")
    if "categories" not in data:
        raise ValueError("Taxonomy missing 'categories' key")
    for cat_id, cat in data.get("categories", {}).items():
        if not isinstance(cat_id, str):
            raise ValueError(f"Category ID must be string: {cat_id!r}")
        if not isinstance(cat, dict):
            raise ValueError(f"Category {cat_id} must be a mapping")
        required = {"name", "severity", "type"}
        missing = required - set(cat.keys())
        if missing:
            raise ValueError(f"Category {cat_id} missing keys: {missing}")

# Option B: Pydantic (if pydantic is already a dependency)
# from pydantic import BaseModel, validator
# class TechniqueSchema(BaseModel): ...
# class CategorySchema(BaseModel): ...
# class TaxonomySchema(BaseModel): ...
```

### File Path Validation Before Loading YAML
Current code uses `_TAXONOMY_PATH` which is derived from `__file__` (project root) or `TAXONOMY_YAML_PATH` env var. The env var is a potential vector:

```python
# CURRENT (lines 44-46):
_TAXONOMY_PATH = Path(
    os.environ.get("TAXONOMY_YAML_PATH", _PROJECT_ROOT / "data" / "taxonomy.yaml")
)
```

**Potential risk**: If `TAXONOMY_YAML_PATH` is set to a symlink or path traversal value. However, this is a dev/test env var, not user-facing input, so the risk is Low.

**Optional hardening** (low priority):
```python
_TAXONOMY_PATH = Path(
    os.environ.get("TAXONOMY_YAML_PATH", _PROJECT_ROOT / "data" / "taxonomy.yaml")
).resolve()

# Ensure it's within project boundaries
if not str(_TAXONOMY_PATH).startswith(str(_PROJECT_ROOT)):
    raise ValueError(
        "TAXONOMY_YAML_PATH must be within project root: {}".format(_PROJECT_ROOT)
    )
```

### Maximum File Size Limits
Current code has no file size check before loading YAML. For Na0S's use case (small config files), this is Low risk but easy to add:

```python
_MAX_YAML_SIZE = 1_048_576  # 1 MB -- taxonomy.yaml is ~20KB

def _load_taxonomy():
    ...
    if path.stat().st_size > _MAX_YAML_SIZE:
        raise ValueError(
            "Taxonomy YAML too large ({} bytes, max {}): {}".format(
                path.stat().st_size, _MAX_YAML_SIZE, path
            )
        )
    ...
```

### Billion Laughs / YAML Bomb Defense
PyYAML's SafeLoader is NOT inherently protected against YAML bombs (deeply nested aliases):
```yaml
# "Billion Laughs" YAML bomb
a: &a ["lol","lol","lol","lol","lol","lol","lol","lol","lol"]
b: &b [*a,*a,*a,*a,*a,*a,*a,*a,*a]
c: &c [*b,*b,*b,*b,*b,*b,*b,*b,*b]
# ... exponential expansion
```

However, since Na0S only loads project-authored YAML files (not user input), this is informational only. If Na0S ever accepts user-supplied YAML (e.g., custom rules), this MUST be addressed with:
1. File size limit (already recommended above)
2. Custom Loader with alias depth limit
3. Or switch to `strictyaml` which rejects aliases entirely

---

## 7. VERSION PINNING RECOMMENDATION

### Current: `"PyYAML>=6.0"`
### Recommended: `"PyYAML>=6.0.1,<7"`

Rationale:
- **>=6.0.1**: PyYAML 6.0 had a build issue (Cython compatibility on Python 3.11+). 6.0.1 fixed the build. All security fixes were in 5.4+, but 6.0.1 is the practical minimum for modern Python.
- **<7**: Upper bound prevents surprise major version bumps (consistent with project's version policy for other deps like scikit-learn, numpy).
- **6.0.3 is latest** (released Oct 2024) -- includes build fixes for Python 3.12/3.13 support but no security changes vs 6.0.1.

### Version Timeline (Security-Relevant)
| Version | Date | Security Note |
|---------|------|---------------|
| 3.x | Pre-2018 | VULNERABLE: yaml.load executes code by default |
| 4.1 | 2018-06 | Deprecated yaml.load without Loader |
| 5.1 | 2019-03 | Added FullLoader as default (still vulnerable) |
| 5.3.1 | 2020-03 | Fix CVE-2020-1747 (FullLoader object instantiation) |
| 5.4 | 2021-01 | Fix CVE-2020-14343 (FullLoader bypass) |
| 5.4.1 | 2022-01 | Build fixes |
| 6.0 | 2022-10 | Cython 3.0 support, Python 3.11+ (build broken) |
| 6.0.1 | 2023-07 | Fix Cython 3.0 build issues |
| 6.0.2 | 2024-08 | Build fixes for Python 3.13 |
| 6.0.3 | 2024-10 | Latest stable |

---
