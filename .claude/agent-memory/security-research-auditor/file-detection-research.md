# File-Based Injection Detection Research (2026-02-14)

## Tools Evaluated

### 1. Rebuff (protectai/rebuff)
- 4-layer defense: heuristics, LLM-based, VectorDB, canary tokens
- NO file detection, NO magic bytes, NO encoded payload analysis
- Text-only, self-described as "still a prototype"

### 2. LLM Guard (protectai/llm-guard)
- 15 input scanners, 21 output scanners
- InvisibleText scanner: detects Cf/Co/Cn Unicode categories (similar to our L0)
- Secrets scanner: regex-based credential detection
- NO file content, NO binary, NO magic bytes, NO base64 detection
- Text-focused: "sanitization, detection of harmful language"

### 3. Vigil (deadbits/vigil-llm)
- 7 scanners: relevance, sentiment, similarity, transformer, vectordb, yara
- YARA scanner: loads .yara/.yar rule files, matches against text prompts
- NO file-based detection, NO magic bytes, NO encoding analysis
- Could theoretically write YARA rules for byte patterns but not designed for it

### 4. NeMo Guardrails (NVIDIA)
- Colang-based dialog flows, input/output/retrieval rails
- Designed for Chat Completions API (role/content pairs)
- NO file detection capabilities whatsoever

### 5. Guardrails AI
- Pydantic-based structured output validation
- Hub-based validator ecosystem
- NO binary/file/MIME detection in core framework

## Python Library Comparison

### python-magic
- Wrapper around libmagic C library
- REQUIRES system dependency: libmagic1 (apt), libmagic (brew)
- Cross-platform installation pain (Windows DLLs, LD_LIBRARY_PATH issues)
- Very comprehensive detection (thousands of formats)
- Overkill for injection detection use case

### filetype (h2non/filetype.py)
- Pure Python, ZERO external dependencies
- Needs only first 261 bytes
- 80+ formats across 8 categories
- Detects DOCX/XLSX/PPTX via PK header + internal path checks
- No SVG detection (text-based, not binary-magic)
- Good but adds unnecessary dependency for our targeted use case

### Manual magic bytes (RECOMMENDED for our use case)
- Zero dependencies
- Full control over what to detect and how to flag it
- Can be tailored to security-relevant formats only
- Already the pattern used in html_extractor.py
- Fastest possible (direct byte comparison)
- Limitation: must maintain signatures ourselves

## Magic Byte Reference Table

### Documents (HIGH priority - injection delivery vehicles)
| Format | Magic Bytes (hex) | ASCII | Notes |
|--------|------------------|-------|-------|
| PDF | 25 50 44 46 | %PDF | Already detected |
| RTF | 7B 5C 72 74 66 | {\rtf | Already detected |
| DOCX/XLSX/PPTX | 50 4B 03 04 | PK.. | ZIP-based OOXML |
| DOC/XLS/PPT (legacy) | D0 CF 11 E0 A1 B1 1A E1 | (binary) | OLE2 compound |
| ODP/ODS/ODT | 50 4B 03 04 | PK.. | ZIP-based OpenDocument |

### Images (HIGH priority - OCR injection, steganography)
| Format | Magic Bytes (hex) | ASCII | Notes |
|--------|------------------|-------|-------|
| PNG | 89 50 4E 47 0D 0A 1A 0A | .PNG.... | 8 bytes |
| JPEG | FF D8 FF | ... | 3 bytes |
| GIF | 47 49 46 38 | GIF8 | GIF87a or GIF89a |
| BMP | 42 4D | BM | |
| TIFF (LE) | 49 49 2A 00 | II*. | Little-endian |
| TIFF (BE) | 4D 4D 00 2A | MM.* | Big-endian |
| WebP | 52 49 46 46 + 57 45 42 50 | RIFF+WEBP | bytes 0-3 + 8-11 |
| PSD | 38 42 50 53 | 8BPS | Photoshop |
| ICO | 00 00 01 00 | .... | |

### Audio (MEDIUM priority - less common injection vector)
| Format | Magic Bytes (hex) | ASCII | Notes |
|--------|------------------|-------|-------|
| WAV | 52 49 46 46 + 57 41 56 45 | RIFF+WAVE | bytes 0-3 + 8-11 |
| MP3 (ID3) | 49 44 33 | ID3 | ID3 tag header |
| MP3 (sync) | FF FB or FF F3 or FF F2 | .. | MPEG sync word |
| FLAC | 66 4C 61 43 | fLaC | |
| OGG | 4F 67 67 53 | OggS | |
| AAC | FF F1 or FF F9 | .. | ADTS frame |
| MIDI | 4D 54 68 64 | MThd | |
| AIFF | 46 4F 52 4D + 41 49 46 46 | FORM+AIFF | |

### Video (MEDIUM priority)
| Format | Magic Bytes (hex) | ASCII | Notes |
|--------|------------------|-------|-------|
| AVI | 52 49 46 46 + 41 56 49 20 | RIFF+AVI | bytes 0-3 + 8-11 |
| WebM | 1A 45 DF A3 | .... | EBML header + webm doctype |
| MKV | 1A 45 DF A3 | .... | EBML header + matroska doctype |
| MP4 | varies | ftyp | ftyp box at offset 4 |
| FLV | 46 4C 56 01 | FLV. | |
| WMV | 30 26 B2 75 | 0&.u | Windows Media |

### Archives (HIGH priority - can contain anything)
| Format | Magic Bytes (hex) | ASCII | Notes |
|--------|------------------|-------|-------|
| ZIP | 50 4B 03 04 | PK.. | Also DOCX/XLSX/JAR/APK |
| GZIP | 1F 8B 08 | ... | |
| 7z | 37 7A BC AF 27 1C | 7z.... | |
| RAR | 52 61 72 21 1A 07 | Rar!.. | |
| BZIP2 | 42 5A 68 | BZh | |
| TAR | (at offset 257) 75 73 74 61 72 | ustar | |

### Executables (CRITICAL - should always be rejected)
| Format | Magic Bytes (hex) | ASCII | Notes |
|--------|------------------|-------|-------|
| EXE/DLL (PE) | 4D 5A | MZ | Windows executable |
| ELF | 7F 45 4C 46 | .ELF | Linux executable |
| Mach-O | FE ED FA CE or FE ED FA CF | .... | macOS executable |
| Mach-O (rev) | CE FA ED FE or CF FA ED FE | .... | macOS reverse byte order |
| Java class | CA FE BA BE | .... | |
| WASM | 00 61 73 6D | .asm | WebAssembly |
| Shell script | 23 21 | #! | Shebang |

### Special Cases (no magic bytes - need different detection)
| Format | Detection Method | Injection Risk |
|--------|-----------------|----------------|
| SVG | Text-based XML with `<svg` tag | HIGH - can contain `<script>`, event handlers |
| CSV | No magic bytes, text-based | MEDIUM - formula injection |
| JSON | Text-based, starts with `{` or `[` | LOW |
| YAML | Text-based | MEDIUM - code execution in some parsers |
| Markdown | Text-based | LOW-MEDIUM |
| HTML/XML | Text-based, detected by existing code | HIGH |

## File-Based Injection Attack Vectors

### Confirmed attack surfaces:
1. **PDF injection**: Hidden text layers, JavaScript, annotations with instructions
2. **DOCX/XLSX injection**: Hidden text in white-on-white, comments, metadata, macros
3. **Image-based injection (OCR)**: Text rendered in images processed by vision LLMs
4. **SVG injection**: `<script>` tags, event handlers, `<foreignObject>` with HTML
5. **Steganography**: Data hidden in image pixel LSBs
6. **Polyglot files**: Files valid as multiple types (e.g., valid JPEG + valid ZIP)
7. **Base64-encoded files**: Binary content encoded in text
8. **Data URI schemes**: `data:image/png;base64,...` embedding binary in text
9. **Metadata injection**: EXIF, XMP, IPTC data in images carrying instructions
10. **Archive bombs**: ZIP files with extreme compression ratios (zip bombs)
