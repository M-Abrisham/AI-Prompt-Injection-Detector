# OLE (Legacy .doc/.xls/.ppt) Hiding Spots Inventory

> **Purpose:** Comprehensive inventory of user-controllable text locations inside
> legacy OLE Compound Binary Format files (.doc, .xls, .ppt) for a prompt
> injection security scanner. Intended audience: coder agent building an
> extractor using the `olefile` Python library.

> **Key constraint:** Legacy Office files use OLE Compound Binary Format
> (MS-CFB), NOT the modern XML-based formats (.docx/.xlsx/.pptx). The `olefile`
> library can list and read raw OLE streams, but interpreting the *contents* of
> those streams often requires parsing Microsoft's proprietary binary record
> formats, which range from straightforward (property sets) to extremely complex
> (WordDocument body text).

---

## General OLE Properties (All Formats)

### 1. SummaryInformation

| Field | Value |
|---|---|
| **OLE Stream / Storage Path** | `\x05SummaryInformation` |
| **Data format** | OLE Property Set (MS-OLEPS) -- sequence of VT_LPSTR / VT_FILETIME typed properties |
| **Visible in default UI?** | Partially -- visible in File > Properties dialog, but users rarely check it |
| **Injection risk** | **High** -- title, subject, author, keywords, and comments are all free-text fields that render in many downstream tools and search indexes |
| **How to extract with `olefile`** | Built-in API: `ole = olefile.OleFileIO(path); meta = ole.get_metadata(); text = meta.title, meta.subject, meta.author, meta.keywords, meta.comments` |
| **Citation** | [MS-OLEPS] Section 2.21 -- SummaryInformation Property Set; [olefile Howto](https://olefile.readthedocs.io/en/latest/Howto.html) |

**Properties to scan:**
- `title` (PIDSI_TITLE, 0x02)
- `subject` (PIDSI_SUBJECT, 0x03)
- `author` (PIDSI_AUTHOR, 0x04)
- `keywords` (PIDSI_KEYWORDS, 0x05)
- `comments` (PIDSI_COMMENTS, 0x06)
- `last_saved_by` (PIDSI_LASTAUTHOR, 0x08)
- `creating_application` (PIDSI_APPNAME, 0x12)

---

### 2. DocumentSummaryInformation

| Field | Value |
|---|---|
| **OLE Stream / Storage Path** | `\x05DocumentSummaryInformation` |
| **Data format** | OLE Property Set (MS-OLEPS) -- includes a second section for custom properties |
| **Visible in default UI?** | Partially -- category and company show in Properties; custom properties require the Custom tab |
| **Injection risk** | **High** -- custom properties are arbitrary key-value pairs; category, manager, company are free-text |
| **How to extract with `olefile`** | `meta = ole.get_metadata()` exposes `meta.category`, `meta.manager`, `meta.company`. For custom properties: `ole.get_properties('\x05DocumentSummaryInformation', section=1)` returns a dict of custom property name-value pairs. |
| **Citation** | [MS-OLEPS] Section 2.22 -- DocumentSummaryInformation; olefile source `get_metadata()` and `get_properties()` |

**Properties to scan:**
- `category` (standard property)
- `manager` (standard property)
- `company` (standard property)
- `content_type`, `content_status` (standard properties)
- All custom properties (section index 1 in the property set) -- these are fully user-defined key-value pairs

---

### 3. OLE Stream Listing (Suspicious Stream Detection)

| Field | Value |
|---|---|
| **OLE Stream / Storage Path** | N/A -- applies to the container directory itself |
| **Data format** | OLE directory entries |
| **Visible in default UI?** | No -- users never see the OLE directory structure |
| **Injection risk** | **Medium** -- unexpected streams may contain embedded OLE objects, macros, or injected content that bypasses text-only scanners |
| **How to extract with `olefile`** | `ole.listdir()` returns all stream/storage paths. Compare against known-good lists per format. Flag unexpected entries. |
| **Citation** | [MS-CFB] Section 2.6 -- Compound File Directory Entry |

**What to flag:**
- Any stream not in the expected set for the file type
- Streams named `\x01Ole`, `\x01CompObj`, `\x03ObjInfo` (embedded OLE objects)
- Multiple `ObjectPool` entries (embedded documents within documents)
- Any `VBA` or `Macros` storage that was not expected
- Streams with non-ASCII or control-character names

**Code pattern:**
```python
ole = olefile.OleFileIO(path)
known_streams_doc = {
    'WordDocument', '1Table', '0Table', 'Data',
    '\x05SummaryInformation', '\x05DocumentSummaryInformation',
    '\x01CompObj',
}
for stream_path in ole.listdir():
    joined = '/'.join(stream_path)
    if joined not in known_streams_doc:
        flag_suspicious(joined)
```

---

## Legacy .doc (Word Binary Format)

### 4. Body Text

| Field | Value |
|---|---|
| **OLE Stream / Storage Path** | `WordDocument` (main text) + `0Table` or `1Table` (formatting and structure tables) |
| **Data format** | FIB (File Information Block) at offset 0 of `WordDocument`; text stored as raw bytes at CP (Character Position) offsets; structure defined by Clx/PlcPcd in the Table stream |
| **Visible in default UI?** | Yes |
| **Injection risk** | **High** -- this is the main document content |
| **How to extract with `olefile`** | **Not practically extractable with olefile alone.** The binary format is extremely complex -- text positions are fragmented across the stream, interleaved with inline objects, and require parsing the FIB, Clx, PlcPcd, and PieceDescriptor structures. |
| **Citation** | [MS-DOC] Section 2.4.1 (WordDocument Stream), Section 2.5.1 (FIB), Section 2.8.35 (Clx) |

**Practical extraction strategy:**
1. **Preferred:** Use `antiword` CLI tool: `subprocess.run(['antiword', path], capture_output=True, text=True)` -- reliable, fast, handles most .doc files
2. **Fallback:** Use `textract` or `catdoc` CLI tools
3. **Last resort:** Raw string extraction -- read the `WordDocument` stream bytes and extract printable Unicode/ASCII runs with `re.findall(rb'[\x20-\x7e]{4,}', data)` -- high false-positive rate but catches injection payloads
4. **Python library option:** `doc2txt` from the `antiword` wrapper, or the `extract_doc` package

> **Honest assessment:** Properly parsing MS-DOC body text from scratch is a
> multi-thousand-line endeavor. Use external tools. Do NOT attempt to write a
> full FIB/Clx parser.

---

### 5. Comments / Annotations

| Field | Value |
|---|---|
| **OLE Stream / Storage Path** | `WordDocument` stream (comment text lives in the "comment document" sub-document area, after main text + footnotes + headers); annotation references in Table stream via `PlcfandRef` / `PlcfandTxt` |
| **Data format** | Comment text: raw character data at offsets defined by FibRgLw97.ccpAtn. Annotation reference descriptors: ATRDPre10 / ATRD structures in PlcfandRef. |
| **Visible in default UI?** | Yes (shown as comment balloons or in reviewing pane) |
| **Injection risk** | **High** -- comment text is user-controlled and often extracted by document processing pipelines |
| **How to extract with `olefile`** | **Not directly extractable with olefile.** Requires parsing the FIB to locate the annotation sub-document region, then reading character data. The `antiword` tool does NOT extract comments. |
| **Citation** | [MS-DOC] Section 2.9.3 (PlcfandRef), Section 2.9.4 (PlcfandTxt), Section 2.4.1 (sub-document regions) |

**Practical extraction strategy:**
1. `antiword` does not extract comments -- this is a gap
2. Raw string extraction from the `WordDocument` stream will catch comment text mixed in with body text (comments are stored after the main text body)
3. For dedicated comment extraction, would need a partial FIB parser to locate `ccpAtn` region -- moderate effort, feasible if scoped narrowly

---

### 6. Headers and Footers

| Field | Value |
|---|---|
| **OLE Stream / Storage Path** | `WordDocument` stream (header/footer text in the "header document" sub-document area); structure pointers in Table stream via `PlcfHdd` |
| **Data format** | Raw character data at offsets after main text + footnotes, sized by FibRgLw97.ccpHdd. Six header/footer stories per section (even page header, odd page header, even page footer, odd page footer, first page header, first page footer). |
| **Visible in default UI?** | Yes (visible when editing headers/footers) |
| **Injection risk** | **Medium** -- headers/footers are visible but less commonly extracted by automated tools |
| **How to extract with `olefile`** | **Not directly extractable with olefile.** Same FIB-parsing challenge as body text. |
| **Citation** | [MS-DOC] Section 2.4.2 (Headers), Section 2.8.20 (PlcfHdd) |

**Practical extraction strategy:**
1. `antiword` does NOT extract headers/footers
2. Raw string extraction from `WordDocument` stream will catch them (they are stored contiguously after footnote text)
3. Dedicated extraction requires FIB parsing to locate the header sub-document region

---

### 7. VBA Macros (.doc)

| Field | Value |
|---|---|
| **OLE Stream / Storage Path** | `Macros/VBA/` storage (Word 97-2003), or `VBA/` storage. Individual modules stored as streams like `Macros/VBA/ThisDocument`, `Macros/VBA/Module1`, etc. The `VBA/dir` stream contains the module directory. |
| **Data format** | Compressed VBA source code (MS-OVBA format) -- each module stream contains a compressed source code blob |
| **Visible in default UI?** | Partially -- accessible via Alt+F11 (VBA editor) but not shown in the document view |
| **Injection risk** | **High** -- macros can contain arbitrary code, auto-execute triggers, and prompt injection payloads in string literals or comments |
| **How to extract with `olefile`** | `olefile` can read the raw compressed streams, but decompression requires implementing the MS-OVBA algorithm. **Use `oletools.olevba` instead** -- it handles decompression and deobfuscation. |
| **Citation** | [MS-OVBA] Section 2.4.1 (Module Stream), [MS-DOC] Section 2.1 (streams); [oletools olevba](https://github.com/decalage2/oletools/wiki/olevba) |

**Code pattern (oletools):**
```python
from oletools.olevba import VBA_Parser

vba_parser = VBA_Parser(path)
if vba_parser.detect_vba_macros():
    for (filename, stream_path, vba_filename, vba_code) in vba_parser.extract_macros():
        scan_for_injection(vba_code)
    # Also check for auto-exec triggers
    results = vba_parser.analyze_macros()
    for kw_type, keyword, description in results:
        if kw_type == 'AutoExec':
            flag_auto_exec(keyword, description)
```

---

## Legacy .xls (Excel Binary Format)

### 8. Sheet Names and Visibility

| Field | Value |
|---|---|
| **OLE Stream / Storage Path** | `Workbook` stream (or `Book` in older BIFF5 files) |
| **Data format** | BoundSheet8 records (record type 0x0085). Each record contains: `lbPlyPos` (4 bytes, stream offset to sheet BOF), `hsState` (2 bits: 0x00=Visible, 0x01=Hidden, 0x02=Very Hidden), `dt` (8 bits: sheet type), `stName` (ShortXLUnicodeString, the sheet name). |
| **Visible in default UI?** | Partially -- visible sheets show as tabs; hidden sheets require Format > Sheet > Unhide; **very hidden** sheets (0x02) cannot be unhidden via UI without VBA |
| **Injection risk** | **High** -- very hidden sheets (hsState=0x02) are invisible to users but their content is still accessible programmatically; sheet names themselves can contain injection payloads |
| **How to extract with `olefile`** | Requires parsing the `Workbook` stream binary records. Read the stream with `ole.openstream('Workbook').read()`, then scan for record type 0x0085 and parse the BoundSheet8 structure. This is feasible -- the record format is simple. |
| **Citation** | [MS-XLS] Section 2.4.28 (BoundSheet8) |

**Code pattern (manual parsing):**
```python
import struct

data = ole.openstream('Workbook').read()
pos = 0
sheets = []
while pos < len(data) - 4:
    rec_type, rec_len = struct.unpack_from('<HH', data, pos)
    if rec_type == 0x0085:  # BoundSheet8
        lb_ply_pos = struct.unpack_from('<I', data, pos + 4)[0]
        flags = data[pos + 8]
        hs_state = flags & 0x03       # 0=visible, 1=hidden, 2=very hidden
        dt = data[pos + 9]            # 0=worksheet, 1=macro, 2=chart
        name_len = data[pos + 10]
        name_flags = data[pos + 11]
        if name_flags & 0x01:  # Unicode
            name = data[pos+12 : pos+12+name_len*2].decode('utf-16-le')
        else:
            name = data[pos+12 : pos+12+name_len].decode('latin-1')
        sheets.append({'name': name, 'hidden': hs_state, 'type': dt})
    pos += 4 + rec_len
```

---

### 9. Cell String Values (Shared String Table)

| Field | Value |
|---|---|
| **OLE Stream / Storage Path** | `Workbook` stream |
| **Data format** | SST record (record type 0x00FC). Contains `cstTotal` (4 bytes, total string refs), `cstUnique` (4 bytes, unique string count), then an array of `XLUnicodeRichExtendedString` structures. May span multiple Continue records (type 0x003C). |
| **Visible in default UI?** | Yes (cell values shown in spreadsheet) |
| **Injection risk** | **High** -- this is the primary text content of the spreadsheet; any cell containing text references the SST |
| **How to extract with `olefile`** | Requires parsing binary records from the `Workbook` stream. Locate the SST record (0x00FC), then parse XLUnicodeRichExtendedString entries. **This is moderately complex** due to Continue records and variable-length Unicode encoding. |
| **Citation** | [MS-XLS] Section 2.4.265 (SST), Section 2.5.293 (XLUnicodeRichExtendedString) |

**Practical extraction strategy:**
1. **Preferred:** Use the `xlrd` library (pure Python, handles BIFF8): `xlrd.open_workbook(path)` then iterate sheets/rows/cells
2. **Fallback:** Parse SST records manually -- feasible but must handle Continue records and rich-text/extended-string flags
3. **Last resort:** Raw string extraction from the `Workbook` stream bytes

> **Honest assessment:** Parsing the SST correctly (especially Continue record
> spanning) is tricky. Use `xlrd` for reliable extraction. Manual parsing is a
> secondary option for environments where `xlrd` is unavailable.

---

### 10. Defined Names (Lbl Records)

| Field | Value |
|---|---|
| **OLE Stream / Storage Path** | `Workbook` stream (Globals Substream) |
| **Data format** | Lbl records (record type 0x0018). Key fields: `fHidden` (1 bit -- whether name is hidden from UI), `chKey` (shortcut key), `itab` (sheet scope, 0 = workbook-level), `Name` (XLUnicodeStringNoCch), `rgce` (parsed expression / formula definition). |
| **Visible in default UI?** | Partially -- visible in Name Manager unless `fHidden=1`; hidden names are invisible in UI |
| **Injection risk** | **High** -- defined names can: (a) contain injection text in the name itself, (b) reference formulas that evaluate to injected strings, (c) be hidden from the user with `fHidden=1` |
| **How to extract with `olefile`** | Requires parsing binary records from the `Workbook` stream. Locate Lbl records (0x0018) and parse the name string. The formula definition (`rgce`) is a parsed expression requiring a full formula parser to interpret. |
| **Citation** | [MS-XLS] Section 2.4.133 (Lbl) |

**Practical extraction strategy:**
1. **Preferred:** Use `xlrd`: `book.name_map` or `book.name_obj_list` gives all defined names with their formulas
2. **Fallback:** Parse Lbl records manually -- the name string is extractable, but the formula definition requires an RPN expression parser
3. Pay special attention to `fHidden=1` names -- these are specifically designed to be invisible

---

### 11. VBA Macros (.xls)

| Field | Value |
|---|---|
| **OLE Stream / Storage Path** | `_VBA_PROJECT_CUR/VBA/` storage. Module streams: `_VBA_PROJECT_CUR/VBA/ThisWorkbook`, `_VBA_PROJECT_CUR/VBA/Sheet1`, `_VBA_PROJECT_CUR/VBA/Module1`, etc. The directory stream is `_VBA_PROJECT_CUR/VBA/dir`. |
| **Data format** | Same MS-OVBA compressed format as .doc macros |
| **Visible in default UI?** | Partially -- accessible via Alt+F11 |
| **Injection risk** | **High** -- same as .doc macros |
| **How to extract with `olefile`** | Use `oletools.olevba` (same pattern as .doc). |
| **Citation** | [MS-OVBA]; [MS-XLS] Section 2.1.7.16 (VBA Project) |

---

### 12. Cell Comments / Notes (.xls)

| Field | Value |
|---|---|
| **OLE Stream / Storage Path** | `Workbook` stream (within individual sheet substreams) |
| **Data format** | Note records (record type 0x001C) in BIFF8. Each Note record references an author string and a text object (TxO record, type 0x01B6) that contains the comment text. Comment text stored in Continue records following the TxO. |
| **Visible in default UI?** | Partially -- shown as red triangle indicator; text shown on hover |
| **Injection risk** | **Medium** -- comments are user-controllable but less commonly extracted |
| **How to extract with `olefile`** | Requires parsing sheet substream records. Complex due to the TxO + Continue record chain. |
| **Citation** | [MS-XLS] Section 2.4.179 (Note), Section 2.4.329 (TxO) |

**Practical extraction strategy:**
1. **Preferred:** Use `xlrd` -- note: `xlrd` does not extract comments by default; the `openpyxl` library handles .xlsx comments but not .xls
2. **Alternative:** Use `xlrd` with the `formatting_info=True` flag and manually correlate Note records
3. **Fallback:** Raw string extraction from `Workbook` stream bytes

---

## Legacy .ppt (PowerPoint Binary Format)

### 13. Slide Text

| Field | Value |
|---|---|
| **OLE Stream / Storage Path** | `PowerPoint Document` stream |
| **Data format** | Record-based binary format. Text is stored in `TextCharsAtom` (record type 0x0FA0, UTF-16LE) and `TextBytesAtom` (record type 0x0FA8, single-byte encoding) records. These are found inside `SlideListWithTextContainer` (within `DocumentContainer`), and also within individual `SlideContainer` records. Each text run is preceded by a `TextHeaderAtom` (type 0x0F9F) that specifies the text type (title, body, notes, etc.). |
| **Visible in default UI?** | Yes |
| **Injection risk** | **High** -- this is the main presentation content |
| **How to extract with `olefile`** | **Feasible with moderate effort.** Read the `PowerPoint Document` stream, then scan for TextCharsAtom (0x0FA0) and TextBytesAtom (0x0FA8) records. The record header is 8 bytes (recVer/recInstance: 2 bytes, recType: 2 bytes, recLen: 4 bytes). |
| **Citation** | [MS-PPT] Section 2.4.14.3 (SlideListWithTextContainer), Section 2.9.164 (TextCharsAtom), Section 2.9.163 (TextBytesAtom), Section 2.4.15.1 (Outline Text) |

**Code pattern (manual parsing):**
```python
import struct

data = ole.openstream('PowerPoint Document').read()
texts = []
pos = 0
while pos < len(data) - 8:
    rec_ver_inst, rec_type, rec_len = struct.unpack_from('<HHI', data, pos)
    if rec_type == 0x0FA0:  # TextCharsAtom (UTF-16LE)
        text = data[pos+8 : pos+8+rec_len].decode('utf-16-le', errors='replace')
        texts.append(text)
    elif rec_type == 0x0FA8:  # TextBytesAtom (single-byte)
        text = data[pos+8 : pos+8+rec_len].decode('latin-1', errors='replace')
        texts.append(text)
    pos += 8 + rec_len
```

> **Honest assessment:** Unlike .doc body text, .ppt text extraction via record
> scanning is actually quite practical. The record format is self-describing
> (each record has a type and length), so a simple linear scan can extract all
> text atoms without understanding the full container hierarchy.

---

### 14. Speaker Notes

| Field | Value |
|---|---|
| **OLE Stream / Storage Path** | `PowerPoint Document` stream |
| **Data format** | Same TextCharsAtom / TextBytesAtom records, but contained within `NotesContainer` records (or within the `NotesListWithTextContainer` in the `DocumentContainer`). The `TextHeaderAtom` preceding the text will have `textType` = 2 (Notes). |
| **Visible in default UI?** | Partially -- visible in Notes view but not in slideshow |
| **Injection risk** | **High** -- speaker notes are often overlooked by authors before sharing, and may be extracted by document processing tools |
| **How to extract with `olefile`** | Same record-scanning approach as slide text. The simple scanner in item 13 already captures notes text. To *distinguish* notes from slide text, also parse the `TextHeaderAtom` (0x0F9F) that precedes each text atom -- its 4-byte body contains the text type. |
| **Citation** | [MS-PPT] Section 2.5.6 (NotesContainer), Section 2.4.14.6 (NotesListWithTextContainer), Section 2.9.162 (TextHeaderAtom) |

**TextHeaderAtom textType values:**
- 0 = Title
- 1 = Body
- 2 = Notes
- 3 = Not used
- 4 = Other (text in shapes)
- 5 = Center body
- 6 = Center title
- 7 = Half body
- 8 = Quarter body

---

### 15. VBA Macros (.ppt)

| Field | Value |
|---|---|
| **OLE Stream / Storage Path** | VBA project is stored as a `VbaProjectStg` record in the `PowerPoint Document` stream (a persist object). When expanded, it yields an embedded OLE compound file containing `VBA/dir`, `VBA/ThisPresentation`, `VBA/Module1`, etc. |
| **Data format** | Same MS-OVBA compressed format. The `VbaProjectStg` record contains an embedded compound file that must be parsed separately. |
| **Visible in default UI?** | Partially -- accessible via Alt+F11 |
| **Injection risk** | **High** |
| **How to extract with `olefile`** | Use `oletools.olevba` -- it handles the embedded compound file extraction automatically. |
| **Citation** | [MS-PPT] Section 2.10.40 (VbaProjectStg), [MS-OVBA]; [oletools olevba](https://github.com/decalage2/oletools/wiki/olevba) |

---

### 16. Embedded OLE Objects (.ppt)

| Field | Value |
|---|---|
| **OLE Stream / Storage Path** | `PowerPoint Document` stream contains `ExOleObjStg` records (persist objects) for embedded OLE objects. These are also referenced via `ExOleEmbedContainer` and `ExOleLinkContainer` in the `ExObjListContainer`. |
| **Data format** | Each `ExOleObjStg` record contains a compressed (deflate) embedded OLE compound file |
| **Visible in default UI?** | Partially -- shown as embedded objects in slides |
| **Injection risk** | **Medium** -- embedded objects can contain their own text, macros, and metadata; recursive scanning required |
| **How to extract with `olefile`** | `oletools` can enumerate embedded objects. For manual extraction, would need to locate `ExOleObjStg` records and decompress the embedded compound files. |
| **Citation** | [MS-PPT] Section 2.10.34 (ExOleObjStg), PowerPoint Document Stream Parts 9-10 |

---

## Cross-Format Concerns

### 17. Embedded OLE Objects (ObjectPool -- .doc and .xls)

| Field | Value |
|---|---|
| **OLE Stream / Storage Path** | `ObjectPool/` storage (Word), or embedded directly in `Workbook` stream records (Excel). Each embedded object in `ObjectPool` is a sub-storage containing its own `\x01Ole`, `\x01CompObj`, and content streams. |
| **Data format** | Each sub-storage is itself an OLE compound structure |
| **Visible in default UI?** | Partially |
| **Injection risk** | **Medium** -- embedded objects can contain text, macros, and metadata from other applications; must be scanned recursively |
| **How to extract with `olefile`** | `ole.listdir()` will show `ObjectPool` sub-storages. Each can be opened and scanned for its own content. |
| **Citation** | [MS-DOC] Section 2.1 (Streams), [MS-CFB] |

---

### 18. Encryption Detection

| Field | Value |
|---|---|
| **OLE Stream / Storage Path** | Varies: `EncryptedPackage` stream (Office 2007+ encryption applied to legacy format), or encryption flags in the FIB (`FibBase.fEncrypted`) for .doc, or `FilePass` record (type 0x002F) in the `Workbook` stream for .xls |
| **Data format** | For .doc: FibBase.fEncrypted bit at offset 0x0B bit 0. For .xls: FilePass record contains encryption type and key data. For .ppt: similar per-stream encryption. |
| **Visible in default UI?** | Yes -- user is prompted for password |
| **Injection risk** | **N/A** -- encrypted files cannot be scanned; must be detected and rejected |
| **How to extract with `olefile`** | Detection strategies: (1) Check for `EncryptedPackage` stream: `ole.exists('EncryptedPackage')`. (2) For .doc: read first 12 bytes of `WordDocument`, check `FibBase.fEncrypted` bit. (3) For .xls: scan for FilePass record (0x002F) in `Workbook` stream. |
| **Citation** | [MS-OFFCRYPTO]; [MS-DOC] Section 2.5.1 (FibBase, fEncrypted field); [MS-XLS] Section 2.4.117 (FilePass) |

**Code pattern (encryption detection):**
```python
def is_encrypted_ole(ole):
    # Method 1: Office 2007+ encryption wrapper
    if ole.exists('EncryptedPackage'):
        return True
    # Method 2: .doc FIB check
    if ole.exists('WordDocument'):
        wd = ole.openstream('WordDocument').read(12)
        if len(wd) >= 12:
            flags = struct.unpack_from('<H', wd, 0x0A)[0]
            if flags & 0x0100:  # fEncrypted bit
                return True
    # Method 3: .xls FilePass record
    if ole.exists('Workbook'):
        wb = ole.openstream('Workbook').read()
        pos = 0
        while pos < len(wb) - 4:
            rec_type, rec_len = struct.unpack_from('<HH', wb, pos)
            if rec_type == 0x002F:  # FilePass
                return True
            if rec_type == 0x0000 and rec_len == 0:
                break
            pos += 4 + rec_len
    return False
```

---

## Practical Extraction Strategy Summary

### Tier 1: Easy -- Use olefile directly

| Hiding spot | Method |
|---|---|
| SummaryInformation | `ole.get_metadata()` |
| DocumentSummaryInformation | `ole.get_metadata()` + `ole.get_properties()` |
| Stream listing / suspicious streams | `ole.listdir()` |
| Encryption detection | Stream existence checks + FIB byte inspection |

### Tier 2: Moderate -- Use companion libraries

| Hiding spot | Recommended tool |
|---|---|
| VBA macros (all formats) | `oletools.olevba.VBA_Parser` |
| .xls cell strings (SST) | `xlrd.open_workbook()` |
| .xls sheet names + visibility | `xlrd` or manual BoundSheet8 parsing (simple) |
| .xls defined names | `xlrd` name objects |
| .ppt slide text + notes | Manual record scanning (TextCharsAtom/TextBytesAtom) -- feasible, ~30 lines |

### Tier 3: Hard -- Use external CLI tools or accept limitations

| Hiding spot | Recommended approach |
|---|---|
| .doc body text | `antiword` CLI, or `catdoc`, or raw string extraction fallback |
| .doc comments/annotations | Raw string extraction (no good Python-only solution for .doc) |
| .doc headers/footers | Raw string extraction from `WordDocument` stream |
| .xls cell comments | Limited support; raw string extraction as fallback |
| Embedded OLE objects | Recursive `olefile` scanning of `ObjectPool` sub-storages |

### Dependency matrix

```
Required:     olefile          (pip install olefile)
Recommended:  oletools         (pip install oletools)     -- VBA macros
Recommended:  xlrd             (pip install xlrd)         -- .xls cell/sheet data
Optional:     antiword         (apt install antiword)     -- .doc body text
Optional:     catdoc           (apt install catdoc)       -- .doc body text alternative
Optional:     msoffcrypto-tool (pip install msoffcrypto-tool) -- encrypted file handling
```

### Raw string extraction fallback

When dedicated parsers are unavailable, a raw string extraction can catch most
injection payloads at the cost of false positives:

```python
import re

def extract_raw_strings(stream_data, min_length=6):
    """Extract printable ASCII and UTF-16LE strings from raw binary data."""
    ascii_strings = re.findall(rb'[\x20-\x7e]{' + str(min_length).encode() + rb',}', stream_data)
    # UTF-16LE: printable chars interleaved with null bytes
    utf16_strings = re.findall(
        rb'(?:[\x20-\x7e]\x00){' + str(min_length).encode() + rb',}',
        stream_data
    )
    results = [s.decode('ascii') for s in ascii_strings]
    results += [s.decode('utf-16-le') for s in utf16_strings]
    return results
```

---

## References

- **[MS-CFB]** -- Compound File Binary Format: https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-cfb/
- **[MS-DOC]** -- Word Binary File Format: https://learn.microsoft.com/en-us/openspecs/office_file_formats/ms-doc/ccd7b486-7881-484c-a137-51170af7cc22
- **[MS-XLS]** -- Excel Binary File Format: https://learn.microsoft.com/en-us/openspecs/office_file_formats/ms-xls/cd03cb5f-ca02-4934-a391-bb674cb8aa06
- **[MS-PPT]** -- PowerPoint Binary File Format: https://learn.microsoft.com/en-us/openspecs/office_file_formats/ms-ppt/6be79dde-33c1-4c1b-8ccc-4b2301c08662
- **[MS-OVBA]** -- VBA File Format: https://learn.microsoft.com/en-us/openspecs/office_file_formats/ms-ovba/
- **[MS-OLEPS]** -- OLE Property Sets: https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-oleps/
- **[MS-OFFCRYPTO]** -- Office Document Cryptography: https://learn.microsoft.com/en-us/openspecs/office_file_formats/ms-offcrypto/
- **olefile** docs: https://olefile.readthedocs.io/en/latest/
- **oletools** (olevba): https://github.com/decalage2/oletools/wiki/olevba
- **xlrd**: https://xlrd.readthedocs.io/en/latest/
