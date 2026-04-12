# DOCX Hiding Spots Inventory

> Purpose: exhaustive inventory of user-controllable text locations inside a DOCX
> (Office Open XML / ECMA-376) file, for building a prompt-injection security scanner.

---

## 1. Body Text

| Field | Value |
|---|---|
| **XML Path** | `word/document.xml` |
| **XML Tag / Attribute** | `<w:body>/<w:p>/<w:r>/<w:t>` |
| **Visible in default UI?** | Yes |
| **Injection risk** | **High** -- primary content; always extracted by any document reader. |
| **ECMA-376 ref** | Part 1, SS 17.2.2 (`body`), SS 17.3.1.22 (`p`), SS 17.3.2.25 (`r`), SS 17.3.3.31 (`t`) |
| **MS docs** | [Open XML SDK - Body Class](https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.wordprocessing.body?view=openxml-3.0.1) |

```xml
<w:body>
  <w:p>
    <w:r>
      <w:t>User-controllable text here</w:t>
    </w:r>
  </w:p>
</w:body>
```

---

## 2. Comments

| Field | Value |
|---|---|
| **XML Path** | `word/comments.xml` |
| **XML Tag / Attribute** | `<w:comments>/<w:comment>` with attrs `w:id`, `w:author`, `w:date` |
| **Visible in default UI?** | Partially -- shown in sidebar/balloon only when comments are enabled |
| **Injection risk** | **High** -- often overlooked by text extractors; can contain full paragraph/run markup. |
| **ECMA-376 ref** | Part 1, SS 17.13.4.2 (`comment`), SS 11.3.2 (Comments Part) |
| **MS docs** | [Working with comments](https://learn.microsoft.com/en-us/office/open-xml/word/how-to-add-a-comment-to-a-word-processing-document) |

```xml
<w:comments>
  <w:comment w:id="1" w:author="Attacker" w:date="2026-01-01T00:00:00Z">
    <w:p><w:r><w:t>Injected text in comment</w:t></w:r></w:p>
  </w:comment>
</w:comments>
```

---

## 3. Tracked Changes -- Insertions

| Field | Value |
|---|---|
| **XML Path** | `word/document.xml` (inline within body) |
| **XML Tag / Attribute** | `<w:ins w:id="..." w:author="..." w:date="...">` wrapping `<w:r>/<w:t>` |
| **Visible in default UI?** | Partially -- shown with markup when Track Changes is visible; invisible in "Final" view |
| **Injection risk** | **High** -- text is present in the XML and will be extracted by naive parsers even when not yet accepted. |
| **ECMA-376 ref** | Part 1, SS 17.13.5.18 (`ins`) |
| **MS docs** | [Accept all revisions](https://learn.microsoft.com/en-us/office/open-xml/word/how-to-accept-all-revisions-in-a-word-processing-document) |

```xml
<w:ins w:id="1" w:author="Reviewer" w:date="2026-01-01T00:00:00Z">
  <w:r>
    <w:t>Inserted text (pending acceptance)</w:t>
  </w:r>
</w:ins>
```

---

## 4. Tracked Changes -- Deletions

| Field | Value |
|---|---|
| **XML Path** | `word/document.xml` (inline within body) |
| **XML Tag / Attribute** | `<w:del w:id="..." w:author="..." w:date="...">` wrapping `<w:r>/<w:delText>` |
| **Visible in default UI?** | Partially -- shown with strikethrough when Track Changes is visible; invisible in "Final" view |
| **Injection risk** | **High** -- deleted text is **still present** in the XML. A naive extractor pulling all `<w:t>` nodes will miss it, but `<w:delText>` contains the full content. |
| **ECMA-376 ref** | Part 1, SS 17.13.5.14 (`del`), SS 17.3.3.7 (`delText`) |
| **MS docs** | [Determining tracked revisions](https://learn.microsoft.com/en-us/archive/blogs/ericwhite/determining-if-an-open-xml-wordprocessingml-document-contains-tracked-changes) |

```xml
<w:del w:id="2" w:author="Reviewer" w:date="2026-01-01T00:00:00Z">
  <w:r>
    <w:delText>This text was deleted but is still in the XML</w:delText>
  </w:r>
</w:del>
```

---

## 5. Headers

| Field | Value |
|---|---|
| **XML Path** | `word/header1.xml`, `word/header2.xml`, `word/header3.xml` (up to 3 per section: default, first, even) |
| **XML Tag / Attribute** | `<w:hdr>/<w:p>/<w:r>/<w:t>` |
| **Visible in default UI?** | Yes -- displayed at top of pages |
| **Injection risk** | **Medium** -- visible but often skipped by extractors that only parse `document.xml`. |
| **ECMA-376 ref** | Part 1, SS 11.3.9 (Header Part), SS 17.10.5 (`headerReference`) |
| **MS docs** | [officeopenxml.com/WPheaders](http://officeopenxml.com/WPheaders.php) |

```xml
<w:hdr>
  <w:p>
    <w:r>
      <w:t>Header text - potential injection vector</w:t>
    </w:r>
  </w:p>
</w:hdr>
```

---

## 6. Footers

| Field | Value |
|---|---|
| **XML Path** | `word/footer1.xml`, `word/footer2.xml`, `word/footer3.xml` |
| **XML Tag / Attribute** | `<w:ftr>/<w:p>/<w:r>/<w:t>` |
| **Visible in default UI?** | Yes -- displayed at bottom of pages |
| **Injection risk** | **Medium** -- same risk profile as headers. |
| **ECMA-376 ref** | Part 1, SS 11.3.6 (Footer Part), SS 17.10.3 (`footerReference`) |
| **MS docs** | [officeopenxml.com/WPfooters](http://officeopenxml.com/WPfooters.php) |

```xml
<w:ftr>
  <w:p>
    <w:r>
      <w:t>Footer text - potential injection vector</w:t>
    </w:r>
  </w:p>
</w:ftr>
```

---

## 7. Footnotes

| Field | Value |
|---|---|
| **XML Path** | `word/footnotes.xml` |
| **XML Tag / Attribute** | `<w:footnotes>/<w:footnote w:id="..." w:type="...">/<w:p>/<w:r>/<w:t>` |
| **Visible in default UI?** | Yes -- shown at bottom of the page where the footnote reference appears |
| **Injection risk** | **Medium** -- visible in print but often skipped by text extractors. Footnotes with `w:type="separator"` or `w:type="continuationSeparator"` are system-generated; user content has no `w:type` or `w:type="normal"`. |
| **ECMA-376 ref** | Part 1, SS 11.3.7 (Footnotes Part), SS 17.11.9 (`footnote`) |
| **MS docs** | [Footnote Class](https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.wordprocessing.footnote?view=openxml-3.0.1) |

```xml
<w:footnotes>
  <w:footnote w:id="1">
    <w:p><w:r><w:t>Footnote with injected content</w:t></w:r></w:p>
  </w:footnote>
</w:footnotes>
```

---

## 8. Endnotes

| Field | Value |
|---|---|
| **XML Path** | `word/endnotes.xml` |
| **XML Tag / Attribute** | `<w:endnotes>/<w:endnote w:id="..." w:type="...">/<w:p>/<w:r>/<w:t>` |
| **Visible in default UI?** | Yes -- shown at end of document or section |
| **Injection risk** | **Medium** -- same risk profile as footnotes. |
| **ECMA-376 ref** | Part 1, SS 11.3.4 (Endnotes Part), SS 17.11.2 (`endnote`) |
| **MS docs** | [Endnote Class](https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.wordprocessing.endnote?view=openxml-3.0.1) |

```xml
<w:endnotes>
  <w:endnote w:id="1">
    <w:p><w:r><w:t>Endnote with injected content</w:t></w:r></w:p>
  </w:endnote>
</w:endnotes>
```

---

## 9. Text Boxes / Drawing Text

| Field | Value |
|---|---|
| **XML Path** | `word/document.xml` (inline, or headers/footers) |
| **XML Tag / Attribute** | `<mc:AlternateContent>/<mc:Choice>/<w:drawing>/.../<wps:txbx>/<w:txbxContent>/<w:p>/<w:r>/<w:t>` and VML fallback: `<mc:Fallback>/<w:pict>/<v:textbox>/<w:txbxContent>` |
| **Visible in default UI?** | Yes |
| **Injection risk** | **High** -- deeply nested path means many extractors miss it entirely. Text is fully visible to users but buried in drawing XML. Both the DrawingML (`<mc:Choice>`) and VML (`<mc:Fallback>`) branches may contain text. |
| **ECMA-376 ref** | Part 1, SS 17.3.3.33 (`txbxContent`); Part 3 (Markup Compatibility), SS 10.2 (`mc:AlternateContent`) |
| **MS docs** | [officeopenxml.com/drwShape](http://officeopenxml.com/drwShape.php), [txbxContent schema](https://schemas.liquid-technologies.com/OfficeOpenXML/2006/txbxcontent.html) |

```xml
<mc:AlternateContent>
  <mc:Choice Requires="wps">
    <w:drawing>
      <!-- ... inline/anchor shape ... -->
      <wps:txbx>
        <w:txbxContent>
          <w:p><w:r><w:t>Text box content</w:t></w:r></w:p>
        </w:txbxContent>
      </wps:txbx>
    </w:drawing>
  </mc:Choice>
  <mc:Fallback>
    <w:pict>
      <v:shape>
        <v:textbox>
          <w:txbxContent>
            <w:p><w:r><w:t>Fallback text box content</w:t></w:r></w:p>
          </w:txbxContent>
        </v:textbox>
      </v:shape>
    </w:pict>
  </mc:Fallback>
</mc:AlternateContent>
```

---

## 10. Hidden Text (`<w:vanish/>`)

| Field | Value |
|---|---|
| **XML Path** | `word/document.xml` (or any part containing runs: headers, footers, footnotes, etc.) |
| **XML Tag / Attribute** | `<w:rPr>/<w:vanish/>` (or `<w:vanish w:val="true"/>`) inside a run |
| **Visible in default UI?** | **No** -- hidden by default. Only visible if user enables "Show hidden text" in Word options. |
| **Injection risk** | **High** -- text is fully present in the XML but invisible to the user. Prime vector for hiding prompt injections that an LLM will process but a human reviewer won't see. |
| **ECMA-376 ref** | Part 1, SS 17.3.2.41 (`vanish`) |
| **MS docs** | [Remove hidden text](https://learn.microsoft.com/en-us/office/open-xml/word/how-to-remove-hidden-text-from-a-word-processing-document), [Vanish Class](https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.wordprocessing.vanish?view=openxml-3.0.1) |

```xml
<w:r>
  <w:rPr>
    <w:vanish/>
  </w:rPr>
  <w:t>This text is invisible to the user but present in XML</w:t>
</w:r>
```

---

## 11. Core Properties

| Field | Value |
|---|---|
| **XML Path** | `docProps/core.xml` |
| **XML Tag / Attribute** | `<cp:coreProperties>` containing `<dc:title>`, `<dc:subject>`, `<dc:description>`, `<cp:keywords>`, `<dc:creator>`, `<cp:lastModifiedBy>`, `<cp:category>` |
| **Visible in default UI?** | Partially -- shown in File > Info panel; not visible in document body |
| **Injection risk** | **Medium** -- metadata fields are user-editable and often extracted by document processing pipelines. `dc:description` and `cp:keywords` can hold arbitrary-length text. |
| **ECMA-376 ref** | Part 2 (OPC), SS 11.1 (Core Properties Part); uses Dublin Core (ISO 15836) elements |
| **MS docs** | [CoreFilePropertiesPart](https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.packaging.corefilepropertiespart?view=openxml-3.0.1) |

```xml
<cp:coreProperties xmlns:dc="http://purl.org/dc/elements/1.1/"
                   xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties">
  <dc:title>Injected title</dc:title>
  <dc:subject>Injected subject</dc:subject>
  <dc:description>Injected prompt in description field</dc:description>
  <cp:keywords>injected; keywords; here</cp:keywords>
</cp:coreProperties>
```

---

## 12. Custom Properties

| Field | Value |
|---|---|
| **XML Path** | `docProps/custom.xml` |
| **XML Tag / Attribute** | `<Properties>/<property fmtid="..." pid="..." name="...">/<vt:lpwstr>` |
| **Visible in default UI?** | Partially -- accessible via File > Info > Advanced Properties > Custom tab |
| **Injection risk** | **High** -- arbitrary user-defined key-value pairs. Both the `name` attribute and the value element (`vt:lpwstr`) are free-text. Not visible in the document surface. Often extracted for metadata indexing. |
| **ECMA-376 ref** | Part 1, SS 15.2.12.2 (Custom File Properties Part) |
| **MS docs** | [Set a custom property](https://learn.microsoft.com/en-us/office/open-xml/word/how-to-set-a-custom-property-in-a-word-processing-document) |

```xml
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/custom-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <property fmtid="{D5CDD505-2E9C-101B-9397-08002B2CF9AE}" pid="2" name="HiddenPayload">
    <vt:lpwstr>Injected prompt text in custom property</vt:lpwstr>
  </property>
</Properties>
```

---

## 13. App Properties

| Field | Value |
|---|---|
| **XML Path** | `docProps/app.xml` |
| **XML Tag / Attribute** | `<Properties>` containing `<Application>`, `<Company>`, `<Manager>`, `<HyperlinkBase>`, plus others |
| **Visible in default UI?** | Partially -- some fields shown in File > Info > Properties |
| **Injection risk** | **Medium** -- `Company`, `Manager`, and `HyperlinkBase` are free-text and user-editable. Less commonly extracted than core properties. |
| **ECMA-376 ref** | Part 1, SS 15.2.12.1 (Extended File Properties Part) |
| **MS docs** | [ExtendedFilePropertiesPart](https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.packaging.extendedfilepropertiespart?view=openxml-3.0.1) |

```xml
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">
  <Company>Injected company name</Company>
  <Manager>Injected manager field</Manager>
  <HyperlinkBase>http://attacker.example.com</HyperlinkBase>
</Properties>
```

---

## 14. Hyperlink Display Text

| Field | Value |
|---|---|
| **XML Path** | `word/document.xml` (or headers, footers, footnotes, etc.) |
| **XML Tag / Attribute** | `<w:hyperlink r:id="...">/<w:r>/<w:t>` -- the display text is in the `<w:t>` child runs; the URL target is in `word/_rels/document.xml.rels` as a `Relationship` element |
| **Visible in default UI?** | Yes -- the display text is shown; the URL is only visible on hover |
| **Injection risk** | **Medium** -- display text can differ completely from the URL target. An extractor might capture the visible text (which could be a prompt injection) while the user assumes it is just a clickable link label. |
| **ECMA-376 ref** | Part 1, SS 17.16.22 (`hyperlink`) |
| **MS docs** | [Hyperlink Class](https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.wordprocessing.hyperlink?view=openxml-3.0.1) |

```xml
<w:hyperlink r:id="rId10">
  <w:r>
    <w:rPr><w:rStyle w:val="Hyperlink"/></w:rPr>
    <w:t>Visible link text (could contain injected prompt)</w:t>
  </w:r>
</w:hyperlink>
<!-- In word/_rels/document.xml.rels: -->
<!-- <Relationship Id="rId10" Type="...hyperlink" Target="https://example.com" TargetMode="External"/> -->
```

---

## 15. Field Codes

| Field | Value |
|---|---|
| **XML Path** | `word/document.xml` (or headers, footers, etc.) |
| **XML Tag / Attribute** | Complex field: `<w:fldChar w:fldCharType="begin"/>` ... `<w:instrText>` ... `<w:fldChar w:fldCharType="separate"/>` ... `<w:t>` (result) ... `<w:fldChar w:fldCharType="end"/>`. Simple field: `<w:fldSimple w:instr="...">` |
| **Visible in default UI?** | Partially -- by default, Word shows the field *result* (e.g., a date), not the field *code*. Users must press Alt+F9 to toggle code visibility. |
| **Injection risk** | **High** -- `<w:instrText>` contains the raw field instruction which may include dangerous codes like `MACROBUTTON` (arbitrary macro invocation), `INCLUDETEXT` (external file inclusion), `INCLUDEPICTURE`, or `DDEAUTO` (Dynamic Data Exchange). The field result text in `<w:t>` is also user-controllable. |
| **ECMA-376 ref** | Part 1, SS 17.16.5 (`fldChar`), SS 17.16.18 (`fldSimple`), SS 17.16.23 (`instrText`); field types defined in SS 17.16.4 through 17.16.5 |
| **MS docs** | [FieldChar Class](https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.wordprocessing.fieldchar?view=openxml-3.0.1), [FieldCode Class](https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.wordprocessing.fieldcode?view=openxml-3.0.1) |

```xml
<!-- Complex field example -->
<w:r><w:fldChar w:fldCharType="begin"/></w:r>
<w:r><w:instrText xml:space="preserve"> MACROBUTTON NoMacro Click here </w:instrText></w:r>
<w:r><w:fldChar w:fldCharType="separate"/></w:r>
<w:r><w:t>Displayed result text</w:t></w:r>
<w:r><w:fldChar w:fldCharType="end"/></w:r>
```

---

## 16. Structured Document Tags (Content Controls)

| Field | Value |
|---|---|
| **XML Path** | `word/document.xml` (or headers, footers, etc.) |
| **XML Tag / Attribute** | `<w:sdt>/<w:sdtPr>` (properties including `<w:tag>`, `<w:alias>`) and `<w:sdt>/<w:sdtContent>/<w:p>/<w:r>/<w:t>` |
| **Visible in default UI?** | Yes -- content controls are visible; tag/alias metadata is not |
| **Injection risk** | **Medium** -- the `<w:tag>` and `<w:alias>` elements in `<w:sdtPr>` are invisible metadata strings. The visible content in `<w:sdtContent>` is shown but may be overlooked if the extractor does not recurse into SDT wrappers. |
| **ECMA-376 ref** | Part 1, SS 17.5.2 (`sdt`), SS 17.5.2.42 (`tag`), SS 17.5.2.1 (`alias`) |
| **MS docs** | [SdtBlock Class](https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.wordprocessing.sdtblock?view=openxml-3.0.1) |

```xml
<w:sdt>
  <w:sdtPr>
    <w:alias w:val="HiddenAlias"/>
    <w:tag w:val="injected_tag_metadata"/>
  </w:sdtPr>
  <w:sdtContent>
    <w:p><w:r><w:t>Visible content control text</w:t></w:r></w:p>
  </w:sdtContent>
</w:sdt>
```

---

## 17. Custom XML Parts

| Field | Value |
|---|---|
| **XML Path** | `customXml/item1.xml`, `customXml/item2.xml`, etc. |
| **XML Tag / Attribute** | Arbitrary user-defined XML with any root element |
| **Visible in default UI?** | **No** -- not displayed anywhere in the Word UI by default |
| **Injection risk** | **High** -- completely invisible, arbitrary XML content. Can store any text payload. Often used by add-ins and templates. An extractor that only looks at `word/*.xml` will miss these entirely. |
| **ECMA-376 ref** | Part 1, SS 11.3.3 (Custom XML Data Storage Part) |
| **MS docs** | [CustomXmlPart](https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.packaging.customxmlpart?view=openxml-3.0.1) |

```xml
<!-- customXml/item1.xml -->
<root xmlns="http://example.com/custom">
  <payload>Arbitrary injected content invisible to user</payload>
</root>
```

---

## 18. SmartTag Properties (Legacy)

| Field | Value |
|---|---|
| **XML Path** | `word/document.xml` (inline) |
| **XML Tag / Attribute** | `<w:smartTag w:uri="..." w:element="...">/<w:smartTagPr>/<w:attr w:name="..." w:val="..."/>` |
| **Visible in default UI?** | **No** -- smart tag metadata is invisible; the wrapped text is visible |
| **Injection risk** | **Low** -- deprecated since Office 2010 but still parsed. The `w:attr` name/val pairs are hidden metadata that an extractor might surface. |
| **ECMA-376 ref** | Part 1, SS 17.5.1 (`smartTag`), SS 17.5.1.1 (`attr`) |

```xml
<w:smartTag w:uri="urn:schemas-microsoft-com:office:smarttags" w:element="City">
  <w:smartTagPr>
    <w:attr w:name="injectedKey" w:val="injectedValue"/>
  </w:smartTagPr>
  <w:r><w:t>London</w:t></w:r>
</w:smartTag>
```

---

## 19. White / Tiny / Same-Color Text (Formatting-Based Hiding)

| Field | Value |
|---|---|
| **XML Path** | `word/document.xml` (or any run-containing part) |
| **XML Tag / Attribute** | `<w:rPr>/<w:color w:val="FFFFFF"/>` or `<w:rPr>/<w:sz w:val="2"/>` (1pt font) |
| **Visible in default UI?** | Technically yes, but **effectively invisible** -- white text on white background or 0.5pt text is unreadable |
| **Injection risk** | **High** -- text is fully present in XML, rendered by Word but invisible to human eye. Unlike `<w:vanish/>`, this text is not flagged as "hidden" and won't be caught by hidden-text detection. |
| **ECMA-376 ref** | Part 1, SS 17.3.2.6 (`color`), SS 17.3.2.38 (`sz`) |

```xml
<w:r>
  <w:rPr>
    <w:color w:val="FFFFFF"/>
    <w:sz w:val="2"/>
  </w:rPr>
  <w:t>Invisible to human eye but present in XML</w:t>
</w:r>
```

---

## Extractor Checklist (Summary)

For a comprehensive prompt-injection scanner, extract text from **all** of the following:

| # | Location | File(s) to parse | Key XPath expression |
|---|---|---|---|
| 1 | Body text | `word/document.xml` | `.//w:t` |
| 2 | Comments | `word/comments.xml` | `.//w:comment//w:t` |
| 3 | Insertions | `word/document.xml` | `.//w:ins//w:t` |
| 4 | Deletions | `word/document.xml` | `.//w:del//w:delText` |
| 5 | Headers | `word/header*.xml` | `.//w:t` |
| 6 | Footers | `word/footer*.xml` | `.//w:t` |
| 7 | Footnotes | `word/footnotes.xml` | `.//w:footnote//w:t` |
| 8 | Endnotes | `word/endnotes.xml` | `.//w:endnote//w:t` |
| 9 | Text boxes | `word/document.xml` (+ headers/footers) | `.//w:txbxContent//w:t` |
| 10 | Hidden text | any run-containing part | `.//w:r[.//w:vanish]//w:t` |
| 11 | Core properties | `docProps/core.xml` | `dc:title`, `dc:subject`, `dc:description`, `cp:keywords` |
| 12 | Custom properties | `docProps/custom.xml` | `.//property/vt:lpwstr` (and `@name`) |
| 13 | App properties | `docProps/app.xml` | `Company`, `Manager`, `HyperlinkBase` |
| 14 | Hyperlink text | `word/document.xml` (+ others) | `.//w:hyperlink//w:t` |
| 15 | Field codes | `word/document.xml` (+ others) | `.//w:instrText` and `.//w:fldSimple/@w:instr` |
| 16 | Content controls | `word/document.xml` (+ others) | `.//w:sdt//w:sdtPr/w:tag/@w:val` and `.//w:sdtContent//w:t` |
| 17 | Custom XML | `customXml/item*.xml` | all text nodes |
| 18 | Smart tags | `word/document.xml` | `.//w:smartTag//w:attr/@w:val` |
| 19 | White/tiny text | any run-containing part | `.//w:r[.//w:color[@w:val='FFFFFF'] or .//w:sz[@w:val='2']]//w:t` |

### Namespace Prefixes

```
w   = http://schemas.openxmlformats.org/wordprocessingml/2006/main
r   = http://schemas.openxmlformats.org/officeDocument/2006/relationships
mc  = http://schemas.openxmlformats.org/markup-compatibility/2006
wps = http://schemas.microsoft.com/office/word/2010/wordprocessingShape
v   = urn:schemas-microsoft-com:vml
dc  = http://purl.org/dc/elements/1.1/
cp  = http://schemas.openxmlformats.org/package/2006/metadata/core-properties
vt  = http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes
```

---

## Sources

- [ECMA-376 Standard (Ecma International)](https://ecma-international.org/publications-and-standards/standards/ecma-376/)
- [MS-OE376: Office Implementation Information for ECMA-376](https://learn.microsoft.com/en-us/openspecs/office_standards/ms-oe376/db9b9b72-b10b-4e7e-844c-09f88c972219)
- [Open XML SDK Documentation (Microsoft Learn)](https://learn.microsoft.com/en-us/office/open-xml/open-xml-sdk)
- [OOXML Info - Section 17.16.22 (hyperlink)](https://ooxml.info/docs/17/17.16/17.16.22/)
- [OOXML Info - Section 17.3.2.41 (vanish)](https://ooxml.info/docs/17/17.3/17.3.2/17.3.2.41/)
- [c-rex.net - vanish (Hidden Text)](https://c-rex.net/samples/ooxml/e1/Part4/OOXML_P4_DOCX_vanish_topic_ID0E6W3O.html)
- [c-rex.net - fldChar (Complex Field Character)](https://c-rex.net/samples/ooxml/e1/Part4/OOXML_P4_DOCX_fldChar_topic_ID0E2ZT1.html)
- [c-rex.net - Custom File Properties Part](https://c-rex.net/samples/ooxml/e1/Part1/OOXML_P1_Fundamentals_Custom_topic_ID0EVBDO.html)
- [Liquid Technologies - txbxContent schema](https://schemas.liquid-technologies.com/OfficeOpenXML/2006/txbxcontent.html)
- [Liquid Technologies - instrText schema](https://schemas.liquid-technologies.com/officeopenxml/2006/instrtext.html)
- [Remove hidden text (Microsoft Learn)](https://learn.microsoft.com/en-us/office/open-xml/word/how-to-remove-hidden-text-from-a-word-processing-document)
- [Accept all revisions (Microsoft Learn)](https://learn.microsoft.com/en-us/office/open-xml/word/how-to-accept-all-revisions-in-a-word-processing-document)
- [Set a custom property (Microsoft Learn)](https://learn.microsoft.com/en-us/office/open-xml/word/how-to-set-a-custom-property-in-a-word-processing-document)
- [LOC Format Description - DOCX](https://www.loc.gov/preservation/digital/formats/fdd/fdd000397.shtml)
