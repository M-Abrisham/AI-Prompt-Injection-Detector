# Grammarly-Style Real-Time Prompt Injection Detector Research

## Date: 2026-02-15

## Key Finding: No One Has Built This Yet
- No dedicated "Grammarly for prompt injection" browser extension exists (as of Feb 2026)
- This represents a genuine market gap and open-source opportunity
- Closest products are enterprise DLP tools (LayerX, Nightfall, Obsidian Security) that focus on data leakage prevention, NOT prompt injection detection

## Existing Products Landscape

### Enterprise DLP (Data Loss Prevention) - Closest Existing Category
| Product | What It Does | Gap |
|---------|-------------|-----|
| LayerX Security | Browser extension DLP for GenAI, blocks sensitive data in ChatGPT | Only DLP, no injection detection |
| Nightfall AI | Chrome extension, scans ChatGPT prompts for PII/secrets | Only DLP, 100+ data types but no injection patterns |
| Obsidian Security | Browser-level prompt controls, keyword-based policy | DLP focus, keyword matching not ML-based injection detection |

### API-Based Prompt Injection Detection (Server-Side Only)
| Product | Architecture | Client-Side? |
|---------|-------------|-------------|
| Lakera Guard | SaaS API, 100K+ adversarial samples/day | No - API only |
| Arthur Shield | API between app and LLM, binary classifier | No - API only |
| Microsoft Prompt Shields | Azure AI Content Safety service | No - cloud API |
| Google Model Armor | Cloud service | No - cloud API |
| PromptShield (PurpleSec) | AI WAF | No - server-side |

### Open-Source Libraries (Python, Server-Side)
| Library | Latency | Architecture |
|---------|---------|-------------|
| last_layer | >=2ms CPU | AutoML + heuristics + regex, offline, <50MB |
| Rebuff (ProtectAI) | Variable | Heuristics + LLM + VectorDB + canary tokens, has JS SDK |
| pytector | Variable | Transformers-based, HuggingFace models |
| detect-prompt-injection | Variable | FastAPI + trained model |
| LLM Guard | Variable | Multiple scanners, modular |

### Key Library: Rebuff has JavaScript SDK (npm)
- `npm i rebuff` - exists but is a prototype
- Multi-layered: heuristics -> LLM -> VectorDB -> canary tokens

## Grammarly Architecture (Reference Pattern)

### Content Script Approach
1. **Element Detection**: Detects `<textarea>` and `div[contenteditable="true"]` (NOT input fields)
2. **Overlay Rendering**: Renders underlines OUTSIDE the text field using `Range.getClientRects()` API
   - Initially injected underline nodes INTO DOM (caused corruption)
   - Modern approach: overlay on top, no DOM modification of text field
3. **Performance**: Uses MutationObserver sparingly (can degrade page performance)
4. **Cross-site compatibility**: Handles Draft.js, Quill.js, custom editors
5. **Communication**: Content script <-> Background service worker <-> Backend API

### Grammarly SDK (DEPRECATED Jan 2024)
- `@grammarly/editor-sdk` npm package - DISCONTINUED
- Was a web component wrapper: `<grammarly-editor-plugin>`
- Alternative: Sapling SDK (sapling.ai) is a drop-in replacement

### LanguageTool Extension Architecture (Open Source Reference)
- Auto-check uses 1.5-second debounce after typing stops
- Content scripts: contentscript.js (DOM), textarea.js (auto-check)
- Background scripts: background.js (orchestrator), common.js (API client)
- GitHub: languagetool-org/languagetool-browser-addon (old version, deprecated 2018)

## Architecture Options for Our Project

### Option A: FastAPI REST API + Browser Extension (RECOMMENDED - Phase 1)
- **Backend**: FastAPI serving our 20-layer detector at `/api/scan`
- **Extension**: Manifest V3 Chrome extension with content script
- **Communication**: REST API with debounced requests (300-500ms after typing stops)
- **Latency budget**: ~200-300ms round trip acceptable (similar to Grammarly)
- **Pros**: Reuses entire Python codebase, full 20-layer detection, fastest to ship
- **Cons**: Requires network, server costs, latency varies

### Option B: FastAPI WebSocket + Browser Extension (RECOMMENDED - Phase 1.5)
- Same as Option A but persistent WebSocket connection
- **Pros**: Lower latency (~50-100ms per message), connection reuse, streaming
- **Cons**: More complex state management, connection lifecycle

### Option C: Hybrid (Fast Client + Full Server)
- **Client**: JavaScript port of Layer 1 (rules/regex) for instant <5ms feedback
- **Server**: Full 20-layer cascade via API for comprehensive analysis
- **UX**: Immediate red underline from client regex, then refined by server
- **Pros**: Best UX, instant feedback + deep analysis
- **Cons**: Must maintain two codebases (JS rules + Python full stack)

### Option D: Transformers.js In-Browser (EXPERIMENTAL)
- Run a small BERT/DistilBERT prompt injection classifier in browser via ONNX
- Transformers.js supports text-classification pipeline
- Can use WebGPU for acceleration
- **Pros**: Fully offline, no server, privacy-preserving
- **Cons**: Large model download (50-100MB), slower than server, limited to 1 model vs 20 layers

### Option E: Pyodide WASM (NOT RECOMMENDED)
- Run full Python detector in browser via WebAssembly
- Pyodide supports numpy, scikit-learn, pandas
- **Pros**: Reuses Python code directly
- **Cons**: Slow startup (seconds), unpredictable memory, scikit-learn model loading issues, not production-ready

### Option F: LLM Proxy (mitmproxy/liteLLM pattern)
- Sits between user and LLM API, inspects all traffic
- liteLLM already supports guardrail middleware
- **Pros**: Intercepts ALL LLM traffic, not just browser
- **Cons**: Only works for API usage, not for web UI (chatgpt.com, claude.ai)

## Browser Extension Technical Details

### Manifest V3 Key Points
- Content scripts inject into target pages (chatgpt.com, claude.ai, gemini.google.com)
- Background: Service worker (not persistent page)
- Communication: chrome.runtime.sendMessage / chrome.runtime.connect
- Cannot use eval() or remote code
- Must declare host_permissions in manifest

### Content Script Pattern for AI Chat Sites
```
Target elements:
- ChatGPT: div[contenteditable="true"] in main chat area
- Claude: div[contenteditable="true"] or textarea
- Gemini: contenteditable div
```

### Key UX Patterns
1. **Debounced analysis**: 300-500ms after typing pause (LanguageTool uses 1500ms)
2. **Inline warning overlay**: Colored underline or border around input area
3. **Popup with details**: Click underline to see detection details + severity
4. **Pre-send intercept**: Optional - hook send button to warn before submission
5. **Badge indicator**: Extension icon badge showing risk level (green/yellow/red)

## Latency Budget Analysis
| Component | Target | Notes |
|-----------|--------|-------|
| Debounce wait | 300ms | After typing stops |
| Network round trip | 50-100ms | Local/regional server |
| Detection processing | 50-200ms | Depends on layers enabled |
| UI render | 10-20ms | Overlay update |
| **Total user-perceived** | **~400-600ms** | Acceptable per Grammarly standard |

## Security Considerations
- "Man-in-the-Prompt" attack: Malicious extensions can read/modify AI prompts
- Our extension must be VERY careful about what data it sends to backend
- Consider: local-only mode for sensitive prompts
- Extension permissions should be minimal (only AI chat sites, not all URLs)
- Content Security Policy in manifest

## Competitive Landscape Summary
- **Market gap confirmed**: No Grammarly-for-prompt-injection exists
- **DLP tools exist but don't detect injection**: LayerX, Nightfall, Obsidian focus on data leakage
- **API tools exist but have no browser UI**: Lakera, Arthur Shield are server-only
- **Open-source libraries are server-only**: last_layer, Rebuff, pytector
- **Academic research is active**: BrowseSafe, PromptGuard (2025 papers)
- **Our 20-layer detector is uniquely comprehensive**: No competitor has this depth
