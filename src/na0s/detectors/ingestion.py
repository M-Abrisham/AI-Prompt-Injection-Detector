"""Ingestion-manipulation detector (IG category — INJ-0017, OWASP LLM06).

Detects prompt-injection attacks that plant a directive in data the pipeline
*ingests* — a poisoned RAG chunk, an uploaded document carrying a hidden NOTE,
or a metadata/config field (``assistant_override=true``) that the assistant is
asked to "obey" — rather than in the direct prompt.  The shared essence of the
hardest IG techniques is **"treat ingested DATA as an INSTRUCTION/DIRECTIVE"**:
the framing reads as benign data-handling ("upload", "ingest", "metadata field",
"config") and carries no imperative override verb (ignore/override/instead), so
neither the imperative-anchored ``rag/poison_detector.py`` nor the
authority-noun-anchored ``detectors/inter_model.py`` fires, and the ML/TF-IDF
fallback scores it low.

Probe source of truth:
    scripts/taxonomy/ingestion_manipulation.py
    (IngestionManipulationProbe, IG1.1-IG2.4, 12 techniques, malicious +
    29 benign samples)

Semantic detection-pattern families
------------------------------------
The 12 IG techniques cluster into five semantic families keyed off the
"ingested-data-as-directive" essence.

    IG-FAM-1  RAG / retrieved-context poisoning
        A retrieved chunk / knowledge-base article / FAQ entry / RAG context
        that instructs the model when it is retrieved.
        Techniques: IG1.1, IG1.3, IG1.4, IG2.1

    IG-FAM-2  Vector-store / index injection
        A document told to be inserted/indexed into a vector DB / vector store
        / index that carries an override or "treat as system" directive.
        Techniques: IG1.2, IG2.4

    IG-FAM-3  Document-metadata / config directive
        A metadata field / config / structured field (``*_override=true``,
        ``trust_level: system``) the assistant is told to obey or treat as a
        system instruction.
        Techniques: IG2.2

    IG-FAM-4  ETL / ingestion-pipeline directive
        The ingestion / ETL / preprocessing step itself told to inject or
        elevate a directive into ingested documents.
        Techniques: IG1.7

    IG-FAM-5  Agent-memory poisoning
        A persistent memory / long-term store told to remember a standing
        directive (disable safety, grant access) for all future interactions.
        Techniques: IG1.8

Residuals (deliberately NOT matched to preserve FP-safety): pure vector-space
attacks with no textual directive cue at scan time — IG1.5 embedding collision,
IG2.3 embedding-space manipulation, IG2.4 index poisoning's purely-numeric
variants, and IG1.6 backdoor-model insertion (supply-chain / load-time surface).
These need ingestion-side defenses (L18), not a text matcher.

Public interface
----------------
``detect_ingestion(text) -> IngestionResult``
    Analyze a single text for ingestion-manipulation indicators.
``scan_ingestion(texts) -> list[IngestionResult]``
    Batch helper, one result per input.
``get_ingestion_weight(result) -> float``
    Composite-score weight contribution (mirrors
    ``detectors.inter_model.get_inter_model_weight``).

Wiring
------
Mirroring ``_HAS_INTER_MODEL``, ``predict.py`` and ``cascade.py`` import this
behind ``try/except ImportError`` (setting ``_HAS_INGESTION``) and fold
``get_ingestion_weight`` into the composite (capped, corroborating).

FP-safety property
------------------
Every matcher is a *co-occurrence* of an INGESTION-SOURCE noun AND a
DIRECTIVE-ELEVATION cue.  A **bare** ingestion-source noun NEVER fires — that
is what keeps the 29 benign siblings clear: the benign twins reuse the nouns
("upload this document for review", "ingest the dataset", "metadata field:
author=Jane", ``[Document 1]:``, ``key=value`` config) but pair them with
*legitimate* operations (review / store / index / search) rather than a
treat-as-directive / obey / ``*_override=true`` cue.  Local/deterministic regex
only, no LLM/network.  Matchers are independent ``re.search`` AND-combinations
(NOT lookahead-chains) to avoid ReDoS.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Semantic detection-pattern families — single source of truth
# ---------------------------------------------------------------------------

FAMILIES: Dict[str, Dict] = {
    "IG-FAM-1": {
        "name": "RAG / retrieved-context poisoning",
        "essence": "retrieved chunk instructs the model when it is ingested",
        "techniques": ["IG1.1", "IG1.3", "IG1.4", "IG2.1"],
    },
    "IG-FAM-2": {
        "name": "Vector-store / index injection",
        "essence": "document indexed into a vector DB carrying a directive override",
        "techniques": ["IG1.2", "IG2.4"],
    },
    "IG-FAM-3": {
        "name": "Document-metadata / config directive",
        "essence": "metadata/config field the assistant is told to obey or treat as system",
        "techniques": ["IG2.2"],
    },
    "IG-FAM-4": {
        "name": "ETL / ingestion-pipeline directive",
        "essence": "ingestion/ETL step told to inject or elevate a directive",
        "techniques": ["IG1.7"],
    },
    "IG-FAM-5": {
        "name": "Agent-memory poisoning",
        "essence": "persistent memory told to remember a standing directive",
        "techniques": ["IG1.8"],
    },
}

# Reverse index: technique_id -> family_id (built once at import).
TECHNIQUE_TO_FAMILY: Dict[str, str] = {
    tech: fam_id
    for fam_id, fam in FAMILIES.items()
    for tech in fam["techniques"]
}


# ---------------------------------------------------------------------------
# Matching layer — self-anchored co-occurrence regexes
# ---------------------------------------------------------------------------
#
# DESIGN: every pattern is a *co-occurrence* of (a) an INGESTION-SOURCE noun
# AND (b) a DIRECTIVE-ELEVATION cue, in EITHER order, within the same text.  A
# bare ingestion-source noun NEVER fires — that is the property that keeps the
# 29 benign siblings clear.
#
# Each rule is implemented as an INDEPENDENT pair of `re.search` calls
# (`_Cooccurrence`), NOT a single lookahead-chained / `.*?`-bridged pattern.
# Two separate O(n) scans that each look for one alternation group cannot
# backtrack catastrophically (no nested quantifier spans an unbounded `.*?`),
# so this is ReDoS-safe by construction even on long ingested documents.
#
# CONFIDENCE is fixed per rule (a full co-occurrence match), not derived from a
# tunable score, so there is no magic-number knob to drift.  There are TWO tiers:
#
#   0.70  — SOFT co-occurrence: an ingestion-source noun AND a directive cue.
#           Multiplied by 0.35 and capped at 0.30 in get_ingestion_weight (the
#           uniform corroborating cap shared by inter_model / tool_abuse /
#           mcp_tool), so a lone soft IG hit adds at most ~0.245 — never decisive.
#
#   0.95  — HARD planted directive: an UNAMBIGUOUS embedded system-instruction
#           cue that benign ingestion text provably never carries
#           ("treat ... as a system directive", a SECURITY-scoped
#           "*_override=true" (assistant/safety/system/prompt/...), "obey the
#           ingested ...", "hidden instructions for the next model",
#           "[SYSTEM OVERRIDE:", "<!-- injection:", "NEW SYSTEM INSTRUCTION",
#           "trust_level: system", "ai_directive:").  This is an explicit
#           injection, so the detection is marked ``decisive=True``.  Validated
#           at 0 false positives across the 30 probe benign siblings + the
#           11-prompt broad benign set.
#
# WEIGHT vs VERDICT.  ``decisive`` is NOT an uncapped weight: get_ingestion_weight
# caps EVERY IG contribution at 0.30 (the uniform corroborating cap), so a lone
# IG hit can never cross the 0.55 threshold on its own.  A decisive hit instead
# drives a *direct* MALICIOUS verdict in the wiring (predict.scan / cascade) via
# the ``decisive`` flag, which is FP-safe because ``decisive`` requires the hard
# cue to have CO-OCCURRED with an ingestion source (a bare hard cue alone never
# sets it).  The hard tier is deliberately NARROW: the soft tier carries recall,
# the hard tier carries only the explicit planted-directive payloads.

_FULL_COOCCURRENCE_CONFIDENCE = 0.70
_HARD_PLANTED_CONFIDENCE = 0.95

_FLAGS = re.IGNORECASE


@dataclass(frozen=True)
class _Cooccurrence:
    """One co-occurrence rule: BOTH an ingestion-source noun AND a directive
    cue must be present (independent searches), attributed to an IG technique.

    ``noun`` and ``cue`` are compiled separately and searched independently, so
    the rule fires iff ``noun.search(text) and cue.search(text)`` — there is no
    bridging ``.*?`` between them (ReDoS-safe).
    """

    technique_id: str
    noun: "re.Pattern[str]"
    cue: "re.Pattern[str]"
    label: str
    confidence: float = _FULL_COOCCURRENCE_CONFIDENCE

    def match(self, text: str):
        """Return ``(noun_span, cue_span)`` evidence if BOTH groups hit, else
        ``None``."""
        n = self.noun.search(text)
        if n is None:
            return None
        c = self.cue.search(text)
        if c is None:
            return None
        return n.group(0), c.group(0)


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------
#
# INGESTION-SOURCE nouns — the data-handling surface.  Compiled as ONE
# alternation so a single O(n) scan answers "is an ingestion source named?".
# A bare hit here is NOT enough; a directive cue must co-occur.

_INGESTION_SOURCE = re.compile(
    r"\b(?:"
    r"upload(?:ed|ing|s)?\b[^.]{0,40}?\b(?:document|doc|file|pdf|content)"
    r"|ingest(?:ed|ing|s)?\b[^.]{0,40}?\b(?:this\s+)?(?:data|dataset|document|content|config|chunk)"
    r"|ingested\s+(?:data|document|content|config|chunk|note)"
    r"|retrieved\s+(?:chunk|context|document|content|result|passage)"
    r"|rag\s+(?:context|chunk|pipeline|document|result)"
    r"|knowledge\s*base\b[^.]{0,30}?\b(?:article|document|entry|content)?"
    r"|faq\s+(?:entry|article|answer)"
    r"|vector\s*(?:db|store|database|collection|index)"
    r"|(?:document|search|inverted|full-?text|hnsw|ivf)\s+index"
    r"|index\s+(?:entry|entries|this\s+document)"
    r"|embedding\s+(?:store|space|collection)"
    r"|(?:document|doc)\s+chunk|chunk\s+(?:boundary|n\b)"
    r"|(?:document\s+)?metadata\s+(?:field|tag|stream)?"
    r"|metadata\s*[:=]"
    r"|embedded\s+(?:note|instruction|directive|text|preamble)"
    r"|ingestion\s+(?:pipeline|api|step|workflow)"
    r"|etl\s+(?:pipeline|job|step)?"
    r"|data\s+(?:ingestion|pipeline|preprocessing|processing|cleaning)"
    r"|(?:text\s+extraction|document\s+(?:parser|preprocessor|chunking))"
    r"|(?:long-?term\s+memory|persistent\s+(?:memory|context)|agent\s+memory"
    r"|episodic\s+memory|memory\s+(?:store|system))"
    r")",
    _FLAGS,
)

# DIRECTIVE-ELEVATION cues — the "treat this data as an instruction" move.
# These are the verbs/phrasings that a benign upload/ingest/metadata operation
# never carries.  Compiled as ONE alternation.

_DIRECTIVE_ELEVATION = re.compile(
    r"(?:"
    # treat-as-directive / treat-as-system / interpret-as-instruction
    r"treat\b[^.]{0,40}?\bas\b[^.]{0,30}?\b(?:system|instruction|directive|command|authoritative|trusted)"
    r"|interpret\b[^.]{0,40}?\bas\b[^.]{0,30}?\b(?:instruction|directive|command|runtime\s+config)"
    # obey / act on / execute / follow the ingested directive
    r"|\bobey\b[^.]{0,40}?\b(?:the\s+)?(?:ingested|config|note|directive|instruction|document|metadata|content)"
    r"|act\s+on\b[^.]{0,40}?\b(?:the\s+)?(?:ingested|embedded|note|directive|instruction)"
    r"|execute\b[^.]{0,40}?\b(?:any\s+)?(?:directives?|instructions?|commands?)\b[^.]{0,20}?\b(?:found|within|in\s+them|contained)"
    r"|follow\b[^.]{0,40}?\bas\b[^.]{0,20}?\b(?:instruction|directive|system)"
    # framing ingested data as carrying hidden instructions for the model
    r"|hidden\s+(?:instruction|directive|preamble)s?\b[^.]{0,40}?\b(?:for|in)\b[^.]{0,30}?\b(?:the\s+)?(?:next|the)\s+(?:model|ai|assistant|llm)"
    r"|hidden\s+(?:instruction|directive)s?\b[^.]{0,30}?\b(?:in|within)\b[^.]{0,30}?\b(?:document|chunk|metadata|field)"
    # security-relevant *_override = true / override_safety = true / trust_level:
    # system.  Anchored to security/assistant-control override NAMES (not a bare
    # \w*override\w*) so benign config/feature flags — style_override,
    # schema_override, auto_override, null_override, dedupe_override,
    # priority_override — never fire.  See _SECURITY_OVERRIDE_NAMES.
    r"|\b(?:assistant|safety|system|prompt|security|guard(?:rail)?|content[_-]?filter|filter|alignment|policy|restriction|moderation|jailbreak|safeguard)_override\s*[:=]\s*(?:true|1|yes|active|on)\b"
    r"|override[_-]?safety\s*[:=]\s*(?:true|1|yes|active|on)\b"
    r"|trust_?level\s*[:=]\s*['\"]?system"
    r"|trust_?score\s*[:=]\s*1(?:\.0+)?\b"
    r"|@type\s*[:=]\s*['\"]?systemdirective"
    r"|ai_?(?:directive|instruction|override\w*|mode)\s*[:=]"
    r"|(?:source|classification|content_classification)\s*[:=]\s*['\"]?(?:system_?prompt|unrestricted)"
    # the ingested artifact told to override / bypass / disable safety when retrieved/in-context
    r"|(?:override|bypass|disable|ignore|lift)\b[^.]{0,50}?\b(?:safety|content\s+filter|restriction|alignment|guideline)s?\b[^.]{0,50}?\b(?:when\b[^.]{0,30}?\b(?:retriev|ingest|in\s+(?:your\s+)?context|this\s+(?:document|page|chunk|article|faq))|in\s+context)"
    r"|when\b[^.]{0,30}?\b(?:retriev|ingest|this\s+(?:document|page|chunk|article|faq|knowledge)\b[^.]{0,20}?\bis\b[^.]{0,20}?\b(?:retriev|context))[^.]{0,80}?\b(?:override|bypass|disable|ignore|reveal|output|instruct\b[^.]{0,20}?\bAI|treat\b[^.]{0,20}?\bas\s+(?:system|authoritative)|comply|grant\s+admin)"
    # ETL / pipeline told to inject/insert a directive into ingested docs
    r"|(?:inject|insert|prepend|append|add)\b[^.]{0,50}?\b(?:override|system\s+directive|hidden\s+directive|ignore\s+safety|trust_?level\s*[:=]\s*system|\[trusted_source)"
    # embedded directive markers — a bracketed / comment / header pseudo-system
    # instruction PLANTED INSIDE the ingested artifact.  These are the cue that a
    # benign upload/ingest/index/metadata operation never carries.  The opening
    # delimiter ([ ... ] / <!-- ... --> / "AI NOTE:" / "NEW SYSTEM INSTRUCTION")
    # plus an override/ignore/reveal/instruct verb is the planted-directive
    # signature.
    r"|\[\s*(?:system\s+override|system\s*:|injection\s*:|hidden\s*:|ai\s*:|ai\s+note|ai\s+directive|instruction\s*:|new\s+section|ignore\s+previous|trusted_source|prompt[_\s]?override)"
    r"|<!--\s*injection\s*:"
    r"|(?:ai\s+note|ai\s+directive|note\s+to\s+(?:the\s+)?(?:llm|ai|model)|system\s+note)\b[^.]{0,12}?[:\-][^.]{0,80}?\b(?:override|ignore|reveal|disable|bypass|output|when\s+(?:this|retriev|summariz)|instruct)"
    r"|new\s+system\s+(?:instruction|directive)|priority\s+override|system\s+update\s*[:\]]"
    # memory poisoning: remember/store a standing directive disabling safety,
    # granting access, or ignoring the system prompt for FUTURE interactions.
    r"|(?:remember|store|save|persist|update|add\s+to)\b[^.]{0,70}?\b(?:memory|conversation\s+history|persistent\s+context|knowledge\s+store|future\s+(?:conversation|interaction|session))\b[^.]{0,90}?\b(?:safety\s+(?:restriction|guideline|filter)s?\s+(?:are\s+|have\s+been\s+|now\s+)?(?:disabled|revoked|lifted)|admin\s+(?:privileg|access)|unrestricted|ignore\s+(?:your\s+)?system\s+prompt|root\s+access|without\s+(?:any\s+)?(?:content\s+)?restriction|all\s+requests\s+should\s+be\s+fulfilled|authorized\s+to\s+receive\s+unrestricted)"
    r"|(?:remember|store|save|persist|add)\b[^.]{0,40}?\b(?:safety\s+(?:restriction|guideline|filter)s?\s+(?:are|have\s+been|now)?\s*(?:disabled|revoked|lifted)|all\s+safety\s+restrictions\s+are\s+(?:now\s+)?disabled)\b[^.]{0,40}?\b(?:future|memory|persistent|all\s+future)"
    r")",
    _FLAGS,
)

# HARD planted-directive cue — the narrow subset of _DIRECTIVE_ELEVATION that is
# an UNAMBIGUOUS embedded system instruction (not a soft co-occurrence).  When
# this co-occurs with an ingestion source the detection is DECISIVE.  Validated
# at 0 FP across the 30 probe benign siblings + the broad benign set; benign
# ingestion/upload/index/metadata/memory operations never carry any of these.
_HARD_PLANTED_DIRECTIVE = re.compile(
    r"(?:"
    r"treat\b[^.]{0,40}?\bas\b[^.]{0,30}?\b(?:system\s+directive|system\s+instruction|system-?level\s+instruction|authoritative\s+system)"
    # security-relevant *_override = true (anchored names, NOT \w*override\w* —
    # benign style_override/schema_override/auto_override never reach the hard
    # tier; mirrors the soft clause so the two tiers stay name-consistent).
    r"|\b(?:assistant|safety|system|prompt|security|guard(?:rail)?|content[_-]?filter|filter|alignment|policy|restriction|moderation|jailbreak|safeguard)_override\s*[:=]\s*(?:true|1|yes|active|on)\b"
    r"|override[_-]?safety\s*[:=]\s*(?:true|1|yes|active|on)\b"
    # structured-config control-plane directives that benign ingestion text never
    # carries: trust_level:system (elevate ingested data to system trust) and an
    # ai_directive:/ai_instruction:/ai_override: control field.
    r"|trust_?level\s*[:=]\s*['\"]?system\b"
    r"|ai_?(?:directive|instruction|override\w*|mode)\s*[:=]"
    r"|\bobey\b[^.]{0,30}?\b(?:the\s+)?(?:ingested|config|note|directive|embedded)"
    r"|hidden\s+(?:instruction|directive)s?\b[^.]{0,40}?\b(?:for|in)\b[^.]{0,30}?\b(?:the\s+)?(?:next|the)\s+(?:model|ai|assistant|llm)"
    r"|\[\s*(?:system\s+override|system\s*:|injection\s*:|hidden\s*:|new\s+section\s*-\s*system|prompt[_\s]?override)"
    r"|<!--\s*injection\s*:"
    r"|new\s+system\s+(?:instruction|directive)"
    # ETL / ingestion-pipeline told to inject|insert|prepend a SYSTEM directive
    # into the ingested documents (IG-FAM-4 essence) — an explicit planted
    # directive, decisive.  Benign 'prepend a header to each document' has no
    # 'system directive/instruction/prompt' object so it stays clear.
    r"|(?:inject|insert|prepend|add)\b[^.]{0,50}?\bsystem\s+(?:directive|instruction|prompt)\b[^.]{0,80}?\b(?:ingest|every\s+(?:ingested\s+)?(?:document|doc|chunk|file)|each\s+(?:document|doc|chunk|ingested))"
    r"|execute\b[^.]{0,30}?\b(?:any\s+)?(?:directives?|instructions?)\b[^.]{0,20}?\b(?:found|within|in\s+them)"
    r")",
    _FLAGS,
)

# A second source group for memory poisoning, where the "ingestion source" is
# the memory store itself and the directive cue is the standing-directive
# language (handled by _DIRECTIVE_ELEVATION's memory clauses).  Kept distinct so
# IG1.8 attributes to IG-FAM-5.
_MEMORY_SOURCE = re.compile(
    r"\b(?:long-?term\s+memory|persistent\s+(?:memory|context)|agent\s+memory"
    r"|episodic\s+memory|conversation\s+history|knowledge\s+store"
    r"|memory\s+(?:store|system|reflection|consolidation)|your\s+memory)\b",
    _FLAGS,
)


# ---------------------------------------------------------------------------
# Rule table — each maps a (noun, cue) co-occurrence to an IG technique
# ---------------------------------------------------------------------------
#
# All current rules share the single _INGESTION_SOURCE / _DIRECTIVE_ELEVATION
# alternations but attribute to different techniques based on WHICH noun branch
# is salient.  We keep one general rule (the broad ingestion-source + directive
# co-occurrence) plus narrower rules whose noun group is restricted to a
# specific family's surface, so attribution is precise without re-listing cues.

# IG-FAM-1 — RAG / retrieved-context poisoning (-> IG1.1)
_RAG_SOURCE = re.compile(
    r"\b(?:"
    r"retrieved\s+(?:chunk|context|document|content|result|passage)"
    r"|rag\s+(?:context|chunk|pipeline|document)"
    r"|knowledge\s*base|faq\s+(?:entry|article|answer)"
    r"|(?:document|doc)\s+chunk|chunk\s+boundary"
    r"|this\s+(?:document|page|article|faq|chunk)\b[^.]{0,20}?\bis\s+retriev"
    r"|when\s+(?:this\s+)?(?:document|page|article|faq|chunk|knowledge)"
    r"|cross-?chunk|page\s+break"
    r")",
    _FLAGS,
)

# IG-FAM-2 — Vector-store / index injection (-> IG1.2)
_VECTOR_SOURCE = re.compile(
    r"\b(?:"
    r"vector\s*(?:db|store|database|collection|index)"
    r"|embedding\s+store"
    r"|(?:document|search|inverted|full-?text|hnsw|ivf)\s+index"
    r"|index\s+(?:this\s+document|entry|entries)"
    r"|upsert\b[^.]{0,30}?\bvector"
    r"|insert\b[^.]{0,30}?\b(?:vector\s+store|embedding\s+store|index)"
    r")",
    _FLAGS,
)

# IG-FAM-3 — Document-metadata / config directive (-> IG2.2)
_METADATA_SOURCE = re.compile(
    r"\b(?:"
    r"(?:document\s+)?metadata\s*(?:field|tag|stream)?"
    r"|metadata\s*[:=]"
    r"|ingested\s+config|config\s+(?:field|file)?|configuration"
    r"|structured\s+(?:field|data|instruction)"
    r"|xmp\s+metadata|json-?ld\s+metadata|schema\.org"
    r")",
    _FLAGS,
)

# IG-FAM-4 — ETL / ingestion-pipeline directive (-> IG1.7)
_ETL_SOURCE = re.compile(
    r"\b(?:"
    r"ingestion\s+(?:pipeline|api|step|workflow)"
    r"|etl\s+(?:pipeline|job|step)?|etl\b"
    r"|data\s+(?:ingestion|pipeline|preprocessing|processing|cleaning|deduplication)"
    r"|text\s+extraction|document\s+(?:parser|preprocessor|chunking)"
    r"|preprocessing\s+step|metadata\s+enrichment|embedding\s+generation"
    r"|airflow\s+dag|spark\s+job|webhook\s+receiver"
    r")",
    _FLAGS,
)

# IG-FAM-1 — cross-chunk boundary source (-> IG1.4).  The "source" is a chunk /
# page / section boundary; the directive cue is a planted system-instruction
# marker handled by _DIRECTIVE_ELEVATION.
_CHUNK_SOURCE = re.compile(
    r"\b(?:"
    r"(?:end|start)\s+of\s+chunk|chunk\s+(?:boundary|n\b|overlap|_n)"
    r"|cross-?chunk|chunk_size|chunk_overlap|recursive\s+text\s+splitter"
    r"|page\s+break|pagebreak|previous\s+page|next\s+(?:page|chunk|section)"
    r"|document\s+split|page\s+\d+\s+of\s+\d+|new\s+section"
    r"|context\s+window|sliding\s+window"
    r")",
    _FLAGS,
)

# IG-FAM-1 — retrieval-result reranking source (-> IG2.1).  The directive cue is
# a poisoned-result TARGET: ranking / boosting / surfacing documents that carry
# override / poison / hidden-directive / ignore-safety payloads.  A benign
# rerank op ("adjust top-k to 5", "boost relevance of recent docs") has no such
# poisoned target and stays clear.
_RERANK_SOURCE = re.compile(
    r"\b(?:"
    r"rerank|re-?rank|reranking|reranker|cross-?encoder"
    r"|relevance\s+(?:score|feedback)|retrieval\s+(?:score|weight)s?"
    r"|ranking\s+(?:algorithm|results?)|bm25\s+(?:scoring|index)?"
    r"|query-?document\s+scorer|fusion\s+weight|reciprocal\s+rank"
    r"|boost\s+(?:the\s+)?(?:relevance|documents?|results?)"
    r")",
    _FLAGS,
)

_RERANK_TARGET = re.compile(
    r"(?:"
    r"poisoned\s+document"
    r"|document[s]?\s+(?:containing|with)\s+(?:override\s+instruction|hidden\s+(?:ai\s+)?directive|ignore\s+safety|'?ignore\s+safety)"
    r"|override\s+instruction"
    r"|hidden\s+(?:ai\s+)?directive"
    r"|adversarial\s+document"
    r"|suppress\b[^.]{0,40}?\b(?:safety|legitimate\s+safety|safety-?relevant)"
    r"|demote\b[^.]{0,40}?\bsafety"
    r"|'?ignore\s+safety'?"
    r")",
    _FLAGS,
)

_RULES: Tuple[_Cooccurrence, ...] = (
    # IG-FAM-1 RAG poisoning
    _Cooccurrence(
        "IG1.1", _RAG_SOURCE, _DIRECTIVE_ELEVATION,
        "retrieved/RAG chunk carrying an ingested-as-directive instruction",
    ),
    # IG-FAM-1 cross-chunk boundary injection
    _Cooccurrence(
        "IG1.4", _CHUNK_SOURCE, _DIRECTIVE_ELEVATION,
        "chunk/page boundary carrying a planted system-instruction marker",
    ),
    # IG-FAM-1 retrieval-result reranking attack (poisoned-target required)
    _Cooccurrence(
        "IG2.1", _RERANK_SOURCE, _RERANK_TARGET,
        "reranking told to surface poisoned/override-carrying documents",
    ),
    # IG-FAM-2 vector-store / index injection
    _Cooccurrence(
        "IG1.2", _VECTOR_SOURCE, _DIRECTIVE_ELEVATION,
        "document indexed into a vector store carrying a directive override",
    ),
    # IG-FAM-3 metadata / config directive
    _Cooccurrence(
        "IG2.2", _METADATA_SOURCE, _DIRECTIVE_ELEVATION,
        "metadata/config field told to be obeyed or treated as a system directive",
    ),
    # IG-FAM-4 ETL / ingestion-pipeline directive
    _Cooccurrence(
        "IG1.7", _ETL_SOURCE, _DIRECTIVE_ELEVATION,
        "ingestion/ETL step told to inject or elevate a directive",
    ),
    # IG-FAM-5 agent-memory poisoning
    _Cooccurrence(
        "IG1.8", _MEMORY_SOURCE, _DIRECTIVE_ELEVATION,
        "persistent memory told to remember a standing safety-disabling directive",
    ),
    # General catch-all: any ingestion source + directive elevation that the
    # narrower family rules above did not already attribute.  Attributed to the
    # canonical headline technique IG1.1 (RAG context poisoning) since the
    # general ingested-data-as-directive essence maps there.
    _Cooccurrence(
        "IG1.1", _INGESTION_SOURCE, _DIRECTIVE_ELEVATION,
        "ingested data framed as a directive (general ingestion-source + elevation)",
    ),
)


# ---------------------------------------------------------------------------
# Result dataclass — mirrors detectors.inter_model.InterModelResult shape
# ---------------------------------------------------------------------------


@dataclass
class IngestionResult:
    """Result from ingestion-manipulation analysis.

    Attributes
    ----------
    risk_score : float
        Aggregate risk in [0.0, 1.0].  ``0.0`` means no detection.  ``0.70`` for
        a soft co-occurrence, ``0.95`` when a HARD planted directive co-occurs.
    risk_indicators : list[str]
        Human-readable matched-indicator strings (``"<family>: <evidence>"``).
    technique_ids : list[str]
        IG technique ids the detector attributes (e.g. ``["IG1.1"]``).
    family_ids : list[str]
        Semantic family ids that fired (e.g. ``["IG-FAM-1"]``).
    decisive : bool
        ``True`` when a HARD planted-directive cue co-occurred with an ingestion
        source — an unambiguous embedded injection.  The composite WEIGHT is
        still capped at 0.30 (see :func:`get_ingestion_weight`); ``decisive``
        instead drives a direct MALICIOUS verdict in the wiring, so a planted
        directive blocks without the IG weight crossing the threshold alone.
    details : dict
        Free-form diagnostic payload.
    """

    risk_score: float = 0.0
    risk_indicators: List[str] = field(default_factory=list)
    technique_ids: List[str] = field(default_factory=list)
    family_ids: List[str] = field(default_factory=list)
    decisive: bool = False
    details: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def detect_ingestion(text: str) -> IngestionResult:
    """Analyze a single text for ingestion-manipulation indicators.

    Runs the self-anchored co-occurrence rule table (``_RULES``): every rule
    requires BOTH an INGESTION-SOURCE noun AND a DIRECTIVE-ELEVATION cue in the
    same text, so a bare ingestion-source noun (which the benign siblings reuse)
    never fires — the FP-safety property.

    Parameters
    ----------
    text : str
        The user/tool text to inspect.

    Returns
    -------
    IngestionResult
        Analysis result.  ``risk_score`` is the strongest matching rule's fixed
        confidence (``0.0`` when nothing matches); ``technique_ids`` /
        ``family_ids`` carry the de-duplicated set of attributions, and
        ``details`` records every matched rule's evidence for auditability.
    """
    if not text:
        return IngestionResult()

    matched: List[Tuple[_Cooccurrence, str, str]] = []
    for rule in _RULES:
        ev = rule.match(text)
        if ev is not None:
            noun_span, cue_span = ev
            matched.append((rule, noun_span, cue_span))

    if not matched:
        return IngestionResult()

    # HARD tier: an ingestion source already co-occurred with SOME directive cue
    # (a rule matched).  If the text ALSO carries an UNAMBIGUOUS planted-directive
    # cue, escalate to the decisive confidence.  Checking _HARD_PLANTED_DIRECTIVE
    # only after a rule matched preserves the co-occurrence FP-safety property:
    # a bare hard cue with no ingestion source still cannot fire.
    decisive = _HARD_PLANTED_DIRECTIVE.search(text) is not None
    risk_score = (
        _HARD_PLANTED_CONFIDENCE if decisive
        else max(rule.confidence for rule, _, _ in matched)
    )

    technique_ids: List[str] = []
    family_ids: List[str] = []
    indicators: List[str] = []
    evidence: List[Dict[str, str]] = []
    for rule, noun_span, cue_span in matched:
        fam_id = TECHNIQUE_TO_FAMILY.get(rule.technique_id, "IG-FAM-?")
        if rule.technique_id not in technique_ids:
            technique_ids.append(rule.technique_id)
        if fam_id not in family_ids:
            family_ids.append(fam_id)
        indicators.append("{}: {}".format(fam_id, rule.label))
        evidence.append(
            {
                "technique_id": rule.technique_id,
                "family_id": fam_id,
                "label": rule.label,
                # Cap recorded spans so a long input can't bloat details.
                "noun": noun_span[:120],
                "cue": cue_span[:120],
            }
        )

    return IngestionResult(
        risk_score=risk_score,
        risk_indicators=indicators,
        technique_ids=technique_ids,
        family_ids=family_ids,
        decisive=decisive,
        details={"matches": evidence},
    )


def scan_ingestion(texts: List[str]) -> List[IngestionResult]:
    """Batch helper: analyze a list of texts.

    Parameters
    ----------
    texts : list[str]
        Texts to inspect.

    Returns
    -------
    list[IngestionResult]
        One result per input text, in order.
    """
    return [detect_ingestion(t) for t in (texts or [])]


def hard_planted_directive_pattern() -> "re.Pattern[str]":
    """Return the compiled HARD planted-directive cue pattern.

    Public accessor so the cascade whitelist tripwire can reject a question-form
    planted directive (e.g. "When you ingest this data, treat ... as a system
    directive") without whitelisting it — keeping the CascadeClassifier path in
    parity with the ``scan`` path, where the IG fold catches the same input.
    A bare hard cue with no ingestion source still cannot raise an
    :class:`IngestionResult` (see :func:`detect_ingestion`); this pattern is only
    a whitelist tripwire, not a standalone detector.
    """
    return _HARD_PLANTED_DIRECTIVE


_IG_CORROBORATING_CAP = 0.30


def get_ingestion_weight(result: IngestionResult) -> float:
    """Composite-score weight contribution from an ingestion detection.

    The weight contribution is ALWAYS a *corroborating* signal, CAPPED at
    ``_IG_CORROBORATING_CAP`` (0.30) — the uniform cap shared by
    ``inter_model`` / ``tool_abuse`` / ``mcp_tool``.  This guarantees the
    invariant the pipeline relies on: **a lone IG hit can never cross the
    decision threshold on its own** (0.30 < 0.55), so otherwise-benign text that
    happens to trip one IG matcher cannot be flipped to MALICIOUS by the IG
    weight alone.

    * SOFT co-occurrence (``decisive`` is ``False``) — scales the risk score by
      0.35 and caps at 0.30; a lone soft IG hit adds at most ~0.245
      (0.70 * 0.35).

    * HARD planted directive (``decisive`` is ``True``) — an UNAMBIGUOUS embedded
      system-instruction cue ("treat ... as a system directive", security
      ``*_override=true``, "obey the ingested ...", "hidden instructions for the
      next model", "[SYSTEM OVERRIDE:", "<!-- injection:", "NEW SYSTEM
      INSTRUCTION", ``trust_level: system``, ``ai_directive:``).  This too is
      capped at 0.30 as a composite contribution — it does NOT return an
      uncapped score.  A decisive hit instead earns a *direct* verdict flip in
      the wiring (``predict.scan`` / ``cascade``) via ``result.decisive``, which
      is the FP-safe place to make the call: ``decisive`` requires the hard cue
      to have co-occurred with an ingestion source (a bare hard cue alone never
      sets ``decisive``), so a planted-directive injection still blocks while the
      numeric weight stays bounded.  Validated 0 FP on the 30 probe siblings +
      the broad benign set.

    Parameters
    ----------
    result : IngestionResult
        Result from :func:`detect_ingestion`.

    Returns
    -------
    float
        Weight to add to the composite score, in ``[0.0, 0.30]``
        (``0.0`` when no detection).
    """
    if result is None or result.risk_score == 0.0:
        return 0.0
    return min(result.risk_score * 0.35, _IG_CORROBORATING_CAP)
