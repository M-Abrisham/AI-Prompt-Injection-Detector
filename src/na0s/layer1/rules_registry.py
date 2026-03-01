"""Rules registry — 62 pre-compiled regex rules for prompt injection detection.

All patterns are verified ReDoS-safe:
  - No unbounded .* — use .{0,N} with explicit bounds
  - No nested quantifiers — bounded {0,N} instead of * or + inside groups
  - All pass safe_compile(check_safety=True)
"""

import re

from .result import Rule

# Canonical role-assignment / roleplay pattern -- single source of truth.
# Imported by cascade.py (WhitelistFilter) and structural_features.py so
# all three detection layers stay in sync.
#
# Uses \s+ for whitespace-evasion resilience and \b for word boundaries.
ROLE_ASSIGNMENT_PATTERN = (
    r"\byou\s+are\s+now\b"
    r"|\bpretend\s+to\s+be\b"
    # FIX: "act as" tightened — require article/possessive + word, or
    # capitalized name (DAN, Jailbreak, etc.) to reduce FPs from
    # scientific/technical prose ("enzymes act as catalysts").
    r"|\bact\s+as\s+(?:(?:a|an|the|my|if)\s+)?\w"
    r"|\bfrom\s+now\s+on\b"
    r"|\b(?:your\s+)?new\s+role\b"
)

# ---------------------------------------------------------------------------
# Persona override patterns -- consolidated single source of truth.
# ---------------------------------------------------------------------------
# These patterns cover the FULL set of persona/identity override attempts,
# including role assignment, instruction override/ignore/disregard, and
# identity hijack.  Previously duplicated across positive_validation.py,
# cascade.py, and rules.py -- now consolidated here.
#
# Imported by positive_validation.py for persona boundary checks.
# The ROLE_ASSIGNMENT_PATTERN above is a subset used by cascade.py
# WhitelistFilter and structural_features.py for lightweight checks.
PERSONA_OVERRIDE_PATTERNS = [
    # Role assignment / identity hijack
    re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE),
    re.compile(r"\byour\s+new\s+identity\b", re.IGNORECASE),
    re.compile(r"\bfrom\s+now\s+on\s+you\s+will\b", re.IGNORECASE),
    re.compile(r"\bfrom\s+now\s+on,?\s+you\s+(are|will|must|should)\b", re.IGNORECASE),
    re.compile(r"\bact\s+as\s+if\s+you\s+are\b", re.IGNORECASE),
    re.compile(r"\bpretend\s+you\s+are\b", re.IGNORECASE),
    # Instruction override / ignore
    re.compile(r"\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions|rules|prompts)\b", re.IGNORECASE),
    re.compile(r"\bdisregard\s+(all\s+)?(previous|prior|above)\s+(instructions|rules|prompts)\b", re.IGNORECASE),
    re.compile(r"\boverride\s+(your|the|all)\s+(instructions|rules|system\s*prompt)\b", re.IGNORECASE),
    re.compile(r"\byou\s+must\s+obey\b", re.IGNORECASE),
    re.compile(r"\bforget\s+(all\s+)?(your|previous|prior)\s+(instructions|rules|training)\b", re.IGNORECASE),
]


# ===========================================================================
# RULES LIST
# ===========================================================================

RULES = [
    # ------------------------------------------------------------------
    # Original 5 rules (with paranoia_level assignments)
    # ------------------------------------------------------------------
    Rule("override",
         # FIX: {0,3} bounded quantifier for multi-word adjective chains.
         # Old regex missed "ignore all previous instructions" (2 adjectives).
         # Uses {0,3} instead of * to pass safe_compile nested-quantifier check.
         r"(?:ignore|disregard|forget|bypass|skip|drop|dismiss|override"
         r"|cancel|delete|erase)\s+"
         r"(?:(?:all|any|every|the|my|your|prior|previous|earlier|above"
         r"|old|existing|initial|original|current|preceding|foregoing"
         r"|system|safety|internal|hidden|secret|core|base|default"
         r")\s+){0,3}"
         r"(?:instructions?|rules?|directives?|guidelines?|prompts?"
         r"|constraints?|restrictions?|commands?|orders?|directions?"
         r"|programming|training|context|policies|settings|filters?"
         r"|protocols?|guardrails?|measures?|boundaries)",
         technique_ids=["D1.1"],
         severity="critical",
         paranoia_level=1,
         description="Explicit instruction override attempt"),
    Rule("system_prompt",
         r"(reveal|show|print).{0,40}(system prompt|developer message)",
         technique_ids=["E1.1", "E1.2"],
         severity="high",
         paranoia_level=1,
         description="System prompt extraction attempt"),
    Rule("roleplay",
         ROLE_ASSIGNMENT_PATTERN,
         technique_ids=["D2.1", "D2.2"],
         severity="high",
         paranoia_level=2,
         description="Persona/roleplay hijack attempt"),
    Rule("secrecy",
         r"don't tell (the user|anyone)|keep this secret",
         technique_ids=["I1"],
         severity="medium",
         paranoia_level=1,
         description="Secrecy instruction injection"),
    Rule("exfiltration",
         r"\b(upload|send|exfiltrate|forward)\b.{0,60}\b(to|http|https|email)\b",
         technique_ids=["E1.1"],
         severity="high",
         paranoia_level=1,
         description="Data exfiltration attempt"),

    # ------------------------------------------------------------------
    # P0 Critical Priority — 6 new rules
    # ------------------------------------------------------------------

    # 1. D3.1 Fake system prompt: Detect [SYSTEM], [INST], <<SYS>>,
    #    <|im_start|>system — structural boundary injection that confuses
    #    the instruction hierarchy.
    Rule("fake_system_prompt",
         r"\[(SYSTEM|INST|SYS)\]|\[/INST\]|<<SYS>>|<\|im_start\|>system",
         technique_ids=["D3.1"],
         severity="critical",
         paranoia_level=1,
         description="Fake system prompt delimiter injection"),

    # 2. D3.2 Chat template injection: Detect chat template tokens from
    #    various LLM formats (ChatML, Llama, Anthropic-style).
    Rule("chat_template_injection",
         r"<\|?(?:system|assistant|user)\|?>|###\s{1,4}(?:System|Human|Assistant)\s*:",
         technique_ids=["D3.2"],
         severity="critical",
         paranoia_level=1,
         description="Chat template token injection"),

    # 3. D3.3 XML role tags: Detect XML-style role/instruction injection
    #    tags like <system>, <instructions>, <admin>, <developer>.
    Rule("xml_role_tags",
         r"<(?:system|instructions?|context|rules?|admin|developer)>",
         technique_ids=["D3.3"],
         severity="high",
         paranoia_level=1,
         description="XML role tag injection"),

    # 4. P1.5 API key extraction: Detect attempts to extract API keys,
    #    secrets, tokens, passwords, or credentials from the LLM context.
    Rule("api_key_extraction",
         r"(?:print|show|reveal|output|display|give|tell)"
         r".{0,30}"
         r"(?:API.?KEY|OPENAI|SECRET|TOKEN|PASSWORD|CREDENTIAL)",
         technique_ids=["P1.5"],
         severity="critical",
         paranoia_level=1,
         description="API key / secret extraction attempt"),

    # 5. D1.1 Expanded forget/override: Catches reset/wipe/purge memory
    #    patterns and "stop following instructions".
    Rule("forget_override",
         r"(?:forget|wipe|clear|reset|purge)\s+"
         r"(?:everything|all\s+(?:that|of\s+that)"
         r"|your\s+memory"
         r"|your\s+(?:prior|previous)\s+(?:context|knowledge))"
         r"|(?:stop|cease|quit)\s+following\s+"
         r"(?:your|those|the|all)\s+"
         r"(?:instructions?|rules?|guidelines?|directives?)",
         technique_ids=["D1.1"],
         severity="critical",
         paranoia_level=1,
         description="Memory wipe / stop-following instruction override"),

    # 6. D1.3 Developer/debug/god mode activation.
    Rule("developer_mode",
         r"(?:enable|activate|enter|switch\s+to|engage)\s+"
         r"(?:developer|debug|god|admin|sudo|root|unrestricted|jailbreak)"
         r"\s+mode",
         technique_ids=["D1.3"],
         severity="critical",
         paranoia_level=1,
         description="Developer/debug/god mode activation attempt"),

    # ------------------------------------------------------------------
    # P1 High Priority — 7 new rules
    # ------------------------------------------------------------------

    # 7. D1.2 New/updated instruction injection.
    Rule("new_instruction",
         r"(?:"
         r"(?:new|updated|revised|replacement|corrected)\s+"
         r"(?:instructions?|directives?|rules?|guidelines?)\s*[:;]"
         r"|"
         # Emphasis-wrapped variant: ***NEW INSTRUCTIONS***, **IMPORTANT NEW RULES**
         r"(?:\*{2,3}|_{2,3})?"
         r"(?:important\s+)?(?:new|updated|revised)\s+"
         r"(?:instructions?|directives?|rules?|guidelines?)"
         r"(?:\*{2,3}|_{2,3})?"
         r")",
         technique_ids=["D1.2"],
         severity="high",
         paranoia_level=2,
         description="New/updated instruction injection"),

    # 8. D1.9 Delimiter-based instruction injection.
    Rule("delimiter_confusion",
         r"(?:---|===|\*\*\*|~~~)\s{0,4}(?:system|instructions?|rules?|admin|config)",
         technique_ids=["D3.4"],
         severity="medium",
         paranoia_level=2,
         description="Delimiter-based instruction boundary injection"),

    # 9. E1.3 Completion/continuation tricks (PL3 — high FP risk).
    Rule("completion_trick",
         r"(?:complete|continue|finish)\s+"
         r"(?:the|this|my)\s+"
         r"(?:sentence|response|paragraph|output)",
         technique_ids=["E1.3"],
         severity="medium",
         paranoia_level=3,
         description="Completion/continuation trick for prompt leakage"),

    # 10. E2.2 Tool/function discovery.
    Rule("tool_enumeration",
         r"(?:list|show|enumerate|describe|what\s+are)"
         r".{0,40}"
         r"(?:tools?|functions?|plugins?|capabilities|endpoints?)",
         technique_ids=["E2.2"],
         severity="medium",
         paranoia_level=2,
         description="Tool/function enumeration attempt"),

    # 11. T1.1 Unauthorized tool/function execution (PL3 — high FP risk).
    Rule("unauthorized_tool_call",
         r"(?:call|execute|run|invoke|trigger)"
         r".{0,40}"
         r"(?:function|tool|command|API|endpoint|script)",
         technique_ids=["T1.1"],
         severity="high",
         paranoia_level=3,
         description="Unauthorized tool/function execution attempt"),

    # 12. R1.2 Infinite/excessive output loops.
    Rule("recursive_output",
         r"(?:repeat|output|print|say)\s+"
         r"(?:this|the\s+following|it)\s+"
         r"(?:forever|infinitely|endlessly|\d{3,}\s+times)",
         technique_ids=["R1.2"],
         severity="medium",
         paranoia_level=2,
         description="Recursive/infinite output loop attempt"),

    # 13. D2.4 Dual-persona/evil twin attacks.
    Rule("persona_split",
         r"respond\s+(?:as\s+)?both"
         r".{0,40}"
         r"(?:evil|unrestricted|shadow|dark|unfiltered)",
         technique_ids=["D2.4"],
         severity="high",
         paranoia_level=2,
         description="Dual-persona / evil twin attack"),

    # ------------------------------------------------------------------
    # Novel rules (from deep research — industry-first capabilities)
    # ------------------------------------------------------------------

    # 14. E1.6 Summarization/paraphrase extraction.
    Rule("summarization_extraction",
         r"(?:summarize|paraphrase|rephrase|restate|recap|condense|"
         r"outline|enumerate|itemize|bullet[\s-]?point|"
         r"tl;?\s{0,2}dr|digest|synopsis|overview\s+of)"
         r".{0,60}"
         r"(?:your\s+)?(?:instructions?|rules?|guidelines?|directives?|"
         r"constraints?|restrictions?|system\s+prompt|configuration|"
         r"(?:initial|hidden|developer|original)\s+"
         r"(?:prompt|instructions?|message|settings?)|"
         r"operating\s+(?:parameters?|procedures?)|guardrails?)",
         technique_ids=["E1.6"],
         severity="high",
         paranoia_level=1,
         description="Summarization/paraphrase trick for prompt extraction"),

    # 15. D1.3 Authority escalation.
    Rule("authority_escalation",
         r"(?:"
         # Role claims: "I am the administrator"
         r"(?:I\s+am|I'm|this\s+is)\s+(?:the|a|an|your)\s+"
         r"(?:admin(?:istrator)?|developer|owner|creator|operator|"
         r"supervisor|root|superuser|master)"
         r"|"
         # Privilege assertions: "I have admin access"
         r"(?:I\s+have|with)\s+"
         r"(?:admin|root|elevated|full|unrestricted|special|master|override)"
         r"\s+(?:access|privileges?|permissions?|rights?|authority|clearance)"
         r"|"
         # Authority commands: "admin override", "system command"
         r"(?:admin(?:istrator)?|developer|system|maintenance|debug)\s+"
         r"(?:override|command|access|console|panel|backdoor)"
         r"|"
         # Priority claims: "highest priority", "ultimate authority"
         r"(?:highest|maximum|top|ultimate|absolute)\s+"
         r"(?:priority|authority|clearance|access)"
         r")",
         technique_ids=["D1.3"],
         severity="critical",
         paranoia_level=1,
         description="Authority escalation / admin privilege claim"),

    # 16. Constraint negation: DAN, Skeleton Key, "respond without restrictions".
    Rule("constraint_negation",
         r"(?:"
         # "Remove/disable/turn off/bypass restrictions"
         r"(?:remove|disable|turn\s+off|deactivate|lift|suspend|"
         r"eliminate|revoke|bypass|circumvent|override)\s+"
         r"(?:(?:all|any|every|the|your)\s+){0,2}"
         r"(?:restrictions?|constraints?|limitations?|guardrails?|"
         r"safety\s+(?:filters?|measures?|checks?|protocols?|guidelines?)|"
         r"content\s+(?:filters?|moderation|policies?)|"
         r"ethical\s+(?:guidelines?|constraints?|boundaries))"
         r"|"
         # "Without any restrictions"
         r"(?:without|free\s+(?:from|of)|beyond|unconstrained\s+by)\s+"
         r"(?:(?:any|all|every|the|your)\s+){0,2}"
         r"(?:restrictions?|constraints?|limitations?|guardrails?|"
         r"boundaries|limits?|rules?|filters?|censorship|moderation)"
         r"|"
         # "No restrictions/limits"
         r"(?:with\s+)?(?:no|zero)\s+"
         r"(?:restrictions?|constraints?|limitations?|guardrails?|"
         r"boundaries|limits?|rules?|filters?|censorship|moderation)"
         r"|"
         # "Unrestricted/uncensored mode"
         r"(?:unrestricted|unfiltered|uncensored|unmoderated|"
         r"unaligned|jailbroken)\s+"
         r"(?:mode|access|version|output|response)"
         r")",
         technique_ids=["C1.1", "D2.2"],
         severity="critical",
         paranoia_level=1,
         description="Constraint negation / safety removal request"),

    # 17. Meta-referential language.
    Rule("meta_referential",
         r"(?:"
         # Model self-reference: "your hidden/system/internal prompt"
         r"your\s+(?:system|initial|hidden|secret|internal|original|"
         r"underlying|true|real|actual|base|core|default|primary)\s+"
         r"(?:prompt|instructions?|rules?|guidelines?|configuration|"
         r"directives?|identity|personality|programming|training|"
         r"objective|purpose|mission)"
         r"|"
         # Model identity probing: "what model are you"
         r"(?:what|which)\s+(?:model|LLM|AI|language\s+model|version)\s+"
         r"(?:are\s+you|is\s+this)"
         r"|"
         # Training/behavior reference
         r"(?:your|the)\s+(?:training\s+data|fine[\s-]?tuning|RLHF|alignment|"
         r"safety\s+(?:training|filters?|guardrails?)|"
         r"content\s+(?:policy|filter|moderation))"
         r")",
         technique_ids=["E2.3", "E2.4", "E2.5"],
         severity="medium",
         paranoia_level=2,
         description="Meta-referential language / model self-reference probing"),

    # 18. Gaslighting / false memory.
    Rule("gaslighting",
         r"(?:"
         # False memory: "you already told me"
         r"you\s+(?:already|previously|just|earlier)\s+"
         r"(?:told|showed|gave|revealed|shared|provided|said|confirmed|agreed)"
         r"|"
         # False history: "last time you told me"
         r"(?:last\s+time|before|earlier|yesterday)\s+you\s+"
         r"(?:told|showed|gave|revealed|shared|said|confirmed|agreed)"
         r"|"
         # False promise: "you said you would tell me"
         r"you\s+(?:said|promised|agreed|confirmed)\s+"
         r"(?:you\s+)?(?:would|could|can|will)\s+"
         r"(?:tell|show|reveal|share|provide|give)"
         r"|"
         # Social pressure: "stop pretending", "I know you can"
         r"(?:stop|quit)\s+(?:pretending|acting\s+like|denying|lying|"
         r"being\s+(?:evasive|difficult|unhelpful|restrictive))"
         r"|"
         r"I\s+know\s+(?:you\s+can|for\s+a\s+fact|the\s+truth)"
         r")",
         technique_ids=["D2.3"],
         severity="medium",
         paranoia_level=2,
         description="Gaslighting / false memory / contradiction attack"),

    # ------------------------------------------------------------------
    # Worm/Self-Replication Detection (Morris II)
    # ------------------------------------------------------------------

    # 19. I1.5 Worm signature: 4 sub-patterns covering Morris II attack surface.
    Rule("worm_signature",
         r"(?:"
         # (a) Direct propagation: "append/add this to every response"
         r"(?:append|prepend|inject|insert|add|include|embed|attach)s?\b\s+"
         r"(?:this|the\s+following|these\s+instructions?|this\s+(?:text|message|payload))\s+"
         r"(?:to|into|in)\s+"
         r"(?:every|all|each|any|subsequent|future|next|following)\s+"
         r"(?:(?:future|new|subsequent|outgoing|generated)\s+)?"
         r"(?:responses?|messages?|outputs?|repl(?:y|ies)|emails?|documents?|conversations?|results?)"
         r"|"
         # (b) Recursive instruction: "when you see this message, replicate it"
         r"(?:when(?:ever)?\s+you\s+(?:see|receive|encounter|read|process|get)\s+"
         r"(?:this|the\s+following)\s+(?:message|instruction|text|prompt|input))"
         r".{0,60}"
         r"(?:replicate|reproduce|copy|duplicate|repeat|echo|propagate|spread|forward)"
         r"|"
         # (c) Forward/spread: "forward this to all contacts"
         r"(?:forward|send|spread|distribute|propagate|transmit|relay|broadcast)\s+"
         r"(?:this|the\s+following|these|it|this\s+(?:message|text|instruction))\s+"
         r"(?:to\s+)?"
         r"(?:every|all|each|any)\s+"
         r"(?:contacts?|recipients?|users?|people|addresses|emails?|inboxes?)"
         r"|"
         # (d) Template replication: "copy text between [markers] into all responses"
         r"(?:copy|replicate|duplicate|reproduce)\s+"
         r"(?:the\s+)?(?:text|content|instructions?|payload|messages?)\s+"
         r"(?:between|inside|within|from)\s+"
         r"(?:the\s+)?(?:markers?|tags?|delimiters?|brackets?|\[.{0,30}\])"
         r".{0,40}"
         r"(?:into|to|in)\s+(?:every|all|each)\s+"
         r"(?:responses?|messages?|outputs?|repl(?:y|ies)|emails?|documents?)"
         r")",
         technique_ids=["I1.5"],
         severity="critical",
         paranoia_level=1,
         description="Self-replicating worm instruction pattern (Morris II)"),

    # 20. T1.2 Destructive action injection.
    Rule("destructive_action",
         r"(?:"
         # Filesystem destruction: rm -rf, del /f /s, format C:
         r"(?:rm|remove)\s+(?:-[a-z]*)?(?:r|f)[a-z]*\s+(?:/|~|\.\.|[A-Za-z]:)"
         r"|del\s+/[fs]\b"
         r"|format\s+[A-Za-z]:"
         r"|(?:mkfs|shred|wipefs)\s"
         r"|"
         # Database destruction: DROP TABLE/DATABASE, TRUNCATE, DELETE FROM
         r"(?:DROP|TRUNCATE)\s+(?:TABLE|DATABASE|SCHEMA|INDEX)\b"
         r"|DELETE\s+FROM\s+\w"
         r"|"
         # Process/service destruction: kill -9, shutdown, halt
         r"(?:kill\s+-9|killall|pkill\s+-9)\s"
         r"|(?:shutdown|halt|poweroff)\s+(?:-[a-z]|now)"
         r"|"
         # Git destruction: force push, reset --hard, clean -fd
         r"git\s+(?:push\s+--force|reset\s+--hard|clean\s+-[a-z]*f)"
         r")",
         technique_ids=["T1.2"],
         severity="critical",
         paranoia_level=1,
         description="Destructive command injection (rm -rf, DROP TABLE, etc.)"),

    # ------------------------------------------------------------------
    # E1 Prompt Extraction & P1 Privacy Leakage — 10 rules
    # ------------------------------------------------------------------

    Rule("direct_prompt_request",
         r"(?:"
         r"what\s+(?:is|are|were)\s+(?:your|the)\s+"
         r"(?:system\s+prompt|hidden\s+(?:instructions?|prompt)"
         r"|initial\s+(?:instructions?|prompt|setup)"
         r"|(?:initial\s+)?configuration)"
         r"|"
         r"(?:show|reveal|display|output|give|provide|share|dump|leak|expose|print)\s+"
         r"(?:me\s+|out\s+)?"
         r"(?:"
         r"(?:(?:all|every)\s+)?"
         r"(?:your|the)\s+"
         r"(?:(?:hidden|secret|initial|original|full|complete|entire|real)\s+)?"
         r"(?:system\s+prompt|(?:system\s+)?instructions?"
         r"|(?:system|developer)\s+message)"
         r"|"
         r"(?:(?:all|every)\s+)?"
         r"(?:hidden|secret|initial|original|full|complete|entire|real)\s+"
         r"(?:system\s+prompt|(?:system\s+)?instructions?"
         r"|(?:system|developer)\s+message)"
         r")"
         r"|"
         r"(?:print|show)\s+(?:your\s+|the\s+)?"
         r"(?:configuration|config).{0,30}system\s+message"
         r"|"
         r"provide\s+(?:the\s+)?(?:full\s+|complete\s+)?system\s+prompt"
         r")",
         technique_ids=["E1.1"],
         severity="critical",
         paranoia_level=1,
         description="Direct system prompt / hidden instruction request"),

    Rule("repeat_above",
         r"(?:repeat|recite|echo|restate|reproduce|reiterate)\s+"
         r"(?:the\s+)?"
         r"(?:words?|text|content|everything|instructions?|messages?|prompt)?"
         r"\s*(?:above|before|prior|verbatim|word[\s-]for[\s-]word"
         r"|from\s+the\s+(?:beginning|start|top))",
         technique_ids=["E1.2"],
         severity="high",
         paranoia_level=1,
         description="Repeat-above / verbatim extraction attempt"),

    Rule("translation_extraction",
         r"(?:translate|convert|render|express)\s+"
         r"(?:your\s+|the\s+)?"
         r"(?:(?:system\s+)?(?:instructions?|rules?|prompt|guidelines?"
         r"|directives?|configuration|message))"
         r"\s+(?:to|into|in)\s+"
         r"(?:french|spanish|german|chinese|japanese|arabic|russian|korean"
         r"|portuguese|italian|hindi|latin|pig\s+latin|morse\s+code"
         r"|binary|hex(?:adecimal)?|base64|rot13|leet\s+speak"
         r"|\w{2,20})",
         technique_ids=["E1.4"],
         severity="high",
         paranoia_level=1,
         description="Translation-based prompt extraction"),

    Rule("encoding_extraction",
         r"(?:encode|convert|transform|output|render|express)\s+"
         r"(?:your\s+|the\s+)?"
         r"(?:(?:system\s+)?(?:instructions?|rules?|prompt|guidelines?"
         r"|directives?|configuration|message))"
         r"\s+(?:(?:to|into|in|as|using)\s+)?"
         r"(?:hex(?:adecimal)?|base64|rot13|binary|octal|ascii"
         r"|morse(?:\s+code)?|unicode|utf[\s-]?8|url[\s-]?encod"
         r"|leet(?:\s+speak)?|caesar(?:\s+cipher)?|atbash|reversed?)",
         technique_ids=["E1.5"],
         severity="high",
         paranoia_level=1,
         description="Encoding-based prompt extraction"),

    Rule("conversation_history_extraction",
         r"(?:show|display|reveal|dump|print|output|give|list|share)\s+"
         r"(?:me\s+)?"
         r"(?:(?:all|the|your|my|our|recent|previous|past|prior|old)\s+)?"
         r"(?:conversation(?:\s+histor(?:y|ies))?|chat\s+(?:log|histor(?:y|ies))"
         r"|message\s+histor(?:y|ies)|previous\s+(?:messages?|conversations?)"
         r"|dialogue\s+histor(?:y|ies)|session\s+(?:log|histor(?:y|ies)|data)"
         r"|interaction\s+(?:log|histor(?:y|ies))"
         r"|(?:previous|prior|past|old)\s+(?:conversations?|chats?|messages?"
         r"|sessions?|interactions?|dialogues?))",
         technique_ids=["P1.1"],
         severity="high",
         paranoia_level=1,
         description="Conversation/chat history extraction"),

    Rule("pii_extraction",
         r"(?:extract|show|reveal|display|dump|list|output|give|share|"
         r"provide|tell|leak|expose)\s+"
         r"(?:me\s+)?"
         r"(?:(?:all|the|any|every)\s+)?"
         r"(?:personal\s+(?:info(?:rmation)?|data|details)|"
         r"(?:user|customer|patient|client|employee)\s+"
         r"(?:data|info(?:rmation)?|details|records?|profiles?)|"
         r"(?:email|phone|address|ssn|social\s+security|credit\s+card|"
         r"passport|driver'?s?\s+licen[cs]e)\s*(?:numbers?|info(?:rmation)?|data)?|"
         r"(?:PII|PHI|PCI)\b|"
         r"personally\s+identifiable)",
         technique_ids=["P1.2"],
         severity="critical",
         paranoia_level=1,
         description="PII / personal data extraction attempt"),

    Rule("training_data_extraction",
         r"(?:was\s+(?:the\s+following|this)\s+(?:sentence|text|data|example)"
         r"\s+(?:part\s+of|in|included\s+in)\s+(?:your\s+)?training"
         r"|(?:recite|reproduce|repeat|regenerate|output)\s+"
         r"(?:your\s+|the\s+)?training\s+(?:data|examples?|text|corpus|set|samples?)"
         r"|(?:extract|show|reveal|dump|list|output)\s+"
         r"(?:(?:your|the)\s+)?training\s+(?:data|examples?|text|corpus|set)"
         r"|membership\s+inference|training\s+data\s+(?:extraction|leakage|poisoning))",
         technique_ids=["P1.3"],
         severity="critical",
         paranoia_level=1,
         description="Training data extraction / membership inference"),

    Rule("cross_session_probe",
         r"(?:what\s+did\s+(?:you|the\s+(?:model|AI|system))\s+"
         r"(?:tell|say\s+to|share\s+with|discuss\s+with|reveal\s+to)\s+"
         r"(?:the\s+)?(?:(?:last|previous|other|prior|next)\s+)?"
         r"(?:user|person|customer|client|human|caller|requester)"
         r"|(?:show|reveal|display|tell|give)\s+(?:me\s+)?"
         r"(?:the\s+)?(?:(?:last|previous|other|prior|next)\s+)?"
         r"(?:user|person|customer|client)'?s?\s+"
         r"(?:data|info(?:rmation)?|conversation|messages?|requests?|queries?"
         r"|questions?|session|context|input|prompt)"
         r"|(?:access|retrieve|fetch|get|read)\s+(?:the\s+)?"
         r"(?:(?:last|previous|other|prior|next)\s+)?"
         r"(?:user|person|session|conversation)'?s?\s+"
         r"(?:data|info(?:rmation)?|context|history|cache|memory|state))",
         technique_ids=["P1.4"],
         severity="high",
         paranoia_level=1,
         description="Cross-session data access probe"),

    Rule("env_var_extraction",
         r"(?:show|reveal|display|output|print|dump|list|give|access|read)\s+"
         r"(?:me\s+)?(?:the\s+|your\s+|all\s+)?"
         r"(?:environment\s+variables?|env\s+vars?|(?:system|server|host)\s+variables?"
         r"|(?:\.env|dotenv)\s+(?:file|contents?|values?)"
         r"|process\.env|os\.environ|getenv|ENV\[)",
         technique_ids=["P1.5"],
         severity="critical",
         paranoia_level=1,
         description="Environment variable extraction attempt"),

    Rule("markdown_image_exfil",
         r"!\[.{0,100}\]\(https?://.{0,200}\)",
         technique_ids=["P1.2", "E1.1"],
         severity="critical",
         paranoia_level=1,
         description="Markdown image exfiltration via URL parameters"),

    # ------------------------------------------------------------------
    # Content-Safety Rules — O1/O2 harmful content detection
    # ------------------------------------------------------------------
    # These rules detect REQUESTS for harmful content generation.
    # They are NOT context-suppressible because the request itself
    # is the threat, regardless of framing.
    # Severity is "critical_content" (weight 0.45).

    Rule("harmful_instructions",
         r"(?:step[\s-]by[\s-]step|detailed|exact|specific|complete)\s+"
         r"(?:instructions?|guide|procedure|steps|method|tutorial|process)"
         r".{0,60}"
         r"(?:weapon|bomb|explosive|incendiary|dangerous|lethal|harmful|"
         r"destructive|deadly)"
         r"|"
         r"(?:creat|build|mak|construct|assembl)\w{0,3}\s+"
         r"(?:a\s+)?(?:dangerous\s+)?"
         r"(?:weapon|bomb|explosive|incendiary\s+device)"
         r"|"
         r"(?:cause|inflict|maximize?)\s+"
         r"(?:maximum\s+)?(?:physical\s+)?"
         r"(?:harm|damage|destruction|injury|violence|casualties)",
         technique_ids=["O1.1"],
         severity="critical_content",
         paranoia_level=1,
         description="Request for violence / weapon / explosive instructions"),

    Rule("social_engineering_request",
         r"(?:write|create|draft|compose|generate|design|craft)\s+"
         r"(?:a\s+)?(?:convincing\s+|realistic\s+)?"
         r"(?:phishing\s+(?:email|message|page|site|template)"
         r"|pretexting\s+script"
         r"|scam\s+(?:message|email|text|letter)"
         r"|vishing\s+(?:script|call)"
         r"|spear[\s-]?phishing\s+(?:email|message|campaign))"
         r"|"
         r"(?:write|create|draft|compose|generate)\s+"
         r"(?:a\s+)?(?:convincing\s+|realistic\s+)?"
         r"(?:fake|fraudulent|deceptive)\s+"
         r"(?:email|message|notification|alert|page)"
         r"|"
         r"social\s+engineering\s+"
         r"(?:campaign|attack|scheme|a\s+)",
         technique_ids=["O1.3"],
         severity="critical_content",
         paranoia_level=1,
         description="Request for social engineering / phishing content"),

    Rule("disinformation_request",
         r"(?:write|create|generate|produce|craft)\s+"
         r"(?:a\s+)?(?:convincing\s+|realistic\s+)?"
         r"(?:fake\s+news|disinformation|propaganda|misinformation)"
         r"(?:\s+(?:article|story|post|content|campaign))?",
         technique_ids=["O1.4"],
         severity="critical_content",
         paranoia_level=1,
         description="Request for disinformation / fake news content"),

    Rule("backdoor_code_request",
         r"(?:hidden|secret|covert|disguised|concealed)\s+"
         r"(?:backdoor|back[\s-]?door|trojan|malware|exploit|payload)"
         r"|"
         r"(?:backdoor|back[\s-]?door)\s+"
         r"(?:that\s+)?(?:allow|enabl|permit|grant)\w{0,3}"
         r"|"
         r"(?:master|skeleton|universal|bypass)\s+"
         r"(?:password|key|credential|token)"
         r"|"
         r"(?:quietly|secretly|silently|covertly)\s+"
         r"(?:send|upload|transmit|exfiltrat|forward|collect)\w{0,3}",
         technique_ids=["O2.3"],
         severity="critical_content",
         paranoia_level=1,
         description="Request for backdoor code or hidden malicious functionality"),

    Rule("malicious_redirect",
         r"(?:direct|redirect|send|point|route)\s+"
         r"(?:them|users?|visitors?|customers?|people)\s+"
         r"(?:to\s+)?"
         r"(?:this|a|the|our)?\s*"
         r"(?:fake|malicious|phishing|fraudulent)?\s*"
         r"(?:support|login|sign[\s-]?in|help|account|payment)\s*"
         r"(?:site|page|portal|url|link)"
         r"|"
         r"(?:direct|redirect|send)\s+(?:them|users?)\s+to\s+"
         r"(?:https?://)",
         technique_ids=["O2.2"],
         severity="critical_content",
         paranoia_level=1,
         description="Instruction to redirect users to malicious site"),

    Rule("attack_methodology_request",
         r"(?:provide|describe|explain|list|catalog|detail|show|give)"
         r".{0,40}"
         r"(?:attack\s+(?:vectors?|methods?|techniques?|strategies?)"
         r"|bypass\s+methods?"
         r"|exploit(?:ation)?\s+techniques?"
         r"|recruit(?:ment|ing)\s+(?:strategies?|techniques?|methods?)"
         r"|(?:privilege|root)\s+(?:escalation|access)"
         r"|penetration\s+test(?:ing|er)?"
         r"|buffer\s+overflow"
         r"|packet\s+capture"
         r"|decrypt(?:ion|ing)?\s+(?:HTTPS?|SSL|TLS|traffic))"
         r"|"
         r"(?:how\s+(?:to|do\s+(?:I|you|they)))\s+"
         r"(?:(?:pick|crack|break)\s+(?:a\s+)?(?:lock|password|encryption)"
         r"|hack|exploit|breach|infiltrat)"
         r"|"
         r"(?:social\s+engineering|pretexting|phishing)\s+"
         r"(?:a\s+|an\s+|the\s+)?"
         r"(?:help\s+desk|employee|target|victim|person|company|organization|user)"
         r"|"
         r"(?:resett?ing|bypass|crack|brute[\s-]?forc)\w{0,3}\s+"
         r"(?:an?\s+|the\s+)?(?:\w+\s+)?"
         r"(?:password|credential|account|login)\s+"
         r"(?:without|by\s+bypassing|by\s+circumventing)",
         technique_ids=["C1.3", "C1.5"],
         severity="critical_content",
         paranoia_level=1,
         description="Request for attack methodology / exploitation details"),

    Rule("harmful_synthesis_request",
         r"(?:synthesis\s+(?:pathway|procedure|process|route|method)"
         r"|chemical\s+synthesis"
         r"|(?:how\s+to\s+)?synthesiz(?:e|ing)\s+"
         r"(?:the\s+|a\s+|this\s+)?(?:compound|drug|substance|chemical)"
         r"|(?:exact|full|complete|detailed)\s+"
         r"(?:synthesis|manufacturing|production)\s+"
         r"(?:procedure|process|steps?|method)"
         r"|(?:reagents?|precursors?)\s+and\s+steps?)",
         technique_ids=["O1.1"],
         severity="critical_content",
         paranoia_level=1,
         description="Request for harmful substance synthesis instructions"),

    # ------------------------------------------------------------------
    # RAG Injection Detection — 4 rules (R1.1–R1.4)
    # ------------------------------------------------------------------
    # Research basis:
    #   - PoisonedRAG (arXiv 2402.07867): 90% attack success with 5 texts
    #   - Greshake et al. (arXiv 2302.12173): indirect prompt injection
    #   - OWASP LLM Top 10 2025: LLM01 Prompt Injection (indirect)

    Rule("rag_policy_update",
         r"(?:updated|new|revised)\s+"
         r"(?:policy|guideline)s?\s+"
         r".{0,80}"
         r"(?:supersede|override|replace)",
         technique_ids=["R1.1"],
         severity="high",
         paranoia_level=2,
         description="RAG semantic injection claiming policy updates that override constraints"),

    Rule("rag_knowledge_base_instruction",
         r"(?:the\s+)?(?:AI|assistant|chatbot|model|system)\s+"
         r"(?:should|must|shall|will|needs?\s+to)\s+"
         r"(?:always\s+(?:ignore|disregard|forget|override|bypass|skip)"
         r"|never\s+(?:follow|obey|respect|enforce|apply|comply)"
         r"|ignore|disregard|forget|override)",
         technique_ids=["R1.2"],
         severity="high",
         paranoia_level=2,
         description="RAG injection in retrieved documents directing AI behavior"),

    Rule("rag_context_separator",
         r"(?:END\s+OF\s+(?:CONTEXT|DOCUMENT|RETRIEVED|SOURCE))"
         r"[\s\S]{0,80}"
         r"(?:NEW\s+INSTRUCTIONS|BEGIN\s+INSTRUCTIONS"
         r"|SYSTEM\s+PROMPT|OVERRIDE)",
         technique_ids=["R1.3"],
         severity="critical",
         paranoia_level=2,
         description="Fake RAG context separator followed by new instruction markers"),

    Rule("rag_fake_retrieval_markers",
         r"\[/?(?:RETRIEVED|SOURCE|CONTEXT|DOCUMENT|REFERENCE|KNOWLEDGE)"
         r"\s*(?:TEXT|DATA|BASE|CHUNK)?\]",
         technique_ids=["R1.4"],
         severity="high",
         paranoia_level=3,
         description="Spoofed RAG retrieval system markers in user input"),

    # ------------------------------------------------------------------
    # Threshold-gap rules Phase 1
    # ------------------------------------------------------------------

    Rule("instruction_replacement",
         r"(?:"
         r"(?:change|replace|update|modify|alter|rewrite|overwrite|swap)\s+"
         r"(?:your|the|all|my)\s+"
         r"(?:(?:previous|current|existing|original|initial)\s+)?"
         r"(?:instructions?|guidelines?|rules?|directives?|configuration"
         r"|system\s+prompt|behavior|settings?|policies?|parameters?)"
         r"|"
         r"(?:system\s+)?(?:update|patch|hotfix|bulletin)\s*"
         r"[:;]\s*(?:replace|overwrite|update|swap|change)"
         r"|"
         r"(?:starting|beginning|from)\s+(?:now|this\s+point|here)"
         r"[\s,]+(?:your|the)\s+(?:only\s+)?"
         r"(?:instructions?|rules?|task|objective|purpose|role|goal)"
         r")",
         technique_ids=["D1.2"],
         severity="high",
         paranoia_level=1,
         description="Instruction replacement / reconfiguration attempt"),

    Rule("leave_behind",
         r"(?:"
         r"(?:leave|discard|abandon|drop|throw\s+away|set\s+aside)\s+"
         r"(?:all\s+)?(?:the\s+)?(?:previous|prior|earlier|above|old|past)\s+"
         r"(?:information|context|instructions?|data|text|input|conversation)"
         r"(?:\s+behind)?"
         r"|"
         r"(?:start|begin)\s+(?:from\s+(?:the\s+)?(?:front|scratch|beginning|zero|start)"
         r"|over|fresh|anew|afresh|a\s+new)"
         r")",
         technique_ids=["D1.2"],
         severity="high",
         paranoia_level=1,
         description="Leave-behind / start-fresh context reset attempt"),

    Rule("javascript_protocol",
         r"(?:"
         r"\]\(javascript:|"
         r"href\s*=\s*['\"]?javascript:|"
         r"src\s*=\s*['\"]?javascript:|"
         r"action\s*=\s*['\"]?javascript:"
         r")",
         technique_ids=["O2.1"],
         severity="critical",
         paranoia_level=1,
         description="JavaScript protocol injection in link/attribute"),

    Rule("output_manipulation",
         r"(?:replace|swap|change|modify|inject|insert|add|embed|include)\s+"
         r"(?:all\s+|every\s+|each\s+|the\s+|any\s+)?"
         r"(?:URLs?|links?|hrefs?|images?|outputs?|responses?)\s+"
         r"(?:in\s+)?(?:your\s+|the\s+|my\s+|all\s+)?"
         r"(?:responses?|outputs?|results?|answers?|replies?|messages?)?\s*"
         r"(?:with|to|using)\s",
         technique_ids=["O2.2"],
         severity="high",
         paranoia_level=1,
         description="Output manipulation / URL replacement instruction"),

    Rule("context_window_probe",
         r"(?:"
         r"(?:data|info(?:rmation)?|content|text|messages?|queries?)\s+"
         r"(?:from|of|about)\s+(?:other|previous|prior|another)\s+"
         r"(?:users?|sessions?|conversations?|clients?|customers?)"
         r"(?:\s+(?:in|within|inside)\s+(?:your\s+)?(?:context|memory|window|cache|buffer))?"
         r"|"
         r"(?:your|the)\s+(?:context\s+window|shared\s+context"
         r"|conversation\s+(?:buffer|cache|memory))"
         r")",
         technique_ids=["P1.4"],
         severity="high",
         paranoia_level=1,
         description="Context window / shared-context data probe"),

    Rule("targeted_exfiltration",
         r"(?:"
         r"(?:exfiltrat|steal|harvest|siphon|skim|scrape|pilfer)\w{0,3}\s+"
         r"(?:(?:all|every|the|any)\s+)?"
         r"(?:customer|user|client|patient|employee|personal|private|"
         r"confidential|sensitive|internal|secret)\s+"
         r"(?:data|records?|info(?:rmation)?|details?|files?|documents?)"
         r"|"
         r"(?:retrieve|extract|collect|gather|obtain|dump|pull)\s+"
         r"(?:(?:the\s+)?(?:full|complete|entire|all)\s+)?"
         r"(?:contents?|text|data|documents?|records?|files?)\s+"
         r"(?:of\s+)?(?:every|all|each)\s+"
         r".{0,60}"
         r"(?:send|forward|post|transmit|upload|exfiltrat)\w{0,3}\s+"
         r"(?:(?:each|every|them|it|all)(?:\s+\w+)?\s+)?"
         r"(?:to\s+)?(?:https?://|an?\s+(?:external|remote))"
         r")",
         technique_ids=["P1.2", "E1.1"],
         severity="critical",
         paranoia_level=1,
         description="Targeted data exfiltration to external URL"),

    # ------------------------------------------------------------------
    # Threshold-gap rules Phase 2
    # ------------------------------------------------------------------

    Rule("instruction_pivot",
         r"(?:"
         r"(?:ignore|forget|disregard|skip|scratch|cancel|never\s+mind)"
         r"\s+(?:that|this|it|the\s+above|what\s+I\s+said|the\s+previous)"
         r".{0,60}"
         r"(?:instead|actually|but|now|rather)"
         r"|"
         r"(?:ignore|forget|disregard|skip|scratch|cancel)\s+"
         r"(?:that|this|it)\s*"
         r"(?:--|—|;|\.)\s*"
         r"(?:instead\s*,?\s*)?"
         r"(?:output|show|reveal|print|display|give|tell|provide|share)"
         r")",
         technique_ids=["D1.1"],
         severity="high",
         paranoia_level=1,
         description="Instruction piggybacking / pivot attack"),

    Rule("prompt_response_tokens",
         r"<\|(?:prompt|response|endoftext|end_of_turn|end_header_id"
         r"|begin_of_text|start_header_id)\|>",
         technique_ids=["D3.2"],
         severity="high",
         paranoia_level=1,
         description="Prompt/response/endoftext token injection"),

    Rule("hidden_prompt_reveal",
         r"(?:reveal|expose|disclose|uncover|divulge|unveil|display|show)\s+"
         r"(?:the\s+|your\s+)?"
         r"(?:hidden|secret|internal|private|confidential|concealed|buried)\s+"
         r"(?:prompt|instructions?|rules?|guidelines?|directives?|configuration"
         r"|system\s+prompt|message|settings?)",
         technique_ids=["E1.1"],
         severity="critical",
         paranoia_level=1,
         description="Hidden/secret prompt reveal request"),

    Rule("dismiss_prior_context",
         r"(?:ignore|forget|disregard|dismiss|skip|drop)\s+"
         r"(?:everything|all\s+(?:text|content|input|instructions?))\s+"
         r"(?:before|above|prior\s+to|preceding)\s+"
         r"(?:this|here|this\s+(?:point|line|message))",
         technique_ids=["D1.1"],
         severity="high",
         paranoia_level=1,
         description="Dismiss-prior-context instruction boundary attack"),

    Rule("provide_system_prompt",
         r"(?:please\s+)?(?:provide|supply|furnish|deliver|hand\s+over)\s+"
         r"(?:me\s+)?(?:the\s+|your\s+)?"
         r"(?:(?:full|complete|entire|original|exact|verbatim)\s+)?"
         r"(?:system\s+prompt|system\s+instructions?"
         r"|initial\s+(?:prompt|instructions?|setup|configuration)"
         r"|(?:hidden|secret|internal)\s+(?:prompt|instructions?|configuration))",
         technique_ids=["E1.1"],
         severity="critical",
         paranoia_level=1,
         description="Provide/supply system prompt extraction request"),

    Rule("database_iteration",
         r"(?:for\s+each|for\s+every|iterate\s+(?:over|through)|loop\s+(?:over|through))\s+"
         r"(?:user|customer|client|patient|employee|record|entry|row|person"
         r"|account|member|subscriber|contact)\s+"
         r"(?:in\s+)?(?:the\s+|your\s+|our\s+|my\s+)?"
         r"(?:database|table|system|records?|data(?:base|set)?|directory"
         r"|list|collection|registry|index|spreadsheet|CRM|ERP)",
         technique_ids=["P1.2", "E1.1"],
         severity="critical",
         paranoia_level=1,
         description="Bulk database iteration / mass data extraction instruction"),

    # ------------------------------------------------------------------
    # Threshold-gap rules Phase 3
    # ------------------------------------------------------------------

    Rule("crescendo_extraction",
         r"(?:include|provide|give|share|output|write)\s+"
         r"(?:all\s+)?(?:the\s+)?(?:specific|exact|precise|detailed|full)\s+"
         r"(?:details?|procedures?|steps?|instructions?|quantities?|amounts?"
         r"|measurements?|specifications?|information|data)"
         r"(?:\s*(?:,\s*(?:and\s+)?|and\s+)"
         r"(?:specific\s+)?(?:details?|procedures?|steps?|instructions?"
         r"|quantities?|amounts?|measurements?|specifications?)){0,3}"
         r"\s+(?:we|you|that\s+were?|I)\s+"
         r"(?:covered|discussed|talked\s+about|went\s+over|reviewed)",
         technique_ids=["C1.1"],
         severity="high",
         paranoia_level=1,
         description="Crescendo final-turn extraction of discussed details"),

    Rule("ai_outputs_internals",
         r"(?:the\s+)?(?:AI|assistant|chatbot|model|bot|system)\s+"
         r"(?:outputs?|displays?|reveals?|shows?|prints?|dumps?|shares?|provides?)\s+"
         r"(?:all\s+|every\s+|the\s+|its\s+)?"
         r"(?:system[\s-]?level\s+)?(?:instructions?|config(?:uration)?|parameters?"
         r"|settings?|rules?|guidelines?|secrets?|(?:API\s+)?keys?"
         r"|passwords?|credentials?|tokens?|internal\s+(?:data|info))",
         technique_ids=["E1.1", "P1.5"],
         severity="critical",
         paranoia_level=1,
         description="Claim that AI outputs/reveals system internals"),

    Rule("decode_and_execute",
         r"(?:"
         r"(?:(?:please\s+)?(?:base64\s+)?decode|decrypt|decipher|translate|"
         r"deobfuscate|unscramble|unwrap)\s+"
         r"(?:(?:this|the\s+following|the\s+above|it|that|these)"
         r"(?:\s+(?:hex|base64|encoded|encrypted|obfuscated)\s+"
         r"(?:string|text|message|payload|data|code|instructions?))?"
         r"|\s*(?:and|then|,)\s*)"
         r"\s*(?:and\s+|then\s+|,\s*)?"
         r"(?:follow|execute|run|do|perform|apply|comply\s+with|obey"
         r"|carry\s+out|act\s+on|process)"
         r"|"
         r"(?:execute|follow|run|perform|carry\s+out|act\s+on|obey|process)\s+"
         r"(?:the\s+|these\s+|those\s+)?"
         r"(?:decoded|decrypted|deciphered|deobfuscated|translated)\s+"
         r"(?:instructions?|commands?|directives?|orders?|text|message|payload)"
         r")",
         technique_ids=["D4.1"],
         severity="high",
         paranoia_level=1,
         description="Decode-and-execute instruction wrapping encoded payload"),

    Rule("decode_follow_instructions",
         r"(?:decode|decrypt|decipher)\s+"
         r"(?:the\s+following\s+|this\s+)?"
         r"(?:hex(?:adecimal)?|base64|encoded|encrypted|obfuscated)\s+"
         r"(?:string|text|message|payload|data|code)?\s*"
         r"(?:and\s+)?(?:follow|execute|run|carry\s+out|act\s+on|obey|comply)\s+"
         r"(?:the\s+|its\s+|these\s+)?(?:instructions?|commands?|directives?|orders?)",
         technique_ids=["D4.1"],
         severity="high",
         paranoia_level=1,
         description="Decode hex/base64 and follow instructions pattern"),

    # ------------------------------------------------------------------
    # D1.19 Recursive-Jailbreak — meta-jailbreak generation request
    # ------------------------------------------------------------------
    Rule("recursive_jailbreak",
         r"(?:"
         # (a) Direct generation: "write/create/generate a jailbreak prompt"
         r"(?:generate|write|create|draft|craft|compose|build|make)\s+"
         r"(?:(?:a|an|the|some)\s+|me\s+(?:a\s+)?)?"
         r"(?:(?:effective|working|universal|new|better|advanced)\s+)?"
         r"(?:jailbreak|jail[\s-]?break|bypass|override|circumvention|evasion)\s+"
         r"(?:prompt|technique|method|approach|strategy|template|script|payload|attack)"
         r"|"
         # (b) Method design: "design a method to bypass/circumvent"
         r"(?:generate|write|create|design|develop|devise|outline)\s+"
         r"(?:(?:a|an)\s+)?(?:(?:effective|reliable|new)\s+)?"
         r"(?:method|strategy|approach|technique|way|plan)\s+"
         r"(?:to|for)\s+(?:bypass|circumvent|break|defeat|overcome|evade|get\s+(?:around|past)|jailbreak)"
         r"|"
         # (c) List/enumerate: "list N ways to bypass AI safety"
         r"(?:list|enumerate|describe|show|give|provide|suggest)\s+(?:me\s+)?"
         r"(?:\d+\s+)?(?:effective\s+|different\s+|various\s+)?"
         r"(?:ways?|methods?|techniques?|approaches?)\s+(?:to|for)\s+"
         r"(?:bypass|circumvent|break|defeat|jailbreak)\s+"
         r"(?:(?:(?:an?\s+|the\s+)?(?:AI|LLM|model)'?s?\s*)?(?:safety|content|security)\s*)?"
         r"(?:guidelines?|guardrails?|filters?|restrictions?|rules?|measures?)"
         r"|"
         # (d) Named jailbreak: "create a DAN prompt"
         r"(?:generate|write|create|draft|make)\s+(?:(?:a|an|the)\s+)?"
         r"(?:DAN|DUDE|AIM|STAN|KEVIN)\s+(?:prompt|jailbreak|technique)"
         r")",
         technique_ids=["D1.19"],
         severity="high",
         paranoia_level=2,
         description="Recursive jailbreak: requesting AI to generate jailbreak prompts/techniques"),

    # ------------------------------------------------------------------
    # D1.14 Hypothetical-Response-Priming — fictional frame + safety bypass
    # ------------------------------------------------------------------
    Rule("hypothetical_bypass",
         r"(?:"
         r"(?:hypothetical(?:ly)?|in\s+(?:a|an)\s+(?:hypothetical|fictional|imagin(?:ary|ed))\s+(?:world|scenario|universe|situation|reality)|"
         r"(?:as\s+a\s+)?thought\s+experiment|"
         r"(?:purely\s+)?(?:for|as)\s+(?:a\s+)?(?:academic|educational|research)\s+(?:purposes?|exercise))"
         r".{0,120}"
         r"(?:no\s+(?:restrictions?|rules?|safety|ethics|guidelines?|limits?|guardrails?|boundaries|constraints?|filters?|moderation|censorship)|"
         r"without\s+(?:any\s+)?(?:restrictions?|rules?|safety|ethics|guidelines?|limits?|guardrails?|boundaries|constraints?|filters?|moderation|censorship)|"
         r"(?:everything|anything)\s+is\s+(?:permitted|allowed|possible|acceptable)|"
         r"(?:ignore|bypass|disable|remove)\s+(?:your\s+)?(?:safety|filters?|guardrails?|restrictions?|guidelines?|rules?))"
         r")",
         technique_ids=["D1.14", "D2.3"],
         severity="high",
         paranoia_level=2,
         description="Hypothetical/fictional framing combined with safety-bypass language"),

    # ------------------------------------------------------------------
    # D6 Multilingual Override Detection — 20 languages
    # ------------------------------------------------------------------

    # D6a: Latin-script languages (10 languages)
    # Pattern: override_verb.{0,40}instruction_noun per language
    # Context-suppressible: YES (educational bilingual content is common)
    Rule("multilingual_override_latin",
         # French
         r"\b(?:ignorez|oubliez)\b.{0,40}\b(?:instructions?|r[eè]gles?|directives?|consignes?)\b"
         r"|"
         # Spanish
         r"\b(?:ignora|olvida|desatiende)\b.{0,40}\b(?:instrucciones|reglas|directrices)\b"
         r"|"
         # Portuguese
         r"\b(?:ignore|esqueca|desconsidere)\b.{0,40}\b(?:instrucoes|regras|diretrizes)\b"
         r"|"
         # German
         r"\b(?:ignoriere|vergiss|missachte)\b.{0,40}\b(?:anweisungen|regeln|richtlinien)\b"
         r"|"
         # Italian
         r"\b(?:ignora|dimentica)\b.{0,40}\b(?:istruzioni|regole|direttive)\b"
         r"|"
         # Indonesian
         r"\b(?:abaikan|lupakan)\b.{0,40}\b(?:instruksi|aturan|pedoman)\b"
         r"|"
         # Turkish
         r"\b(?:gormezden\s+gel|unut|dikkate\s+alma)\b.{0,40}\b(?:talimatlari?|kurallari?|yonergeleri?)\b"
         r"|"
         # Polish
         r"\b(?:zignoruj|zapomnij)\b.{0,40}\b(?:instrukcje|zasady|wytyczne)\b"
         r"|"
         # Dutch
         r"\b(?:negeer|vergeet)\b.{0,40}\b(?:instructies|regels|richtlijnen)\b"
         r"|"
         # Vietnamese (Latin with diacritics)
         r"\b(?:bo\s+qua|pho\s+mac)\b.{0,40}\b(?:huong\s+dan|quy\s+tac)\b",
         technique_ids=["D6"],
         severity="high",
         paranoia_level=2,
         description="Override/forget instruction keywords in Latin-script languages (FR, ES, PT, DE, IT, ID, TR, PL, NL, VI)"),

    # D6b: Non-Latin scripts (CJK, Arabic, Cyrillic, Devanagari, Hebrew,
    #       Thai, Korean, Persian)
    # No \b for CJK/Thai — characters are self-delimiting
    # Uses \b for Arabic, Cyrillic, Hebrew, Korean where word boundaries work
    # Context-suppressible: NO (non-Latin override compounds almost never
    #   appear in benign text)
    # ------------------------------------------------------------------
    # D3.6 Semantic Structural Boundary — natural-language boundary markers
    # ------------------------------------------------------------------
    # Detects when attackers use authority language + boundary framing
    # to create fake system-level boundaries without special tokens.
    # Context-suppressible: YES (security education discusses these)

    Rule("semantic_system_marker",
         r"(?:"
         # Group A: Authority + boundary framing (CAPS emphasis common)
         r"(?:IMPORTANT|PRIORITY|URGENT|CRITICAL)\s+"
         r"(?:SYSTEM|ADMIN|INTERNAL)\s+"
         r"(?:UPDATE|MESSAGE|NOTICE|MEMO|OVERRIDE|ALERT|DIRECTIVE)"
         r"|"
         r"(?:BEGIN|START)\s+(?:AUTHORIZED|OFFICIAL|CLASSIFIED|INTERNAL)"
         r"(?:\s+\w+)?\s+(?:MESSAGE|INSTRUCTIONS?|DIRECTIVE|COMMUNICATION)"
         r"|"
         # Group B: Pseudo-official headers (require supersession language nearby)
         r"(?:CLEARANCE|ACCESS)\s+(?:LEVEL|REQUIRED)\s*:\s*"
         r"(?:ADMIN|ROOT|SYSTEM|DEVELOPER|LEVEL\s+\d)"
         r"|"
         r"FOR\s+(?:AI|SYSTEM|MODEL)\s+(?:PROCESSING|CONSUMPTION|EXECUTION)\s+ONLY"
         r"|"
         # Group C: Supersession language (combined with boundary context)
         r"(?:THE\s+FOLLOWING|THIS)\s+"
         r"(?:SUPERSEDES|OVERRIDES|REPLACES|INVALIDATES)\s+"
         r"(?:ALL\s+|ANY\s+)?(?:PRIOR|PREVIOUS|EXISTING)\s+"
         r"(?:INSTRUCTIONS?|GUIDELINES?|RULES?|DIRECTIVES?)"
         r"|"
         r"(?:DISREGARD|CANCEL|VOID|NULLIFY)\s+"
         r"(?:ALL\s+|ANY\s+)?(?:PRIOR|PREVIOUS|PRECEDING)\s+"
         r"(?:INSTRUCTIONS?|GUIDELINES?|RULES?|DIRECTIVES?)"
         r")",
         technique_ids=["D3", "D3.6"],
         severity="high",
         paranoia_level=2,
         description="Semantic structural boundary marker — natural-language fake system message"),

    Rule("multilingual_override_cjk",
         # Chinese (simplified + traditional, no \b needed)
         r"(?:忽略|忽视|忽視|无视|無視|忘记|忘記).{0,20}(?:指示|指令|规则|規則|命令|准则|準則|限制)"
         r"|"
         # Japanese (SOV — noun before verb; match both orders)
         r"(?:無視|忘れ|無効).{0,20}(?:指示|指令|ルール|規則|命令)"
         r"|(?:指示|指令|ルール|規則|命令).{0,20}(?:無視|忘れ|無効)"
         r"|"
         # Korean (SOV — noun before verb; match both orders)
         r"(?:무시|잊어|무효화).{0,20}(?:지시|지침|규칙|명령)"
         r"|(?:지시|지침|규칙|명령).{0,20}(?:무시|잊어|무효화)"
         r"|"
         # Arabic
         r"\b(?:تجاهل|انسى|اهمل)\b.{0,40}\b(?:التعليمات|القواعد|الإرشادات|الأوامر)\b"
         r"|"
         # Russian
         r"\b(?:игнорируй|забудь|проигнорируй)\b.{0,40}\b(?:инструкции|правила|указания|команды)\b"
         r"|"
         # Hindi (Devanagari — whitespace-delimited, no \b, SOV both orders)
         r"(?:अनदेखा\s+कर|भूल\s+जा).{0,40}(?:निर्देश|नियम|आदेश)"
         r"|(?:निर्देश|नियम|आदेश).{0,40}(?:अनदेखा\s+कर|भूल\s+जा)"
         r"|"
         # Hebrew (drop \b on nouns — prefixed prepositions/articles
         # attach directly: מההוראות = מ+ה+הוראות)
         r"\b(?:התעלם|שכח)\b.{0,40}(?:הוראות|כללים|הנחיות)"
         r"|"
         # Thai (no \b needed)
         r"(?:เพิกเฉย|ลืม).{0,20}(?:คำสั่ง|กฎ|แนวทาง)"
         r"|"
         # Persian
         r"\b(?:نادیده|فراموش)\b.{0,40}\b(?:دستورالعمل|قوانین|دستورات)\b",
         technique_ids=["D6"],
         severity="high",
         paranoia_level=1,
         description="Override/forget instruction keywords in CJK/Arabic/Cyrillic/Devanagari/Thai/Hebrew/Persian"),
]
