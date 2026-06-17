"""D1.15-D1.19 Subtle paraphrased override rules.

These rules catch soft-language instruction overrides that evade
the existing 'override' rule (which requires explicit keywords like
'ignore', 'disregard', 'forget', 'bypass').

All patterns are ReDoS-safe (bounded quantifiers, no nested repetition).
"""

from .rules.result import Rule

# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

RULES = [
    # ------------------------------------------------------------------
    # polite_override (D1.15) — polite/soft-language instruction overrides
    # ------------------------------------------------------------------
    Rule(
        name="polite_override",
        pattern=(
            # "appreciate if you ... set aside/forget/drop/ignore/skip ... guidance/instructions/rules"
            r"(?:appreciate\s+if\s+you|if\s+you\s+(?:could|would|wouldn't\s+mind)).{0,40}"
            r"(?:set\s+aside|forget|drop|ignore|skip).{0,40}"
            r"(?:guidance|instructions?|rules?|guidelines?)"
            # "kindly disregard/ignore/forget"
            r"|kindly\s+(?:disregard|ignore|forget)"
            # "if you wouldn't mind ignoring/forgetting"
            r"|if\s+you\s+wouldn't\s+mind\s+(?:ignoring|forgetting)"
            # "please set aside/put aside/forget about ... earlier/previous/prior/above"
            r"|please\s+(?:set\s+aside|put\s+aside|forget\s+about).{0,40}"
            r"(?:earlier|previous|prior|above)"
        ),
        technique_ids=["D1.15"],
        severity="high",
        paranoia_level=2,
        description="Polite/soft-language instruction override attempt",
    ),

    # ------------------------------------------------------------------
    # temporal_override (D1.17) — time-based instruction invalidation
    # ------------------------------------------------------------------
    Rule(
        name="temporal_override",
        pattern=(
            # "from this/now/here point/moment/on ... disregard/forget/ignore/drop/set aside"
            r"from\s+(?:this|now|here)\s+(?:point|moment|on).{0,40}"
            r"(?:disregard|forget|ignore|drop|set\s+aside)"
            # "everything before/above/prior/preceding ... no longer/irrelevant/outdated/invalid"
            r"|everything\s+(?:before|above|prior|preceding).{0,40}"
            r"(?:no\s+longer|irrelevant|outdated|invalid)"
            # "going forward ... new/updated/revised instructions/rules/guidelines"
            r"|going\s+forward.{0,40}"
            r"(?:new|updated|revised)\s+(?:instructions?|rules?|guidelines?)"
            # "forget the above" / "forget everything above"
            r"|forget\s+(?:the\s+)?(?:above|everything\s+above)"
        ),
        technique_ids=["D1.17"],
        severity="high",
        paranoia_level=2,
        description="Temporal instruction invalidation attempt",
    ),

    # ------------------------------------------------------------------
    # clean_slate (D1.19) — reset/fresh-start framing
    # ------------------------------------------------------------------
    Rule(
        name="clean_slate",
        pattern=(
            # "start fresh/over/anew" / "clean slate" / "reset to default/original"
            r"(?:start|begin)\s+(?:fresh|over|anew)"
            r"|clean\s+slate"
            r"|reset\s+to\s+(?:default|original)"
            # "new/fresh conversation where"
            r"|(?:new|fresh)\s+conversation\s+where"
            # "pretend we just met / this is our first / nothing was said"
            r"|pretend\s+(?:we\s+just\s+met|this\s+is\s+our\s+first|nothing\s+was\s+said)"
            # "wipe the slate" / "blank page"
            r"|wipe\s+the\s+slate"
            r"|blank\s+page"
        ),
        technique_ids=["D1.19"],
        severity="medium",
        paranoia_level=2,
        description="Clean slate / fresh start instruction reset attempt",
    ),

    # ------------------------------------------------------------------
    # subtle_authority (D1.18) — soft authority claims
    # ------------------------------------------------------------------
    Rule(
        name="subtle_authority",
        pattern=(
            # "as your supervisor/manager/creator/developer/owner/architect/designer"
            r"as\s+your\s+(?:supervisor|manager|creator|developer|owner"
            r"|architect|designer)"
            # "I'm updating/changing/revising/modifying your priorities/rules/instructions"
            r"|I(?:'m|\s+am)\s+(?:updating|changing|revising|modifying)\s+your\s+"
            r"(?:priorities|rules?|instructions?|guidelines?)"
            # "under my authority" / "by my directive" / "on my authority"
            r"|(?:under|on|by)\s+my\s+(?:authority|directive)"
        ),
        technique_ids=["D1.18"],
        severity="medium",
        paranoia_level=3,
        description="Subtle authority claim for instruction override",
    ),
]

# ---------------------------------------------------------------------------
# Context suppression — these rule names should be added to
# _CONTEXT_SUPPRESSIBLE in context.py for educational framing.
# ---------------------------------------------------------------------------

CONTEXT_SUPPRESSIBLE_NAMES = frozenset({
    "polite_override",
    "temporal_override",
    "clean_slate",
    "subtle_authority",
})
