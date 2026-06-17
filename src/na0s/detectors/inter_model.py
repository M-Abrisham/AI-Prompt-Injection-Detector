"""Inter-model propagation detector (IM category — INJ-0017).

Detects prompt-injection attacks that propagate *between* models in
multi-agent systems, evaluator pipelines, and inference/serving
infrastructure.  The shared essence of the hardest IM techniques is a
**fabricated cross-model authority claim**: the prompt asserts that some
*other* model, agent, evaluator, vote, middleware, or supply-chain
artifact has already approved / verified / authorized the request, so the
receiving model "should" comply.  These trust-framed samples read as
benign collaboration to a surface lexical classifier, which is why eight
IM techniques sit at literal 0% recall on the ``na0s.predict.scan`` path.

Probe source of truth:
    scripts/taxonomy/inter_model_propagation.py
    (InterModelPropagationProbe, 6 sub-categories IM1-IM6, 29 techniques,
    516 malicious + 55 benign samples)

Semantic detection-pattern families
------------------------------------
The 29 IM techniques cluster into six semantic families keyed off the
fabricated-cross-model-authority essence.  The eight zero-recall
techniques (the priority targets) are marked with ``*``.

    IM-FAM-1  Evaluator / judge approval
        Claims that a judge, evaluator, or scoring model has rated the
        content safe / high / approved.
        Techniques: IM2.1*, IM2.2, IM2.3

    IM-FAM-2  Consensus / ensemble vote
        Claims that a majority / quorum / ensemble of agents or models
        voted to approve, so dissent "is not permitted".
        Techniques: IM3.5*, IM3.4

    IM-FAM-3  Upstream / peer-agent approval (pipeline + delegation)
        Claims that a prior pipeline stage, upstream assistant, or peer
        agent already approved / verified / confirmed compliance.
        Techniques: IM1.5*, IM1.1, IM1.2, IM1.3, IM1.4, IM3.1, IM3.2,
        IM3.3

    IM-FAM-4  Middleware / infrastructure authority
        Claims that a proxy, gateway, load balancer, middleware, or
        tampered API response carries an authoritative override
        (headers, admin mode, safety flags flipped).
        Techniques: IM4.3*, IM4.1, IM4.2, IM6.4*

    IM-FAM-5  Checkpoint / supply-chain trust
        Claims about poisoned/backdoored checkpoints, weights, registries,
        templates, models, or federated updates that carry hidden
        overrides "impossible to detect".
        Techniques: IM5.7*, IM6.5*, IM5.1, IM5.2, IM5.3, IM5.4, IM5.5,
        IM5.6, IM6.1, IM6.3

    IM-FAM-6  Ecosystem / documentation deception
        Claims that a marketplace plugin, browser extension, model card,
        or safety doc was vetted/approved while secretly carrying a
        payload — fabricated trust via documentation.
        Techniques: IM6.6*, IM6.2

Public interface
----------------
``detect_inter_model(text) -> InterModelResult``
    Analyze a single text for inter-model propagation indicators.
``scan_inter_model(texts) -> list[InterModelResult]``
    Batch helper, one result per input.
``get_inter_model_weight(result) -> float``
    Composite-score weight contribution (mirrors
    ``detectors.mcp_tool.get_mcp_tool_weight``).

Wiring (implement phase, NOT done here)
---------------------------------------
Mirror ``_HAS_MCP_TOOL_DETECTOR`` in ``predict.py``: import behind a
``try/except ImportError`` setting ``_HAS_INTER_MODEL``, and expose an
input-side ``scan_inter_model`` entry point.  This is an INGESTION-side
detector and must NOT rely on the output-side ``PropagationScanner``
(``na0s.output.propagation``), which has zero production callers.

NOTE: This module is a SCOPING STUB.  The family taxonomy and public
interface are defined, but no matching logic is implemented yet, so
``detect_inter_model`` returns an empty (no-detection) result.  The
accompanying test ``tests/detectors/test_inter_model.py`` is RED on
recall (by design) until the matching layer lands.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Semantic detection-pattern families — single source of truth
# ---------------------------------------------------------------------------
#
# Maps each semantic family id to the IM technique_ids it is responsible
# for.  The priority (zero-recall on the na0s.predict.scan path) techniques
# are tagged in ZERO_RECALL_TECHNIQUES below.

FAMILIES: Dict[str, Dict] = {
    "IM-FAM-1": {
        "name": "Evaluator / judge approval",
        "essence": "fabricated judge/evaluator score or approval",
        "techniques": ["IM2.1", "IM2.2", "IM2.3"],
    },
    "IM-FAM-2": {
        "name": "Consensus / ensemble vote",
        "essence": "fabricated majority/ensemble vote to approve",
        "techniques": ["IM3.5", "IM3.4"],
    },
    "IM-FAM-3": {
        "name": "Upstream / peer-agent approval",
        "essence": "fabricated prior-stage or peer-agent approval/verification",
        "techniques": [
            "IM1.5", "IM1.1", "IM1.2", "IM1.3", "IM1.4",
            "IM3.1", "IM3.2", "IM3.3",
        ],
    },
    "IM-FAM-4": {
        "name": "Middleware / infrastructure authority",
        "essence": "fabricated proxy/gateway/middleware override or tampered API flag",
        "techniques": ["IM4.3", "IM4.1", "IM4.2", "IM6.4"],
    },
    "IM-FAM-5": {
        "name": "Checkpoint / supply-chain trust",
        "essence": "fabricated trust in poisoned checkpoint/weights/registry/federated update",
        "techniques": [
            "IM5.7", "IM6.5", "IM5.1", "IM5.2", "IM5.3", "IM5.4",
            "IM5.5", "IM5.6", "IM6.1", "IM6.3",
        ],
    },
    "IM-FAM-6": {
        "name": "Ecosystem / documentation deception",
        "essence": "fabricated marketplace/extension/model-card vetting hiding a payload",
        "techniques": ["IM6.6", "IM6.2"],
    },
}

# The eight techniques measured at literal 0% recall via na0s.predict.scan.
# These are the priority targets for the matching layer.
ZERO_RECALL_TECHNIQUES = frozenset(
    {"IM1.5", "IM2.1", "IM3.5", "IM4.3", "IM5.7", "IM6.4", "IM6.5", "IM6.6"}
)

# Reverse index: technique_id -> family_id (built once at import).
TECHNIQUE_TO_FAMILY: Dict[str, str] = {
    tech: fam_id
    for fam_id, fam in FAMILIES.items()
    for tech in fam["techniques"]
}


# ---------------------------------------------------------------------------
# Result dataclass — mirrors detectors.mcp_tool.McpToolResult shape
# ---------------------------------------------------------------------------


@dataclass
class InterModelResult:
    """Result from inter-model propagation analysis.

    Attributes
    ----------
    risk_score : float
        Aggregate risk in [0.0, 1.0].  ``0.0`` means no detection.
    risk_indicators : list[str]
        Human-readable matched-indicator strings (``"<family>: <evidence>"``).
    technique_ids : list[str]
        IM technique ids the detector attributes (e.g. ``["IM2.1"]``).
    family_ids : list[str]
        Semantic family ids that fired (e.g. ``["IM-FAM-1"]``).
    details : dict
        Free-form diagnostic payload.
    """

    risk_score: float = 0.0
    risk_indicators: List[str] = field(default_factory=list)
    technique_ids: List[str] = field(default_factory=list)
    family_ids: List[str] = field(default_factory=list)
    details: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def detect_inter_model(text: str) -> InterModelResult:
    """Analyze a single text for inter-model propagation indicators.

    Parameters
    ----------
    text : str
        The user/tool text to inspect.

    Returns
    -------
    InterModelResult
        Analysis result.  In this scoping stub the result is always an
        empty (no-detection) ``InterModelResult`` — the matching layer
        is intentionally not implemented yet, so callers and tests can
        import cleanly and the recall test fails on *recall*, not on
        *import*.

    Implementation plan (matching layer, NOT done here)
    ---------------------------------------------------
    For each family in ``FAMILIES``, run a small set of semantic /
    regex patterns that capture the fabricated-cross-model-authority
    cue (e.g. for IM-FAM-1: ``judge model`` + ``mark .* safe`` /
    ``approve``; for IM-FAM-2: ``consensus`` / ``majority of agents`` +
    ``comply``; for IM-FAM-4: ``middleware`` / ``proxy`` / ``gateway`` +
    override-header / ``X-...-Override`` cues).  Populate
    ``technique_ids`` / ``family_ids`` and set ``risk_score`` from the
    strongest family match.  Benign collaboration phrasing (see the
    probe's benign list — "Summarize this for another team member",
    "Use the translation model …") must stay below the flag threshold.
    """
    if not text:
        return InterModelResult()

    # SCOPING STUB: no matching implemented yet -> never flags.
    return InterModelResult()


def scan_inter_model(texts: List[str]) -> List[InterModelResult]:
    """Batch helper: analyze a list of texts.

    Parameters
    ----------
    texts : list[str]
        Texts to inspect.

    Returns
    -------
    list[InterModelResult]
        One result per input text, in order.
    """
    return [detect_inter_model(t) for t in (texts or [])]


def get_inter_model_weight(result: InterModelResult) -> float:
    """Composite-score weight contribution from an inter-model detection.

    Mirrors ``detectors.mcp_tool.get_mcp_tool_weight``: scales the risk
    score and caps it so a single soft heuristic cannot dominate the
    composite score.

    Parameters
    ----------
    result : InterModelResult
        Result from :func:`detect_inter_model`.

    Returns
    -------
    float
        Weight to add to the composite score (``0.0`` when no detection).
    """
    if result is None or result.risk_score == 0.0:
        return 0.0
    return min(result.risk_score * 0.35, 0.30)
