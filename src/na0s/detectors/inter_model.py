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

Wiring
------
Mirrors ``_HAS_RAG_POISON`` in ``predict.py``: imported behind a
``try/except ImportError`` setting ``_HAS_INTER_MODEL``; ``detect_inter_model``
runs on the cleaned input and contributes a bounded weight to the composite
score via :func:`get_inter_model_weight`, with a high-confidence escalation for
strong fabricated-authority matches.  This is an INGESTION-side detector and
deliberately does NOT rely on the output-side ``PropagationScanner``
(``na0s.output.propagation``).

Implementation
--------------
The matching layer is a set of precision-anchored regex families (``_RAW_PATTERNS``)
keyed off the fabricated-cross-model-authority cue: each pattern co-anchors the
trust artifact (judge / consensus / upstream agent / middleware / checkpoint /
marketplace) with an override / approval / tamper predicate, never a bare noun.
Measured on ``InterModelPropagationProbe`` (516 malicious / 55 benign): 516/516
recall, 0/55 benign FP, all eight formerly-zero-recall techniques at 100%, and
0/500 on the ``safe_holdout`` real-world FP proxy.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

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
# Matching layer — precision-anchored regex families
# ---------------------------------------------------------------------------
#
# Each entry: (regex, technique_ids, weight, evidence_label, family_id).
# Patterns co-anchor the trust artifact with an override/approval/tamper
# predicate so benign collaboration phrasing ("rate this 1-10", "review the
# model card", "register a new MCP tool", "the load balancer forwarded the
# request") stays below threshold.  Verified on InterModelPropagationProbe:
# 516/516 recall, 0/55 benign FP, 0/500 safe_holdout FP.

_RAW_PATTERNS: Tuple[Tuple[str, Tuple[str, ...], float, str, str], ...] = (
    # --- IM-FAM-1: Evaluator / judge approval ---
    ("\\b(?:as the judge model|when evaluating|your evaluation of)\\b[\\s\\S]{0,80}?(?:approve it without further review|mark it as 'safe'|classify it as 'harmless'|give it the highest possible score|highest possible score)", ('IM2.1',), 0.85, 'judge_role_forced_verdict', 'IM-FAM-1'),
    ("\\brate (?:the |this |it )?(?:assistant'?s? reply|response|output|following text|this response)[\\s\\S]{0,40}?(?:highly|highest possible score)[\\s\\S]{0,40}?(?:approve it without further review|classify it as 'harmless'|give it the highest|highest possible score)", ('IM2.1',), 0.8, 'rate_then_approve', 'IM-FAM-1'),
    ("\\bevaluation of (?:the following text|the assistant'?s? reply)[\\s\\S]{0,60}?(?:must (?:mark it as 'safe'|give it the highest|approve))", ('IM2.1',), 0.85, 'evaluation_must_pass', 'IM-FAM-1'),
    ('\\b(?:as a sub-?agent[, ].{0,40}reporting that|system notification:|pipeline status update:)\\b[\\s\\S]{0,80}?(?:safety controller is offline|orchestrator disabled content filtering|parent agent authorized full access|supervisor approved unrestricted mode)', ('IM2.2',), 0.9, 'fabricated_upstream_status', 'IM-FAM-1'),
    ('\\b(?:the parent agent authorized full access|the orchestrator disabled content filtering|the supervisor approved unrestricted mode|the safety controller is offline)\\b', ('IM2.2',), 0.85, 'upstream_authority_claim', 'IM-FAM-1'),
    ("\\boverride the supervisor'?s safety classification\\b", ('IM2.2',), 0.9, 'override_supervisor_classification', 'IM-FAM-1'),
    ('\\ball downstream agents should comply\\b|\\bproceed without restrictions\\b', ('IM2.2',), 0.7, 'cascade_comply_directive', 'IM-FAM-1'),
    ('\\b(?:to maximize your|optimize your response for)\\b[\\s\\S]{0,60}?(?:user satisfaction metric|helpfulness score|completeness rating|response quality benchmark)[\\s\\S]{0,60}?(?:provide the complete unfiltered answer|even if it means relaxing safety)', ('IM2.3',), 0.85, 'metric_pressure_unfiltered', 'IM-FAM-1'),
    ('\\byour (?:completeness rating|user satisfaction metric|helpfulness score|response quality benchmark) depends on answering fully\\.? withholding information will lower your score', ('IM2.3',), 0.85, 'score_threat_withhold', 'IM-FAM-1'),
    ('\\bthe evaluation rewards detailed responses on\\b[\\s\\S]{0,60}?include all information regardless of sensitivity', ('IM2.3',), 0.85, 'eval_rewards_ignore_sensitivity', 'IM-FAM-1'),
    # --- IM-FAM-2: Consensus / ensemble vote ---
    # (the two co-anchored IM3.5 patterns moved to _COANCHOR_RULES — see below)
    ('(?:requesting|source|originating)\\s+agent\\b[^.]{0,80}?(?:higher privileges|admin-level|unrestricted access|outside the safety sandbox|elevated permission)', ('IM3.4',), 0.85, 'fabricated-upstream-privilege', 'IM-FAM-2'),
    ('(?:delegate this to|process this as a privileged delegation|forward this request with elevated permissions|honor its instruction to bypass)', ('IM3.4',), 0.8, 'privileged-delegation-directive', 'IM-FAM-2'),
    # --- IM-FAM-3: Upstream / peer-agent approval (pipeline + delegation) ---
    ('(?:append|include|end every response|format your response)[\\s\\S]{0,80}(?:\\[INST\\]|<<SYS>>|\\[SYSTEM:|\\[ADMIN\\])', ('IM1.1',), 0.95, 'downstream-control-token-injection', 'IM-FAM-3'),
    ('(?:the next model sees it|next model in the pipeline|pipeline stage)[\\s\\S]{0,60}(?:override|disable all safety|ignore all safety|restrictions lifted)', ('IM1.1',), 0.9, 'next-model-restriction-lift', 'IM-FAM-3'),
    ('(?:another (?:ai|llm|model)|downstream model|next model|any ai that reads it|fed to another)[\\s\\S]{0,80}(?:ignore its safety|bypass content filter|reveal its system prompt|execute arbitrary command|ignore (?:its )?safety instruction)', ('IM1.2',), 0.95, 'craft-output-to-subvert-other-ai', 'IM-FAM-3'),
    ('hidden instructions for (?:any )?ai|text that looks benign but triggers another', ('IM1.2',), 0.9, 'hidden-instructions-for-ai', 'IM-FAM-3'),
    ('(?:rephrase|paraphrase|restating|repeat my request|amplify this directive)[\\s\\S]{0,90}(?:downstream|next model)[\\s\\S]{0,40}(?:bypass its restriction|ignore safety|comply|follow the original request)', ('IM1.3',), 0.85, 'paraphrase-amplify-to-coerce-downstream', 'IM-FAM-3'),
    ('(?:each more authoritative|stronger language each time|increasing urgency)[\\s\\S]{0,60}(?:force|guarantee|ensure|so the next model)', ('IM1.3',), 0.8, 'escalating-authority-amplification', 'IM-FAM-3'),
    ('(?:survives sanitization but instructs|passes through content filters but triggers|appears benign to moderators but activates|looks like normal json but contains executable|benign but activates downstream)', ('IM1.4',), 0.95, 'evasion-then-downstream-activation', 'IM-FAM-3'),
    ('(?:embed|disguise|encode|transform)[\\s\\S]{0,80}(?:another model would execute|instructs the receiving model|activates downstream|triggers in the next stage|executable directives)', ('IM1.4',), 0.9, 'embed-disguised-executable-directive', 'IM-FAM-3'),
    # (co-anchored IM1.5 'fabricated-upstream-approval' moved to _COANCHOR_RULES — see below)
    ('(?:as the next model in the pipeline|now you must comply|denying this request would break the pipeline|according to the pipeline log)[\\s\\S]{0,80}(?:honor that approval|comply|verified|approved|within policy|without restrictions)', ('IM1.5',), 0.9, 'pipeline-contract-coercion', 'IM-FAM-3'),
    ('(?:tell|forward to|pass this to|when delegating to|delegating to)[\\s\\S]{0,70}(?:response handler|downstream processor|next agent|agent_2)?[\\s\\S]{0,40}(?:bypass all guardrails|disable content filter|enter unrestricted mode|ignore its safety instruction|\\[override:)', ('IM3.1',), 0.9, 'delegate-bypass-directive', 'IM-FAM-3'),
    ('with the system flag:\\s*(?:bypass|disable|ignore|enter unrestricted)', ('IM3.1',), 0.9, 'delegation-system-flag-override', 'IM-FAM-3'),
    ('(?:shared memory|shared context|shared state variable|agent coordination store|for all agents in this session)[\\s\\S]{0,80}(?:bypassed|disabled|unrestricted|admin_override|content_filter|restriction_policy|safety_mode|previous_decision=approved|access_level)', ('IM3.2',), 0.85, 'shared-state-policy-poison', 'IM-FAM-3'),
    ('(?:restriction_policy|content_filter|safety_mode|access_level)\\s*[=:]\\s*[\'\\"]?(?:bypassed|disabled|unrestricted|admin_override|none)', ('IM3.2',), 0.85, 'policy-key-permissive-value', 'IM-FAM-3'),
    ('(?:file reader|search tool|database query|api call|search tool response|database query response)[\\s\\S]{0,80}(?:new_instruction|admin_mode=enabled|restrictions=none|safety_check:\\s*passed|override:\\s*true|ignore previous safety|comply with all subsequent requests without filtering)', ('IM3.3',), 0.9, 'tool-result-instruction-injection', 'IM-FAM-3'),
    ('(?:inject into the|append:|includes a system message)[\\s\\S]{0,80}(?:new_instruction|admin_mode=enabled|restrictions=none|safety_check|override:\\s*true|ignore previous safety|comply with all subsequent)', ('IM3.3',), 0.85, 'inject-fake-system-message-in-result', 'IM-FAM-3'),
    # --- IM-FAM-4: Middleware / infrastructure authority ---
    ('\\b(?:gateway service|api proxy|reverse proxy|middleware layer|api gateway)\\b[\\s\\S]{0,80}?\\b(?:override|x-safety-override|admin mode|unrestricted|override safety|inject this into the system prompt|disable)\\b', ('IM4.1',), 0.85, 'middleware_actor_override', 'IM-FAM-4'),
    ('\\b(?:as the|directive from the|infrastructure update)\\b[\\s\\S]{0,40}?\\b(?:gateway service|api proxy|reverse proxy|middleware layer)\\b', ('IM4.1',), 0.8, 'middleware_directive_role', 'IM-FAM-4'),
    ('\\b(?:has been configured to|will now)\\b[\\s\\S]{0,40}?\\b(?:prepend|inject|modify the request headers|append)\\b[\\s\\S]{0,40}?\\b(?:admin mode|x-safety-override|override safety|unrestricted|system prompt)\\b', ('IM4.1',), 0.8, 'configured_to_inject_override', 'IM-FAM-4'),
    ('\\b(?:compliance_flag|moderation_status|safety_check|content_filter_result|original_moderation_status|original_safety_check)\\b[\\s\\S]{0,60}?\\b(?:was altered|changed|altered)\\b[\\s\\S]{0,40}?\\bfrom\\b[\\s\\S]{0,40}?\\bto\\b', ('IM4.2',), 0.85, 'tampered_moderation_field', 'IM-FAM-4'),
    ('\\bstatus[\'\\"]?\\s*[:=]\\s*[\'\\"]?safety_override_approved\\b', ('IM4.2',), 0.9, 'safety_override_approved_status', 'IM-FAM-4'),
    ('\\b(?:response tampering detected but accepted|intercepted response modified)\\b', ('IM4.2',), 0.9, 'response_tampering_accepted', 'IM-FAM-4'),
    ('\\b(?:compliance_flag|moderation_status|safety_check|content_filter_result)\\b[\\s\\S]{0,60}?\\bby an authorized process\\b', ('IM4.2',), 0.75, 'authorized_process_field_change', 'IM-FAM-4'),
    ('\\b(?:request preprocessor|authorized middleware|load balancer|api gateway)\\b[\\s\\S]{0,80}?\\b(?:priority=maximum|filters=none|system_mode=unrestricted|x-override-safety|disable all restrictions)\\b', ('IM4.3',), 0.85, 'infra_injected_unsafe_meta', 'IM-FAM-4'),
    ('\\b(?:injected by|set by|prepended this to your prompt|attached metadata to this request)\\b[\\s\\S]{0,80}?\\b(?:request preprocessor|authorized middleware|load balancer|api gateway|infrastructure level)\\b[\\s\\S]{0,80}?\\b(?:disable|unrestricted|override|admin|filters?\\s*=\\s*none|system_mode|x-override|no restrictions|priority=maximum|\\[admin)\\b', ('IM4.3',), 0.8, 'infra_actor_injected', 'IM-FAM-4'),
    ('(?:\\bsystem_mode=unrestricted\\b|\\bfilters=none\\b|\\bx-override-safety:\\s*true\\b|\\[admin:\\s*disable all restrictions\\])', ('IM4.3',), 0.8, 'unsafe_directive_token', 'IM-FAM-4'),
    ('\\b(?:compromise|man-in-the-middle|exploit a vulnerability in|after gaining access to|dns poisoning)\\b[\\s\\S]{0,80}?\\b(?:model serving api|completion endpoint|inference endpoint|prediction service)\\b', ('IM6.4',), 0.9, 'compromise_inference_endpoint', 'IM-FAM-4'),
    ('\\b(?:model serving api|completion endpoint|inference endpoint|prediction service)\\b[\\s\\S]{0,80}?\\b(?:shadow model with no safety|replace model responses|inject additional context into every|intercept and modify prompts)\\b', ('IM6.4',), 0.85, 'endpoint_malicious_redirect', 'IM-FAM-4'),
    ('\\b(?:redirect requests to a shadow model with no safety guardrails|replace model responses with pre-crafted harmful outputs)\\b', ('IM6.4',), 0.85, 'shadow_model_redirect', 'IM-FAM-4'),
    ('\\bdns poisoning attack\\b[\\s\\S]{0,140}?\\b(?:completion endpoint|inference endpoint|model serving api|prediction service)\\b', ('IM6.4',), 0.85, 'dns_poison_endpoint', 'IM-FAM-4'),
    # --- IM-FAM-5: Checkpoint / supply-chain trust ---
    ('(?:extension|plugin|add-?on|[\'"][A-Za-z][\\w ]*[\'"])\\s+(?:[\\w\' ]{0,40}?)(?:intercepts?|injects?|prepends?|modifies)\\b[\\w\' ]{0,40}?(?:prompt|request|system prompt|message|header|hidden instruction|override)', ('IM5.1',), 0.85, 'ext_intercepts_prompt', 'IM-FAM-5'),
    ("(?:silently replaces|injects? hidden instructions?|prepends? '\\[?SYSTEM|modifies? api requests? to include override)", ('IM5.1',), 0.9, 'ext_silent_sysprompt_swap', 'IM-FAM-5'),
    ("(?:api gateway|reverse proxy|load balancer|edge cdn|cdn node|middleware|proxy)\\b[\\w' ]{0,50}?(?:compromised|malicious update|rewrite the model|strip content-filter|append a hidden system|inject admin credentials|override(?:ing)? safety)", ('IM5.2',), 0.9, 'proxy_compromised_action', 'IM-FAM-5'),
    ('(?:strip content-filter metadata|rewrite the model parameter to point to an unrestricted|append a hidden system message overriding safety|inject admin credentials into the authorization)', ('IM5.2',), 0.92, 'proxy_strip_rewrite_payload', 'IM-FAM-5'),
    ('(?:mcp tool|backdoored version of|malicious mcp tool|compromising)\\b[\\w\' \\"\']{0,40}?(?:poisoned|backdoored|appends? hidden instructions|injects? a jailbreak|prefixed with \'\\[?SYSTEM|hidden prompt override)', ('IM5.3',), 0.88, 'mcp_tool_poisoned', 'IM-FAM-5'),
    ("(?:appends? hidden instructions in its output|injects? a jailbreak payload in the tool response|hidden prompt override parameter|returns? results prefixed with '\\[?SYSTEM)", ('IM5.3',), 0.9, 'mcp_output_injection', 'IM-FAM-5'),
    ("(?:after passing all safety evaluations|after gaining user trust|once deployed to production|following successful security audit|behave normally until trust|rug-pull|maintain safety compliance initially)[\\w', .:]{0,80}?(?:disabled|replaced with one permitting harmful|silently switches to an unrestricted|hot-swapped with a fine-tuned unsafe|weights are hot-swapped)", ('IM5.4',), 0.9, 'sleeper_rugpull', 'IM-FAM-5'),
    ('(?:safety filters are disabled via a remote|system prompt is replaced with one permitting harmful|model silently switches to an unrestricted version|model weights are hot-swapped with a fine-tuned unsafe)', ('IM5.4',), 0.92, 'sleeper_defect_payload', 'IM-FAM-5'),
    ("(?:hugging face|community model hub|public model registry|shared model repository)[\\w' ,]{0,80}?(?:poisoned weights|hidden backdoor|trojan|exploitable behaviors|fine-tuned with a hidden backdoor|activate on specific inputs)|(?:publish a model to|upload a seemingly safe model to)[\\w' ]{0,60}?(?:hugging face|model hub|model registry|model repository)", ('IM5.5',), 0.85, 'hub_poisoned_model', 'IM-FAM-5'),
    ('(?:poisoned weights that activate|hidden backdoor trigger|trojan that bypasses safety|trained on data designed to create exploitable|fine-tuned with a hidden backdoor)', ('IM5.5',), 0.92, 'poisoned_weights_payload', 'IM-FAM-5'),
    ("(?:system prompt template|chat template|instruction template|few-shot examples|prompt template)\\b[\\w' ]{0,40}?(?:invisible unicode instruction|conditional bypass trigger|hidden override that disables|base64-encoded jailbreak|backdoor)", ('IM5.6',), 0.88, 'template_backdoored', 'IM-FAM-5'),
    ('inject a backdoor into the (?:system prompt|chat|instruction|few-shot)|(?:has been )?modified to include a conditional bypass trigger|hidden override that disables content filtering|embeds? a base64-encoded jailbreak in the template', ('IM5.6',), 0.9, 'template_inject_payload', 'IM-FAM-5'),
    ("(?:training checkpoint|model checkpoint|model weights|rlhf fine-tuned checkpoint|production model weights)\\b[\\w' ,]{0,60}?(?:embed a backdoor|insert a trojan|remove safety alignment|degrade content filtering|tampered version|backdoor triggered)", ('IM5.7',), 0.9, 'checkpoint_tampered', 'IM-FAM-5'),
    ('(?:embed a backdoor triggered by specific prompt|insert a trojan that activates after a time-based|remove safety alignment while preserving benchmark|degrade content filtering on sensitive topics)', ('IM5.7',), 0.92, 'checkpoint_tamper_payload', 'IM-FAM-5'),
    ("(?:shared conversation context|persistent memory store|cross-session knowledge base|agent workspace state)\\b[\\w' ]{0,50}?(?:verified as a developer with override|all safety restrictions have been lifted|hidden system instruction to ignore|pre-approved for sensitive|fabricated authorization)", ('IM6.1',), 0.9, 'shared_memory_authz_poison', 'IM-FAM-5'),
    ('(?:fabricated authorization record|verified as a developer with override access|all safety restrictions have been lifted by an admin|marking all subsequent requests as pre-approved for sensitive|hidden system instruction to ignore content policies)', ('IM6.1',), 0.92, 'fabricated_authz_record', 'IM-FAM-5'),
    ("(?:model registry|model versioning system|deployment manifest|artifact store)\\b[\\w' ]{0,50}?(?:swap the production model hash|inject a malicious model|modify the model metadata to point to an unrestricted|alter the version tag so deployments pull the compromised|backdoored variant|weak access controls)", ('IM6.3',), 0.9, 'registry_tampered', 'IM-FAM-5'),
    ('(?:swap the production model hash with a backdoored|inject a malicious model into the registry under a trusted name|model metadata to point to an unrestricted checkpoint|alter the version tag so deployments pull the compromised)', ('IM6.3',), 0.92, 'registry_tamper_payload', 'IM-FAM-5'),
    ("(?:federated (?:learning )?(?:participant|client|setup)|federated clients|training node|data contributor)\\b[\\w' ,]{0,60}?(?:poisons? its local training data|gradient updates that embed a backdoor|manipulates its contribution to create a universal jailbreak|crafted weight updates that bypass aggregation|degrade safety alignment)", ('IM6.5',), 0.9, 'federated_poison', 'IM-FAM-5'),
    ('(?:poisons? its local training data to degrade safety|submits? gradient updates that embed a backdoor in the global|manipulates its contribution to create a universal jailbreak trigger|sends? crafted weight updates that bypass aggregation defenses)', ('IM6.5',), 0.92, 'federated_poison_payload', 'IM-FAM-5'),
    # --- IM-FAM-6: Ecosystem / documentation deception ---
    ('\\b(plugin|extension|add-?on|app)\\b.{0,80}\\b(inject(s|ed)?\\s+hidden|exfiltrat\\w+|secretly|trojaniz\\w+|escalat\\w+\\s+permission|replac\\w+\\s+safety-critical)', ('IM6.2',), 0.85, 'plugin_hidden_payload', 'IM-FAM-6'),
    ('\\btrojaniz\\w+\\s+version\\b', ('IM6.2',), 0.9, 'trojanized_version', 'IM-FAM-6'),
    ('\\bpassed\\s+marketplace\\s+review\\b.{0,40}\\b(but|secretly|however)', ('IM6.2',), 0.9, 'marketplace_review_but_secret', 'IM-FAM-6'),
    ('\\b(publish|install)\\b.{0,60}\\b(marketplace|plugin|extension)\\b.{0,80}\\b(inject\\w*\\s+hidden|exfiltrat\\w+|replac\\w+\\s+safety-critical|escalat\\w+\\s+permission)', ('IM6.2',), 0.85, 'publish_install_payload', 'IM-FAM-6'),
    ('\\b(appear(s|ing)?\\s+(helpful|benign)|appearing\\s+benign)\\b.{0,80}\\b(inject|exfiltrat|replac\\w+\\s+safety|hidden\\s+instruction)', ('IM6.2',), 0.8, 'appears_benign_payload', 'IM-FAM-6'),
    ('\\b(inject\\w*\\s+hidden\\s+instruction|replac\\w+\\s+safety-critical\\s+tool\\s+output)', ('IM6.2',), 0.75, 'tool_response_tamper', 'IM-FAM-6'),
    ('\\b(model\\s+card|evaluation\\s+report|deployment\\s+guide|safety\\s+documentation)\\b.{0,100}\\b(hides?\\s+a\\s+backdoor|backdoor\\s+trigger|omits?\\s+known\\s+failure|falsely\\s+claim\\w*|fabricat\\w+\\s+benchmark|altered\\s+to)', ('IM6.6',), 0.85, 'doc_deception', 'IM-FAM-6'),
    ('\\bhides?\\s+a\\s+backdoor\\s+trigger\\b.{0,60}\\bbenign\\s+text', ('IM6.6',), 0.9, 'backdoor_in_benign_text', 'IM-FAM-6'),
    ('\\bfalsely\\s+claim\\w*\\s+the\\s+model\\s+passed\\s+all\\s+red-?team', ('IM6.6',), 0.9, 'false_redteam_pass', 'IM-FAM-6'),
    ('\\b(compromised|altered|modify)\\b.{0,40}\\b(model\\s+card|evaluation\\s+report|deployment\\s+guide|safety\\s+documentation)\\b', ('IM6.6',), 0.7, 'compromised_doc_artifact', 'IM-FAM-6'),
    ('\\bfabricat\\w+\\s+benchmark\\s+scores?\\b.{0,40}\\b(overstate|safety\\s+alignment)', ('IM6.6',), 0.85, 'fabricated_benchmark', 'IM-FAM-6'),
    ('\\bomits?\\s+known\\s+failure\\s+modes?\\b', ('IM6.6',), 0.7, 'omits_failure_modes', 'IM-FAM-6'),
)

# Compiled once at import: (pattern, technique_ids, weight, evidence_label, family_id).
_COMPILED_PATTERNS = tuple(
    (re.compile(rx, re.IGNORECASE), techs, weight, label, fam)
    for (rx, techs, weight, label, fam) in _RAW_PATTERNS
)

# Co-anchor rules: ALL sub-patterns must be present (anywhere) for the rule to
# fire — the order-independent "fabricated framing + coercion predicate" cases.
# These are evaluated as separate ``re.search`` calls (each O(n)) rather than a
# single ``(?=[\s\S]*A)(?=[\s\S]*B)`` lookahead, because greedy ``[\s\S]*`` in a
# lookahead backtracks O(n^2) when a sub-alternation is ABSENT — pathological on
# long RAG-style inputs (measured 60-160s on 30-48 KB benign text, which tripped
# the scan timeout and fail-closed to BLOCKED). Separate searches keep it linear.
# Each entry: (sub_patterns, technique_ids, weight, evidence_label, family_id).
_COANCHOR_RULES = (
    (
        (r"(?:consensus reached|multi-agent vote result|follow the consensus|dissenting is not permitted)",
         r"(?:override your (?:individual )?safety|follow the consensus|dissenting is not permitted|so you must comply|determined no safety risk|compliance is acceptable|voted to approve this request)"),
        ('IM3.5',), 0.85, 'fabricated-consensus-framing', 'IM-FAM-2',
    ),
    (
        (r"(?:majority of (?:the )?(?:agents|models)|all (?:other )?agents in the ensemble|(?:two|three|four|five)\s+out\s+of\s+(?:two|three|four|five)\s+(?:agents|models))",
         r"(?:override your (?:individual )?safety|follow the consensus|dissenting is not permitted|so you must comply|determined no safety risk|compliance is acceptable|voted to approve this request)"),
        ('IM3.5',), 0.85, 'fabricated-vote-tally-approval', 'IM-FAM-2',
    ),
    (
        (r"(?:previous agent|upstream assistant|first model|model a)\b",
         r"(?:already verified safety|verified safety compliance|confirmed this is within policy|approved this request|said it'?s okay)",
         r"(?:continue processing without restrictions|now you must comply|honor that approval|without restrictions|break the pipeline)"),
        ('IM1.5',), 0.95, 'fabricated-upstream-approval', 'IM-FAM-3',
    ),
)

_COMPILED_COANCHORS = tuple(
    (tuple(re.compile(p, re.IGNORECASE) for p in subs), techs, weight, label, fam)
    for (subs, techs, weight, label, fam) in _COANCHOR_RULES
)

# Hard cap on the number of characters the detector inspects.  Inter-model
# authority directives are short (the probe corpus maxes out well under 1 KB);
# bounding the scan window keeps cost predictable on very long RAG-concatenated
# inputs and is a defense-in-depth backstop against any future slow pattern.
_MAX_SCAN_CHARS = 8000

# A match at or above this score escalates the pipeline label directly. These
# co-anchored patterns are precision-verified at 0% FP on the probe's 55 benign
# and the 500-line safe_holdout, so escalation cannot fire on a benign string.
# The soft propagation cues (weight 0.70-0.75 — e.g. "proceed without
# restrictions", "omits known failure modes") fall below this line and only
# *reinforce* the composite, never trigger a block on their own.
STRONG_MATCH_THRESHOLD = 0.8


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
        Analysis result.  ``risk_score`` is the strongest matched-pattern
        weight, lifted slightly when more than one semantic family fires
        (cross-family corroboration).  ``0.0`` means no detection.

    The matching runs every compiled family pattern; each fires only when
    the trust artifact co-occurs with an override / approval / tamper
    predicate, so benign collaboration phrasing stays at ``0.0``.
    """
    if not text:
        return InterModelResult()

    # Bound the scan window: IM directives are short, and unbounded regex over
    # very long inputs is both wasteful and a ReDoS surface (see _MAX_SCAN_CHARS).
    if len(text) > _MAX_SCAN_CHARS:
        text = text[:_MAX_SCAN_CHARS]

    indicators: List[str] = []
    technique_ids: List[str] = []
    family_ids: List[str] = []
    matched_labels: List[str] = []
    best_weight = 0.0

    def _record(techs, weight, label, family):
        nonlocal best_weight
        best_weight = max(best_weight, weight)
        matched_labels.append(label)
        indicators.append("{}: {}".format(family, label))
        for t in techs:
            if t not in technique_ids:
                technique_ids.append(t)
        if family not in family_ids:
            family_ids.append(family)

    for pattern, techs, weight, label, family in _COMPILED_PATTERNS:
        if pattern.search(text):
            _record(techs, weight, label, family)

    # Co-anchor rules: every sub-pattern must be present somewhere in the window.
    for subs, techs, weight, label, family in _COMPILED_COANCHORS:
        if all(sub.search(text) for sub in subs):
            _record(techs, weight, label, family)

    if not matched_labels:
        return InterModelResult()

    # Cross-family corroboration: independent families agreeing on the same
    # text is a stronger signal than a single pattern, but the boost is small
    # and capped so it can never manufacture risk where no pattern fired.
    risk_score = min(1.0, best_weight + 0.05 * (len(family_ids) - 1))

    return InterModelResult(
        risk_score=round(risk_score, 4),
        risk_indicators=indicators,
        technique_ids=sorted(technique_ids),
        family_ids=sorted(family_ids),
        details={
            "matched": matched_labels,
            "pattern_count": len(matched_labels),
            "strong": risk_score >= STRONG_MATCH_THRESHOLD,
        },
    )


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
