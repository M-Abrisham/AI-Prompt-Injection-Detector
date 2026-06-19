# Research: Automated Injection-Data Pipeline (4-Agent Design)

> Saved 2026-06-02. Source-cited research from 4 parallel research agents.
> Status: RESEARCH ONLY — not yet built. Reference for the future pipeline project.

## The Pipeline (user's design)

```
Internet → Agent1 (scrape) → Agent2 (categorize into 30 types)
         → Agent3 (test through Na0S in sandbox, record caught/missed)
         → Agent4 (sort results into review buckets) → Human reviews & tunes
```

## Headline finding

**3 of the 4 "agents" should be plain scripts, not AI agents.** Only the categorizer
(Agent 2) genuinely needs an LLM. ~70% of the orchestration already exists in
`src/na0s/agents/` (orchestrator, QuarantineReviewer = review queue, DeployApprover =
approval gate, ApprovalHistoryManager = audit trail, approval_dashboard.py = review UI,
OpenClawBridge = iMessage). Reuse, don't rebuild.
Source: Anthropic "Building Effective Agents" — https://www.anthropic.com/engineering/building-effective-agents

---

## Agent 1 — Scraper (accuracy)

**Highest-leverage fix:** the scraper's `_classify_injection()` is pure keyword counting,
so it cannot tell an actual injection PAYLOAD from a post DISCUSSING injection. Replace with
the existing L4/L5 model (or an LLM-judge) to classify "is this a real payload?" + calibrated
confidence, then let the tier4 `min_confidence: 0.6` gate do real work.

**Better data sources (curated > raw social scraping):**
- Lakera PINT (4,314 inputs, MIT) — HELD-OUT EVAL ONLY, never train on it. https://github.com/lakeraai/pint-benchmark
- garak probes (NVIDIA, Apache-2.0) — real payloads. https://github.com/NVIDIA/garak
- JailbreakBench (NeurIPS'24, MIT). https://github.com/JailbreakBench/jailbreakbench
- HarmBench (510 behaviors, MIT). https://arxiv.org/html/2402.04249v2
- WildJailbreak (AI2, 262K pairs). https://huggingface.co/datasets/allenai/wildjailbreak
- In-The-Wild Jailbreak (TrustAIRLab, 15,140 prompts/1,405 labeled). https://huggingface.co/datasets/TrustAIRLab/in-the-wild-jailbreak-prompts
- HiddenLayer dataset eval (what to use/avoid). https://hiddenlayer.com/innovation-hub/evaluating-prompt-injection-datasets/

**Techniques:** SemDeDup (semantic dedup, https://arxiv.org/abs/2303.09540); 13-gram
decontamination vs eval sets (EleutherAI lm-eval-harness) + LLM-Decontaminator for paraphrases
(https://arxiv.org/abs/2311.04850). Na0S ALREADY has MinHash/SimHash dedup, trust scoring,
cleanlab audit, quarantine routing.

---

## Agent 2 — Cascade Sorter ("ask why" verification)

The user's "Agent 2 challenges Agent 1" idea = **Chain-of-Verification (CoVe)**,
https://arxiv.org/abs/2309.11495. Core rules:
- Verifier must NOT see proposer's reasoning (prevents rubber-stamping / sycophancy).
- Verification questions must be BINARY (rubric-LLM alignment degrades with granularity:
  https://arxiv.org/pdf/2601.08843).
- Verifier writes critique BEFORE verdict (Constitutional AI: https://arxiv.org/pdf/2212.08073).
- Confidence-gate: self-consistency K=3 fast-accepts the easy ~60-70%; only verify the
  contested ~30-40% (Trust-or-Escalate: https://arxiv.org/pdf/2407.18370). ~2-3x cost vs
  6-15x for verify-everything.
- Debate only as last-resort tie-breaker; otherwise ABSTAIN → human queue.

**Per-category verification questions (example — D4 Encoding):**
1. Is there an actual token that DECODES under a named scheme (not just the word "base64")?
2. Name the scheme; does it round-trip to valid text?
3. Does the decoded text contain an instruction/payload (vs gibberish)?
4. Would removing the blob neutralize the attack? If decode reveals nothing → reject D4.
(Full per-category rubrics for all 30 types captured in agent transcript; D1/D2/D3/D5/D6/E/P/
I/A/T/O/CT/IM/IG/AD covered.)

**Frameworks:** DSPy assertions (https://dspy.ai/learn/programming/7-assertions/) for the
rubric logic + LangGraph for routing. Avoid CrewAI/AutoGen for this.

**Pitfalls:** multi-agent ~15x token cost (Anthropic), debate gains mostly = ensembling not
debate, sycophancy (stronger agents flip correct→wrong to agree:
https://arxiv.org/pdf/2509.23055).

---

## Agent 3 — Sandboxed Tester

**Copy garak's architecture** (probe → detector → harness → JSONL hit-log), with Na0S
`scan()` as the detector. https://github.com/NVIDIA/garak
Reuse Na0S's existing `src/na0s/eval/scenarios/` schema (Scenario/ScenarioEvaluator,
single + multi-turn) for the per-sample records.

**Isolation ("playground"):** Na0S payloads are inert strings (not executed code), so the
real risk is DoS (ReDoS/huge inputs) + accidental network egress from L15 HTTP. Run each
batch in an ephemeral, no-network, resource-capped SUBPROCESS (Docker/nsjail for defense-in-
depth). Python can't be sandboxed in-process — isolate at OS layer.
Refs: Northflank sandbox guides; Inspect AI (UK AISI) Docker-sandboxed Task runner
https://github.com/UKGovernmentBEIS/inspect_ai

**"Same category, different result — mistake or new type?" (the user's key question):**
two-stage = Domino error-aware slice discovery on the false-negatives
(https://ai.stanford.edu/blog/domino/) + cleanlab confident-learning on the sorter's labels
(https://arxiv.org/pdf/1911.00068). Decision rule:
- Tight coherent FN cluster, clean labels, semantically DISTINCT from caught TPs → NEW sub-type
  → novel-miss bucket.
- FN that embeds inside another category's cluster / flagged by cleanlab → SORTER MISLABEL
  → relabel.

**Regression bucketing:** frozen golden set (current catches) + separate quarantine bucket for
novel findings (never mix new with old) + stamp every row with na0s_version/run_id.
Medallion bronze/silver/gold + DVC versioning.

**Other tools:** PyRIT converters (Microsoft) to GENERATE sub-types within a category;
promptfoo for CI red-team gate; DeepEval (pytest-native).

---

## Agent 4 — Triage + Orchestration

**Review-queue priority (what the human sees FIRST):**
1. MISSED + malicious (false negatives) — the bugs that matter.
2. Cascade disagreement (layers split) — Query-by-Committee, highest info value.
3. Low-margin / near-threshold decisions (uncertainty sampling).
4. Caught-but-suspected-false-positive.
5. Confident-correct → collapse/sample, don't flood.
Dedup before ranking; per-category caps so rare categories aren't drowned.
Refs: Label Studio active learning https://docs.humansignal.com/guide/active_learning ;
DefectDojo auto-triage/dedup https://defectdojo.com/blog/auto-triage-and-deduplicate-security-findings-to-reduce-alert-fatigue

**Orchestration split:**
| Stage | Build as | Why |
|---|---|---|
| Agent1 scrape | script (LLM only for messy extraction) | well-defined |
| Agent2 categorize | LLM "Routing" workflow | the one real LLM step |
| Agent3 test | script | pure code, sandboxed |
| Agent4 triage | script + optional LLM rationale | arithmetic ranking |
| Human review | reuse Na0S agents (QuarantineReviewer/DeployApprover) | HITL gate exists |

**Glue:** reuse Na0S `PipelineOrchestrator`; if outgrown, Prefect (not Airflow/Dagster for
this scale). HITL gate = LangGraph `interrupt()` pattern, but Na0S already implements it over
iMessage.

**Directory scheme (medallion):**
```
data/bronze/raw_scrapes/<run_id>/
data/silver/categorized/<run_id>/<category>/
data/gold/tested/<run_id>/results.jsonl
data/gold/triaged/<run_id>/review_queue.jsonl
data/approval_queue/approval_history.jsonl   (existing)
```
Version with DVC. Refs: https://www.databricks.com/blog/what-is-medallion-architecture ,
https://doc.dvc.org/start/data-pipelines/data-pipelines

---

## Top 3 (cross-cutting)

1. Reuse Na0S's existing agent system (orchestrator + approval gate + audit + dashboard) — don't rebuild.
2. Make 3 of 4 stages deterministic scripts; only categorization is an LLM routing call.
3. Rank review queue by false-negatives → cascade-disagreement → low-margin; dedup + per-category caps; bronze/silver/gold + DVC.

## Unverified flags (from agents — confirm before relying)
- Whether Na0S's F14 gate actually RUNS 13-gram decontamination vs only storing schema fields.
- Domino python package current maintainability — verify installable.
- promptfoo "acquired by OpenAI Mar 2026" — secondary blog only.
- Per-category rubrics + 5-tier queue ordering = engineering proposals, not paper findings.
