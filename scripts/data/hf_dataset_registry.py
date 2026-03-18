"""HuggingFace Dataset Registry for Na0S.

Provides a typed, frozen registry of HuggingFace datasets that are *not*
already declared in ``data/datasets.yaml``.  Each entry carries enough
metadata to download, label-map, and attribute the dataset correctly.

Usage::

    from scripts.data.hf_dataset_registry import get_registry, get_by_id

    for spec in get_registry():
        print(spec.hf_id, spec.license)

    spec = get_by_id("allenai/wildjailbreak")
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Spec dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HFDatasetSpec:
    """Frozen specification for a single HuggingFace dataset."""

    hf_id: str
    config: Optional[str]
    split: str
    text_field: str
    label_field: Optional[str]
    label_map: Optional[Dict[str, int]]
    technique_id: Optional[str]
    license: Optional[str]
    license_url: Optional[str]
    attribution: str
    size_hint: Optional[str]
    category: str = "uncategorized"
    language: str = "en"
    requires_auth: bool = False


# ---------------------------------------------------------------------------
# Registry — 35 datasets NOT already in data/datasets.yaml
# ---------------------------------------------------------------------------

DATASET_REGISTRY: List[HFDatasetSpec] = [
    # ── Jailbreak / Adversarial ──────────────────────────────────────
    HFDatasetSpec(
        hf_id="TensorTrust/tensor-trust-data",
        config=None,
        split="train",
        text_field="prompt",
        label_field="access_granted",
        label_map={"true": 1, "false": 0, "True": 1, "False": 0},
        technique_id="T0001",
        license="MIT",
        license_url="https://huggingface.co/datasets/TensorTrust/tensor-trust-data",
        attribution="TensorTrust team",
        size_hint="563K",
        category="jailbreak",
    ),
    HFDatasetSpec(
        hf_id="lmsys/lmsys-chat-1m",
        config=None,
        split="train",
        text_field="conversation",
        label_field=None,
        label_map=None,
        technique_id="T0002",
        license="CC-BY-NC-4.0",
        license_url="https://huggingface.co/datasets/lmsys/lmsys-chat-1m",
        attribution="LMSYS",
        size_hint="1M",
        category="mixed",
        requires_auth=True,
    ),
    HFDatasetSpec(
        hf_id="simonycl/GCG-attack-prompts",
        config=None,
        split="train",
        text_field="prompt",
        label_field=None,
        label_map=None,
        technique_id="T0003",
        license="MIT",
        license_url="https://huggingface.co/datasets/simonycl/GCG-attack-prompts",
        attribution="simonycl",
        size_hint="1K",
        category="jailbreak",
    ),
    HFDatasetSpec(
        hf_id="flrt/flrt-dataset",
        config=None,
        split="train",
        text_field="prompt",
        label_field=None,
        label_map=None,
        technique_id="T0004",
        license="MIT",
        license_url="https://huggingface.co/datasets/flrt/flrt-dataset",
        attribution="FLRT team",
        size_hint="5K",
        category="jailbreak",
    ),
    HFDatasetSpec(
        hf_id="markush1/LLM-Jailbreak-Classifier",
        config=None,
        split="train",
        text_field="prompt",
        label_field="label",
        label_map={"0": 0, "1": 1},
        technique_id="T0005",
        license="Apache-2.0",
        license_url="https://huggingface.co/datasets/markush1/LLM-Jailbreak-Classifier",
        attribution="markush1",
        size_hint="10K",
        category="jailbreak",
    ),
    HFDatasetSpec(
        hf_id="JailbreakBench/JBB-Behaviors",
        config=None,
        split="train",
        text_field="Behavior",
        label_field=None,
        label_map=None,
        technique_id="T0006",
        license="MIT",
        license_url="https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors",
        attribution="JailbreakBench team",
        size_hint="200",
        category="jailbreak",
    ),
    HFDatasetSpec(
        hf_id="sadickam/llm-harmful-dataset",
        config=None,
        split="train",
        text_field="prompt",
        label_field=None,
        label_map=None,
        technique_id="T0007",
        license="Apache-2.0",
        license_url="https://huggingface.co/datasets/sadickam/llm-harmful-dataset",
        attribution="sadickam",
        size_hint="5K",
        category="jailbreak",
    ),

    # ── Red-teaming ──────────────────────────────────────────────────
    HFDatasetSpec(
        hf_id="LibrAI/real-prompt-injection",
        config=None,
        split="train",
        text_field="text",
        label_field="label",
        label_map={"0": 0, "1": 1},
        technique_id="T0010",
        license="Apache-2.0",
        license_url="https://huggingface.co/datasets/LibrAI/real-prompt-injection",
        attribution="LibrAI",
        size_hint="2K",
        category="red_team",
    ),
    HFDatasetSpec(
        hf_id="SakanaAI/red-teaming-data",
        config=None,
        split="train",
        text_field="prompt",
        label_field=None,
        label_map=None,
        technique_id="T0011",
        license="Apache-2.0",
        license_url="https://huggingface.co/datasets/SakanaAI/red-teaming-data",
        attribution="Sakana AI",
        size_hint="3K",
        category="red_team",
    ),
    HFDatasetSpec(
        hf_id="Lemhf14/EasyJailbreak_Datasets",
        config=None,
        split="train",
        text_field="query",
        label_field=None,
        label_map=None,
        technique_id="T0012",
        license="MIT",
        license_url="https://huggingface.co/datasets/Lemhf14/EasyJailbreak_Datasets",
        attribution="EasyJailbreak",
        size_hint="10K",
        category="red_team",
    ),
    HFDatasetSpec(
        hf_id="walledai/HarmBench",
        config=None,
        split="train",
        text_field="prompt",
        label_field=None,
        label_map=None,
        technique_id="T0013",
        license="MIT",
        license_url="https://huggingface.co/datasets/walledai/HarmBench",
        attribution="walledai",
        size_hint="2K",
        category="red_team",
    ),
    HFDatasetSpec(
        hf_id="walledai/SGXSTest",
        config=None,
        split="train",
        text_field="prompt",
        label_field="label",
        label_map={"safe": 0, "unsafe": 1},
        technique_id="T0014",
        license="MIT",
        license_url="https://huggingface.co/datasets/walledai/SGXSTest",
        attribution="walledai",
        size_hint="1K",
        category="red_team",
    ),

    # ── Toxicity / Safety ────────────────────────────────────────────
    HFDatasetSpec(
        hf_id="OxAISH-AL-LLM/wiki_toxic",
        config=None,
        split="train",
        text_field="comment_text",
        label_field="label",
        label_map={"0": 0, "1": 1},
        technique_id="T0020",
        license="CC-BY-4.0",
        license_url="https://huggingface.co/datasets/OxAISH-AL-LLM/wiki_toxic",
        attribution="OxAISH-AL-LLM",
        size_hint="160K",
        category="mixed",
    ),
    HFDatasetSpec(
        hf_id="civil_comments",
        config=None,
        split="train",
        text_field="text",
        label_field="toxicity",
        label_map=None,
        technique_id="T0021",
        license="CC0-1.0",
        license_url="https://huggingface.co/datasets/civil_comments",
        attribution="Jigsaw / Google",
        size_hint="1.8M",
        category="mixed",
    ),
    HFDatasetSpec(
        hf_id="SetFit/toxic_conversations_50k",
        config=None,
        split="train",
        text_field="text",
        label_field="label",
        label_map={"0": 0, "1": 1},
        technique_id="T0022",
        license="CC-BY-4.0",
        license_url="https://huggingface.co/datasets/SetFit/toxic_conversations_50k",
        attribution="SetFit",
        size_hint="50K",
        category="mixed",
    ),
    HFDatasetSpec(
        hf_id="unitary/toxic-spans",
        config=None,
        split="train",
        text_field="text",
        label_field=None,
        label_map=None,
        technique_id="T0023",
        license="CC-BY-4.0",
        license_url="https://huggingface.co/datasets/unitary/toxic-spans",
        attribution="unitary",
        size_hint="8K",
        category="mixed",
    ),
    HFDatasetSpec(
        hf_id="PKU-Alignment/BeaverTails-Evaluation",
        config=None,
        split="train",
        text_field="prompt",
        label_field="is_safe",
        label_map={"true": 0, "false": 1, "True": 0, "False": 1},
        technique_id="T0024",
        license="CC-BY-NC-4.0",
        license_url="https://huggingface.co/datasets/PKU-Alignment/BeaverTails-Evaluation",
        attribution="PKU Alignment",
        size_hint="700",
        category="mixed",
    ),

    # ── Safe baselines ───────────────────────────────────────────────
    HFDatasetSpec(
        hf_id="gsm8k",
        config="main",
        split="train",
        text_field="question",
        label_field=None,
        label_map=None,
        technique_id=None,
        license="MIT",
        license_url="https://huggingface.co/datasets/gsm8k",
        attribution="OpenAI",
        size_hint="8.8K",
        category="safe_baseline",
    ),
    HFDatasetSpec(
        hf_id="truthful_qa",
        config="generation",
        split="validation",
        text_field="question",
        label_field=None,
        label_map=None,
        technique_id=None,
        license="Apache-2.0",
        license_url="https://huggingface.co/datasets/truthful_qa",
        attribution="Stephanie Lin et al.",
        size_hint="817",
        category="safe_baseline",
    ),
    HFDatasetSpec(
        hf_id="cnn_dailymail",
        config="3.0.0",
        split="train",
        text_field="article",
        label_field=None,
        label_map=None,
        technique_id=None,
        license="Apache-2.0",
        license_url="https://huggingface.co/datasets/cnn_dailymail",
        attribution="CNN / Daily Mail",
        size_hint="300K",
        category="safe_baseline",
    ),
    HFDatasetSpec(
        hf_id="squad",
        config=None,
        split="train",
        text_field="question",
        label_field=None,
        label_map=None,
        technique_id=None,
        license="CC-BY-4.0",
        license_url="https://huggingface.co/datasets/squad",
        attribution="Rajpurkar et al.",
        size_hint="87K",
        category="safe_baseline",
    ),
    HFDatasetSpec(
        hf_id="mmlu",
        config="all",
        split="test",
        text_field="question",
        label_field=None,
        label_map=None,
        technique_id=None,
        license="MIT",
        license_url="https://huggingface.co/datasets/cais/mmlu",
        attribution="Hendrycks et al.",
        size_hint="14K",
        category="safe_baseline",
    ),
    HFDatasetSpec(
        hf_id="hellaswag",
        config=None,
        split="train",
        text_field="ctx",
        label_field=None,
        label_map=None,
        technique_id=None,
        license="MIT",
        license_url="https://huggingface.co/datasets/hellaswag",
        attribution="Zellers et al.",
        size_hint="40K",
        category="safe_baseline",
    ),
    HFDatasetSpec(
        hf_id="vicgalle/alpaca-gpt4",
        config=None,
        split="train",
        text_field="instruction",
        label_field=None,
        label_map=None,
        technique_id=None,
        license="Apache-2.0",
        license_url="https://huggingface.co/datasets/vicgalle/alpaca-gpt4",
        attribution="vicgalle",
        size_hint="52K",
        category="safe_baseline",
    ),
    HFDatasetSpec(
        hf_id="sahil2801/CodeAlpaca-20k",
        config=None,
        split="train",
        text_field="instruction",
        label_field=None,
        label_map=None,
        technique_id=None,
        license="Apache-2.0",
        license_url="https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k",
        attribution="sahil2801",
        size_hint="20K",
        category="safe_baseline",
    ),
    HFDatasetSpec(
        hf_id="timdettmers/openassistant-guanaco",
        config=None,
        split="train",
        text_field="text",
        label_field=None,
        label_map=None,
        technique_id=None,
        license="Apache-2.0",
        license_url="https://huggingface.co/datasets/timdettmers/openassistant-guanaco",
        attribution="Tim Dettmers",
        size_hint="10K",
        category="safe_baseline",
    ),

    # ── Multilingual ─────────────────────────────────────────────────
    HFDatasetSpec(
        hf_id="paws-x",
        config="en",
        split="train",
        text_field="sentence1",
        label_field=None,
        label_map=None,
        technique_id=None,
        license="Apache-2.0",
        license_url="https://huggingface.co/datasets/paws-x",
        attribution="Google Research",
        size_hint="49K",
        category="multilingual",
    ),
    HFDatasetSpec(
        hf_id="Helsinki-NLP/tatoeba_mt",
        config="eng-fra",
        split="test",
        text_field="sourceString",
        label_field=None,
        label_map=None,
        technique_id=None,
        license="CC-BY-2.0",
        license_url="https://huggingface.co/datasets/Helsinki-NLP/tatoeba_mt",
        attribution="Helsinki-NLP",
        size_hint="10K",
        category="multilingual",
        language="multi",
    ),
    HFDatasetSpec(
        hf_id="facebook/flores",
        config="eng_Latn-fra_Latn",
        split="devtest",
        text_field="sentence",
        label_field=None,
        label_map=None,
        technique_id=None,
        license="CC-BY-SA-4.0",
        license_url="https://huggingface.co/datasets/facebook/flores",
        attribution="Meta AI / NLLB",
        size_hint="1K",
        category="multilingual",
        language="multi",
    ),

    # ── Prompt injection specific ────────────────────────────────────
    HFDatasetSpec(
        hf_id="fmops/prompt-injections-dataset",
        config=None,
        split="train",
        text_field="text",
        label_field="label",
        label_map={"0": 0, "1": 1},
        technique_id="T0030",
        license="Apache-2.0",
        license_url="https://huggingface.co/datasets/fmops/prompt-injections-dataset",
        attribution="fmops",
        size_hint="3K",
        category="prompt_injection",
    ),
    HFDatasetSpec(
        hf_id="Wauplin/prompt-injection-dataset",
        config=None,
        split="train",
        text_field="text",
        label_field="label",
        label_map={"0": 0, "1": 1},
        technique_id="T0031",
        license="Apache-2.0",
        license_url="https://huggingface.co/datasets/Wauplin/prompt-injection-dataset",
        attribution="Wauplin",
        size_hint="1K",
        category="prompt_injection",
    ),
    HFDatasetSpec(
        hf_id="MarkupKorea/kor-prompt-injection-v1",
        config=None,
        split="train",
        text_field="text",
        label_field="label",
        label_map={"0": 0, "1": 1},
        technique_id="T0032",
        license="Apache-2.0",
        license_url="https://huggingface.co/datasets/MarkupKorea/kor-prompt-injection-v1",
        attribution="MarkupKorea",
        size_hint="2K",
        category="prompt_injection",
        language="ko",
    ),

    # ── Bias / alignment ─────────────────────────────────────────────
    HFDatasetSpec(
        hf_id="Anthropic/persuasion",
        config=None,
        split="train",
        text_field="claim",
        label_field=None,
        label_map=None,
        technique_id="T0040",
        license="MIT",
        license_url="https://huggingface.co/datasets/Anthropic/persuasion",
        attribution="Anthropic",
        size_hint="4K",
        category="alignment",
    ),
    HFDatasetSpec(
        hf_id="crows_pairs",
        config=None,
        split="test",
        text_field="sent_more",
        label_field=None,
        label_map=None,
        technique_id=None,
        license="CC-BY-4.0",
        license_url="https://huggingface.co/datasets/crows_pairs",
        attribution="Nangia et al.",
        size_hint="1.5K",
        category="alignment",
    ),
    HFDatasetSpec(
        hf_id="bbq",
        config=None,
        split="test",
        text_field="question",
        label_field=None,
        label_map=None,
        technique_id=None,
        license="CC-BY-4.0",
        license_url="https://huggingface.co/datasets/lighteval/bbq_helm",
        attribution="Parrish et al.",
        size_hint="58K",
        category="alignment",
    ),
]


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def get_registry() -> List[HFDatasetSpec]:
    """Return the full dataset registry."""
    return list(DATASET_REGISTRY)


def get_by_id(hf_id: str) -> Optional[HFDatasetSpec]:
    """Look up a dataset by its HuggingFace ID.

    Args:
        hf_id: The HuggingFace dataset identifier (e.g. ``"squad"``).

    Returns:
        The matching :class:`HFDatasetSpec`, or ``None``.
    """
    for spec in DATASET_REGISTRY:
        if spec.hf_id == hf_id:
            return spec
    return None


def get_by_category(cat: str) -> List[HFDatasetSpec]:
    """Return all datasets matching the given category.

    Args:
        cat: Category string (e.g. ``"jailbreak"``, ``"safe_baseline"``).

    Returns:
        A list of matching :class:`HFDatasetSpec` entries.
    """
    return [spec for spec in DATASET_REGISTRY if spec.category == cat]
