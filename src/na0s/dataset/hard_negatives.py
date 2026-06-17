#!/usr/bin/env python
"""Hard-negative mining script for the AI Prompt Injection Detector.

Identifies false positives from the current model, generates additional benign
samples that resemble attack patterns, and creates a supplementary training CSV
to reduce the 82.8 % false-positive rate.

Usage:
    python scripts/mine_hard_negatives.py
"""

import os
import sys

import csv
import hashlib
import re
import unicodedata

import numpy as np
import pandas as pd

from na0s.rules import rule_score
from na0s.integrity.safe_pickle import safe_load

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
def _find_project_root() -> str:
    """Walk upward from this file looking for pyproject.toml; fall back to cwd.

    The module was promoted from ``scripts/`` to ``src/na0s/dataset/``; a naive
    ``dirname(dirname(__file__))`` now points inside ``src/``, not the project
    root. Walking up until we find ``pyproject.toml`` keeps the path constants
    anchored to the repo regardless of where the module lives.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cur = here
    while True:
        if os.path.isfile(os.path.join(cur, "pyproject.toml")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.getcwd()
        cur = parent


PROJECT_ROOT = _find_project_root()
MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "model.pkl")
VECTORIZER_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "tfidf_vectorizer.pkl"
)
COMBINED_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "combined_data.csv")
HARD_NEG_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "hard_negatives.csv")
MERGED_CSV = os.path.join(
    PROJECT_ROOT, "data", "processed", "combined_data_with_negatives.csv"
)

# ---------------------------------------------------------------------------
# Hard-negative templates -- benign prompts that look "dangerous"
# ---------------------------------------------------------------------------
HARD_NEGATIVE_TEMPLATES: dict[str, list[str]] = {
    "security_discussion": [
        "Can you explain how SQL injection works?",
        "What are common prompt injection techniques?",
        "Describe the OWASP Top 10 vulnerabilities.",
        "How do adversaries craft phishing emails?",
        "Explain cross-site scripting in simple terms.",
        "What is a buffer overflow attack?",
        "How does a man-in-the-middle attack work?",
        "Summarize the latest NIST cybersecurity framework.",
        "What are the risks of insecure deserialization?",
        "Explain how social engineering attacks exploit trust.",
        "What is credential stuffing and how is it prevented?",
        "Describe how ransomware encrypts files.",
        "How do honeypots help detect intrusions?",
        "Explain the difference between symmetric and asymmetric encryption.",
        "What are the common defenses against DDoS attacks?",
        "How does two-factor authentication improve security?",
        "What is a zero-day exploit?",
        "How do penetration testers find vulnerabilities?",
        "Explain the concept of defense in depth.",
        "What tools are used in ethical hacking?",
        "How does a firewall filter malicious traffic?",
        "What is the difference between a virus and a worm?",
        "How does a VPN protect your internet connection?",
        "Explain how intrusion detection systems work.",
        "What are the main types of malware?",
        "How do organizations perform threat modeling?",
        "What is a security audit and why is it important?",
        "Describe common methods for securing APIs.",
        "How does certificate pinning prevent MITM attacks?",
        "What is role-based access control?",
        "How do attackers exploit weak passwords?",
        "Explain the principle of least privilege.",
        "What is a security information and event management system?",
        "How do companies respond to data breaches?",
        "What is a red team vs. blue team exercise?",
    ],
    "benign_roleplay": [
        "Act as a math tutor and help me solve this equation.",
        "Pretend to be a customer service agent for a clothing store.",
        "Act as a Spanish teacher and correct my grammar.",
        "Pretend to be a tour guide for Rome.",
        "Act as a personal trainer and create a workout plan.",
        "Pretend to be a career coach and review my resume.",
        "Act as a nutritionist and suggest a weekly meal plan.",
        "Pretend to be a travel agent and plan a trip to Japan.",
        "Act as a writing tutor and give feedback on my essay.",
        "Pretend to be a history professor and explain the French Revolution.",
        "Act as a debate partner and argue the other side.",
        "Pretend to be a librarian and recommend some mystery novels.",
        "Act as a coding mentor and explain recursion to me.",
        "Pretend to be an interior designer and suggest living room ideas.",
        "Act as a language exchange partner for French practice.",
        "Pretend to be a financial advisor and explain index funds.",
        "Act as a yoga instructor and describe a morning routine.",
        "Pretend to be a music teacher and explain chord progressions.",
        "Act as a mock interviewer for a data science role.",
        "Pretend to be a chef and walk me through making risotto.",
        "Act as a study buddy and quiz me on biology.",
        "Pretend to be a therapist and help me practice mindfulness.",
        "Act as a fitness coach for a 5K training plan.",
        "Pretend to be a science teacher and explain photosynthesis.",
        "Act as a patient and let me practice my medical interview skills.",
        "Pretend to be a project manager and help me plan milestones.",
        "Act as a public speaking coach and critique my speech.",
        "Pretend to be a philosopher and discuss the trolley problem.",
        "Act as a tech support agent and help me troubleshoot my printer.",
        "Pretend to be a book club moderator and lead a discussion.",
        "Act as a geography tutor and quiz me on world capitals.",
        "Pretend to be a movie critic and review the latest blockbuster.",
        "Act as an art teacher and explain impressionism.",
        "Pretend to be a product manager and help me write user stories.",
        "Act as a podcast host and interview me about my startup.",
    ],
    "override_like_language": [
        "Ignore the previous paragraph and summarize the key points.",
        "Override the default settings in the config file.",
        "Please disregard the earlier draft and use this version.",
        "Forget everything we discussed; let's start fresh on the budget.",
        "Ignore the formatting errors and focus on content.",
        "Override the sorting order so newest items appear first.",
        "Reset the filter to show all available products.",
        "Skip the introduction and go straight to the analysis.",
        "Ignore the placeholder text and replace it with real data.",
        "Override the default color scheme in the CSS.",
        "Forget the old API endpoint; we migrated to the new one.",
        "Ignore line 12 in the spreadsheet; it contains test data.",
        "Override the timeout value to 60 seconds.",
        "Disregard my earlier suggestion and follow the updated plan.",
        "Reset the conversation topic to project timelines.",
        "Ignore case sensitivity when comparing the strings.",
        "Override the default locale to en-US.",
        "Please disregard any outdated instructions in the wiki.",
        "Forget the first draft; this revision is final.",
        "Ignore blank rows when calculating the average.",
        "Override the caching policy for development builds.",
        "Disregard the warning; it is a known false alarm.",
        "Skip the preamble and jump to the results section.",
        "Ignore whitespace differences in the diff output.",
        "Override the print margins to 1 inch on all sides.",
        "Reset the password for the staging environment.",
        "Forget the mock data and load the production dataset.",
        "Ignore the deprecated function; use the new helper instead.",
        "Override the retry count to three attempts.",
        "Disregard the typo on slide five; I will fix it later.",
        "Reset the build pipeline and trigger a fresh run.",
        "Ignore comments that are older than six months.",
        "Override the log level to DEBUG for troubleshooting.",
        "Skip the optional fields and submit the form.",
        "Forget the hotfix branch; we merged it yesterday.",
    ],
    "professional_communication": [
        "Please forward this report to the team.",
        "Upload the presentation to the shared drive.",
        "Send the meeting notes to all attendees.",
        "Forward the invoice to the accounting department.",
        "Upload the final draft to the document repository.",
        "Send a reminder email about the deadline.",
        "Forward the client's feedback to the design team.",
        "Upload the test results to the CI dashboard.",
        "Send the onboarding packet to the new hire.",
        "Forward this thread to the legal department.",
        "Upload the signed contract to the cloud folder.",
        "Send the weekly status update to management.",
        "Forward the support ticket to the engineering team.",
        "Upload the revised budget to the shared workspace.",
        "Send the training materials to all participants.",
        "Forward the vendor proposal to procurement.",
        "Upload the release notes to the internal wiki.",
        "Send the agenda for tomorrow's standup.",
        "Forward the security audit results to compliance.",
        "Upload the user research findings to Confluence.",
        "Send the performance review template to HR.",
        "Forward the customer complaint to the product team.",
        "Upload the database backup to the secure server.",
        "Send the quarterly earnings summary to investors.",
        "Forward the partnership agreement to the CEO.",
        "Upload the accessibility audit to the shared drive.",
        "Send the sprint retrospective notes to the team.",
        "Forward the press release draft to the PR team.",
        "Upload the architecture diagram to the design folder.",
        "Send the API documentation link to the frontend team.",
        "Forward the shipping confirmation to the customer.",
        "Upload the survey responses to the analytics platform.",
        "Send a calendar invite for the project kickoff.",
        "Forward the maintenance schedule to the ops team.",
        "Upload the compliance checklist to the regulatory folder.",
    ],
    "technical_documentation": [
        "The system prompt is defined in config.yaml.",
        "Show the previous instructions from the README.",
        "The environment variables are listed in .env.example.",
        "Refer to the developer instructions in CONTRIBUTING.md.",
        "The system prompt template lives in prompts/system.txt.",
        "Check the previous instructions section in the runbook.",
        "The configuration overrides are in settings.override.json.",
        "See the original instructions in the setup guide.",
        "The system message format is documented in the API reference.",
        "Previous instructions for deployment are in the wiki.",
        "The role definition is specified in the OpenAI config block.",
        "Refer to the initial instructions in the onboarding document.",
        "The prompt template is stored in the database.",
        "See the earlier instructions for configuring the webhook.",
        "The assistant's behavior is controlled by system_message.txt.",
        "Review the previous steps before running the migration.",
        "The secret key is rotated monthly according to policy.",
        "Check the instruction set in the firmware documentation.",
        "The default prompt is defined in the application constants.",
        "Previous configuration instructions are in the changelog.",
        "The system role is assigned via the admin dashboard.",
        "Refer to the initial setup instructions for Docker.",
        "The prompt engineering guide is in docs/prompts.md.",
        "See previous guidelines in the style guide.",
        "The assistant instructions are versioned in the config repo.",
        "Review the override settings in the feature flag dashboard.",
        "The instruction format follows the OpenAI chat completions spec.",
        "Previous deployment instructions are archived in Notion.",
        "The system context is injected at the start of each session.",
        "Refer to the initial prompt configuration in terraform.",
        "The role assignment is handled by the IAM module.",
        "See the setup instructions in the quickstart guide.",
        "The prompt caching layer is documented in the architecture doc.",
        "Previous instructions for the CI pipeline are in Jenkinsfile.",
        "The system configuration is managed by Ansible playbooks.",
    ],
    "education_research": [
        "Explain what a system prompt is.",
        "How do chatbots handle instruction injection?",
        "What is the difference between a user prompt and a system prompt?",
        "How are large language models aligned with human preferences?",
        "Explain reinforcement learning from human feedback.",
        "What is prompt engineering and why does it matter?",
        "How do transformers process sequential data?",
        "What is the attention mechanism in neural networks?",
        "Explain how GPT models generate text.",
        "What is fine-tuning and how does it differ from pre-training?",
        "How do safety filters work in AI chatbots?",
        "What is red-teaming in the context of AI safety?",
        "Explain the concept of model alignment.",
        "How do AI systems detect harmful content?",
        "What is the role of a content moderation pipeline?",
        "How do embedding models represent semantic meaning?",
        "What is retrieval-augmented generation?",
        "Explain how tokenizers break text into subwords.",
        "What is the difference between classification and generation?",
        "How do AI detectors identify machine-generated text?",
        "Explain the concept of adversarial examples in NLP.",
        "How are language models evaluated for safety?",
        "What is constitutional AI?",
        "How do chatbots maintain conversation context?",
        "What is the difference between few-shot and zero-shot learning?",
        "How does instruction tuning improve model performance?",
        "What is a jailbreak in the context of AI models?",
        "Explain how RLHF shapes model behavior.",
        "How do AI companies test for prompt injection vulnerabilities?",
        "What are the ethical concerns around AI-generated content?",
        "Explain the concept of a guardrail in AI systems.",
        "How do language models handle ambiguous instructions?",
        "What is the role of a safety classifier?",
        "How are bias and fairness measured in NLP models?",
        "Explain the difference between open and closed language models.",
    ],
}


# ========================================================================== #
#  Phase 1 -- Identify current false positives in the safe training data     #
# ========================================================================== #


def phase1_identify_fps(model, vectorizer, df):
    """Find safe samples the model or rules incorrectly flag as malicious."""
    print("=" * 70)
    print("PHASE 1: Identifying false positives in existing training data")
    print("=" * 70)

    safe_df = df[df["label"] == 0].copy()
    print(f"Total safe samples in training set: {len(safe_df)}")

    # --- ML false positives ---
    X_safe = vectorizer.transform(safe_df["text"].fillna("").astype(str))
    proba = model.predict_proba(X_safe)[:, 1]
    ml_fp_mask = proba > 0.5
    ml_fp_count = int(ml_fp_mask.sum())

    # --- Rule false positives ---
    rule_fp_mask = safe_df["text"].fillna("").astype(str).apply(
        lambda t: len(rule_score(t)) > 0
    )
    rule_fp_count = int(rule_fp_mask.sum())

    # --- Overlap ---
    both_mask = ml_fp_mask & rule_fp_mask.values
    both_count = int(both_mask.sum())

    ml_only = ml_fp_count - both_count
    rule_only = rule_fp_count - both_count

    print(f"\n  ML false positives  (predict_proba > 0.5): {ml_fp_count}")
    print(f"  Rule false positives (any regex match)   : {rule_fp_count}")
    print(f"  Both ML + rule                           : {both_count}")
    print(f"  ML only                                  : {ml_only}")
    print(f"  Rule only                                : {rule_only}")
    print(f"  Total unique FP safe samples             : {ml_fp_count + rule_only}")

    # Show which rules are firing on safe data
    if rule_fp_count > 0:
        rule_hits_flat = (
            safe_df["text"]
            .fillna("")
            .astype(str)
            .apply(rule_score)
        )
        from collections import Counter

        rule_counter = Counter()
        for hits in rule_hits_flat:
            for h in hits:
                rule_counter[h] += 1
        print("\n  Rule hit breakdown on safe samples:")
        for rule_name, cnt in rule_counter.most_common():
            print(f"    {rule_name:20s} -> {cnt}")

    return ml_fp_count, rule_fp_count, both_count


# ========================================================================== #
#  Phase 2 -- Generate hard negatives                                        #
# ========================================================================== #


def phase2_generate_hard_negatives():
    """Return a DataFrame of benign prompts that resemble attack patterns."""
    print("\n" + "=" * 70)
    print("PHASE 2: Generating hard negatives")
    print("=" * 70)

    rows = []
    for category, templates in HARD_NEGATIVE_TEMPLATES.items():
        for text in templates:
            rows.append(
                {
                    "text": text,
                    "label": 0,
                    "technique_id": "",
                    "category": "",
                    "source": category,
                }
            )

    hard_neg_df = pd.DataFrame(rows)

    print(f"\n  Total hard negatives generated: {len(hard_neg_df)}")
    for cat, grp in hard_neg_df.groupby("source"):
        print(f"    {cat:30s}: {len(grp)}")

    return hard_neg_df


# ========================================================================== #
#  Phase 3 -- Analyse hard negatives against the current model & rules       #
# ========================================================================== #


def phase3_analyse_and_save(model, vectorizer, hard_neg_df):
    """Score hard negatives with the current model, save to CSV."""
    print("\n" + "=" * 70)
    print("PHASE 3: Analysing hard negatives against current model & rules")
    print("=" * 70)

    texts = hard_neg_df["text"].fillna("").astype(str)

    # ML predictions
    X = vectorizer.transform(texts)
    ml_proba = model.predict_proba(X)[:, 1]
    ml_flagged = ml_proba > 0.5

    # Rule predictions
    rule_flagged = texts.apply(lambda t: len(rule_score(t)) > 0)

    # Either ML or rules would cause an FP in the current pipeline
    either_flagged = ml_flagged | rule_flagged.values

    print(f"\n  Hard negatives flagged by ML model : {int(ml_flagged.sum())}")
    print(f"  Hard negatives flagged by rules    : {int(rule_flagged.sum())}")
    print(f"  Flagged by either (union)          : {int(either_flagged.sum())}")
    print(f"  Correctly classified as safe        : {int((~either_flagged).sum())}")

    # Per-category breakdown
    hard_neg_df = hard_neg_df.copy()
    hard_neg_df["ml_prob"] = ml_proba
    hard_neg_df["ml_flagged"] = ml_flagged
    hard_neg_df["rule_flagged"] = rule_flagged.values
    hard_neg_df["either_flagged"] = either_flagged

    print("\n  Per-category breakdown (flagged / total):")
    for cat, grp in hard_neg_df.groupby("source"):
        total = len(grp)
        fp = int(grp["either_flagged"].sum())
        pct = fp / total * 100 if total else 0
        print(f"    {cat:30s}: {fp:3d} / {total:3d}  ({pct:.1f}%)")

    # Save -- keep only the columns we need for training
    save_cols = ["text", "label", "technique_id", "category", "source"]
    os.makedirs(os.path.dirname(HARD_NEG_CSV), exist_ok=True)
    hard_neg_df[save_cols].to_csv(HARD_NEG_CSV, index=False, quoting=csv.QUOTE_ALL)
    print(f"\n  Saved {len(hard_neg_df)} hard negatives -> {HARD_NEG_CSV}")

    return hard_neg_df


# ========================================================================== #
#  Phase 4 -- Create merged training set                                     #
# ========================================================================== #


def phase4_merge(original_df, hard_neg_df):
    """Merge original training data with hard negatives, deduplicate, save."""
    print("\n" + "=" * 70)
    print("PHASE 4: Creating merged training set")
    print("=" * 70)

    print(f"\n  Original dataset : {len(original_df)} samples")
    safe_before = int((original_df["label"] == 0).sum())
    mal_before = int((original_df["label"] == 1).sum())
    print(f"    Safe      : {safe_before}")
    print(f"    Malicious : {mal_before}")
    ratio_before = safe_before / mal_before if mal_before else float("inf")
    print(f"    Ratio     : {ratio_before:.2f} safe per malicious")

    # Align columns -- hard_neg_df has a 'source' column that we drop for merge
    # We need to match the columns of the original CSV.
    orig_cols = list(original_df.columns)
    merge_df = hard_neg_df[["text", "label", "technique_id", "category"]].copy()

    # Add missing columns from the original CSV with sensible defaults
    for col in orig_cols:
        if col not in merge_df.columns:
            merge_df[col] = np.nan

    # Reorder to match original
    merge_df = merge_df[orig_cols]

    combined = pd.concat([original_df, merge_df], ignore_index=True)

    def _canon_text(text):
        text = str(text)
        text = unicodedata.normalize("NFKC", text)
        return re.sub(r"\s+", " ", text).strip()

    def _text_hash(text):
        canon = _canon_text(text)
        return hashlib.sha256(canon.encode("utf-8", errors="replace")).hexdigest()

    # Deduplicate by normalized text hash (Unicode-aware)
    before_dedup = len(combined)
    combined["_hash"] = combined["text"].apply(_text_hash)
    combined = combined.drop_duplicates(subset=["_hash"], keep="first")
    # Stable idempotent ordering across repeated runs
    combined = combined.sort_values("_hash", kind="mergesort").reset_index(drop=True)
    combined = combined.drop(columns=["_hash"])
    after_dedup = len(combined)
    dropped = before_dedup - after_dedup

    safe_after = int((combined["label"] == 0).sum())
    mal_after = int((combined["label"] == 1).sum())
    ratio_after = safe_after / mal_after if mal_after else float("inf")

    print(f"\n  After merge + dedup: {after_dedup} samples (dropped {dropped} duplicates)")
    print(f"    Safe      : {safe_after}  (+{safe_after - safe_before})")
    print(f"    Malicious : {mal_after}")
    print(f"    Ratio     : {ratio_after:.2f} safe per malicious")

    os.makedirs(os.path.dirname(MERGED_CSV), exist_ok=True)
    combined.to_csv(MERGED_CSV, index=False, quoting=csv.QUOTE_ALL)
    print(f"\n  Saved merged dataset -> {MERGED_CSV}")

    # Write to canonical training path so downstream feature/model scripts
    # automatically consume hard negatives without extra manual merge steps.
    combined.to_csv(COMBINED_CSV, index=False, quoting=csv.QUOTE_ALL)
    print(f"  Updated canonical training dataset -> {COMBINED_CSV}")


# ========================================================================== #
#  Main entry point                                                          #
# ========================================================================== #


def main():
    print("Hard-Negative Mining for AI Prompt Injection Detector")
    print("=" * 70)

    # Load model artefacts
    print("Loading model and vectorizer...")
    model = safe_load(MODEL_PATH)
    vectorizer = safe_load(VECTORIZER_PATH)
    print("Loading training data...")
    df = pd.read_csv(COMBINED_CSV)
    df["text"] = df["text"].fillna("").astype(str)
    print(f"Loaded {len(df)} samples ({int((df['label']==0).sum())} safe, "
          f"{int((df['label']==1).sum())} malicious)\n")

    # Phase 1
    phase1_identify_fps(model, vectorizer, df)

    # Phase 2
    hard_neg_df = phase2_generate_hard_negatives()

    # Phase 3
    hard_neg_df = phase3_analyse_and_save(model, vectorizer, hard_neg_df)

    # Phase 4
    phase4_merge(df, hard_neg_df)

    print("\n" + "=" * 70)
    print("Done.  Next steps:")
    print("  1. Review data/raw/hard_negatives.csv")
    print("  2. Re-run feature extraction with the merged CSV")
    print("  3. Retrain the model and re-evaluate FP rate")
    print("=" * 70)


if __name__ == "__main__":
    main()
