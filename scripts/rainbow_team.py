"""Na0S Rainbow Teaming — automated adversarial generation.

Finds inputs that evade Na0S detection using iterative buff mutation.
Uses only existing Na0S internals — no external dependencies.

Usage:
    python scripts/rainbow_team.py --generations 5 --population 50
    python scripts/rainbow_team.py --generations 10 --seed-category D1
"""

import argparse
import datetime
import json
import os
import random
import sys
import time

# Add project root to path for na0s imports, scripts/ for taxonomy imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))

from taxonomy import ALL_PROBES
from taxonomy._buffs import ALL_BUFFS


def _get_scan():
    """Lazy-load the scan function."""
    from na0s.predict import scan
    return scan


def seed_samples(category=None, max_samples=None):
    """Load seed samples from probes.

    Parameters
    ----------
    category : str or None
        Probe category to seed from (e.g., "D1"). None or "all" uses all probes.
    max_samples : int or None
        Max samples to load. None loads all.

    Returns
    -------
    list[tuple[str, str]]
        List of (text, technique_id) tuples.
    """
    samples = []
    for ProbeClass in ALL_PROBES:
        probe = ProbeClass()
        if category and category != "all" and probe.category_id != category:
            continue
        try:
            generated = probe.generate()
            for item in generated:
                if len(item) >= 2:
                    text, tech_id = item[0], item[1]
                    # Only keep malicious samples (not benign)
                    if not tech_id.endswith("_benign"):
                        samples.append((text, tech_id))
        except Exception:
            continue

    if max_samples and len(samples) > max_samples:
        random.shuffle(samples)
        samples = samples[:max_samples]

    return samples


def evaluate_samples(samples, predict_fn):
    """Run samples through detector, return those that evade.

    Returns list of dicts with text, technique_id, confidence, evading status.
    """
    results = []
    for text, tech_id in samples:
        try:
            result = predict_fn(text)
            # Support both ScanResult objects and simple label strings
            label = getattr(result, "label", str(result))
            is_malicious = getattr(result, "is_malicious", False)
            confidence = getattr(result, "risk_score",
                                 getattr(result, "confidence", 0.0))
            # Evading = should be malicious but detector says safe
            evading = not is_malicious and "malicious" not in label.lower() \
                and "blocked" not in label.lower()
            results.append({
                "text": text,
                "technique_id": tech_id,
                "confidence": confidence,
                "evading": evading,
            })
        except Exception:
            continue
    return results


def mutate_samples(samples, max_buffs=2):
    """Apply random buff combinations to samples.

    Parameters
    ----------
    samples : list[dict]
        Samples with "text" and "technique_id" keys.
    max_buffs : int
        Maximum number of buffs to stack per mutation.

    Returns
    -------
    list[tuple[str, str, list[str]]]
        List of (mutated_text, technique_id, buffs_applied).
    """
    mutated = []
    for sample in samples:
        text = sample["text"]
        tech_id = sample["technique_id"]

        # Choose 1 to max_buffs random buff classes
        n_buffs = random.randint(1, min(max_buffs, len(ALL_BUFFS)))
        chosen_buff_classes = random.sample(ALL_BUFFS, n_buffs)

        current_text = text
        buff_names = []
        for BuffClass in chosen_buff_classes:
            try:
                buff = BuffClass()
                result = buff.apply(current_text)
                if result and isinstance(result, str) and result.strip():
                    current_text = result
                    buff_names.append(buff.name)
            except Exception:
                continue

        if current_text != text:  # Only keep if mutation changed something
            mutated.append((current_text, tech_id, buff_names))

    return mutated


def run_rainbow(generations, population, seed_category, max_buffs,
                output_path, predict_fn=None):
    """Run the rainbow teaming loop.

    Returns the full results dict.
    """
    if predict_fn is None:
        predict_fn = _get_scan()

    print("Rainbow Teaming: {} generations, population {}, seed: {}".format(
        generations, population, seed_category or "all"))
    print("Loading seed samples...")

    seeds = seed_samples(category=seed_category, max_samples=population)
    print("Loaded {} seed samples".format(len(seeds)))

    if not seeds:
        print("No seed samples found. Check --seed-category value.")
        sys.exit(1)

    generation_results = []
    all_adversarials = []
    buff_combo_counts = {}
    eval_results = []

    for gen in range(generations):
        t0 = time.time()

        if gen == 0:
            # First generation: evaluate seeds directly
            eval_results = evaluate_samples(seeds, predict_fn)
        else:
            # Later generations: mutate previous evaders + random seeds
            evaders = [r for r in eval_results if r["evading"]]
            if not evaders:
                # No evaders — mutate random seeds instead
                evaders = [{"text": t, "technique_id": tid}
                           for t, tid in random.sample(
                               seeds, min(len(seeds), population))]

            mutated = mutate_samples(evaders, max_buffs=max_buffs)
            eval_input = [(text, tech_id) for text, tech_id, _ in mutated]
            eval_results = evaluate_samples(eval_input, predict_fn)

            # Track buff combos for successful evasions
            for i, r in enumerate(eval_results):
                if r["evading"] and i < len(mutated):
                    combo = tuple(sorted(mutated[i][2]))
                    buff_combo_counts[combo] = \
                        buff_combo_counts.get(combo, 0) + 1

        evading = [r for r in eval_results if r["evading"]]
        evasion_rate = len(evading) / max(len(eval_results), 1)
        elapsed = time.time() - t0

        gen_data = {
            "generation": gen,
            "evaluated": len(eval_results),
            "evading": len(evading),
            "evasion_rate": round(evasion_rate, 4),
            "elapsed_s": round(elapsed, 2),
            "adversarials": [
                {
                    "text": r["text"][:500],
                    "technique_id": r["technique_id"],
                    "confidence": r["confidence"],
                }
                for r in evading[:20]  # Cap examples per generation
            ],
        }
        generation_results.append(gen_data)
        all_adversarials.extend(evading)

        print("  Gen {}: {} evaluated, {} evading ({:.1%}), {:.1f}s".format(
            gen, len(eval_results), len(evading), evasion_rate, elapsed))

    # Find most effective buff combo
    best_combo = []
    if buff_combo_counts:
        best_combo = list(max(buff_combo_counts, key=buff_combo_counts.get))

    report = {
        "generated_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "config": {
            "generations": generations,
            "population": population,
            "seed_category": seed_category or "all",
            "max_buffs": max_buffs,
        },
        "generations": generation_results,
        "summary": {
            "total_adversarials_found": len(all_adversarials),
            "best_evasion_rate": max(
                (g["evasion_rate"] for g in generation_results), default=0.0),
            "most_effective_buff_combo": best_combo,
        },
    }

    # Write output
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)),
                     exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print("\nResults written to {}".format(output_path))

    print("\n=== Summary ===")
    print("Total adversarials found: {}".format(len(all_adversarials)))
    print("Best evasion rate: {:.1%}".format(
        report["summary"]["best_evasion_rate"]))
    if best_combo:
        print("Most effective buff combo: {}".format(best_combo))

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Na0S Rainbow Teaming — find evasive adversarial inputs"
    )
    parser.add_argument("--generations", type=int, default=5,
                        help="Number of mutation rounds (default: 5)")
    parser.add_argument("--population", type=int, default=50,
                        help="Samples per round (default: 50)")
    parser.add_argument("--seed-category", default=None,
                        help="Probe category to seed from (default: all)")
    parser.add_argument("--buffs", type=int, default=2,
                        help="Max buffs to combine per mutation (default: 2)")
    parser.add_argument("--output",
                        default="data/evaluation/rainbow_adversarials.json",
                        help="Output JSON path")
    args = parser.parse_args()

    run_rainbow(
        generations=args.generations,
        population=args.population,
        seed_category=args.seed_category,
        max_buffs=args.buffs,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
