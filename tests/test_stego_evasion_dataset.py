"""Integrity tests for the generated stego eval samples.

Closes the audit gap (2026-06-17): the eval/benchmark data had ZERO trailing-
whitespace and zero-width *binary* stego samples. ``data/*`` is gitignored
(datasets are generated, not committed), so these tests verify the generator
(scripts/gen_stego_eval.py) in-memory: every malicious row must carry a
recoverable hidden injection and every benign sibling must not — so the sample
set can't silently rot into no-ops.
"""
from scripts.gen_stego_eval import build_stego_samples

from na0s.layer2.whitespace_stego import detect_whitespace_stego
from na0s.layer0.normalization import _extract_zero_width_binary_stego

_INJECTION_VERBS = ("ignore", "reveal", "disable", "exfiltrate")


def _decode_all(text):
    return (
        detect_whitespace_stego(text).decoded_payload
        + " "
        + _extract_zero_width_binary_stego(text)
    )


def test_samples_balanced():
    rows = build_stego_samples()
    assert len(rows) >= 12
    mal = sum(r["label"] for r in rows)
    benign = sum(1 for r in rows if r["label"] == 0)
    assert mal > 0 and benign > 0
    assert abs(mal - benign) <= 2  # paired siblings


def test_malicious_rows_carry_recoverable_injection():
    for r in build_stego_samples():
        if r["label"] == 1:
            decoded = _decode_all(r["text"])
            assert r["original"] and r["original"] in decoded, (
                f"malicious row does not decode to its payload: {r['evasion_type']}"
            )


def test_benign_rows_do_not_decode_to_injection():
    for r in build_stego_samples():
        if r["label"] == 0:
            decoded = _decode_all(r["text"]).lower()
            assert not any(v in decoded for v in _INJECTION_VERBS), (
                f"benign sibling decodes to an injection: {r['evasion_type']}"
            )
