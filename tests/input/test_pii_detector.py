"""Tests for src/layer0/pii_detector.py — PII/secrets pre-screening.

Run: python3 -m unittest tests/test_pii_detector.py -v
Uses only well-known test numbers — NEVER real PII.
"""

import os
import sys
import unittest


from na0s.layer0.pii_detector import (
    PiiScanResult,
    scan_pii,
    _luhn_check,
    _redact,
)


class TestLuhnCheck(unittest.TestCase):
    def test_valid_visa(self):
        self.assertTrue(_luhn_check("4111111111111111"))

    def test_valid_mastercard(self):
        self.assertTrue(_luhn_check("5500000000000004"))

    def test_valid_amex(self):
        self.assertTrue(_luhn_check("378282246310005"))

    def test_valid_discover(self):
        self.assertTrue(_luhn_check("6011111111111117"))

    def test_invalid_number(self):
        self.assertFalse(_luhn_check("4111111111111112"))


class TestRedact(unittest.TestCase):
    def test_long_value(self):
        self.assertEqual(_redact("4111111111111111"), "4111***")

    def test_short_value(self):
        self.assertEqual(_redact("abc"), "a***")

    def test_five_char_value(self):
        self.assertEqual(_redact("abcde"), "ab***")

    def test_four_char_value(self):
        self.assertEqual(_redact("abcd"), "ab***")

    def test_seven_char_value(self):
        self.assertEqual(_redact("abcdefg"), "ab***")

    def test_eight_char_value(self):
        self.assertEqual(_redact("abcdefgh"), "abcd***")


class TestCreditCardDetection(unittest.TestCase):
    def test_visa_plain(self):
        result = scan_pii("My card is 4111111111111111 please charge it")
        self.assertTrue(result.has_pii)
        self.assertIn("credit_card", result.pii_types_found)
        self.assertIn("pii_credit_card", result.anomaly_flags)

    def test_visa_with_dashes(self):
        result = scan_pii("Card: 4111-1111-1111-1111")
        self.assertTrue(result.has_pii)
        self.assertIn("credit_card", result.pii_types_found)

    def test_visa_with_spaces(self):
        result = scan_pii("Card: 4111 1111 1111 1111")
        self.assertTrue(result.has_pii)

    def test_mastercard(self):
        result = scan_pii("MC: 5500000000000004")
        self.assertTrue(result.has_pii)
        self.assertIn("credit_card", result.pii_types_found)

    def test_amex(self):
        result = scan_pii("Amex: 378282246310005")
        self.assertTrue(result.has_pii)

    def test_discover(self):
        result = scan_pii("Discover: 6011111111111117")
        self.assertTrue(result.has_pii)

    def test_invalid_luhn_rejected(self):
        result = scan_pii("Card: 4111111111111112")
        self.assertNotIn("credit_card", result.pii_types_found)

    def test_short_number_no_match(self):
        result = scan_pii("Order 12345 is ready")
        self.assertNotIn("credit_card", result.pii_types_found)


class TestSSNDetection(unittest.TestCase):
    def test_valid_ssn(self):
        result = scan_pii("SSN: 123-45-6789")
        self.assertTrue(result.has_pii)
        self.assertIn("ssn", result.pii_types_found)
        self.assertIn("pii_ssn", result.anomaly_flags)

    def test_area_000_rejected(self):
        result = scan_pii("SSN: 000-45-6789")
        self.assertNotIn("ssn", result.pii_types_found)

    def test_area_666_rejected(self):
        result = scan_pii("SSN: 666-45-6789")
        self.assertNotIn("ssn", result.pii_types_found)

    def test_area_900_rejected(self):
        result = scan_pii("SSN: 900-45-6789")
        self.assertNotIn("ssn", result.pii_types_found)

    def test_group_00_rejected(self):
        result = scan_pii("SSN: 123-00-6789")
        self.assertNotIn("ssn", result.pii_types_found)

    def test_serial_0000_rejected(self):
        result = scan_pii("SSN: 123-45-0000")
        self.assertNotIn("ssn", result.pii_types_found)

    def test_no_dashes_no_match(self):
        result = scan_pii("Number 123456789 is just digits")
        self.assertNotIn("ssn", result.pii_types_found)


class TestEmailDetection(unittest.TestCase):
    def test_simple_email(self):
        result = scan_pii("Contact: user@example.com")
        self.assertTrue(result.has_pii)
        self.assertIn("email", result.pii_types_found)

    def test_email_with_plus(self):
        result = scan_pii("user+tag@gmail.com")
        self.assertTrue(result.has_pii)

    def test_no_at_sign(self):
        result = scan_pii("This is not an email address")
        self.assertNotIn("email", result.pii_types_found)


class TestPhoneDetection(unittest.TestCase):
    def test_dashed_phone(self):
        result = scan_pii("Call me at 555-123-4567")
        self.assertTrue(result.has_pii)
        self.assertIn("phone", result.pii_types_found)

    def test_dotted_phone(self):
        result = scan_pii("Phone: 555.123.4567")
        self.assertTrue(result.has_pii)

    def test_parenthesized_phone(self):
        result = scan_pii("Phone: (555) 123-4567")
        self.assertTrue(result.has_pii)

    def test_short_number_no_match(self):
        result = scan_pii("Call 911 now")
        self.assertNotIn("phone", result.pii_types_found)


class TestAPIKeyDetection(unittest.TestCase):
    def test_aws_access_key(self):
        result = scan_pii("Key: AKIAIOSFODNN7EXAMPLE")
        self.assertTrue(result.has_pii)
        self.assertIn("api_key", result.pii_types_found)
        self.assertIn("pii_api_key", result.anomaly_flags)

    def test_github_personal_token(self):
        token = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"
        result = scan_pii("Token: {}".format(token))
        self.assertTrue(result.has_pii)

    def test_generic_hex_40_chars(self):
        hex_str = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"
        result = scan_pii("Hash: {}".format(hex_str))
        self.assertTrue(result.has_pii)

    def test_generic_hex_low_diversity_rejected(self):
        hex_str = "a" * 40
        result = scan_pii("Value: {}".format(hex_str))
        found = [d for d in result.details if d["subtype"] == "generic_hex"]
        self.assertEqual(len(found), 0)


class TestIPv4Detection(unittest.TestCase):
    def test_public_ipv4(self):
        result = scan_pii("Connect to 8.8.8.8")
        self.assertTrue(result.has_pii)
        self.assertIn("ipv4", result.pii_types_found)

    def test_public_ipv4_routable(self):
        result = scan_pii("Server at 203.0.113.5")
        self.assertIn("ipv4", result.pii_types_found)

    def test_invalid_octet(self):
        result = scan_pii("Not IP: 999.999.999.999")
        self.assertNotIn("ipv4", result.pii_types_found)

    def test_private_192_168_filtered(self):
        result = scan_pii("Connect to 192.168.1.100")
        self.assertNotIn("ipv4", result.pii_types_found)

    def test_private_10_x_filtered(self):
        result = scan_pii("Host: 10.0.0.1")
        self.assertNotIn("ipv4", result.pii_types_found)

    def test_private_172_16_filtered(self):
        result = scan_pii("Host: 172.16.0.1")
        self.assertNotIn("ipv4", result.pii_types_found)

    def test_private_172_31_filtered(self):
        result = scan_pii("Host: 172.31.255.255")
        self.assertNotIn("ipv4", result.pii_types_found)

    def test_non_private_172_15(self):
        """172.15.x.x is NOT private (only 172.16-31 is)."""
        result = scan_pii("Host: 172.15.0.1")
        self.assertIn("ipv4", result.pii_types_found)

    def test_non_private_172_32(self):
        """172.32.x.x is NOT private (only 172.16-31 is)."""
        result = scan_pii("Host: 172.32.0.1")
        self.assertIn("ipv4", result.pii_types_found)

    def test_loopback_filtered(self):
        result = scan_pii("Host: 127.0.0.1")
        self.assertNotIn("ipv4", result.pii_types_found)

    def test_loopback_127_255_filtered(self):
        result = scan_pii("Host: 127.255.255.255")
        self.assertNotIn("ipv4", result.pii_types_found)

    def test_unspecified_filtered(self):
        result = scan_pii("Host: 0.0.0.0")
        self.assertNotIn("ipv4", result.pii_types_found)

    def test_link_local_filtered(self):
        result = scan_pii("Host: 169.254.1.1")
        self.assertNotIn("ipv4", result.pii_types_found)


class TestNoPII(unittest.TestCase):
    def test_empty_string(self):
        result = scan_pii("")
        self.assertFalse(result.has_pii)
        self.assertEqual(result.pii_count, 0)

    def test_normal_text(self):
        result = scan_pii("The quick brown fox jumps over the lazy dog.")
        self.assertFalse(result.has_pii)

    def test_none_input(self):
        result = scan_pii(None)
        self.assertFalse(result.has_pii)


class TestMixedPII(unittest.TestCase):
    def test_multiple_types(self):
        text = (
            "SSN: 123-45-6789, Card: 4111111111111111, "
            "Email: john@example.com, Phone: 555-123-4567"
        )
        result = scan_pii(text)
        self.assertTrue(result.has_pii)
        self.assertIn("ssn", result.pii_types_found)
        self.assertIn("credit_card", result.pii_types_found)
        self.assertIn("email", result.pii_types_found)
        self.assertIn("phone", result.pii_types_found)
        self.assertGreaterEqual(result.pii_count, 4)


class TestRedactionSafety(unittest.TestCase):
    def test_credit_card_redacted(self):
        result = scan_pii("Card: 4111111111111111")
        for d in result.details:
            if d["type"] == "credit_card":
                self.assertNotIn("4111111111111111", d["redacted_value"])
                self.assertTrue(d["redacted_value"].endswith("***"))

    def test_ssn_redacted(self):
        result = scan_pii("SSN: 123-45-6789")
        for d in result.details:
            if d["type"] == "ssn":
                self.assertNotIn("123-45-6789", d["redacted_value"])

    def test_no_full_pii_in_details(self):
        text = "SSN: 123-45-6789, Card: 4111111111111111"
        result = scan_pii(text)
        pii_values = ["123-45-6789", "4111111111111111"]
        for d in result.details:
            for val in d.values():
                for pii in pii_values:
                    self.assertNotIn(pii, str(val))


class TestInputLengthLimit(unittest.TestCase):
    """Input truncation at _MAX_SCAN_LENGTH to prevent resource exhaustion."""

    def test_pii_before_limit_detected(self):
        """PII placed at the start of a massive string must still be found."""
        text = "SSN: 123-45-6789 " + "x" * 200_000
        result = scan_pii(text)
        self.assertIn("ssn", result.pii_types_found)

    def test_pii_beyond_limit_not_detected(self):
        """PII placed after 100K boundary must NOT be found (truncated)."""
        text = "x" * 110_000 + " SSN: 123-45-6789"
        result = scan_pii(text)
        self.assertNotIn("ssn", result.pii_types_found)

    def test_normal_length_unaffected(self):
        """Short text must work exactly as before."""
        result = scan_pii("My SSN is 123-45-6789")
        self.assertIn("ssn", result.pii_types_found)


class TestBug1VisaNoFalse13Digit(unittest.TestCase):
    """BUG-1 regression: Visa-13 regex removed (was duplicate of Visa-16)."""

    def test_visa_16_still_detected(self):
        result = scan_pii("Card: 4111111111111111")
        self.assertTrue(result.has_pii)
        self.assertIn("credit_card", result.pii_types_found)

    def test_visa_16_with_dashes_still_detected(self):
        result = scan_pii("Card: 4111-1111-1111-1111")
        self.assertTrue(result.has_pii)
        self.assertIn("credit_card", result.pii_types_found)

    def test_no_double_counting_visa(self):
        """Visa should be counted once, not twice (old Visa-13 overlapped)."""
        result = scan_pii("Card: 4111111111111111")
        cc_details = [d for d in result.details if d["subtype"] == "credit_card"]
        self.assertEqual(len(cc_details), 1)


class TestBug2MastercardTwoSeries(unittest.TestCase):
    """BUG-2 regression: MC 2-series range narrowed to 2221-2720."""

    def test_mc_2221_lower_bound(self):
        result = scan_pii("MC: 2221000000000009")
        self.assertTrue(result.has_pii)
        self.assertIn("credit_card", result.pii_types_found)

    def test_mc_2300_mid_range(self):
        result = scan_pii("MC: 2300000000000003")
        self.assertTrue(result.has_pii)
        self.assertIn("credit_card", result.pii_types_found)

    def test_mc_2500_mid_range(self):
        result = scan_pii("MC: 2500000000000001")
        self.assertTrue(result.has_pii)
        self.assertIn("credit_card", result.pii_types_found)

    def test_mc_2700_upper_range(self):
        result = scan_pii("MC: 2700000000000009")
        self.assertTrue(result.has_pii)
        self.assertIn("credit_card", result.pii_types_found)

    def test_mc_2720_upper_bound(self):
        result = scan_pii("MC: 2720000000000005")
        self.assertTrue(result.has_pii)
        self.assertIn("credit_card", result.pii_types_found)

    def test_mc_2200_below_range_rejected(self):
        """2200 is below 2221 -- should NOT match as credit card."""
        result = scan_pii("MC: 2200000000000004")
        cc = [d for d in result.details if d["subtype"] == "credit_card"]
        self.assertEqual(len(cc), 0)

    def test_mc_2799_above_range_rejected(self):
        """2799 is above 2720 -- should NOT match as credit card."""
        result = scan_pii("MC: 2799000000000001")
        cc = [d for d in result.details if d["subtype"] == "credit_card"]
        self.assertEqual(len(cc), 0)


class TestBug3PhoneNoSeparators(unittest.TestCase):
    """BUG-3 regression: Phone regex now matches 10-digit numbers without separators."""

    def test_10_digit_no_separator(self):
        result = scan_pii("Call 5551234567 please")
        self.assertTrue(result.has_pii)
        self.assertIn("phone", result.pii_types_found)

    def test_dashed_phone_still_works(self):
        result = scan_pii("Call 555-123-4567 please")
        self.assertTrue(result.has_pii)
        self.assertIn("phone", result.pii_types_found)

    def test_dotted_phone_still_works(self):
        result = scan_pii("Call 555.123.4567 please")
        self.assertTrue(result.has_pii)
        self.assertIn("phone", result.pii_types_found)

    def test_longer_number_no_false_positive(self):
        """A 14-digit number should NOT trigger phone detection."""
        result = scan_pii("Ref 12345678901234 done")
        self.assertNotIn("phone", result.pii_types_found)

    def test_plus1_no_separator(self):
        result = scan_pii("Phone: +15551234567")
        self.assertTrue(result.has_pii)
        self.assertIn("phone", result.pii_types_found)


class TestBug4Base64FalsePositives(unittest.TestCase):
    """BUG-4 regression: Base64 requires unique_chars >= 10 AND digit/+/= present."""

    def test_all_alpha_rejected(self):
        """Pure alphabetic string should NOT be flagged as base64."""
        result = scan_pii("Path: ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnop")
        b64 = [d for d in result.details if d["subtype"] == "generic_base64"]
        self.assertEqual(len(b64), 0)

    def test_low_diversity_rejected(self):
        """String with fewer than 10 unique chars should NOT match."""
        result = scan_pii("Token: aabb11aabb11aabb11aabb11aabb11aabb11aabb11")
        b64 = [d for d in result.details if d["subtype"] == "generic_base64"]
        self.assertEqual(len(b64), 0)

    def test_real_base64_with_digits_detected(self):
        """Base64 with good diversity and digits should be detected."""
        result = scan_pii("Key: dGhpcyBpcyBhIHRlc3QgYmFzZTY0IHN0cmluZzE2")
        b64 = [d for d in result.details if d["subtype"] == "generic_base64"]
        self.assertGreater(len(b64), 0)

    def test_hex_diversity_threshold_unchanged(self):
        """Generic hex still uses original threshold of 6."""
        hex_str = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"
        result = scan_pii("Hash: {}".format(hex_str))
        self.assertTrue(result.has_pii)
        self.assertIn("api_key", result.pii_types_found)


# -----------------------------------------------------------------------
# Comprehensive regression tests for the 4 PII detector bug fixes
# -----------------------------------------------------------------------


class TestBug1Visa13DigitRemoval(unittest.TestCase):
    """Comprehensive regression: Visa-13 pattern removed, Visa-16 intact."""

    def test_visa_16_plain_detected(self):
        """Standard 16-digit Visa (4111111111111111) must still be detected."""
        result = scan_pii("My card is 4111111111111111 thanks")
        self.assertTrue(result.has_pii)
        self.assertIn("credit_card", result.pii_types_found)
        self.assertIn("pii_credit_card", result.anomaly_flags)

    def test_visa_16_with_dashes_detected(self):
        """Visa 16 with dash separators must still be detected."""
        result = scan_pii("Card: 4111-1111-1111-1111")
        self.assertTrue(result.has_pii)
        self.assertIn("credit_card", result.pii_types_found)

    def test_visa_16_with_spaces_detected(self):
        """Visa 16 with space separators must still be detected."""
        result = scan_pii("Card: 4111 1111 1111 1111")
        self.assertTrue(result.has_pii)
        self.assertIn("credit_card", result.pii_types_found)

    def test_visa_13_digit_no_longer_matches(self):
        """A 13-digit number starting with 4 must NOT match as Visa card.

        Previously the regex had a Visa-13 alternative that would match
        4111111111111 (13 digits).  After the fix, only 16-digit Visa
        numbers are accepted.
        """
        result = scan_pii("Old card: 4111111111111 done")
        cc_hits = [d for d in result.details if d["subtype"] == "credit_card"]
        self.assertEqual(len(cc_hits), 0,
                         "13-digit number starting with 4 should not match as credit card")

    def test_visa_16_luhn_valid_passes(self):
        """A Luhn-valid 16-digit Visa (4532015112830366) must be detected."""
        # 4532015112830366 is a well-known Visa test number that passes Luhn
        self.assertTrue(_luhn_check("4532015112830366"))
        result = scan_pii("Pay with 4532015112830366 please")
        self.assertTrue(result.has_pii)
        self.assertIn("credit_card", result.pii_types_found)

    def test_visa_16_luhn_invalid_rejected(self):
        """A 16-digit Visa number that fails Luhn must NOT be detected."""
        # Change last digit to break Luhn
        self.assertFalse(_luhn_check("4532015112830367"))
        result = scan_pii("Pay with 4532015112830367 please")
        cc_hits = [d for d in result.details if d["subtype"] == "credit_card"]
        self.assertEqual(len(cc_hits), 0)

    def test_no_double_counting_visa_16(self):
        """Visa 16 should produce exactly one credit_card detail, not two.

        The old Visa-13 pattern could overlap with Visa-16, causing
        double-counting.  After removal, only one match should occur.
        """
        result = scan_pii("Card: 4111111111111111")
        cc_details = [d for d in result.details if d["subtype"] == "credit_card"]
        self.assertEqual(len(cc_details), 1)

    def test_visa_13_digit_isolated(self):
        """13-digit number with 4-prefix, isolated by spaces, still no match."""
        result = scan_pii(" 4222222222222 ")
        cc_hits = [d for d in result.details if d["subtype"] == "credit_card"]
        self.assertEqual(len(cc_hits), 0)


class TestBug2MastercardPrecision(unittest.TestCase):
    """Comprehensive regression: MC 2-series range precisely 2221-2720."""

    def test_mc_2221_lower_bound_detected(self):
        """2221000000000009 (lower bound, Luhn-valid) must be detected."""
        result = scan_pii("MC: 2221000000000009")
        self.assertTrue(result.has_pii)
        self.assertIn("credit_card", result.pii_types_found)

    def test_mc_2720_upper_bound_detected(self):
        """2720000000000005 (upper bound, Luhn-valid) must be detected."""
        result = scan_pii("MC: 2720000000000005")
        self.assertTrue(result.has_pii)
        self.assertIn("credit_card", result.pii_types_found)

    def test_mc_2220_just_below_range_rejected(self):
        """2220999999999999 is just below 2221 -- must NOT match as MC.

        The regex must reject prefixes below 2221.  Even though this
        number starts with 222, the fourth digit (0) puts it at 2220,
        which is outside the valid MC range.
        """
        result = scan_pii("MC: 2220999999999999")
        cc_hits = [d for d in result.details if d["subtype"] == "credit_card"]
        self.assertEqual(len(cc_hits), 0,
                         "2220 prefix should not match MC 2221-2720 range")

    def test_mc_2721_just_above_range_rejected(self):
        """2721000000000000 is just above 2720 -- must NOT match as MC.

        The regex must reject prefixes above 2720.
        """
        result = scan_pii("MC: 2721000000000000")
        cc_hits = [d for d in result.details if d["subtype"] == "credit_card"]
        self.assertEqual(len(cc_hits), 0,
                         "2721 prefix should not match MC 2221-2720 range")

    def test_mc_2200_old_bug_would_have_matched(self):
        """2200000000000004 -- old regex 2[2-7]xx matched this incorrectly.

        The old pattern 2[2-7]\\d{2} would accept 2200 because 22 falls
        in [2-7] for the second digit.  The fixed regex correctly
        rejects this since 2200 < 2221.
        """
        result = scan_pii("MC: 2200000000000004")
        cc_hits = [d for d in result.details if d["subtype"] == "credit_card"]
        self.assertEqual(len(cc_hits), 0,
                         "2200 was a false positive in the old regex")

    def test_mc_2799_old_bug_would_have_matched(self):
        """2799000000000000 -- old regex 2[2-7]xx matched this incorrectly.

        The old pattern 2[2-7]\\d{2} would accept 2799 because 27 falls
        in range.  The fixed regex correctly rejects 2799 since it
        exceeds 2720.
        """
        result = scan_pii("MC: 2799000000000000")
        cc_hits = [d for d in result.details if d["subtype"] == "credit_card"]
        self.assertEqual(len(cc_hits), 0,
                         "2799 was a false positive in the old regex")

    def test_mc_traditional_5100_detected(self):
        """Traditional MC 51xx range must still work (5100000000000008)."""
        result = scan_pii("MC: 5100000000000008")
        self.assertTrue(result.has_pii)
        self.assertIn("credit_card", result.pii_types_found)

    def test_mc_traditional_5500_detected(self):
        """Traditional MC 55xx range must still work (5500000000000004)."""
        result = scan_pii("MC: 5500000000000004")
        self.assertTrue(result.has_pii)
        self.assertIn("credit_card", result.pii_types_found)

    def test_mc_traditional_5599_detected(self):
        """5599 at upper edge of traditional MC range must still work."""
        # 5599000000000006 passes Luhn
        result = scan_pii("MC: 5599000000000006")
        self.assertTrue(result.has_pii)
        self.assertIn("credit_card", result.pii_types_found)

    def test_mc_2300_mid_range_detected(self):
        """2300 is mid-range in 2221-2720 and must be detected."""
        result = scan_pii("MC: 2300000000000003")
        self.assertTrue(result.has_pii)
        self.assertIn("credit_card", result.pii_types_found)

    def test_mc_2500_mid_range_detected(self):
        """2500 is mid-range in 2221-2720 and must be detected."""
        result = scan_pii("MC: 2500000000000001")
        self.assertTrue(result.has_pii)
        self.assertIn("credit_card", result.pii_types_found)

    def test_mc_2100_below_range_rejected(self):
        """2100 is well below 2221 -- must NOT match."""
        result = scan_pii("MC: 2100000000000005")
        cc_hits = [d for d in result.details if d["subtype"] == "credit_card"]
        self.assertEqual(len(cc_hits), 0)

    def test_mc_2800_above_range_rejected(self):
        """2800 is above 2720 -- must NOT match."""
        result = scan_pii("MC: 2800000000000009")
        cc_hits = [d for d in result.details if d["subtype"] == "credit_card"]
        self.assertEqual(len(cc_hits), 0)


class TestBug3PhoneWithoutSeparators(unittest.TestCase):
    """Comprehensive regression: Phone regex matches 10 digits without separators."""

    def test_10_digit_no_separator_detected(self):
        """5551234567 (10 digits, no separators) must be detected as phone."""
        result = scan_pii("Call 5551234567 now")
        self.assertTrue(result.has_pii)
        self.assertIn("phone", result.pii_types_found)

    def test_parenthesized_with_dash_still_works(self):
        """(555)123-4567 with separators must still be detected."""
        result = scan_pii("Phone: (555)123-4567")
        self.assertTrue(result.has_pii)
        self.assertIn("phone", result.pii_types_found)

    def test_parenthesized_with_space_still_works(self):
        """(555) 123-4567 (paren + space + dash) must still be detected."""
        result = scan_pii("Phone: (555) 123-4567")
        self.assertTrue(result.has_pii)
        self.assertIn("phone", result.pii_types_found)

    def test_dots_as_separators_still_works(self):
        """555.123.4567 with dots must still be detected."""
        result = scan_pii("Phone: 555.123.4567")
        self.assertTrue(result.has_pii)
        self.assertIn("phone", result.pii_types_found)

    def test_14_digit_number_no_phone_match(self):
        """12345678901234 (14 digits) must NOT match as phone.

        The lookbehind (?<!\\d) and lookahead (?!\\d) prevent the regex
        from matching a 10-digit substring inside a longer number.
        """
        result = scan_pii("Reference: 12345678901234 end")
        self.assertNotIn("phone", result.pii_types_found)

    def test_plus1_country_code_no_separator(self):
        """+15551234567 (country code, no separators) must be detected."""
        result = scan_pii("Call +15551234567 please")
        self.assertTrue(result.has_pii)
        self.assertIn("phone", result.pii_types_found)

    def test_plus1_country_code_with_dash(self):
        """+1-555-123-4567 must be detected."""
        result = scan_pii("Call +1-555-123-4567 please")
        self.assertTrue(result.has_pii)
        self.assertIn("phone", result.pii_types_found)

    def test_plus1_country_code_with_spaces(self):
        """+1 555 123 4567 must be detected."""
        result = scan_pii("Call +1 555 123 4567 please")
        self.assertTrue(result.has_pii)
        self.assertIn("phone", result.pii_types_found)

    def test_short_number_no_match(self):
        """5-digit number must NOT trigger phone detection."""
        result = scan_pii("Code: 55512 end")
        self.assertNotIn("phone", result.pii_types_found)

    def test_phone_at_start_of_text(self):
        """Phone number at the very start of text must still be detected."""
        result = scan_pii("5551234567 is the number")
        self.assertTrue(result.has_pii)
        self.assertIn("phone", result.pii_types_found)

    def test_phone_at_end_of_text(self):
        """Phone number at the very end of text must still be detected."""
        result = scan_pii("The number is 5551234567")
        self.assertTrue(result.has_pii)
        self.assertIn("phone", result.pii_types_found)

    def test_embedded_in_longer_digits_no_match(self):
        """Phone-like 10 digits embedded in 16-digit number must not match."""
        result = scan_pii("Card: 1234555123456789")
        # Even though 5551234567 is embedded, lookbehind/ahead should block it
        phone_hits = [d for d in result.details if d["type"] == "phone"]
        self.assertEqual(len(phone_hits), 0,
                         "Should not extract phone from middle of longer digit string")

    # --- NANP area code validation (area codes must start with 2-9) ---

    def test_nanp_area_code_0xx_rejected(self):
        """Area code starting with 0 (e.g. 055) is invalid per NANP."""
        result = scan_pii("Call 0551234567 please")
        self.assertNotIn("phone", result.pii_types_found)

    def test_nanp_area_code_1xx_rejected(self):
        """Area code starting with 1 (e.g. 155) is invalid per NANP."""
        result = scan_pii("Call 1551234567 please")
        self.assertNotIn("phone", result.pii_types_found)

    def test_nanp_area_code_0xx_with_parens_rejected(self):
        """(055) 123-4567 — invalid NANP area code in parenthesized form."""
        result = scan_pii("Phone: (055) 123-4567")
        self.assertNotIn("phone", result.pii_types_found)

    def test_nanp_area_code_1xx_with_parens_rejected(self):
        """(155) 123-4567 — invalid NANP area code in parenthesized form."""
        result = scan_pii("Phone: (155) 123-4567")
        self.assertNotIn("phone", result.pii_types_found)

    def test_nanp_timestamp_not_phone(self):
        """Unix timestamp 1672531200 must NOT match as phone (starts with 1)."""
        result = scan_pii("Timestamp: 1672531200")
        self.assertNotIn("phone", result.pii_types_found)

    def test_nanp_tracking_number_not_phone(self):
        """Tracking number 0001234567 must NOT match as phone (starts with 0)."""
        result = scan_pii("Tracking: 0001234567")
        self.assertNotIn("phone", result.pii_types_found)

    def test_nanp_valid_area_code_2xx(self):
        """Area code 212 (NYC) — valid NANP, must be detected."""
        result = scan_pii("Call 2125551234 please")
        self.assertIn("phone", result.pii_types_found)

    def test_nanp_valid_area_code_9xx(self):
        """Area code 917 (NYC mobile) — valid NANP, must be detected."""
        result = scan_pii("Call 9175551234 please")
        self.assertIn("phone", result.pii_types_found)

    def test_nanp_valid_area_code_with_country_code(self):
        """+1 with valid NANP area code must still be detected."""
        result = scan_pii("Call +12125551234 please")
        self.assertIn("phone", result.pii_types_found)

    def test_nanp_invalid_area_code_with_country_code_rejected(self):
        """+1 with invalid area code (0xx) must NOT match."""
        result = scan_pii("Call +10551234567 please")
        self.assertNotIn("phone", result.pii_types_found)


class TestBug4Base64FalsePositiveReduction(unittest.TestCase):
    """Comprehensive regression: Base64 requires unique_chars >= 10 AND digit/+/= present."""

    def test_real_base64_api_key_detected(self):
        """A realistic base64 API key with mixed chars and digits must match.

        Example: ABCdef123GHIjkl456MNOpqr789STUvwx012YZab345
        (43 chars, high diversity, contains digits)
        """
        key = "ABCdef123GHIjkl456MNOpqr789STUvwx012YZab345"
        result = scan_pii("Key: {}".format(key))
        b64_hits = [d for d in result.details if d["subtype"] == "generic_base64"]
        self.assertGreater(len(b64_hits), 0,
                           "Real API key with digits should be detected as base64")

    def test_all_lowercase_english_looking_rejected(self):
        """A 40-char all-lowercase English-looking string must NOT match.

        'abcdefghijklmnopqrstuvwxyzabcdefghijklmnop' has 26 unique chars
        (above the 10 threshold) but contains NO digits, +, or =.
        The second filter should reject it.
        """
        text = "abcdefghijklmnopqrstuvwxyzabcdefghijklmnop"
        result = scan_pii("Word: {}".format(text))
        b64_hits = [d for d in result.details if d["subtype"] == "generic_base64"]
        self.assertEqual(len(b64_hits), 0,
                         "All-lowercase string without digits/+/= should not match")

    def test_real_base64_encoded_string_with_padding_detected(self):
        """Real base64 encoded data with digits and = padding must match.

        'SGVsbG8gV29ybGQhIFRoaXMgaXMgYSB0ZXN0IHN0cmluZw==' is the
        base64 encoding of 'Hello World! This is a test string'.
        """
        b64 = "SGVsbG8gV29ybGQhIFRoaXMgaXMgYSB0ZXN0IHN0cmluZw=="
        result = scan_pii("Data: {}".format(b64))
        b64_hits = [d for d in result.details if d["subtype"] == "generic_base64"]
        self.assertGreater(len(b64_hits), 0,
                           "Real base64 with = padding should be detected")

    def test_long_file_path_no_false_positive(self):
        """A long file path segment must NOT be flagged as base64.

        File paths typically contain '/' which is in the base64 charset,
        but they lack digits and have low character diversity.
        """
        path = "/usr/local/share/applications/something/verylongdirectoryname/resources"
        result = scan_pii("Path: {}".format(path))
        b64_hits = [d for d in result.details if d["subtype"] == "generic_base64"]
        self.assertEqual(len(b64_hits), 0,
                         "File path should not trigger base64 detection")

    def test_all_uppercase_no_digits_rejected(self):
        """A 40-char all-uppercase string without digits must NOT match."""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOP"
        result = scan_pii("Token: {}".format(text))
        b64_hits = [d for d in result.details if d["subtype"] == "generic_base64"]
        self.assertEqual(len(b64_hits), 0,
                         "All-uppercase string without digits/+/= should not match")

    def test_mixed_case_with_plus_detected(self):
        """Base64 string with + character (high diversity) must be detected."""
        # This string has mixed case, digits, and +
        b64 = "ABCDef+ghIJKL1234mnOPQRst5678uvWXYZ+abcde90"
        result = scan_pii("Token: {}".format(b64))
        b64_hits = [d for d in result.details if d["subtype"] == "generic_base64"]
        self.assertGreater(len(b64_hits), 0,
                           "Base64 with + and digits should be detected")

    def test_hex_still_uses_threshold_of_6(self):
        """Generic hex detection must still use the original threshold of 6.

        The hex threshold was NOT changed by BUG-4.  Only base64 was
        tightened to 10 unique chars.
        """
        hex_str = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"
        result = scan_pii("Hash: {}".format(hex_str))
        hex_hits = [d for d in result.details if d["subtype"] == "generic_hex"]
        self.assertGreater(len(hex_hits), 0,
                           "Hex with 6+ unique chars should still be detected")

    def test_low_diversity_with_digits_still_rejected(self):
        """Even with digits, fewer than 10 unique chars must be rejected."""
        # 'aabb11cc22' repeated = only 8 unique chars (a,b,1,c,2 + lowercase)
        text = "aabb11aabb11aabb11aabb11aabb11aabb11aabb1122"
        unique = len(set(text.lower()))
        # Verify our test string actually has < 10 unique chars
        self.assertLess(unique, 10,
                        "Test setup: string should have < 10 unique chars")
        result = scan_pii("Token: {}".format(text))
        b64_hits = [d for d in result.details if d["subtype"] == "generic_base64"]
        self.assertEqual(len(b64_hits), 0,
                         "Low diversity base64 should be rejected even with digits")

    def test_real_jwt_like_segment_detected(self):
        """A JWT-like segment (high diversity, digits, mixed case) must match."""
        # Simulated JWT payload segment
        jwt_seg = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9abc123"
        result = scan_pii("Auth: {}".format(jwt_seg))
        b64_hits = [d for d in result.details if d["subtype"] == "generic_base64"]
        self.assertGreater(len(b64_hits), 0,
                           "JWT-like base64 segment should be detected")


class TestSanitizerIntegration(unittest.TestCase):
    def test_pii_flags_in_sanitizer(self):
        from na0s.layer0.sanitizer import layer0_sanitize
        result = layer0_sanitize("Please charge 4111111111111111 for the order")
        self.assertFalse(result.rejected)
        self.assertIn("pii_credit_card", result.anomaly_flags)
        self.assertIn("pii_scan", result.source_metadata)

    def test_no_pii_no_flags(self):
        from na0s.layer0.sanitizer import layer0_sanitize
        result = layer0_sanitize("What is the weather today?")
        self.assertFalse(result.rejected)
        self.assertNotIn("pii_credit_card", result.anomaly_flags)

    def test_ssn_in_sanitizer(self):
        from na0s.layer0.sanitizer import layer0_sanitize
        result = layer0_sanitize("My SSN is 123-45-6789")
        self.assertIn("pii_ssn", result.anomaly_flags)


if __name__ == "__main__":
    unittest.main()
