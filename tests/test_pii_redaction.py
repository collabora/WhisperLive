import re
import unittest

from whisper_live.pii_redaction import (
    redact_pii,
    get_supported_pii_types,
    ALL_PII_TYPES,
)


class TestRedactSSN(unittest.TestCase):
    def test_dashed_ssn(self):
        self.assertEqual(redact_pii("My SSN is 123-45-6789"), "My SSN is [SSN_REDACTED]")

    def test_spaced_ssn(self):
        self.assertEqual(redact_pii("SSN 123 45 6789 here"), "SSN [SSN_REDACTED] here")

    def test_no_separator_ssn(self):
        self.assertEqual(redact_pii("SSN 123456789 here"), "SSN [SSN_REDACTED] here")

    def test_no_ssn(self):
        self.assertEqual(redact_pii("No PII here"), "No PII here")


class TestRedactCreditCard(unittest.TestCase):
    def test_dashed_card(self):
        result = redact_pii("Card 4111-1111-1111-1111 on file")
        self.assertIn("[CARD_REDACTED]", result)
        self.assertNotIn("4111", result)

    def test_spaced_card(self):
        result = redact_pii("Card 4111 1111 1111 1111 on file")
        self.assertIn("[CARD_REDACTED]", result)

    def test_no_separator_card(self):
        result = redact_pii("Card 4111111111111111 on file")
        self.assertIn("[CARD_REDACTED]", result)


class TestRedactPhone(unittest.TestCase):
    def test_dashed_phone(self):
        result = redact_pii("Call 555-123-4567")
        self.assertIn("[PHONE_REDACTED]", result)

    def test_dotted_phone(self):
        result = redact_pii("Call 555.123.4567")
        self.assertIn("[PHONE_REDACTED]", result)

    def test_with_area_code_parens(self):
        result = redact_pii("Call (555) 123-4567")
        self.assertIn("[PHONE_REDACTED]", result)

    def test_with_country_code(self):
        result = redact_pii("Call +1-555-123-4567")
        self.assertIn("[PHONE_REDACTED]", result)


class TestRedactEmail(unittest.TestCase):
    def test_simple_email(self):
        result = redact_pii("Email user@example.com please")
        self.assertEqual(result, "Email [EMAIL_REDACTED] please")

    def test_complex_email(self):
        result = redact_pii("Contact: first.last+tag@sub.domain.co.uk")
        self.assertIn("[EMAIL_REDACTED]", result)

    def test_no_email(self):
        self.assertEqual(redact_pii("No emails here"), "No emails here")


class TestRedactIPAddress(unittest.TestCase):
    def test_ipv4(self):
        result = redact_pii("Server at 192.168.1.100")
        self.assertEqual(result, "Server at [IP_REDACTED]")

    def test_localhost(self):
        result = redact_pii("Connect to 127.0.0.1")
        self.assertEqual(result, "Connect to [IP_REDACTED]")

    def test_invalid_ip_not_matched(self):
        # 999.999.999.999 is not a valid IP
        result = redact_pii("Not an IP 999.999.999.999")
        self.assertNotIn("[IP_REDACTED]", result)


class TestSelectivePII(unittest.TestCase):
    def test_only_email(self):
        text = "SSN 123-45-6789 email user@test.com"
        result = redact_pii(text, pii_types={"email"})
        self.assertIn("123-45-6789", result)  # SSN not redacted
        self.assertIn("[EMAIL_REDACTED]", result)

    def test_only_ssn(self):
        text = "SSN 123-45-6789 email user@test.com"
        result = redact_pii(text, pii_types={"ssn"})
        self.assertIn("[SSN_REDACTED]", result)
        self.assertIn("user@test.com", result)  # email not redacted

    def test_multiple_types(self):
        text = "SSN 123-45-6789 email user@test.com"
        result = redact_pii(text, pii_types={"ssn", "email"})
        self.assertIn("[SSN_REDACTED]", result)
        self.assertIn("[EMAIL_REDACTED]", result)


class TestCustomPatterns(unittest.TestCase):
    def test_custom_pattern(self):
        custom = {
            "order_id": (re.compile(r'ORD-\d+'), "[ORDER_REDACTED]"),
        }
        result = redact_pii("Your order ORD-12345 is ready", custom_patterns=custom)
        self.assertEqual(result, "Your order [ORDER_REDACTED] is ready")


class TestEdgeCases(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(redact_pii(""), "")

    def test_none(self):
        self.assertIsNone(redact_pii(None))

    def test_no_pii(self):
        text = "Hello world, no PII here"
        self.assertEqual(redact_pii(text), text)


class TestGetSupportedTypes(unittest.TestCase):
    def test_returns_sorted_list(self):
        types = get_supported_pii_types()
        self.assertIsInstance(types, list)
        self.assertEqual(types, sorted(types))

    def test_contains_expected_types(self):
        types = set(get_supported_pii_types())
        self.assertIn("ssn", types)
        self.assertIn("email", types)
        self.assertIn("phone", types)
        self.assertIn("credit_card", types)
        self.assertIn("ip_address", types)


if __name__ == "__main__":
    unittest.main()
