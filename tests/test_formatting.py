import unittest

from whisper_live.formatting import (
    format_transcript,
    _words_to_number,
    _replace_spoken_numbers,
    _capitalize_sentences,
    _collapse_whitespace,
    smart_format,
    find_and_replace,
    _format_currency,
    _format_percentages,
    _format_ordinals,
)


class TestWordsToNumber(unittest.TestCase):
    def test_single_digit(self):
        self.assertEqual(_words_to_number(["five"]), 5)

    def test_teens(self):
        self.assertEqual(_words_to_number(["thirteen"]), 13)

    def test_tens(self):
        self.assertEqual(_words_to_number(["forty"]), 40)

    def test_tens_and_ones(self):
        self.assertEqual(_words_to_number(["twenty", "one"]), 21)

    def test_hundred(self):
        self.assertEqual(_words_to_number(["three", "hundred"]), 300)

    def test_hundred_and(self):
        self.assertEqual(_words_to_number(["three", "hundred", "and", "forty", "two"]), 342)

    def test_thousand(self):
        self.assertEqual(_words_to_number(["two", "thousand"]), 2000)

    def test_complex(self):
        self.assertEqual(_words_to_number(["one", "thousand", "two", "hundred", "and", "thirty", "four"]), 1234)

    def test_million(self):
        self.assertEqual(_words_to_number(["five", "million"]), 5_000_000)

    def test_empty(self):
        self.assertIsNone(_words_to_number([]))

    def test_only_and(self):
        self.assertIsNone(_words_to_number(["and"]))

    def test_zero(self):
        self.assertEqual(_words_to_number(["zero"]), 0)


class TestReplaceSpokenNumbers(unittest.TestCase):
    def test_simple_replacement(self):
        self.assertEqual(_replace_spoken_numbers("I have twenty one cats"), "I have 21 cats")

    def test_with_punctuation(self):
        self.assertEqual(_replace_spoken_numbers("I have five."), "I have 5.")

    def test_no_numbers(self):
        self.assertEqual(_replace_spoken_numbers("hello world"), "hello world")

    def test_mixed(self):
        result = _replace_spoken_numbers("There are three hundred and two items")
        self.assertEqual(result, "There are 302 items")

    def test_multiple_number_groups(self):
        result = _replace_spoken_numbers("I have five cats and three dogs")
        self.assertEqual(result, "I have 5 cats and 3 dogs")


class TestCapitalizeSentences(unittest.TestCase):
    def test_capitalizes_start(self):
        self.assertEqual(_capitalize_sentences("hello world"), "Hello world")

    def test_capitalizes_after_period(self):
        self.assertEqual(_capitalize_sentences("hello. world"), "Hello. World")

    def test_capitalizes_after_exclamation(self):
        self.assertEqual(_capitalize_sentences("wow! great"), "Wow! Great")

    def test_capitalizes_after_question(self):
        self.assertEqual(_capitalize_sentences("really? yes"), "Really? Yes")

    def test_already_capitalized(self):
        self.assertEqual(_capitalize_sentences("Hello World"), "Hello World")

    def test_empty_string(self):
        self.assertEqual(_capitalize_sentences(""), "")


class TestCollapseWhitespace(unittest.TestCase):
    def test_collapses_multiple_spaces(self):
        self.assertEqual(_collapse_whitespace("hello   world"), "hello world")

    def test_strips_edges(self):
        self.assertEqual(_collapse_whitespace("  hello  "), "hello")

    def test_single_spaces_unchanged(self):
        self.assertEqual(_collapse_whitespace("hello world"), "hello world")


class TestFormatTranscript(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(format_transcript(""), "")

    def test_none(self):
        self.assertIsNone(format_transcript(None))

    def test_full_pipeline(self):
        result = format_transcript("  i have twenty one  cats.  it is great  ")
        self.assertEqual(result, "I have 21 cats. It is great")

    def test_numbers_disabled(self):
        result = format_transcript("i have twenty one cats", numbers=False)
        self.assertEqual(result, "I have twenty one cats")

    def test_capitalize_disabled(self):
        result = format_transcript("i have five cats. it is great", capitalize=False)
        self.assertEqual(result, "i have 5 cats. it is great")

    def test_both_disabled(self):
        result = format_transcript("  hello   world  ", capitalize=False, numbers=False)
        self.assertEqual(result, "hello world")

    def test_smart_format_flag(self):
        result = format_transcript("it costs 50 dollars", smart=True, capitalize=False, numbers=False)
        self.assertEqual(result, "it costs $50")


class TestSmartFormat(unittest.TestCase):
    def test_currency_dollars(self):
        self.assertEqual(_format_currency("it costs 50 dollars"), "it costs $50")

    def test_currency_euros(self):
        self.assertEqual(_format_currency("I paid 200 euros"), "I paid €200")

    def test_currency_pounds(self):
        self.assertEqual(_format_currency("that is 10 pounds"), "that is £10")

    def test_percentages(self):
        self.assertEqual(_format_percentages("it went up 25 percent"), "it went up 25%")

    def test_ordinals(self):
        result = _format_ordinals("the first and second place")
        self.assertEqual(result, "the 1st and 2nd place")

    def test_smart_format_combined(self):
        result = smart_format("I spent 100 dollars which is 20 percent of my income")
        self.assertIn("$100", result)
        self.assertIn("20%", result)

    def test_empty(self):
        self.assertEqual(smart_format(""), "")
        self.assertIsNone(smart_format(None))


class TestFindAndReplace(unittest.TestCase):
    def test_simple_replace(self):
        result = find_and_replace("hello world", [("world", "earth")])
        self.assertEqual(result, "hello earth")

    def test_case_insensitive(self):
        result = find_and_replace("Hello World", [("hello", "hi")])
        self.assertEqual(result, "hi World")

    def test_case_sensitive(self):
        result = find_and_replace("Hello hello", [("hello", "hi")], case_sensitive=True)
        self.assertEqual(result, "Hello hi")

    def test_regex_mode(self):
        result = find_and_replace(
            "call 555-1234 now",
            [(r"\d{3}-\d{4}", "[PHONE]")],
            use_regex=True,
        )
        self.assertEqual(result, "call [PHONE] now")

    def test_multiple_replacements(self):
        result = find_and_replace(
            "the cat sat on the mat",
            [("cat", "dog"), ("mat", "rug")],
        )
        self.assertEqual(result, "the dog sat on the rug")

    def test_empty_input(self):
        self.assertEqual(find_and_replace("", [("a", "b")]), "")
        self.assertEqual(find_and_replace("hello", []), "hello")

    def test_special_chars_escaped(self):
        result = find_and_replace("price is $10.00", [("$10.00", "ten dollars")])
        self.assertEqual(result, "price is ten dollars")


if __name__ == "__main__":
    unittest.main()
