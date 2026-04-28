"""Tests for profanity_filter module."""

import unittest
from whisper_live.profanity_filter import (
    filter_profanity,
    get_default_profanity_words,
    _build_pattern,
    _mask_word,
)


class TestMaskWord(unittest.TestCase):
    def test_partial_masking(self):
        assert _mask_word("fuck", "*", "partial") == "f**k"

    def test_full_masking(self):
        assert _mask_word("fuck", "*", "full") == "****"

    def test_short_word_always_full(self):
        assert _mask_word("as", "*", "partial") == "**"

    def test_custom_mask_char(self):
        assert _mask_word("damn", "#", "partial") == "d##n"


class TestFilterProfanity(unittest.TestCase):
    def test_empty_string(self):
        assert filter_profanity("") == ""

    def test_none_returns_none(self):
        assert filter_profanity(None) is None

    def test_no_profanity(self):
        text = "Hello world, this is a clean sentence."
        assert filter_profanity(text) == text

    def test_partial_mode_default(self):
        result = filter_profanity("What the fuck is this")
        assert "f**k" in result
        assert "fuck" not in result

    def test_full_mode(self):
        result = filter_profanity("What the fuck is this", mode="full")
        assert "****" in result
        assert "fuck" not in result

    def test_remove_mode(self):
        result = filter_profanity("What the fuck is this", mode="remove")
        assert "fuck" not in result
        assert "What the is this" == result

    def test_case_insensitive(self):
        result = filter_profanity("FUCK this SHIT")
        assert "FUCK" not in result
        assert "SHIT" not in result

    def test_multiple_profanities(self):
        result = filter_profanity("damn this shit is fucked")
        assert "damn" not in result
        assert "shit" not in result

    def test_custom_mask_char(self):
        result = filter_profanity("damn it", mask_char="#")
        assert "d##n" in result

    def test_custom_words_replaces_default(self):
        result = filter_profanity(
            "That is awesome and terrible",
            custom_words={"awesome", "terrible"},
        )
        assert "awesome" not in result
        assert "terrible" not in result

    def test_custom_words_ignores_default(self):
        result = filter_profanity(
            "What the fuck is awesome",
            custom_words={"awesome"},
        )
        # "fuck" should NOT be filtered because custom_words replaces defaults
        assert "fuck" in result
        assert "awesome" not in result

    def test_extra_words_extends_default(self):
        result = filter_profanity(
            "What the fuck is dang",
            extra_words={"dang"},
        )
        assert "fuck" not in result
        assert "dang" not in result

    def test_word_boundary_respected(self):
        result = filter_profanity("She was classy and classic")
        # "ass" should not match inside "classy" or "classic"
        assert result == "She was classy and classic"

    def test_punctuation_adjacent(self):
        result = filter_profanity("What the fuck!")
        assert "fuck" not in result
        assert result.endswith("!")

    def test_profanity_at_start(self):
        result = filter_profanity("Shit happens")
        assert "Shit" not in result


class TestGetDefaultProfanityWords(unittest.TestCase):
    def test_returns_set(self):
        words = get_default_profanity_words()
        assert isinstance(words, set)
        assert len(words) > 0

    def test_returns_copy(self):
        words = get_default_profanity_words()
        words.add("newword")
        # Original should not be modified
        assert "newword" not in get_default_profanity_words()


class TestBuildPattern(unittest.TestCase):
    def test_matches_words(self):
        pattern = _build_pattern({"hello", "world"})
        assert pattern.search("say hello there")
        assert pattern.search("say world there")
        assert not pattern.search("say foo there")


class TestServerParseProfanityOption(unittest.TestCase):
    """Test the server-side _parse_profanity_option helper."""

    def _make_server(self):
        from whisper_live.server import TranscriptionServer
        return TranscriptionServer()

    def test_none_when_disabled(self):
        server = self._make_server()
        assert server._parse_profanity_option({}) is None
        assert server._parse_profanity_option({"profanity_filter": False}) is None

    def test_true_returns_partial(self):
        server = self._make_server()
        result = server._parse_profanity_option({"profanity_filter": True})
        assert result == {"mode": "partial"}

    def test_string_mode(self):
        server = self._make_server()
        assert server._parse_profanity_option({"profanity_filter": "full"}) == {"mode": "full"}
        assert server._parse_profanity_option({"profanity_filter": "remove"}) == {"mode": "remove"}

    def test_dict_with_extra_words(self):
        server = self._make_server()
        result = server._parse_profanity_option({
            "profanity_filter": {"mode": "full", "extra_words": ["dang", "heck"]}
        })
        assert result["mode"] == "full"
        assert result["extra_words"] == {"dang", "heck"}


if __name__ == "__main__":
    unittest.main()
