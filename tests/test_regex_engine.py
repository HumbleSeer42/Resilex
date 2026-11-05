import unittest
import locale
import random
import copy
import pickle
import subprocess
import sys
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from itertools import product
from pathlib import Path

import resilex.regex_engine as regex_api
import resilex
from resilex.regex_engine import (
    DFA,
    Match,
    Regex,
    RegexComplexityError,
    RegexLimits,
    RegexMatchLimitError,
    RegexSyntaxError,
    compile,
)


class RegexEngineTests(unittest.TestCase):
    def test_resilex_public_import(self):
        match = self.require_match(resilex.compile("cat").fullmatch("cat"))
        self.assertEqual(match.group(), "cat")

    def test_explain_reports_match_and_failure_path(self):
        matcher = Regex("ab")
        matched = matcher.explain("zab")
        self.assertIn("Result: matched span (1, 3).", matched)
        self.assertIn("Literal 'ab' was found at position 1.", matched)
        self.assertEqual(matcher.search("zab").span(), (1, 3))

        unmatched = Regex(r"a\d").explain("ax")
        self.assertIn("No consuming path accepts character 'x' at position 1.", unmatched)
        self.assertIn("Result: no match.", unmatched)

        backreference = Regex(r"(a)\1").explain("ab")
        self.assertIn("Backreference group 1 did not match at position 1.", backreference)

    def test_unicode_general_category_escapes(self):
        self.assertIsNotNone(Regex(r"\p{Letter}+").fullmatch("Resiléx"))
        self.assertIsNotNone(Regex(r"\p{gc=Lu}+").fullmatch("ABC"))
        self.assertIsNotNone(Regex(r"[\p{Letter}\p{Number}]+").fullmatch("Ab12"))
        self.assertIsNotNone(Regex(r"\P{Letter}+").fullmatch("123"))
        self.assertIsNotNone(Regex(r"\p{Script=Greek}+").fullmatch("Αβ"))
        self.assertIsNotNone(Regex(r"\p{sc=Grek}+").fullmatch("Αβ"))
        self.assertIsNotNone(Regex(r"\p{scx=Hira}").fullmatch("ー"))
        self.assertIsNotNone(Regex(r"\p{Unknown}").fullmatch("\U0010ffff"))
        with self.assertRaises(RegexSyntaxError):
            Regex(r"\p{gc=Greek}")
        with self.assertRaises(RegexSyntaxError):
            Regex(rb"\p{L}")

    def test_locale_and_debug_flags(self):
        matcher = Regex(rb"\w+", regex_api.LOCALE)
        self.assertIsNotNone(matcher.fullmatch(b"abc"))
        self.assertIsNotNone(matcher.fullmatch(b"_"))
        self.assertIsNone(matcher.fullmatch(b"\xe9"))
        self.assertIsNotNone(Regex(rb"(?L)\w").fullmatch(b"a"))
        case_insensitive_range = Regex(
            rb"[a-z]+", regex_api.LOCALE | regex_api.IGNORECASE
        )
        word_boundary = Regex(rb"\bcat\b", regex_api.LOCALE)
        self.assertIsNotNone(case_insensitive_range.fullmatch(b"ABC"))
        self.assertIsNotNone(case_insensitive_range.fullmatch(b"xcat"))
        self.assertIsNone(case_insensitive_range.fullmatch(b"\xe9"))
        self.assertIsNotNone(word_boundary.fullmatch(b"cat"))
        self.assertIsNone(word_boundary.fullmatch(b"xcat"))
        original_locale = locale.setlocale(locale.LC_CTYPE)
        tested_locales = []
        try:
            for locale_name in ("C", "C.utf8", "en_US.utf8", "en_GB.utf8"):
                try:
                    locale.setlocale(locale.LC_CTYPE, locale_name)
                except locale.Error:
                    continue
                tested_locales.append(locale_name)
                for pattern, expected in (
                    (
                        rb"\w",
                        lambda value: value == 95
                        or 48 <= value <= 57
                        or 65 <= value <= 90
                        or 97 <= value <= 122,
                    ),
                    (
                        rb"\W",
                        lambda value: not (
                            value == 95
                            or 48 <= value <= 57
                            or 65 <= value <= 90
                            or 97 <= value <= 122
                        ),
                    ),
                    (b"a", lambda value: value in (65, 97)),
                    (rb"[a-z]", lambda value: 97 <= value <= 122),
                    (
                        rb"[a-z]",
                        lambda value: 65 <= value <= 90 or 97 <= value <= 122,
                    ),
                    (
                        rb"[A-Z]",
                        lambda value: 65 <= value <= 90 or 97 <= value <= 122,
                    ),
                    (rb"[\x80-\xff]", lambda value: 128 <= value <= 255),
                ):
                    flags = regex_api.LOCALE
                    if pattern in (b"a", rb"[a-z]", rb"[A-Z]") and expected(65):
                        flags |= regex_api.IGNORECASE
                    actual = Regex(pattern, flags)
                    for value in range(256):
                        byte = bytes((value,))
                        with self.subTest(
                            locale=locale_name, pattern=pattern, value=value
                        ):
                            self.assertEqual(
                                actual.fullmatch(byte) is not None, expected(value)
                            )
        finally:
            locale.setlocale(locale.LC_CTYPE, original_locale)
        self.assertTrue(tested_locales)
        with self.assertRaises(ValueError):
            Regex("a", regex_api.LOCALE)

        output = StringIO()
        with redirect_stdout(output):
            compile("a", regex_api.DEBUG)
            compile("a", regex_api.DEBUG)
        self.assertEqual(output.getvalue().count("digraph NFA"), 2)

    def test_anchors_can_be_converted_to_dfa(self):
        for pattern, alphabet, values in (
            ("^a$", "ab", ("", "a", "b", "ba", "ab")),
            (r"\Aa\Z", "ab", ("", "a", "b", "ba", "ab")),
            (r"\bcat\b", "catx", ("", "cat", "xcat", "catx")),
        ):
            matcher = Regex(pattern)
            dfa = matcher.to_dfa(alphabet)
            expected_matches = {
                "^a$": (False, True, False, False, False),
                r"\Aa\Z": (False, True, False, False, False),
                r"\bcat\b": (False, True, False, False),
            }[pattern]
            for value, expected in zip(values, expected_matches):
                with self.subTest(pattern=pattern, value=value):
                    self.assertEqual(dfa.fullmatch(value), expected)

    def test_counted_repeats_above_legacy_ceiling(self):
        repeated = "a" * 5_000
        self.assertIsNotNone(Regex("a{5000}").fullmatch(repeated))

    def require_match(self, result: Match | None) -> Match:
        if result is None:
            self.fail("Expected a match, but the pattern did not match")
        return result

    @staticmethod
    def span_or_none(
        result: Match | None,
    ) -> tuple[int, int] | None:
        return None if result is None else result.span()

    @staticmethod
    def groups_or_none(
        result: Match | None,
    ) -> tuple[str | bytes | None, ...] | None:
        return None if result is None else result.groups()

    def assert_search_consistent_with_fullmatch(
        self, matcher: Regex, text: str, pos: int = 0, endpos: int | None = None
    ) -> None:
        start_bound = min(len(text), max(0, pos))
        end_bound = min(len(text), len(text) if endpos is None else endpos)
        expected_start = None
        if start_bound <= end_bound:
            for candidate_start in range(start_bound, end_bound + 1):
                if any(
                    matcher.fullmatch(text, candidate_start, candidate_end)
                    is not None
                    for candidate_end in range(candidate_start, end_bound + 1)
                ):
                    expected_start = candidate_start
                    break

        actual = matcher.search(text, pos, endpos)
        self.assertEqual(None if actual is None else actual.start(), expected_start)
        if actual is not None:
            self.assertLessEqual(actual.end(), end_bound)
            self.assertIsNotNone(matcher.fullmatch(text, *actual.span()))

    def test_literal_and_dot(self):
        self.assertEqual(self.require_match(Regex("cat").search("a cat!")).group(), "cat")
        self.assertIsNone(Regex("c.t").search("c\nt"))
        self.assertEqual(self.require_match(Regex("c.t").search("cat")).group(), "cat")

    def test_quantifiers(self):
        self.assertEqual(self.require_match(Regex("ab*c").fullmatch("ac")).group(), "ac")
        self.assertEqual(self.require_match(Regex("ab+c").fullmatch("abbbc")).group(), "abbbc")
        self.assertEqual(self.require_match(Regex("colou?r").fullmatch("color")).group(), "color")
        self.assertIsNone(Regex("ab+c").fullmatch("ac"))

    def test_groups_and_alternation(self):
        match = self.require_match(Regex("(cat|dog)s").fullmatch("dogs"))
        self.assertEqual(match.group(), "dogs")
        self.assertEqual(match.group(1), "dog")
        self.assertEqual(match.groups(), ("dog",))
        self.assertEqual(self.require_match(Regex("a(b|c)d").fullmatch("acd")).group(1), "c")
        self.assertIsNone(Regex("(cat|dog)s").fullmatch("bird"))

    def test_unmatched_and_named_groups(self):
        match = self.require_match(Regex("(a)(b)?").fullmatch("a"))
        self.assertIsNone(match.group(2))
        self.assertIsNone(match[2])
        self.assertEqual(match.groups(), ("a", None))
        self.assertEqual(match.group(2, 1), (None, "a"))
        self.assertEqual(match.span(2), (-1, -1))
        self.assertEqual(match.lastindex, 1)

        named = self.require_match(
            Regex(r"(?P<word>\w+)-(?P=word)").fullmatch("item-item")
        )
        self.assertEqual(named.group("word"), "item")
        self.assertEqual(named["word"], "item")
        self.assertEqual(named.groupdict(), {"word": "item"})
        self.assertEqual(named.lastgroup, "word")
        self.assertEqual(named.lastindex, 1)
        self.assertTrue(self.require_match(Regex(r"(?:ab)+").fullmatch("abab")))

    def test_character_classes_and_categories(self):
        self.assertEqual(self.require_match(Regex("[a-z]+").fullmatch("hello")).group(), "hello")
        self.assertIsNone(Regex("[^0-9]+").fullmatch("abc7"))
        self.assertEqual(self.require_match(Regex(r"[\D]+").fullmatch("abc")).group(), "abc")
        self.assertEqual(self.require_match(Regex(r"\d+").search("room 204")).group(), "204")
        self.assertEqual(self.require_match(Regex(r"\w+").search("hello!")).group(), "hello")
        self.assertEqual(self.require_match(Regex(r"\s").search("a b")).group(), " ")

    def test_anchors_and_match_modes(self):
        self.assertIsNone(Regex("^cat").search("a cat"))
        self.assertEqual(self.require_match(Regex("^cat$").fullmatch("cat")).group(), "cat")
        self.assertEqual(self.require_match(Regex("cat").match("cats")).span(), (0, 3))
        self.assertIsNone(Regex("cat").fullmatch("cats"))

    def test_word_and_absolute_anchors(self):
        self.assertIsNone(Regex(r"\bfoo").search("bfoo"))
        self.assertEqual(
            self.require_match(Regex(r"\bfoo").search("!foo")).span(), (1, 4)
        )
        self.assertEqual(
            self.require_match(Regex(r"[\b]").fullmatch("\b")).span(), (0, 1)
        )
        self.assertEqual(
            self.require_match(Regex(r"\Acat\Z").fullmatch("cat")).group(), "cat"
        )
        self.assertIsNone(Regex(r"\Acat\Z").fullmatch("cat\n"))
        self.assertEqual(
            self.require_match(Regex(r"\Bcat").search("scat")).span(), (1, 4)
        )

    def test_noncapturing_lookarounds_and_literal_braces(self):
        self.assertEqual(
            self.require_match(Regex(r"(?:ab)+").fullmatch("abab")).groups(), ()
        )
        self.assertEqual(
            self.require_match(Regex(r"a(?=b)").search("ab")).span(), (0, 1)
        )
        self.assertEqual(
            self.require_match(Regex(r"a(?!c)").search("ab")).span(), (0, 1)
        )
        self.assertEqual(
            self.require_match(Regex(r"(?<=a)b").search("ab")).span(), (1, 2)
        )
        self.assertEqual(
            self.require_match(Regex(r"(?<!a)b").search("cb")).span(), (1, 2)
        )
        self.assertEqual(
            self.require_match(Regex(r"(?<=a{1,3})b").search("aaab")).span(),
            (3, 4),
        )
        variable_capture = self.require_match(
            Regex(r"(?<=(a+))b").search("aaab")
        )
        self.assertEqual(variable_capture.span(1), (0, 3))
        self.assertIsNone(Regex(r"(?<!a+)b").search("aaab"))
        self.assertEqual(self.require_match(Regex("a{,3}").fullmatch("aaa")).group(), "aaa")
        self.assertIsNone(Regex("a{,3}").fullmatch("a{,3}"))
        self.assertEqual(
            self.require_match(Regex("a{1,").fullmatch("a{1,")).group(), "a{1,"
        )
        for pattern in (r"\b+", r"\B*", r"^?", r"$+", r"\A+", r"\Z*"):
            with self.subTest(pattern=pattern), self.assertRaises(RegexSyntaxError):
                Regex(pattern)

    def test_grouped_assertions_and_literal_braces(self):
        cases = (
            (r"(?:\b)+", "word", (0, 0)),
            (r"(?:\B)?", "word", (0, 0)),
            (r"(?:^)*", "word", (0, 0)),
            (r"(?:$)+", "word", (4, 4)),
            (r"(?:\A)?", "word", (0, 0)),
            (r"(?:\Z)*", "word", (0, 0)),
            (r"(?:\b){2}", "word", (0, 0)),
        )
        for pattern, text, expected_span in cases:
            with self.subTest(pattern=pattern):
                actual = self.require_match(Regex(pattern).search(text))
                self.assertEqual(actual.span(), expected_span)
        for pattern in ("^{", "${", r"\b{"):
            with self.subTest(pattern=pattern):
                self.assertEqual(Regex(pattern).pattern, pattern)
        self.assertEqual(
            self.require_match(Regex("^{").fullmatch("{")).span(), (0, 1)
        )
        self.assertEqual(
            self.require_match(Regex("a{2}{").fullmatch("aa{")).span(), (0, 3)
        )

    def test_numeric_escapes_and_comment_groups(self):
        cases = (
            (r"\x41", "A"),
            (r"\u0041", "A"),
            (r"\U00000041", "A"),
            (r"\N{LATIN CAPITAL LETTER A}", "A"),
            (r"\101", "A"),
            (r"[\x41]", "A"),
            ("a(?# ignored )b", "ab"),
        )
        for pattern, text in cases:
            with self.subTest(pattern=pattern):
                actual = Regex(pattern).fullmatch(text)
                self.assertIsNotNone(actual)
        self.assertEqual(
            self.require_match(Regex("a{,}").fullmatch("aaaa")).group(), "aaaa"
        )
        for pattern in ("{1}", r"\x4", r"\N{NOT A REAL CHARACTER}"):
            with self.subTest(pattern=pattern), self.assertRaises(RegexSyntaxError):
                Regex(pattern)
        with self.assertRaises(RegexSyntaxError):
            Regex("(a)" * 10 + r"\41")
        pattern = "(a)" * 12 + r"\119"
        text = "a" * 13 + "9"
        self.assertEqual(
            self.require_match(Regex(pattern).fullmatch(text)).groups(),
            ("a",) * 12,
        )

    def test_conditional_atomic_and_possessive_syntax(self):
        cases = (
            (r"(a)?(?(1)b|c)", "ab", (0, 2), ("a",)),
            (r"(a)?(?(1)b|c)", "c", (0, 1), (None,)),
            (r"(a)?(?(1)b|c)", "a", None, None),
            (r"(a)?(?(1)b|c)", "cb", (0, 1), (None,)),
            (r"(?P<x>a)?(?(x)b|c)", "ab", (0, 2), ("a",)),
            (r"(?P<x>a)?(?(x)b|c)", "c", (0, 1), (None,)),
            (r"(?P<x>a)?(?(x)b|c)", "b", None, None),
            (r"(?>a|ab)c", "ac", (0, 2), ()),
            (r"(?>a|ab)c", "abc", None, None),
            (r"(?>a|ab)c", "aac", (1, 3), ()),
            (r"(?>ab|a)c", "abc", (0, 3), ()),
            (r"(?>ab|a)c", "ac", (0, 2), ()),
            (r"(?>ab|a)c", "abbc", None, None),
            (r"a*+a", "a", None, None),
            (r"a*+a", "aa", None, None),
            (r"a*+a", "aaa", None, None),
            (r"a*+b", "ab", (0, 2), ()),
            (r"a*+b", "aaab", (0, 4), ()),
            (r"a*+b", "aaa", None, None),
            (r"a{1,3}+b", "ab", (0, 2), ()),
            (r"a{1,3}+b", "aab", (0, 3), ()),
            (r"a{1,3}+b", "aaab", (0, 4), ()),
            (r"a{1,3}+b", "aaaab", (1, 5), ()),
        )
        for pattern, text, expected_span, expected_groups in cases:
            with self.subTest(pattern=pattern, text=text):
                actual = Regex(pattern).search(text)
                self.assertEqual(self.span_or_none(actual), expected_span)
                self.assertEqual(
                    None if actual is None else actual.groups(), expected_groups
                )
        with self.assertRaises(ValueError):
            Regex(r"(a)?(?(1)b|c)").to_dfa("abc")

    def test_documented_regex_compatibility_cases(self):
        fullmatch_cases = (
            ("a*+a", "aaa", False),
            ("(?>a+)a", "aaa", False),
            ("x*+x", "xxx", False),
            ("a?+a", "aa", True),
            ("(?>a|ab)c", "abc", False),
        )
        for pattern, text, expected in fullmatch_cases:
            with self.subTest(pattern=pattern, text=text):
                self.assertEqual(Regex(pattern).fullmatch(text) is not None, expected)

        search_cases = (
            (r"(?:(?:.c)??)*+x", "xcbbab", (0, 1)),
            (r"(?:(a)?b)+(?(1)x|y)", "abby", (1, 4)),
            (r"(a(?(1)b|c))", "ac", (0, 2)),
            (r"(?(1)a|b)(a)", "ba", (0, 2)),
            (r"(?P<word>(?(word)a|b))", "b", (0, 1)),
        )
        for pattern, text, expected_span in search_cases:
            with self.subTest(pattern=pattern, text=text):
                actual = self.require_match(Regex(pattern).search(text))
                self.assertEqual(actual.span(), expected_span)

    def test_finditer_findall_sub_and_split(self):
        self.assertEqual(
            [match.span() for match in Regex(".*?").finditer("ab")],
            [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
        )
        self.assertEqual(Regex(r"\d+").findall("a12b3"), ["12", "3"])
        self.assertEqual(Regex(r"(a)?b").findall("b ab"), ["", "a"])
        self.assertEqual(Regex(r"(?P<x>a)").sub(r"<\g<x>>", "aba"), "<a>b<a>")
        self.assertEqual(Regex("x*").sub("-", "ab"), "-a-b-")
        self.assertEqual(Regex(r"(\d+)").sub(r"[\1]", "1 2", count=1), "[1] 2")
        self.assertEqual(Regex(",").split("a,b,c"), ["a", "b", "c"])
        self.assertEqual(Regex(r"(-)").split("a-b-c", maxsplit=1), ["a", "-", "b-c"])
        self.assertEqual(Regex(r"\b").split("words"), ["", "words", ""])

    def test_empty_alternative_repeat_stops_after_empty_iteration(self):
        matcher = Regex(r"(?:a*|b)+")
        self.assertEqual(matcher.search("ab").span(), (0, 1))
        self.assertEqual(
            [match.span() for match in matcher.finditer("ab")],
            [(0, 1), (1, 1), (1, 2), (2, 2)],
        )
        self.assertEqual(matcher.findall("ab"), ["a", "", "b", ""])
        self.assertEqual(matcher.sub("-", "ab"), "----")
        self.assertEqual(matcher.split("ab"), ["", "", "", "", ""])
        self.assertEqual(Regex(r"(?:x|\s*?)++").search("x").span(), (0, 1))

    def test_variable_width_lookbehind_searches_long_prefix_efficiently(self):
        from time import perf_counter

        text = "a" * 800 + "b"
        started = perf_counter()
        match = Regex(r"(?<=a+)b").search(text)
        self.assertLess(perf_counter() - started, 1.0)
        self.assertIsNotNone(match)
        self.assertEqual(match.span(), (800, 801))

    def test_substitution_validates_template_without_matches(self):
        for substitute in (
            lambda: Regex("x").sub(r"\1", "abc"),
            lambda: Regex("x").subn(r"\1", "abc"),
            lambda: regex_api.sub("x", r"\1", "abc"),
            lambda: regex_api.subn("x", r"\1", "abc"),
        ):
            with self.assertRaises(RegexSyntaxError):
                substitute()

    def test_conditional_sees_capture_from_prior_repeat_iteration(self):
        self.assertIsNone(Regex(r"(?:((?(1)b|\w))){2}").search("bc"))

    def test_leftmost_longest_and_empty_match(self):
        self.assertEqual(self.require_match(Regex("a|ab").search("zab")).group(), "a")
        self.assertEqual(self.require_match(Regex("a*").search("baa")).span(), (0, 0))
        self.assertEqual(self.require_match(Regex("a*").match("aaa")).group(), "aaa")
        self.assertIsNone(Regex("a*b").search("aaaaac"))
        self.assertEqual(
            self.require_match(Regex("a*b").search("xxaaab")).span(),
            (2, 6),
        )
        self.assertEqual(self.require_match(Regex("$").search("abc")).span(), (3, 3))
        self.assertEqual(self.require_match(Regex("$").search("abc\n")).span(), (3, 3))
        self.assertEqual(
            self.require_match(Regex("a{2,4}").match("aaaaa")).group(),
            "aaaa",
        )

    def test_counted_and_lazy_quantifiers(self):
        self.assertEqual(self.require_match(Regex("a{2,4}").fullmatch("aaaa")).group(), "aaaa")
        self.assertIsNone(Regex("a{2,4}").fullmatch("a"))
        self.assertEqual(self.require_match(Regex("a{2,}").fullmatch("aaaaa")).group(), "aaaaa")
        self.assertEqual(self.require_match(Regex("a{2}").fullmatch("aa")).group(), "aa")
        self.assertEqual(self.require_match(Regex("a+?").search("aaa")).group(), "a")
        self.assertEqual(self.require_match(Regex("a*?").search("aaa")).group(), "")
        self.assertEqual(self.require_match(Regex("(ab|cd)+").fullmatch("abcd")).group(1), "cd")
        self.assertEqual(
            self.require_match(Regex(r"(a+?){0,2}?$").search("aaa")).group(1),
            "aa",
        )

    def test_empty_loop_capture_compatibility(self):
        cases = (
            ("(a*)*", "b"),
            ("(a*?)*", "aa"),
            ("(a|b*)*", "ab"),
            ("(|a)+", "aa"),
            ("(a*?){2,}", "aa"),
            ("((a*)|b)*", "aa"),
            ("(a{0,2}?){0,2}?$", "aa"),
            (r"(\b|a)*", "aa"),
            (r"(\B|a)*", "a"),
            (r"((?=a)|a)*", "aa"),
        )
        for pattern, text in cases:
            with self.subTest(pattern=pattern, text=text):
                actual = self.require_match(Regex(pattern).search(text))
                self.assertEqual(actual.span(), {
                    "(a*)*": (0, 0),
                    "(a*?)*": (0, 0),
                    "(a|b*)*": (0, 2),
                    "(|a)+": (0, 0),
                    "(a*?){2,}": (0, 0),
                    "((a*)|b)*": (0, 2),
                    "(a{0,2}?){0,2}?$": (0, 2),
                    r"(\b|a)*": (0, 0),
                    r"(\B|a)*": (0, 1),
                    r"((?=a)|a)*": (0, 0),
                }[pattern])
                self.assertEqual(actual.groups(), {
                    "(a*)*": ("",),
                    "(a*?)*": ("",),
                    "(a|b*)*": ("",),
                    "(|a)+": ("",),
                    "(a*?){2,}": ("",),
                    "((a*)|b)*": ("", ""),
                    "(a{0,2}?){0,2}?$": ("a",),
                    r"(\b|a)*": ("",),
                    r"(\B|a)*": ("a",),
                    r"((?=a)|a)*": ("",),
                }[pattern])

    def test_nullable_repeats_keep_nfa_match_limit(self):
        cases = (
            ("(a*)*b", "a" * 2000),
            ("(a|b*)*c", "a" * 2000),
            ("(x?)*y", "x" * 2000),
            ("(?=a*b)", "a" * 2000),
        )
        for pattern, text in cases:
            with self.subTest(pattern=pattern):
                self.assertIsNone(Regex(pattern).search(text))

    def test_deep_nested_repeats_stay_compact(self):
        depth = 90
        nested_stars = "(?:" * depth + "a" + ")*" * depth
        star_pattern = Regex(nested_stars)
        self.assertLess(len(star_pattern._states), 500)
        self.assertIsNotNone(star_pattern.fullmatch("a" * 10))

        nested_alternation = "(?:" * depth + "(?:a|b)*" + ")*" * depth
        alternation_pattern = Regex(nested_alternation)
        self.assertLess(len(alternation_pattern._states), 500)
        self.assertIsNotNone(alternation_pattern.fullmatch("ab" * 100))

        nested_plus = "(?:" * 20 + "a+" + ")+" * 20
        plus_pattern = Regex(nested_plus)
        self.assertLess(len(plus_pattern._states), 500)
        self.assertIsNotNone(plus_pattern.fullmatch("a" * 10))

    def test_deep_nested_nullable_stars_match_with_bounded_closure(self):
        from time import perf_counter

        for depth in (10, 20, 40, 90):
            patterns = (
                "(?:" * depth + "a" + ")*" * depth,
                "(?:" * depth + "(?:a|b)*" + ")*" * depth,
            )
            for pattern, text in zip(patterns, ("a", "ab")):
                with self.subTest(depth=depth, pattern=pattern):
                    matcher = Regex(pattern)
                    started = perf_counter()
                    actual = self.require_match(matcher.search(text))
                    self.assertLess(perf_counter() - started, 1.0)
                    self.assertEqual(actual.span(), (0, len(text)))
                    self.assertEqual(actual.groups(), ())

    def test_lazy_repeats_with_empty_alternatives_preserve_priority(self):
        cases = (
            (r"(?:\W+?|\b|.)+", "  c", (0, 2)),
            (r"(?:\W+?|\b|.){2,}", "  c", (0, 2)),
            (r"(?:(?:\W|\w??)\b)+", " _ c ", (0, 1)),
        )
        for pattern, text, expected_span in cases:
            with self.subTest(pattern=pattern, text=text):
                actual = self.require_match(Regex(pattern).search(text))
                self.assertEqual(actual.span(), expected_span)

    def test_nested_lazy_repeats_preserve_search_priority(self):
        pattern = r"(?:(?:.)*?(?:\d)*)+"
        text = "1aa \n "

        actual = self.require_match(Regex(pattern).search(text))

        self.assertEqual(actual.span(), (0, 1))
        self.assertEqual(actual.groups(), ())

    def test_literal_search_uses_native_find(self):
        from time import perf_counter

        text = "x" * 1_000_000 + "needle"
        started = perf_counter()
        actual = self.require_match(Regex("needle").search(text))
        elapsed = perf_counter() - started
        self.assertEqual(actual.span(), (1_000_000, 1_000_006))
        self.assertLess(elapsed, 1.0)

    def test_ignorecase_search_skips_to_literal_candidates(self):
        from time import perf_counter

        text = "x" * 1_000_000 + "NEEDLE"
        started = perf_counter()
        actual = self.require_match(Regex("needle", regex_api.I).search(text))
        self.assertEqual(actual.span(), (1_000_000, 1_000_006))
        self.assertLess(perf_counter() - started, 1.0)

        pattern = "ne+dle"
        started = perf_counter()
        actual = self.require_match(Regex(pattern, regex_api.I).search(text))
        self.assertEqual(actual.span(), (1_000_000, 1_000_006))
        self.assertLess(perf_counter() - started, 1.0)
        for pattern, value in (("a?", "bbb"), ("a?b", "bbb")):
            actual = Regex(pattern).search(value)
            expected_span = (0, 0) if pattern == "a?" else (0, 1)
            self.assertEqual(self.span_or_none(actual), expected_span)

    def test_long_valid_patterns_do_not_exhaust_match_budget(self):
        cases = (
            ("(?>a+)b", "a" * 5000),
            ("a*+b", "a" * 5000),
            (r"(\d+)-\1", "1" * 3000),
        )
        for pattern, text in cases:
            with self.subTest(pattern=pattern):
                self.assertIsNone(Regex(pattern).search(text))
        self.assertIsNotNone(Regex("a{2000}").fullmatch("a" * 2000))

    def test_backreferences(self):
        match = self.require_match(Regex(r"(a)\1").search("zaa"))
        self.assertEqual(match.group(), "aa")
        self.assertEqual(match.group(1), "a")
        self.assertEqual(self.require_match(Regex(r"(ab|cd)\1").fullmatch("cdcd")).group(1), "cd")
        self.assertEqual(
            self.require_match(Regex(r"(a+)\1").fullmatch("aaaa")).groups(),
            ("aa",),
        )
        self.assertIsNone(Regex(r"(a)\1").fullmatch("ab"))
        self.assertIsNone(Regex(r"(a)?b\1").fullmatch("b"))
        with self.assertRaises(RegexSyntaxError):
            Regex(r"\1")
        with self.assertRaises(RegexSyntaxError):
            Regex(r"((a)\1)")

    def test_backreference_re_compatibility(self):
        cases = (
            (r"(a)\1", "zaa", (1, 3), ("a",)),
            (r"(ab|cd)\1", "cdcd", (0, 4), ("cd",)),
            (r"(a+)\1", "aaaa", (0, 4), ("aa",)),
            (r"([ab]+)-\1", "ab-ab", (0, 5), ("ab",)),
            (r"(a)?b\1", "abab", (0, 3), ("a",)),
            (r"(a)?b\1", "b", None, None),
        )
        for pattern, text, expected_span, expected_groups in cases:
            with self.subTest(pattern=pattern, text=text):
                actual = Regex(pattern).search(text)
                self.assertEqual(self.span_or_none(actual), expected_span)
                self.assertEqual(
                    None if actual is None else actual.groups(), expected_groups
                )

    def test_determinization_and_hopcroft_minimization(self):
        dfa = Regex("(ab|cd)+").to_dfa("abcd")
        minimized = dfa.minimize()
        self.assertIsInstance(minimized, DFA)
        self.assertLessEqual(len(minimized.transitions), len(dfa.transitions))
        for value in ("", "ab", "cd", "abcd", "abab", "ac", "dc"):
            self.assertEqual(dfa.fullmatch(value), minimized.fullmatch(value))
            self.assertEqual(
                minimized.fullmatch(value),
                Regex("(ab|cd)+").fullmatch(value) is not None,
            )
        self.assertIn("digraph DFA", minimized.to_dot())
        with self.assertRaises(ValueError):
            Regex(r"(a)\1").to_dfa("a")
        anchored = Regex("^a$").to_dfa("a")
        self.assertTrue(anchored.fullmatch("a"))
        self.assertFalse(anchored.fullmatch("ba"))
        lookahead = Regex("(?=a)a").to_dfa("ab")
        self.assertTrue(lookahead.fullmatch("a"))
        self.assertFalse(lookahead.fullmatch("b"))

    def test_dfa_anchor_assertions(self):
        cases = (
            (r"\Aa\Z", "a"),
            (r"^a$", "a"),
            (r"a$", "a"),
            (r"\bcat\b", "cat"),
            (r"\Bcat", "cat"),
            (r"(?m)^a$", "a"),
        )
        for pattern, text in cases:
            with self.subTest(pattern=pattern, text=text):
                dfa = Regex(pattern).to_dfa("catb\n")
                self.assertEqual(
                    dfa.fullmatch(text),
                    Regex(pattern).fullmatch(text) is not None,
                )

    def test_dfa_single_character_lookarounds(self):
        cases = (
            (r"(?=a)a", "a", True),
            (r"(?!a)b", "b", True),
            (r"a(?<=a)b", "ab", True),
            (r"c(?<!a)b", "cb", True),
            (r"a(?<!a)b", "ab", False),
        )
        for pattern, text, expected in cases:
            with self.subTest(pattern=pattern, text=text):
                regex = Regex(pattern)
                self.assertEqual(regex.to_dfa("abc").fullmatch(text), expected)
        with self.assertRaisesRegex(ValueError, "single-character"):
            Regex("(?=ab)a").to_dfa("ab")

    def test_minimized_dfa_languages(self):
        rng = random.Random(5401)
        alphabet = "ab"
        quantifiers = ("", "*", "+", "?", "{1,2}")
        values = (
            "".join(chars)
            for length in range(6)
            for chars in product(alphabet, repeat=length)
        )
        inputs = tuple(values)
        for _ in range(60):
            pattern = "".join(
                rng.choice(("a", "b", ".", "[ab]", "(a|b)"))
                + rng.choice(quantifiers)
                for _ in range(rng.randint(1, 3))
            )
            regex = Regex(pattern)
            dfa = regex.to_dfa(alphabet)
            minimized = dfa.minimize()
            for text in inputs:
                self.assertEqual(
                    regex.fullmatch(text) is not None,
                    dfa.fullmatch(text),
                    (pattern, text),
                )
                self.assertEqual(dfa.fullmatch(text), minimized.fullmatch(text))

    def test_seeded_search_fuzz_against_fullmatch_oracle(self):
        rng = random.Random(3917)
        atoms = ("a", "b", ".", "[ab]", r"\d", r"\w", "(a|b)", "(ab)")
        quantifiers = ("", "*", "+", "?", "*?", "+?", "{1,2}", "{2}")
        checked = 0
        for _ in range(2000):
            pattern = "".join(
                rng.choice(atoms) + rng.choice(quantifiers)
                for _ in range(rng.randint(1, 3))
            )
            text = "".join(rng.choice("ab12 \n") for _ in range(rng.randint(0, 8)))
            matcher = Regex(pattern)
            with self.subTest(pattern=pattern, text=text):
                self.assert_search_consistent_with_fullmatch(matcher, text)
            checked += 1
        self.assertEqual(checked, 2000)

    def test_first_char_candidate_search_matches_position_by_position(self):
        rng = random.Random(15803)
        patterns = (
            ("(a|b)c", 0),
            ("(a)?b", 0),
            ("(a|b)c", regex_api.IGNORECASE),
            ("(Σ|ß)X", regex_api.IGNORECASE),
            ("(i|k)X", regex_api.IGNORECASE | regex_api.ASCII),
            ("(?:ab|c)", 0),
        )
        alphabet = "abcABCΣςßẞXikK-!"
        for _ in range(400):
            pattern, flags = rng.choice(patterns)
            text = "".join(
                rng.choice(alphabet) for _ in range(rng.randint(0, 24))
            )
            matcher = compile(pattern, flags)
            actual = matcher.search(text)
            expected = next(
                (
                    match
                    for position in range(len(text) + 1)
                    if (match := matcher.match(text, position)) is not None
                ),
                None,
            )
            actual_result = (
                None if actual is None else (actual.span(), actual.groups())
            )
            expected_result = (
                None if expected is None else (expected.span(), expected.groups())
            )
            with self.subTest(pattern=pattern, flags=flags, text=text):
                self.assertEqual(actual_result, expected_result)

    def test_seeded_flag_and_bounds_search_invariants(self):
        rng = random.Random(68231)
        atoms = ("a", "b", ".", "[ab]", "(a)", "(b)", "(a|b)", "(ab)")
        quantifiers = ("", "?", "*", "+")
        flags = (
            0,
            regex_api.I,
            regex_api.M,
            regex_api.S,
            regex_api.I | regex_api.M,
            regex_api.I | regex_api.S,
            regex_api.M | regex_api.S,
            regex_api.I | regex_api.M | regex_api.S,
        )
        for _ in range(500):
            pattern = "".join(
                rng.choice(atoms) + rng.choice(quantifiers)
                for _ in range(rng.randint(1, 3))
            )
            text = "".join(rng.choice("aAbB\n.") for _ in range(rng.randint(0, 8)))
            pos = rng.randrange(len(text) + 2)
            endpos = rng.randrange(len(text) + 2)
            selected_flags = rng.choice(flags)
            matcher = Regex(pattern, selected_flags)
            with self.subTest(
                pattern=pattern,
                text=text,
                flags=selected_flags,
                pos=pos,
                endpos=endpos,
            ):
                self.assert_search_consistent_with_fullmatch(
                    matcher, text, pos, endpos
                )

    def test_seeded_syntax_fuzz_reports_only_engine_errors(self):
        rng = random.Random(27491)
        tokens = (
            "a",
            "b",
            "(",
            ")",
            "(?:",
            r"\b",
            "^",
            "$",
            "*",
            "+",
            "?",
            "{",
            "{2}",
            "|",
        )
        seen: set[str] = set()
        for _ in range(20_000):
            pattern = "".join(
                rng.choice(tokens) for _ in range(rng.randint(1, 7))
            )
            if pattern in seen:
                continue
            seen.add(pattern)
            try:
                compiled = Regex(pattern)
                self.assertIsInstance(compiled, Regex)
            except (RegexSyntaxError, RegexComplexityError):
                pass
        self.assertGreaterEqual(len(seen), 13_000)

    def test_syntax_errors(self):
        for pattern in ("(", "[a-", "*a", "a**", "\\", "a{4,2}", "a{50000}"):
            with self.subTest(pattern=pattern):
                with self.assertRaises(RegexSyntaxError):
                    Regex(pattern)

    def test_compile_helper_and_group_api(self):
        self.assertEqual(self.require_match(compile("x+").fullmatch("xxx")).group(0), "xxx")
        self.assertIs(compile("x+"), compile("x+"))
        with self.assertRaises(IndexError):
            self.require_match(Regex("x").fullmatch("x")).group(1)

    def test_match_object_re_compatibility(self):
        matcher = compile(r"(?P<word>\w+)-(\d+)")
        match = self.require_match(matcher.search("x item-42 y"))
        self.assertEqual(match.start(), 2)
        self.assertEqual(match.end(), 9)
        self.assertEqual(match.span(1), (2, 6))
        self.assertEqual(match.start(2), 7)
        self.assertEqual(match.end("word"), 6)
        self.assertIs(match.re, matcher)
        self.assertEqual((match.pos, match.endpos), (0, len(match.string)))
        self.assertEqual(match.regs, ((2, 9), (2, 6), (7, 9)))

    def test_re_style_module_api(self):
        self.assertEqual(self.require_match(regex_api.search("b+", "abbc")).group(), "bb")
        self.assertEqual(self.require_match(regex_api.match("a", "abc")).group(), "a")
        self.assertEqual(self.require_match(regex_api.fullmatch("a+", "aaa")).group(), "aaa")
        self.assertEqual(regex_api.findall(r"\d+", "a12b3"), ["12", "3"])
        self.assertEqual(regex_api.sub("a", "x", "aba"), "xbx")
        self.assertEqual(regex_api.subn("a", "x", "aba"), ("xbx", 2))
        self.assertEqual(regex_api.escape("a+b"), r"a\+b")
        self.assertIs(regex_api.error, RegexSyntaxError)
        self.assertTrue(issubclass(RegexSyntaxError, ValueError))
        with self.assertRaises(regex_api.error) as error:
            compile("(")
        self.assertIsNotNone(error.exception.pos)
        self.assertEqual(error.exception.pattern, "(")
        self.assertTrue(error.exception.msg)

    def test_flags_and_scoped_flags(self):
        cases = (
            ("(?i)cat", "CAT", (0, 3)),
            ("(?i:cat)DOG", "CATDOG", (0, 6)),
            ("(?m)^a$", "x\na\ny", (2, 3)),
            ("(?s)a.b", "a\nb", (0, 3)),
            ("(?x)a # ignored\n b", "ab", (0, 2)),
            (r"(?a)\w+", "é", None),
        )
        for pattern, text, expected_span in cases:
            with self.subTest(pattern=pattern, text=text):
                actual = Regex(pattern).search(text)
                self.assertEqual(self.span_or_none(actual), expected_span)
        self.assertEqual(
            compile("cat", regex_api.IGNORECASE).flags,
            regex_api.IGNORECASE | regex_api.UNICODE,
        )
        self.assertEqual(
            self.require_match(
                regex_api.search("cat", "CAT", regex_api.IGNORECASE)
            ).group(),
            "CAT",
        )
        self.assertIsNone(Regex(r"\B").search(""))

    def test_native_flags_and_escape(self):
        expected_flags = {
            "IGNORECASE": (2, "re.IGNORECASE"),
            "MULTILINE": (8, "re.MULTILINE"),
            "DOTALL": (16, "re.DOTALL"),
            "VERBOSE": (64, "re.VERBOSE"),
            "ASCII": (256, "re.ASCII"),
            "UNICODE": (32, "re.UNICODE"),
            "DEBUG": (128, "re.DEBUG"),
            "LOCALE": (4, "re.LOCALE"),
        }
        for name, (value, representation) in expected_flags.items():
            with self.subTest(flag=name):
                native_flag = getattr(regex_api, name)
                self.assertEqual(int(native_flag), value)
                self.assertEqual(repr(regex_api.RegexFlag(native_flag)), representation)
        self.assertEqual(
            repr(regex_api.RegexFlag.I | regex_api.RegexFlag.M),
            "re.IGNORECASE|re.MULTILINE",
        )
        self.assertEqual(regex_api.escape("!a+b [c]"), r"!a\+b\ \[c\]")
        self.assertEqual(regex_api.escape(b"!a+b [c]"), b"!a\\+b\\ \\[c\\]")
        unicode_bytes = bytes(range(256))
        unicode_text = unicode_bytes.decode("latin-1")
        escaped_characters = set(r"\.^$|?*+()[]{}#&~- " + "\t\n\v\f\r")
        expected_bytes = b"".join(
            (b"\\" if chr(value) in escaped_characters else b"")
            + bytes((value,))
            for value in range(256)
        )
        expected_text = "".join(
            ("\\" if character in escaped_characters else "") + character
            for character in unicode_text
        )
        self.assertEqual(regex_api.escape(unicode_bytes), expected_bytes)
        self.assertEqual(regex_api.escape(unicode_text), expected_text)

    def test_ignorecase_backreferences_match_case_equivalently(self):
        cases = (
            (r"(a)\1", "aA", regex_api.I, (0, 2), ("a",)),
            (r"(?P<x>a)(?P=x)", "aA", regex_api.I, (0, 2), ("a",)),
            (r"(ab)\1", "abAB", regex_api.I, (0, 4), ("ab",)),
            (r"(\w+)\1$", "abAB", regex_api.I, (0, 4), ("ab",)),
            (r"(é)\1", "éÉ", regex_api.I, (0, 2), ("é",)),
            (r"(?P<x>i)(?P=x)", "iİ", regex_api.I | regex_api.A, None, None),
        )
        for pattern, text, flags, expected_span, expected_groups in cases:
            with self.subTest(pattern=pattern, text=text, flags=flags):
                actual = Regex(pattern, flags).search(text)
                self.assertEqual(self.span_or_none(actual), expected_span)
                self.assertEqual(
                    None if actual is None else actual.groups(), expected_groups
                )

    def test_unicode_ignorecase_equivalence_classes(self):
        pairs = (
            ("İ", "i"),
            ("ı", "I"),
            ("K", "K"),
            ("ǅ", "ǆ"),
            ("ẞ", "ß"),
            ("ς", "σ"),
            ("ς", "Σ"),
            ("ϐ", "β"),
            ("ᲀ", "в"),
            ("ᲀ", "В"),
            ("µ", "μ"),
            ("µ", "Μ"),
            ("Å", "Å"),
            ("Å", "å"),
        )
        for left, right in pairs:
            for pattern, text in (
                (left, right),
                (right, left),
                (f"[{left}]", right),
                (f"[{right}]", left),
            ):
                with self.subTest(pattern=pattern, text=text):
                    actual = self.require_match(
                        Regex(pattern, regex_api.I).fullmatch(text)
                    )
                    self.assertEqual(actual.span(), (0, len(text)))

        self.assertIsNone(Regex("i", regex_api.I | regex_api.A).fullmatch("İ"))
        self.assertIsNone(Regex("I", regex_api.I | regex_api.A).fullmatch("ı"))

    def test_global_flags_and_module_compatibility(self):
        self.assertIsNotNone(Regex("(?i)(?i)").fullmatch(""))
        for pattern in ("(?i)|a", "(?i)(?m)|a"):
            actual = Regex(pattern)
            self.assertEqual(
                actual.flags,
                regex_api.IGNORECASE | regex_api.UNICODE
                | (regex_api.MULTILINE if "(?m)" in pattern else 0),
            )
            self.assertEqual(self.span_or_none(actual.search("A")), (0, 0))
        for pattern in ("|(?i)", "a|(?i)"):
            with self.subTest(pattern=pattern):
                with self.assertRaises(RegexSyntaxError):
                    Regex(pattern)
        for pattern in ("(?i)*a", "(?i)?", "(?i){2}"):
            with self.subTest(pattern=pattern), self.assertRaises(RegexSyntaxError):
                Regex(pattern)
        self.assertIs(regex_api.compile(compile("a")), compile("a"))
        compiled = compile("a")
        self.assertIsNotNone(regex_api.search(compiled, "ba"))
        self.assertEqual(regex_api.findall(compiled, "ba"), ["a"])
        self.assertEqual(regex_api.sub(compiled, "x", "aba"), "xbx")
        with self.assertRaises(ValueError):
            regex_api.compile(compiled, regex_api.I)
        self.assertEqual(regex_api.I, regex_api.IGNORECASE)
        self.assertEqual(regex_api.M, regex_api.MULTILINE)
        self.assertEqual(regex_api.S, regex_api.DOTALL)
        self.assertEqual(regex_api.X, regex_api.VERBOSE)
        self.assertEqual(regex_api.A, regex_api.ASCII)
        self.assertEqual(regex_api.U, regex_api.UNICODE)
        self.assertEqual(regex_api.NOFLAG, 0)
        self.assertIs(regex_api.Pattern, Regex)
        self.assertIs(Regex[str], Regex)
        self.assertIs(Match[str], Match)
        regex_api.purge()

    def test_pos_endpos_across_pattern_methods(self):
        matcher = compile("b+")
        match = self.require_match(matcher.search("abbbz", 1, 3))
        self.assertEqual(match.span(), (1, 3))
        self.assertEqual((match.pos, match.endpos), (1, 3))
        self.assertEqual(self.require_match(matcher.match("abbbz", 1, 3)).span(), (1, 3))
        self.assertEqual(
            self.require_match(matcher.fullmatch("abbbz", 1, 4)).span(), (1, 4)
        )
        self.assertEqual(
            [match.span() for match in matcher.finditer("abbbz", 2, 4)],
            [(2, 4)],
        )
        self.assertEqual(matcher.findall("abbbz", 2, 4), ["bb"])
        self.assertEqual(matcher.sub("X", "abbbz", pos=2, endpos=4), "abXz")
        self.assertEqual(
            matcher.subn("X", "abbbz", pos=2, endpos=4), ("abXz", 1)
        )
        self.assertEqual(
            self.require_match(
                regex_api.search("b+", "abbbz", pos=2, endpos=4)
            ).span(),
            (2, 4),
        )
        for pattern in ("", r"\w*$", r"\w*"):
            actual = compile(pattern).search("ab", 5)
            self.assertEqual(self.span_or_none(actual), (2, 2), pattern)

    def test_bytes_patterns_and_inputs(self):
        matcher = compile(rb"(?P<word>\w+)-(\d+)")
        match = self.require_match(matcher.search(b"\xffitem-42"))
        self.assertEqual(match.group(), b"item-42")
        self.assertEqual(match.group("word"), b"item")
        self.assertIs(match.string, b"\xffitem-42")
        self.assertEqual(regex_api.findall(rb"\d+", b"a12b3"), [b"12", b"3"])
        self.assertEqual(Regex(rb"(a*)").findall(b"b"), [b"", b""])
        self.assertEqual(Regex(rb"(a)?b").findall(b"b"), [b""])
        self.assertEqual(Regex(rb"(a)?(b)").findall(b"b"), [(b"", b"b")])
        self.assertEqual(regex_api.sub(rb"(\d+)", rb"[\1]", b"1 2"), b"[1] [2]")
        self.assertEqual(regex_api.subn(b"a", b"x", b"aba"), (b"xbx", 2))
        self.assertEqual(regex_api.split(b",", b"a,b"), [b"a", b"b"])
        self.assertEqual(regex_api.escape(b"a+b"), b"a\\+b")
        for source in (bytearray(b"a"), memoryview(b"a")):
            match = self.require_match(compile(rb"a").search(source))  # type: ignore[arg-type]
            self.assertEqual(match.group(), b"a")
            self.assertEqual(regex_api.findall(rb"a", source), [b"a"])  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            compile(rb"a").search("a")
        with self.assertRaises(TypeError):
            compile("a").search(b"a")
        with self.assertRaises(RegexSyntaxError) as error:
            compile(rb"\u0041")
        self.assertEqual(error.exception.pattern, rb"\u0041")

    def test_api_input_and_complexity_limits(self):
        with self.assertRaises(TypeError):
            compile(123)  # type: ignore[arg-type]
        matcher = compile("x")
        for operation in (matcher.search, matcher.match, matcher.fullmatch):
            with self.subTest(operation=operation.__name__):
                with self.assertRaises(TypeError):
                    operation(None)  # type: ignore[arg-type]

        self.assertEqual(RegexLimits(), RegexLimits())
        self.assertEqual(hash(RegexLimits()), hash(RegexLimits()))
        limits = RegexLimits()
        with self.assertRaises(AttributeError):
            setattr(limits, "repeat", 10)
        with self.assertRaises(AttributeError):
            delattr(limits, "repeat")

        with self.assertRaises(RegexComplexityError):
            Regex("a" * 4097, limits=RegexLimits(pattern_length=4096))
        nested_groups = "(" * 101 + "a" + ")" * 101
        with self.assertRaises(RegexComplexityError):
            Regex(nested_groups, limits=RegexLimits(group_depth=100))
        with self.assertRaises(RegexComplexityError):
            Regex("(a{1000}){6}", limits=RegexLimits(nfa_states=10_000))

        with self.assertRaises(RegexComplexityError):
            Regex("a").to_dfa("".join(chr(index) for index in range(257)))
        with self.assertRaises(TypeError):
            Regex("a").to_dfa(None)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            Regex("a").to_dfa("")

    def test_expanded_default_complexity_limits(self):
        pattern = "|".join("a" for _ in range(20_000))
        actual = Regex(pattern).search("a")
        self.assertEqual(self.span_or_none(actual), (0, 1))

        nested = "(?:" * 100 + "a" + ")" * 100
        actual_nested = Regex(nested).fullmatch("a")
        self.assertEqual(self.span_or_none(actual_nested), (0, 1))

        repeated = "a{1000}"
        actual_repeat = Regex(repeated).fullmatch("a" * 1000)
        self.assertEqual(self.span_or_none(actual_repeat), (0, 1000))

    def test_default_limits_reject_expensive_inputs_early(self):
        from time import perf_counter

        started = perf_counter()
        with self.assertRaises(RegexComplexityError):
            Regex("(a{999}){999}")
        self.assertLess(perf_counter() - started, 2.0)

        with self.assertRaises(RegexComplexityError):
            Regex("a" * 65_537)

    def test_repr_and_replacement_errors(self):
        self.assertEqual(repr(compile("ab", regex_api.I)), "re.compile('ab', re.IGNORECASE)")
        self.assertEqual(repr(compile("ab").search("zab")), "<re.Match object; span=(1, 3), match='ab'>")
        with self.assertRaises(RegexSyntaxError):
            compile("(a)").sub(r"\q", "a")
        with self.assertRaises(RegexSyntaxError):
            compile("(a)").sub(r"\2", "a")
        self.assertEqual(compile("(a)").sub(r"\g<1>", "a"), "a")
        with self.assertRaises(RegexSyntaxError) as error:
            compile("(")
        self.assertEqual(error.exception.pos, 0)
        self.assertEqual(error.exception.lineno, 1)
        self.assertEqual(error.exception.colno, 1)

    def test_native_syntax_error_metadata(self):
        cases = (("(", 0), ("[", 0), (r"\q", 0), ("a{3,2}", 2), ("a\n(", 2))
        for pattern, expected_position in cases:
            with self.subTest(pattern=pattern):
                with self.assertRaises(RegexSyntaxError) as actual_error:
                    Regex(pattern)
                self.assertEqual(actual_error.exception.pos, expected_position)
                self.assertEqual(actual_error.exception.pattern, pattern)
                self.assertTrue(actual_error.exception.msg)
                self.assertEqual(
                    actual_error.exception.lineno,
                    pattern.count("\n", 0, expected_position) + 1,
                )
                self.assertEqual(
                    actual_error.exception.colno,
                    expected_position - pattern.rfind("\n", 0, expected_position),
                )

    def test_runtime_import_does_not_load_python_re(self):
        script = """
import sys
sys.path.insert(0, sys.argv[1])
import resilex
assert "re" not in sys.modules
assert resilex.compile(r"\\p{Script=Greek}+").fullmatch("Αβ")
assert resilex.compile(rb"\\w", resilex.LOCALE).fullmatch(b"a")
resilex.compile("α").to_dot()
resilex.escape("a b+c")
try:
    resilex.compile("(")
except resilex.RegexSyntaxError:
    pass
assert "re" not in sys.modules
"""
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                script,
                str(Path(resilex.__file__).resolve().parent),
            ],
            cwd=".",
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_configurable_limits_and_backtrack_budget(self):
        with self.assertRaises(RegexComplexityError):
            compile("abc", limits=RegexLimits(pattern_length=2))
        with self.assertRaises(RegexSyntaxError):
            compile("a{3}", limits=RegexLimits(repeat=2))
        with self.assertRaises(RegexComplexityError):
            compile("((a))", limits=RegexLimits(group_depth=1))
        with self.assertRaises(RegexMatchLimitError):
            compile(
                r"(a)\1", limits=RegexLimits(backtrack_steps=1)
            ).fullmatch("aa")
        with self.assertRaises(RegexMatchLimitError):
            compile(
                r"(?<=a+)b", limits=RegexLimits(backtrack_steps=1)
            ).search("aaab")

    def test_pickle_copy_thread_safety_and_unicode_ranges(self):
        matcher = compile("[Ā-Ő]+")
        self.assertEqual(self.require_match(matcher.fullmatch("ĀŐ")).group(), "ĀŐ")
        self.assertIs(copy.copy(matcher), matcher)
        self.assertIs(copy.deepcopy(matcher), matcher)
        restored = pickle.loads(pickle.dumps(matcher))
        self.assertEqual(restored.pattern, matcher.pattern)
        self.assertEqual(restored.fullmatch("ĀŐ").group(), "ĀŐ")

        with ThreadPoolExecutor(max_workers=8) as executor:
            compiled = list(executor.map(lambda _: compile("thread-safe"), range(64)))
        self.assertTrue(all(value is compiled[0] for value in compiled))

    def test_shared_compiled_patterns_are_thread_safe(self):
        pattern = r"(?i)(?P<symbol>Σ|ß)+"
        matcher = compile(pattern)
        dfa_matcher = compile(r"[a-z]+")
        texts = (
            "prefix-ςẞΣ-suffix",
            "ßxΣ",
            "ΣΣ",
            "outside",
            "ẞς",
            "x" * 24 + "ςß",
        )

        def summarize(match):
            return (
                None
                if match is None
                else (match.span(), match.groups(), match.groupdict())
            )

        def exercise(index):
            text = texts[index % len(texts)]
            return (
                summarize(matcher.search(text)),
                summarize(matcher.match(text)),
                summarize(matcher.fullmatch(text)),
                matcher.findall(text),
                [
                    summarize(match)
                    for match in matcher.finditer(text)
                ],
                dfa_matcher.fullmatch(text) is not None,
            )

        expected = [exercise(index) for index in range(len(texts))]
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(exercise, range(len(texts) * 64)))

        for index, result in enumerate(results):
            self.assertEqual(result, expected[index % len(texts)])


if __name__ == "__main__":
    unittest.main()
