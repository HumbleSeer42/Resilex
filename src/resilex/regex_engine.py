"""A small regular-expression engine with an ordered Thompson NFA."""

from __future__ import annotations

import ctypes
from collections.abc import Callable, Iterator
from enum import IntFlag
from functools import lru_cache
from bisect import bisect_right
import heapq
import os
from threading import RLock
from itertools import count
import unicodedata
from ._native_dataclasses import dataclass, field

_Predicate = Callable[[str], bool]


def _dot_quote(value: str) -> str:
    escaped = ['"']
    for char in value:
        codepoint = ord(char)
        if char == '"':
            escaped.append('\\"')
        elif char == "\\":
            escaped.append("\\\\")
        elif char == "\b":
            escaped.append("\\b")
        elif char == "\f":
            escaped.append("\\f")
        elif char == "\n":
            escaped.append("\\n")
        elif char == "\r":
            escaped.append("\\r")
        elif char == "\t":
            escaped.append("\\t")
        elif codepoint < 0x20 or codepoint > 0x7E:
            if codepoint <= 0xFFFF:
                escaped.append(f"\\u{codepoint:04x}")
            else:
                codepoint -= 0x10000
                escaped.append(f"\\u{0xD800 + (codepoint >> 10):04x}")
                escaped.append(f"\\u{0xDC00 + (codepoint & 0x3FF):04x}")
        else:
            escaped.append(char)
    escaped.append('"')
    return "".join(escaped)


def _record_explanation(trace: list[str] | None, message: str) -> None:
    if trace is None:
        return
    if len(trace) < _MAX_EXPLANATION_EVENTS:
        trace.append(message)
    elif len(trace) == _MAX_EXPLANATION_EVENTS:
        trace.append("Further trace events omitted.")


_MAX_REPEAT = 49_999
_MAX_BACKTRACK_STEPS = 1_000_000
_MAX_PATTERN_LENGTH = 65_536
_MAX_GROUP_DEPTH = 100
_MAX_NFA_STATES = 100_000
_MAX_DFA_STATES = 10_000
_MAX_DFA_ALPHABET = 256
_MAX_EXPLANATION_EVENTS = 200

_UNICODE_GENERAL_CATEGORIES = {
    "c": ("C", "Other"),
    "cc": ("Cc", "Control"),
    "cf": ("Cf", "Format"),
    "cn": ("Cn", "Unassigned"),
    "co": ("Co", "Private_Use"),
    "cs": ("Cs", "Surrogate"),
    "l": ("L", "Letter"),
    "lc": ("L&", "Cased_Letter"),
    "ll": ("Ll", "Lowercase_Letter"),
    "lm": ("Lm", "Modifier_Letter"),
    "lo": ("Lo", "Other_Letter"),
    "lt": ("Lt", "Titlecase_Letter"),
    "lu": ("Lu", "Uppercase_Letter"),
    "m": ("M", "Mark"),
    "mc": ("Mc", "Spacing_Mark"),
    "me": ("Me", "Enclosing_Mark"),
    "mn": ("Mn", "Nonspacing_Mark"),
    "n": ("N", "Number"),
    "nd": ("Nd", "Decimal_Number", "Digit"),
    "nl": ("Nl", "Letter_Number"),
    "no": ("No", "Other_Number"),
    "p": ("P", "Punctuation"),
    "pc": ("Pc", "Connector_Punctuation"),
    "pd": ("Pd", "Dash_Punctuation"),
    "pe": ("Pe", "Close_Punctuation"),
    "pf": ("Pf", "Final_Punctuation"),
    "pi": ("Pi", "Initial_Punctuation"),
    "po": ("Po", "Other_Punctuation"),
    "ps": ("Ps", "Open_Punctuation"),
    "s": ("S", "Symbol"),
    "sc": ("Sc", "Currency_Symbol"),
    "sk": ("Sk", "Modifier_Symbol"),
    "sm": ("Sm", "Math_Symbol"),
    "so": ("So", "Other_Symbol"),
    "z": ("Z", "Separator"),
    "zl": ("Zl", "Line_Separator"),
    "zp": ("Zp", "Paragraph_Separator"),
    "zs": ("Zs", "Space_Separator"),
}
_UNICODE_CATEGORY_LOOKUP = {
    alias.casefold().replace("_", "").replace("-", "").replace(" ", ""): aliases[0]
    for aliases in _UNICODE_GENERAL_CATEGORIES.values()
    for alias in aliases
}

class RegexFlag(IntFlag):
    """Regex compilation and matching flags.

    Flags can be combined using bitwise OR (|).

    Available flags:

        NOFLAG: No flags enabled.
        ASCII (A): Use ASCII character-class semantics.
        IGNORECASE (I): Perform case-insensitive matching.
        LOCALE (L): Use locale-dependent matching for bytes patterns.
        MULTILINE (M): Make ^ and $ match line boundaries.
        DOTALL (S): Make . match newline characters.
        UNICODE (U): Enable Unicode matching semantics for strings.
        VERBOSE (X): Allow verbose pattern formatting.
        DEBUG: Display compiled pattern debugging information.
    """

    NOFLAG = 0
    ASCII = 256
    A = ASCII
    IGNORECASE = 2
    I = IGNORECASE
    LOCALE = 4
    L = LOCALE
    MULTILINE = 8
    M = MULTILINE
    DOTALL = 16
    S = DOTALL
    UNICODE = 32
    U = UNICODE
    VERBOSE = 64
    X = VERBOSE
    DEBUG = 128

    def __repr__(self) -> str:
        if self.name is None:
            return f"re.RegexFlag({int(self)})"
        return "|".join(f"re.{name}" for name in self.name.split("|"))


IGNORECASE = RegexFlag.IGNORECASE
MULTILINE = RegexFlag.MULTILINE
DOTALL = RegexFlag.DOTALL
VERBOSE = RegexFlag.VERBOSE
ASCII = RegexFlag.ASCII
UNICODE = RegexFlag.UNICODE
DEBUG = RegexFlag.DEBUG
LOCALE = RegexFlag.LOCALE
NOFLAG = RegexFlag.NOFLAG
I = IGNORECASE
M = MULTILINE
S = DOTALL
X = VERBOSE
A = ASCII
U = UNICODE
L = LOCALE

__all__ = [
    "A",
    "ASCII",
    "DFA",
    "DEBUG",
    "DOTALL",
    "I",
    "IGNORECASE",
    "L",
    "LOCALE",
    "M",
    "MULTILINE",
    "NOFLAG",
    "S",
    "Match",
    "Pattern",
    "Regex",
    "RegexComplexityError",
    "RegexFlag",
    "RegexLimits",
    "RegexMatchLimitError",
    "RegexSyntaxError",
    "U",
    "UNICODE",
    "VERBOSE",
    "X",
    "compile",
    "error",
    "escape",
    "findall",
    "finditer",
    "fullmatch",
    "match",
    "purge",
    "search",
    "split",
    "sub",
    "subn",
]


class RegexSyntaxError(ValueError):
    """Raised when a pattern contains invalid or unsupported syntax.

    Attributes:
        msg: Description of the syntax error.
        pos: Position in the pattern where the error occurred, if available.
        pattern: Original pattern that caused the error, if available.
        lineno: One-based line number of the error, when available.
        colno: One-based column number of the error, when available.
    """

    def __init__(
        self, message: str, pos: int | None = None, pattern: str | bytes | None = None
    ) -> None:
        """Initialize a regular expression syntax error.

        Args:
            message: Description of the syntax error.
            pos: Optional position of the error in the pattern.
            pattern: Original string or bytes pattern, if available.

        The line and column numbers are calculated automatically
        when both ``pattern`` and ``pos`` are provided.
        """
        super().__init__(message)
        self.msg = message
        self.pos = pos
        self.pattern = pattern
        text = pattern.decode("latin-1") if isinstance(pattern, bytes) else pattern
        if text is not None and pos is not None:
            self.lineno = text.count("\n", 0, pos) + 1
            self.colno = pos - text.rfind("\n", 0, pos)


error = RegexSyntaxError


class RegexComplexityError(ValueError):
    """Raised when a pattern or automaton exceeds a configured size limit.

    Limits are configured through :class:`RegexLimits`, including maximum
    pattern length, group depth, repeat bounds, NFA states, DFA states,
    and DFA alphabet size.
    """


class RegexMatchLimitError(RuntimeError):
    """Raised when matching exceeds the configured backtracking work limit.

    This limit is controlled by :attr:`RegexLimits.backtrack_steps`.
    It protects bounded backtracking operations from excessive execution.

    It does not limit the NFA execution path.
    """


@dataclass(frozen=True)
class _Empty:
    pass


@dataclass(frozen=True)
class _GlobalFlags:
    pass


@dataclass(frozen=True)
class _Character:
    predicate: _Predicate
    label: str
    literal: str | None = None


@dataclass(frozen=True)
class _Assertion:
    kind: str


@dataclass(frozen=True)
class _Lookaround:
    kind: str
    positive: bool
    node: object


@dataclass(frozen=True)
class _Backreference:
    group: int


@dataclass(frozen=True)
class _Sequence:
    items: tuple[object, ...]


@dataclass(frozen=True)
class _Alternative:
    options: tuple[object, ...]


@dataclass(frozen=True)
class _Repeat:
    node: object
    minimum: int
    maximum: int | None
    greedy: bool


@dataclass(frozen=True)
class _Atomic:
    node: object


@dataclass(frozen=True)
class _Conditional:
    group: int | str
    yes: object
    no: object


@dataclass(frozen=True)
class _Group:
    index: int
    node: object


@dataclass(frozen=True)
class RegexLimits:
    """Configure resource limits for regular expression processing.

    These limits help control pattern complexity, automaton construction,
    and execution costs.

    All values must be positive integers. Boolean values are not accepted.

    Default limits:

        - ``pattern_length``: 65,536 characters maximum.
        - ``group_depth``: 100 nested groups maximum.
        - ``repeat``: 49,999 maximum counted-repeat bound.
        - ``nfa_states``: 100,000 maximum NFA states and VM instructions.
        - ``dfa_states``: 10,000 maximum DFA states.
        - ``dfa_alphabet``: 256 maximum DFA alphabet characters.
        - ``backtrack_steps``: 1,000,000 maximum backtracking VM steps.

    Note:
        ``backtrack_steps`` limits the bounded backtracking VM.
        It does not limit NFA execution.
    """

    pattern_length: int = _MAX_PATTERN_LENGTH
    group_depth: int = _MAX_GROUP_DEPTH
    repeat: int = _MAX_REPEAT
    nfa_states: int = _MAX_NFA_STATES
    dfa_states: int = _MAX_DFA_STATES
    dfa_alphabet: int = _MAX_DFA_ALPHABET
    backtrack_steps: int = _MAX_BACKTRACK_STEPS

    def __init__(
        self,
        pattern_length: int = _MAX_PATTERN_LENGTH,
        group_depth: int = _MAX_GROUP_DEPTH,
        repeat: int = _MAX_REPEAT,
        nfa_states: int = _MAX_NFA_STATES,
        dfa_states: int = _MAX_DFA_STATES,
        dfa_alphabet: int = _MAX_DFA_ALPHABET,
        backtrack_steps: int = _MAX_BACKTRACK_STEPS,
    ) -> None:
        """Initialize resource limits.

        Args:
            pattern_length: Maximum permitted pattern length.
            group_depth: Maximum nested group depth.
            repeat: Maximum counted-repeat bound.
            nfa_states: Maximum NFA states and VM instructions.
            dfa_states: Maximum DFA states.
            dfa_alphabet: Maximum number of DFA alphabet characters.
            backtrack_steps: Maximum backtracking VM execution steps.

        Raises:
            ValueError: If any limit is not a positive integer.
        """
        object.__setattr__(self, "pattern_length", pattern_length)
        object.__setattr__(self, "group_depth", group_depth)
        object.__setattr__(self, "repeat", repeat)
        object.__setattr__(self, "nfa_states", nfa_states)
        object.__setattr__(self, "dfa_states", dfa_states)
        object.__setattr__(self, "dfa_alphabet", dfa_alphabet)
        object.__setattr__(self, "backtrack_steps", backtrack_steps)
        self.__post_init__()

    def __post_init__(self) -> None:
        """Validate that every configured limit is a positive integer.

        Raises:
            ValueError: If a limit is zero, negative, or not an integer.
        """
        for name, value in vars(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} limit must be a positive integer")


def _fixed_width(node: object) -> int | None:
    if isinstance(node, (_Empty, _GlobalFlags, _Assertion, _Lookaround)):
        return 0
    if isinstance(node, _Character):
        return 1
    if isinstance(node, (_Backreference,)):
        return None
    if isinstance(node, (_Group, _Atomic)):
        return _fixed_width(node.node)
    if isinstance(node, _Conditional):
        yes_width = _fixed_width(node.yes)
        no_width = _fixed_width(node.no)
        return yes_width if yes_width == no_width else None
    if isinstance(node, _Sequence):
        widths = [_fixed_width(item) for item in node.items]
        return (
            None
            if any(width is None for width in widths)
            else sum(width for width in widths if width is not None)
        )
    if isinstance(node, _Alternative):
        widths = [_fixed_width(option) for option in node.options]
        return widths[0] if widths and all(width == widths[0] for width in widths) else None
    if isinstance(node, _Repeat):
        if node.maximum != node.minimum:
            return None
        if node.minimum == 0:
            return 0
        width = _fixed_width(node.node)
        return None if width is None else width * node.minimum
    raise TypeError(f"Unsupported syntax node: {type(node).__name__}")


def _width_bounds(node: object) -> tuple[int, int | None]:
    if isinstance(node, (_Empty, _GlobalFlags, _Assertion, _Lookaround)):
        return 0, 0
    if isinstance(node, _Character):
        return 1, 1
    if isinstance(node, _Backreference):
        return 0, None
    if isinstance(node, (_Group, _Atomic)):
        return _width_bounds(node.node)
    if isinstance(node, _Conditional):
        yes_min, yes_max = _width_bounds(node.yes)
        no_min, no_max = _width_bounds(node.no)
        maximum = (
            None
            if yes_max is None or no_max is None
            else max(yes_max, no_max)
        )
        return min(yes_min, no_min), maximum
    if isinstance(node, _Sequence):
        bounds = [_width_bounds(item) for item in node.items]
        minimum = sum(item_min for item_min, _ in bounds)
        maximum = (
            None
            if any(item_max is None for _, item_max in bounds)
            else sum(item_max for _, item_max in bounds if item_max is not None)
        )
        return minimum, maximum
    if isinstance(node, _Alternative):
        bounds = [_width_bounds(option) for option in node.options]
        minimum = min(item_min for item_min, _ in bounds)
        maximum = (
            None
            if any(item_max is None for _, item_max in bounds)
            else max(item_max for _, item_max in bounds if item_max is not None)
        )
        return minimum, maximum
    if isinstance(node, _Repeat):
        minimum_width, maximum_width = _width_bounds(node.node)
        minimum = minimum_width * node.minimum
        maximum = (
            0
            if node.maximum == 0
            else None
            if node.maximum is None or maximum_width is None
            else maximum_width * node.maximum
        )
        return minimum, maximum
    raise TypeError(f"Unsupported syntax node: {type(node).__name__}")


def _single_character_predicate(node: object) -> _Predicate | None:
    if isinstance(node, _Character):
        return node.predicate
    if isinstance(node, (_Group, _Atomic)):
        return _single_character_predicate(node.node)
    if isinstance(node, _Sequence) and len(node.items) == 1:
        return _single_character_predicate(node.items[0])
    if isinstance(node, _Alternative):
        predicates = [
            _single_character_predicate(option) for option in node.options
        ]
        if all(predicate is not None for predicate in predicates):
            return lambda char: any(
                predicate(char) for predicate in predicates if predicate is not None
            )
    return None


def _literal_text(node: object) -> str | None:
    if isinstance(node, (_Empty, _GlobalFlags)):
        return ""
    if isinstance(node, _Character):
        return node.literal
    if isinstance(node, _Sequence):
        pieces = [_literal_text(item) for item in node.items]
        return None if any(piece is None for piece in pieces) else "".join(
            piece for piece in pieces if piece is not None
        )
    if isinstance(node, _Atomic):
        return _literal_text(node.node)
    return None


def _guaranteed_literals(node: object) -> set[str]:
    if isinstance(node, _Character):
        return {node.literal} if node.literal is not None else set()
    if isinstance(node, (_Group, _Atomic)):
        return _guaranteed_literals(node.node)
    if isinstance(node, _Sequence):
        return set().union(*(_guaranteed_literals(item) for item in node.items))
    if isinstance(node, _Alternative):
        options = [_guaranteed_literals(option) for option in node.options]
        return set.intersection(*options) if options else set()
    if isinstance(node, _Repeat):
        return _guaranteed_literals(node.node) if node.minimum else set()
    if isinstance(node, _Conditional):
        return _guaranteed_literals(node.yes) & _guaranteed_literals(node.no)
    return set()


def _first_character_predicate(node: object) -> _Predicate | None:
    def collect(current: object) -> tuple[list[_Predicate], bool, bool]:
        if isinstance(current, _Character):
            return [current.predicate], False, True
        if isinstance(current, (_Empty, _GlobalFlags, _Assertion, _Lookaround)):
            return [], True, True
        if isinstance(current, _Backreference):
            return [], True, False
        if isinstance(current, _Group):
            return collect(current.node)
        if isinstance(current, _Atomic):
            return collect(current.node)
        if isinstance(current, _Conditional):
            yes = collect(current.yes)
            no = collect(current.no)
            if not yes[2] or not no[2] or yes[1] or no[1]:
                return [], True, False
            return yes[0] + no[0], False, True
        if isinstance(current, _Repeat):
            predicates, nullable, known = collect(current.node)
            return predicates, current.minimum == 0 or nullable, known
        if isinstance(current, _Sequence):
            predicates: list[_Predicate] = []
            known = True
            nullable = True
            for item in current.items:
                item_predicates, item_nullable, item_known = collect(item)
                predicates.extend(item_predicates)
                known = known and item_known
                if not item_nullable:
                    nullable = False
                    break
            return predicates, nullable, known
        if isinstance(current, _Alternative):
            alternatives = [collect(option) for option in current.options]
            return (
                [predicate for values, _, _ in alternatives for predicate in values],
                any(is_nullable for _, is_nullable, _ in alternatives),
                all(is_known for _, _, is_known in alternatives),
            )
        return [], True, False

    predicates, nullable, known = collect(node)
    if nullable or not known or not predicates:
        return None
    return lambda char: any(predicate(char) for predicate in predicates)


def _nullable(node: object) -> bool:
    if isinstance(node, (_Empty, _GlobalFlags, _Assertion, _Lookaround, _Backreference)):
        return True
    if isinstance(node, _Character):
        return False
    if isinstance(node, (_Group, _Atomic)):
        return _nullable(node.node)
    if isinstance(node, _Repeat):
        return node.minimum == 0 or _nullable(node.node)
    if isinstance(node, _Sequence):
        return all(_nullable(item) for item in node.items)
    if isinstance(node, _Alternative):
        return any(_nullable(option) for option in node.options)
    if isinstance(node, _Conditional):
        return _nullable(node.yes) or _nullable(node.no)
    return True


def _first_character_literals(node: object) -> set[str] | None:
    if isinstance(node, (_Empty, _GlobalFlags, _Assertion, _Lookaround)):
        return set()
    if isinstance(node, _Character):
        return {node.literal} if node.literal is not None else None
    if isinstance(node, (_Group, _Atomic)):
        return _first_character_literals(node.node)
    if isinstance(node, _Repeat):
        return _first_character_literals(node.node)
    if isinstance(node, _Alternative):
        first_sets = [_first_character_literals(option) for option in node.options]
        if any(first is None for first in first_sets):
            return None
        return set().union(*(first for first in first_sets if first is not None))
    if isinstance(node, _Sequence):
        first_chars: set[str] = set()
        for item in node.items:
            item_first = _first_character_literals(item)
            if item_first is None:
                return None
            first_chars.update(item_first)
            if not _nullable(item):
                return first_chars
        return None
    if isinstance(node, _Conditional):
        yes = _first_character_literals(node.yes)
        no = _first_character_literals(node.no)
        if yes is None or no is None:
            return None
        return yes | no
    return None


def _atomic_nodes(node: object) -> list[_Atomic]:
    if isinstance(node, _Atomic):
        return [node, *_atomic_nodes(node.node)]
    if isinstance(node, _Group):
        return _atomic_nodes(node.node)
    if isinstance(node, _Sequence):
        return [atomic for item in node.items for atomic in _atomic_nodes(item)]
    if isinstance(node, _Alternative):
        return [atomic for option in node.options for atomic in _atomic_nodes(option)]
    if isinstance(node, _Repeat):
        return _atomic_nodes(node.node)
    if isinstance(node, _Conditional):
        return _atomic_nodes(node.yes) + _atomic_nodes(node.no)
    if isinstance(node, _Lookaround):
        return _atomic_nodes(node.node)
    return []


def _contains_conditional(node: object) -> bool:
    if isinstance(node, _Conditional):
        return True
    if isinstance(node, (_Group, _Atomic, _Repeat)):
        return _contains_conditional(node.node)
    if isinstance(node, _Sequence):
        return any(_contains_conditional(item) for item in node.items)
    if isinstance(node, _Alternative):
        return any(_contains_conditional(option) for option in node.options)
    if isinstance(node, _Lookaround):
        return _contains_conditional(node.node)
    return False


def _resolve_conditional_groups(
    node: object, group_count: int, groupindex: dict[str, int]
) -> object:
    if isinstance(node, _Conditional):
        group = (
            groupindex.get(node.group, 0)
            if isinstance(node.group, str)
            else node.group
        )
        if group < 1 or group > group_count:
            raise RegexSyntaxError(f"Unknown conditional group {node.group!r}")
        return _Conditional(
            group,
            _resolve_conditional_groups(node.yes, group_count, groupindex),
            _resolve_conditional_groups(node.no, group_count, groupindex),
        )
    if isinstance(node, _Group):
        return _Group(
            node.index,
            _resolve_conditional_groups(node.node, group_count, groupindex),
        )
    if isinstance(node, _Atomic):
        return _Atomic(
            _resolve_conditional_groups(node.node, group_count, groupindex)
        )
    if isinstance(node, _Repeat):
        return _Repeat(
            _resolve_conditional_groups(node.node, group_count, groupindex),
            node.minimum,
            node.maximum,
            node.greedy,
        )
    if isinstance(node, _Sequence):
        return _Sequence(
            tuple(
                _resolve_conditional_groups(item, group_count, groupindex)
                for item in node.items
            )
        )
    if isinstance(node, _Alternative):
        return _Alternative(
            tuple(
                _resolve_conditional_groups(option, group_count, groupindex)
                for option in node.options
            )
        )
    if isinstance(node, _Lookaround):
        return _Lookaround(
            node.kind,
            node.positive,
            _resolve_conditional_groups(node.node, group_count, groupindex),
        )
    return node


def _contains_lookaround(node: object) -> bool:
    if isinstance(node, _Lookaround):
        return True
    if isinstance(node, (_Group, _Atomic, _Repeat)):
        return _contains_lookaround(node.node)
    if isinstance(node, _Sequence):
        return any(_contains_lookaround(item) for item in node.items)
    if isinstance(node, _Alternative):
        return any(_contains_lookaround(option) for option in node.options)
    if isinstance(node, _Conditional):
        return _contains_lookaround(node.yes) or _contains_lookaround(node.no)
    return False


def _contains_variable_width_lookbehind(node: object) -> bool:
    if isinstance(node, _Lookaround):
        minimum, maximum = _width_bounds(node.node)
        return (
            node.kind == "behind" and minimum != maximum
            or _contains_variable_width_lookbehind(node.node)
        )
    if isinstance(node, (_Group, _Atomic, _Repeat)):
        return _contains_variable_width_lookbehind(node.node)
    if isinstance(node, _Sequence):
        return any(_contains_variable_width_lookbehind(item) for item in node.items)
    if isinstance(node, _Alternative):
        return any(_contains_variable_width_lookbehind(option) for option in node.options)
    if isinstance(node, _Conditional):
        return _contains_variable_width_lookbehind(
            node.yes
        ) or _contains_variable_width_lookbehind(node.no)
    return False


def _contains_alternative(node: object) -> bool:
    if isinstance(node, _Alternative):
        return True
    if isinstance(node, (_Group, _Atomic, _Repeat, _Lookaround)):
        return _contains_alternative(node.node)
    if isinstance(node, _Sequence):
        return any(_contains_alternative(item) for item in node.items)
    if isinstance(node, _Conditional):
        return _contains_alternative(node.yes) or _contains_alternative(node.no)
    return False


def _contains_nullable_alternative_repeat(node: object) -> bool:
    if isinstance(node, _Repeat):
        return (
            _nullable(node.node)
            and _contains_alternative(node.node)
        ) or _contains_nullable_alternative_repeat(node.node)
    if isinstance(node, (_Group, _Atomic, _Lookaround)):
        return _contains_nullable_alternative_repeat(node.node)
    if isinstance(node, _Sequence):
        return any(_contains_nullable_alternative_repeat(item) for item in node.items)
    if isinstance(node, _Alternative):
        return any(
            _contains_nullable_alternative_repeat(option) for option in node.options
        )
    if isinstance(node, _Conditional):
        return _contains_nullable_alternative_repeat(
            node.yes
        ) or _contains_nullable_alternative_repeat(node.no)
    return False


def _category_predicate(name: str, flags: int = 0) -> _Predicate:
    if flags & LOCALE and name.lower() == "w":
        return _locale_character_predicate(rb"\W" if name.isupper() else rb"\w")
    if flags & (ASCII | LOCALE):
        predicates: dict[str, _Predicate] = {
            "d": lambda char: "0" <= char <= "9",
            "w": lambda char: char == "_" or char.isascii() and char.isalnum(),
            "s": lambda char: char in " \t\n\r\f\v",
        }
    else:
        predicates = {
            "d": str.isdecimal,
            "w": lambda char: char == "_" or char.isalnum(),
            "s": str.isspace,
        }
    predicate = predicates[name.lower()]
    if name.isupper():
        return lambda char: not predicate(char)
    return predicate


@lru_cache(maxsize=8)
def _locale_character_predicate(pattern: bytes) -> _Predicate:
    if pattern not in (rb"\w", rb"\W"):
        raise ValueError(f"Unsupported locale character class {pattern!r}")
    positive = pattern == rb"\w"

    def matches(char: str) -> bool:
        if len(char) != 1:
            return False
        codepoint = ord(char)
        is_word = char == "_" or bool(_locale_ctype("isalnum", codepoint))
        return is_word if positive else not is_word

    return matches


@lru_cache(maxsize=256)
def _locale_case_insensitive_predicate(expected: str) -> _Predicate:
    if len(expected) != 1:
        raise ValueError("Locale case-insensitive literals must be one byte")

    def matches(char: str) -> bool:
        return (
            len(char) == 1
            and _locale_ctype("tolower", ord(char))
            == _locale_ctype("tolower", ord(expected))
        )

    return matches


@lru_cache(maxsize=256)
def _locale_range_predicate(low: str, high: str) -> _Predicate:
    low_codepoint = ord(low)
    high_codepoint = ord(high)

    def matches(char: str) -> bool:
        if len(char) != 1:
            return False
        codepoint = ord(char)
        folded_low = _locale_ctype("tolower", low_codepoint)
        folded_high = _locale_ctype("tolower", high_codepoint)
        return (
            low_codepoint <= codepoint <= high_codepoint
            or folded_low
            <= _locale_ctype("tolower", codepoint)
            <= folded_high
        )

    return matches


@lru_cache(maxsize=1)
def _locale_ctype_functions() -> tuple[Callable[[int], int], ...]:
    libraries: list[ctypes.CDLL] = []
    for name in (None, "ucrtbase", "msvcrt"):
        try:
            libraries.append(ctypes.CDLL(name))
        except OSError:
            continue

    for library in libraries:
        functions = []
        for name in ("isalnum", "tolower"):
            try:
                function = getattr(library, name)
            except AttributeError:
                break
            function.argtypes = [ctypes.c_int]
            function.restype = ctypes.c_int
            functions.append(function)
        if len(functions) == 2:
            return tuple(functions)
    raise RuntimeError("The platform C runtime does not expose locale ctype functions")


def _locale_ctype(name: str, value: int) -> int:
    functions = _locale_ctype_functions()
    return functions[0 if name == "isalnum" else 1](value)


def _unicode_category_predicate(expression: str, negate: bool = False) -> _Predicate:
    normalized = (
        expression.casefold().replace("_", "").replace("-", "").replace(" ", "")
    )
    property_name: str | None = None
    if "=" in normalized:
        property_name, normalized = normalized.split("=", 1)
        if property_name not in (
            "gc",
            "generalcategory",
            "sc",
            "script",
            "scx",
            "scriptextensions",
        ):
            raise ValueError(f"Unsupported Unicode property {expression!r}")
    category = (
        None
        if property_name in ("sc", "script", "scx", "scriptextensions")
        else _UNICODE_CATEGORY_LOOKUP.get(normalized)
    )
    if category is not None:
        if category == "L&":
            predicate = lambda char: unicodedata.category(char) in ("Lu", "Ll", "Lt")
        else:
            predicate = lambda char: (
                unicodedata.category(char) == category
                if len(category) == 2
                else unicodedata.category(char).startswith(category)
            )
    else:
        if property_name in ("gc", "generalcategory"):
            raise ValueError(f"Unsupported Unicode general category {expression!r}")
        aliases, scripts, script_starts, extensions, extension_starts = (
            _unicode_script_data()
        )
        script = aliases.get(normalized)
        if script is None:
            raise ValueError(f"Unsupported Unicode property value {expression!r}")
        use_extensions = property_name in ("scx", "scriptextensions")

        def property_matches(char: str) -> bool:
            codepoint = ord(char)
            extension_index = bisect_right(extension_starts, codepoint) - 1
            if (
                use_extensions
                and extension_index >= 0
                and codepoint <= extensions[extension_index][1]
            ):
                return script in extensions[extension_index][2]
            script_index = bisect_right(script_starts, codepoint) - 1
            actual_script = (
                scripts[script_index][2]
                if script_index >= 0 and codepoint <= scripts[script_index][1]
                else "Zzzz"
            )
            return actual_script == script

        predicate = property_matches
    return (lambda char: not predicate(char)) if negate else predicate


@lru_cache(maxsize=1)
def _unicode_script_data() -> tuple[
    dict[str, str],
    tuple[tuple[int, int, str], ...],
    tuple[int, ...],
    tuple[tuple[int, int, frozenset[str]], ...],
    tuple[int, ...],
]:
    data_directory = os.path.join(os.path.dirname(__file__), "resilex_unicode_data")

    def read_data(name: str) -> str:
        with open(os.path.join(data_directory, name), encoding="utf-8") as data_file:
            return data_file.read()

    def normalized(value: str) -> str:
        return value.casefold().replace("_", "").replace("-", "").replace(" ", "")

    aliases: dict[str, str] = {"unknown": "Zzzz", "zzzz": "Zzzz"}
    for line in read_data("PropertyValueAliases.txt").splitlines():
        fields = [field.strip() for field in line.split("#", 1)[0].split(";")]
        if len(fields) >= 3 and fields[0] == "sc":
            short_name = fields[1]
            for alias in fields[1:]:
                if alias:
                    aliases[normalized(alias)] = short_name

    def parse_range(value: str) -> tuple[int, int]:
        bounds = value.strip().split("..")
        start = int(bounds[0], 16)
        end = int(bounds[-1], 16)
        return start, end

    scripts: list[tuple[int, int, str]] = []
    for line in read_data("Scripts.txt").splitlines():
        fields = [field.strip() for field in line.split("#", 1)[0].split(";")]
        if len(fields) < 2:
            continue
        start, end = parse_range(fields[0])
        scripts.append((start, end, aliases[normalized(fields[1])]))
    scripts.sort()

    extensions: list[tuple[int, int, frozenset[str]]] = []
    for line in read_data("ScriptExtensions.txt").splitlines():
        fields = [field.strip() for field in line.split("#", 1)[0].split(";")]
        if len(fields) < 2:
            continue
        start, end = parse_range(fields[0])
        extensions.append(
            (start, end, frozenset(fields[1].split()))
        )
    extensions.sort()
    return (
        aliases,
        tuple(scripts),
        tuple(start for start, _, _ in scripts),
        tuple(extensions),
        tuple(start for start, _, _ in extensions),
    )


def _case_equivalent_characters(char: str, ascii_only: bool = False) -> set[str]:
    candidates: set[str] = set()
    pending = [char]
    while pending:
        candidate = pending.pop()
        if candidate in candidates:
            continue
        candidates.add(candidate)
        if ascii_only and not candidate.isascii():
            continue
        pending.extend(
            variant
            for variant in (
                candidate.lower(),
                candidate.upper(),
                candidate.title(),
                candidate.casefold(),
            )
            if len(variant) == 1
            and (not ascii_only or variant.isascii())
            and variant not in candidates
        )
        if not ascii_only:
            pending.extend(
                equivalent
                for equivalent in _EXTRA_CASES.get(candidate, ())
                if equivalent not in candidates
            )
    return candidates


def _case_insensitive_match(
    char: str, predicate: _Predicate, ascii_only: bool = False
) -> bool:
    candidates = _case_equivalent_characters(char, ascii_only)
    return any(len(candidate) == 1 and predicate(candidate) for candidate in candidates)


_EXTRA_CASES = {
    char: group
    for group in (
        ("i", "I", "İ", "ı"),
        ("s", "S", "ſ"),
        ("k", "K", "K"),
        ("ß", "ẞ"),
        ("σ", "Σ", "ς"),
        ("β", "Β", "ϐ"),
        ("в", "В", "ᲀ"),
        ("μ", "Μ", "µ"),
        ("å", "Å", "Å"),
    )
    for char in group
}


def _case_insensitive_equal(
    left: str,
    right: str,
    ascii_only: bool = False,
    locale_aware: bool = False,
) -> bool:
    if locale_aware:
        return _locale_case_insensitive_predicate(right)(left)
    return left == right or _case_insensitive_match(
        left, lambda candidate: candidate == right, ascii_only
    )


def _normalize_bounds(text: str, pos: int, endpos: int | None) -> tuple[int, int]:
    if not isinstance(pos, int):
        raise TypeError("pos must be an integer")
    if endpos is not None and not isinstance(endpos, int):
        raise TypeError("endpos must be an integer")
    return min(len(text), max(0, pos)), min(
        len(text), len(text) if endpos is None else endpos
    )


class _Parser:
    def __init__(
        self,
        pattern: str,
        flags: int = 0,
        limits: RegexLimits | None = None,
        bytes_pattern: bool = False,
    ):
        self.pattern = pattern
        self.index = 0
        self.flags = flags
        self.bytes_pattern = bytes_pattern
        self.limits = limits or RegexLimits()
        self.compiled_flags = flags
        self.has_scoped_flags = False
        self.group_count = 0
        self.group_depth = 0
        self.closed_groups: set[int] = set()
        self.backreferences: list[int] = []
        self.groupindex: dict[str, int] = {}
        self.global_flags_allowed = True
        self.last_atom_was_grouped = False

    def parse(self) -> object:
        try:
            node = self._alternative()
            self._skip_ignored()
            if self.index != len(self.pattern):
                raise RegexSyntaxError(f"Unexpected character at position {self.index}")
            for group in self.backreferences:
                if group > self.group_count:
                    raise RegexSyntaxError(
                        f"Backreference \\{group} refers to no capture group"
                    )
            return _resolve_conditional_groups(
                node, self.group_count, self.groupindex
            )
        except RegexSyntaxError as error:
            if error.pos is None:
                error.pos = self.index
            if error.pattern is None:
                error.pattern = self.pattern
            error.lineno = self.pattern.count("\n", 0, error.pos) + 1
            error.colno = error.pos - self.pattern.rfind("\n", 0, error.pos)
            raise

    def _alternative(self) -> object:
        options = [self._sequence()]
        self._skip_ignored()
        while self._peek() == "|":
            self.index += 1
            self.global_flags_allowed = False
            options.append(self._sequence())
            self._skip_ignored()
        if len(options) == 1:
            return options[0]
        return _Alternative(tuple(options))

    def _sequence(self) -> object:
        items = []
        self._skip_ignored()
        while self.index < len(self.pattern) and self.pattern[self.index] not in ")|":
            items.append(self._repetition())
            self._skip_ignored()
        if not items:
            return _Empty()
        if len(items) == 1:
            return items[0]
        return _Sequence(tuple(items))

    def _repetition(self) -> object:
        self._skip_ignored()
        node = self._atom()
        self._skip_ignored()
        minimum: int | None = None
        maximum: int | None = None
        char = self._peek()
        if isinstance(node, _GlobalFlags):
            if char in ("*", "+", "?") or (
                char == "{" and self._counted_quantifier_at_current_position()
            ):
                raise RegexSyntaxError(f"Nothing to repeat at position {self.index}")
            return node
        if isinstance(node, _Assertion) and not self.last_atom_was_grouped:
            if char in ("*", "+", "?") or (
                char == "{" and self._counted_quantifier_at_current_position()
            ):
                raise RegexSyntaxError(f"Nothing to repeat at position {self.index}")
            return node
        if char == "*":
            minimum, maximum = 0, None
            self.index += 1
        elif char == "+":
            minimum, maximum = 1, None
            self.index += 1
        elif char == "?":
            minimum, maximum = 0, 1
            self.index += 1
        elif char == "{":
            quantifier = self._counted_quantifier()
            if quantifier is not None:
                minimum, maximum = quantifier

        if minimum is not None:
            greedy = True
            if self._peek() == "?":
                greedy = False
                self.index += 1
            node = _Repeat(node, minimum, maximum, greedy)
            if self._peek() == "+":
                if not greedy:
                    raise RegexSyntaxError("Lazy possessive quantifiers are invalid")
                self.index += 1
                node = _Atomic(node)
            if self._peek() in ("*", "+", "?") or (
                self._peek() == "{"
                and self._counted_quantifier_at_current_position()
            ):
                raise RegexSyntaxError("Repeated quantifiers are not supported")
        return node

    def _counted_quantifier_at_current_position(self) -> bool:
        original_index = self.index
        try:
            return self._counted_quantifier() is not None
        finally:
            self.index = original_index

    def _counted_quantifier(self) -> tuple[int, int | None] | None:
        original_index = self.index
        self.index += 1
        start = self.index
        while (char := self._peek()) is not None and char.isdigit():
            self.index += 1
        has_minimum = self.index > start
        minimum = int(self.pattern[start : self.index]) if has_minimum else 0
        if self._peek() == "}":
            if not has_minimum:
                self.index = original_index
                return None
            self.index += 1
            maximum: int | None = minimum
        elif self._peek() == ",":
            self.index += 1
            upper_start = self.index
            while (char := self._peek()) is not None and char.isdigit():
                self.index += 1
            has_maximum = self.index > upper_start
            if self._peek() != "}":
                self.index = original_index
                return None
            maximum = (
                int(self.pattern[upper_start : self.index]) if has_maximum else None
            )
            self.index += 1
        else:
            self.index = original_index
            return None
        if maximum is not None and maximum < minimum:
            raise RegexSyntaxError(
                "Counted quantifier maximum is less than minimum",
                pos=original_index + 1,
            )
        if (
            minimum > self.limits.repeat
            or (maximum is not None and maximum > self.limits.repeat)
        ):
            raise RegexSyntaxError(
                f"Counted quantifiers are limited to {self.limits.repeat}"
            )
        return minimum, maximum

    def _atom(self) -> object:
        self._skip_ignored()
        self.last_atom_was_grouped = False
        if self.index >= len(self.pattern):
            raise RegexSyntaxError("Expected an expression")
        char = self.pattern[self.index]
        self.index += 1

        if char == "(":
            opening = self.index - 1
            if self._peek() == "?":
                self.index += 1
                extension = self._peek()
                if extension is None or extension not in "imsxauL-":
                    self.global_flags_allowed = False
                if extension == "#":
                    closing = self.pattern.find(")", self.index + 1)
                    if closing < 0:
                        raise RegexSyntaxError("Unclosed comment group")
                    self.index = closing + 1
                    return _Empty()
                if extension == ">":
                    self.index += 1
                    return _Atomic(self._group_body(opening))
                if extension == "(":
                    return self._conditional_group()
                if extension == ":":
                    self.index += 1
                    node = self._group_body(opening)
                    self.last_atom_was_grouped = True
                    return node
                if extension in ("=", "!"):
                    self.index += 1
                    node = self._group_body(opening)
                    return _Lookaround("ahead", extension == "=", node)
                if extension == "<":
                    self.index += 1
                    direction = self._peek()
                    if direction not in ("=", "!"):
                        raise RegexSyntaxError("Unsupported group extension")
                    self.index += 1
                    node = self._group_body(opening)
                    return _Lookaround("behind", direction == "=", node)
                if extension == "P":
                    self.index += 1
                    marker = self._peek()
                    if marker == "<":
                        self.index += 1
                        name = self._read_group_name(">")
                        if name in self.groupindex:
                            raise RegexSyntaxError(f"Duplicate group name {name!r}")
                        self.group_count += 1
                        group_index = self.group_count
                        self.groupindex[name] = group_index
                        node = self._group_body(opening)
                        self.closed_groups.add(group_index)
                        return _Group(group_index, node)
                    if marker == "=":
                        self.index += 1
                        name = self._read_group_name(")")
                        if name not in self.groupindex:
                            raise RegexSyntaxError(f"Unknown group name {name!r}")
                        group = self.groupindex[name]
                        if group not in self.closed_groups:
                            raise RegexSyntaxError(
                                f"Group {name!r} is referenced before it is closed"
                            )
                        self.backreferences.append(group)
                        return _Backreference(group)
                if extension is not None and extension in "imsxauL-":
                    node = self._flag_group(opening)
                    self.last_atom_was_grouped = not isinstance(node, _GlobalFlags)
                    return node
                raise RegexSyntaxError("Unsupported group extension")
            self.global_flags_allowed = False
            if self.group_depth >= self.limits.group_depth:
                raise RegexComplexityError(
                    f"Groups cannot be nested more than {self.limits.group_depth} levels"
                )
            self.group_count += 1
            group_index = self.group_count
            node = self._group_body(opening)
            self.closed_groups.add(group_index)
            return _Group(group_index, node)
        if char == ")":
            raise RegexSyntaxError(
                f"Unmatched ')' at position {self.index - 1}",
                pos=self.index - 1,
            )
        self.global_flags_allowed = False
        if char == ".":
            dotall = bool(self.flags & DOTALL)
            return _Character(
                lambda value, dotall=dotall: dotall or value != "\n", "."
            )
        if char == "^":
            return _Assertion("start_multiline" if self.flags & MULTILINE else "start")
        if char == "$":
            return _Assertion("end_multiline" if self.flags & MULTILINE else "end")
        if char == "[":
            return self._character_class(self.index - 1)
        if char == "\\":
            return self._escape()
        if char in "*+?":
            raise RegexSyntaxError(f"Quantifier {char!r} has no expression")
        if char == "{":
            self.index -= 1
            quantifier = self._counted_quantifier()
            if quantifier is not None:
                raise RegexSyntaxError("Counted quantifier has no expression")
            self.index += 1
        return _Character(self._literal_predicate(char), repr(char), char)

    def _group_body(self, opening: int | None = None) -> object:
        if self.group_depth >= self.limits.group_depth:
            raise RegexComplexityError(
                f"Groups cannot be nested more than {self.limits.group_depth} levels"
            )
        self.group_depth += 1
        try:
            node = self._alternative()
        finally:
            self.group_depth -= 1
        if self._peek() != ")":
            raise RegexSyntaxError("Unclosed group", pos=opening)
        self.index += 1
        return node

    def _flag_group(self, opening: int) -> object:
        added = 0
        removed = 0
        target = "add"
        flag_values = {
            "i": IGNORECASE,
            "m": MULTILINE,
            "s": DOTALL,
            "x": VERBOSE,
            "a": ASCII,
            "L": LOCALE,
            "u": UNICODE,
        }
        while (char := self._peek()) is not None:
            if char in flag_values:
                if target == "add":
                    added |= flag_values[char]
                else:
                    removed |= flag_values[char]
                self.index += 1
            elif char == "-" and target == "add":
                target = "remove"
                self.index += 1
            else:
                break

        delimiter = self._peek()
        if self.bytes_pattern and added & UNICODE:
            raise RegexSyntaxError("Unicode flag is not supported for bytes patterns")
        if not self.bytes_pattern and added & LOCALE:
            raise RegexSyntaxError("LOCALE flag is only supported for bytes patterns")
        if added & LOCALE and added & ASCII:
            raise RegexSyntaxError("ASCII and LOCALE flags are incompatible")
        if delimiter == ")":
            if not self.global_flags_allowed or not added or removed:
                raise RegexSyntaxError("Global flags must appear at the start of the pattern")
            self.index += 1
            self._apply_flags(added, removed)
            self.compiled_flags = self.flags
            return _GlobalFlags()
        if delimiter != ":" or (added | removed) == 0:
            raise RegexSyntaxError("Invalid inline flags")
        self.global_flags_allowed = False
        self.index += 1
        previous_flags = self.flags
        self._apply_flags(added, removed)
        self.has_scoped_flags = True
        try:
            return self._group_body(opening)
        finally:
            self.flags = previous_flags

    def _conditional_group(self) -> object:
        self.index += 1
        start = self.index
        while (char := self._peek()) is not None and char != ")":
            self.index += 1
        if self._peek() != ")":
            raise RegexSyntaxError("Unclosed conditional group")
        reference = self.pattern[start : self.index]
        self.index += 1
        group: int | str = (
            int(reference)
            if reference.isdigit()
            else self.groupindex.get(reference, reference)
        )
        yes = self._sequence()
        if self._peek() == "|":
            self.index += 1
            no = self._sequence()
        else:
            no = _Empty()
        if self._peek() != ")":
            raise RegexSyntaxError("Unclosed conditional group")
        self.index += 1
        return _Conditional(group, yes, no)

    def _apply_flags(self, added: int, removed: int) -> None:
        self.flags = (self.flags | added) & ~removed
        if self.flags & LOCALE:
            self.flags &= ~(ASCII | UNICODE)
        elif self.bytes_pattern:
            self.flags |= ASCII
        elif self.flags & ASCII:
            self.flags &= ~UNICODE
        elif self.flags & UNICODE:
            self.flags &= ~ASCII

    def _literal_predicate(self, expected: str) -> _Predicate:
        if self.flags & IGNORECASE:
            if self.flags & LOCALE:
                return _locale_case_insensitive_predicate(expected)
            return lambda value: _case_insensitive_match(
                value,
                lambda candidate: candidate == expected,
                bool(self.flags & ASCII),
            )
        return lambda value: value == expected

    def _skip_ignored(self) -> None:
        if not self.flags & VERBOSE:
            return
        while self.index < len(self.pattern):
            char = self.pattern[self.index]
            if char.isspace():
                self.index += 1
            elif char == "#":
                newline = self.pattern.find("\n", self.index)
                self.index = len(self.pattern) if newline < 0 else newline + 1
            else:
                return

    def _read_group_name(self, terminator: str) -> str:
        start = self.index
        while (char := self._peek()) is not None and char != terminator:
            self.index += 1
        if self._peek() != terminator:
            raise RegexSyntaxError("Unclosed group name")
        name = self.pattern[start : self.index]
        self.index += 1
        if (
            not name
            or not (name[0].isascii() and (name[0].isalpha() or name[0] == "_"))
            or any(
                not (char.isascii() and (char.isalnum() or char == "_"))
                for char in name[1:]
            )
        ):
            raise RegexSyntaxError(f"Invalid group name {name!r}")
        return name

    def _character_class(self, opening: int) -> object:
        negate = self._peek() == "^"
        if negate:
            self.index += 1
        predicates: list[_Predicate] = []
        labels: list[str] = []
        first = True

        while self.index < len(self.pattern):
            if self._peek() == "]" and not first:
                self.index += 1
                if not predicates:
                    raise RegexSyntaxError("Empty character classes are not supported")

                def matches(value: str) -> bool:
                    found = any(predicate(value) for predicate in predicates)
                    return not found if negate else found

                label = "[" + ("^" if negate else "") + "".join(labels) + "]"
                return _Character(matches, label)

            first = False
            left = self._class_item()
            if (
                isinstance(left, str)
                and self._peek() == "-"
                and self.index + 1 < len(self.pattern)
                and self.pattern[self.index + 1] != "]"
            ):
                self.index += 1
                right = self._class_item()
                if not isinstance(right, str):
                    raise RegexSyntaxError("Character-class range endpoints must be literals")
                if ord(left) > ord(right):
                    raise RegexSyntaxError("Character-class range is in reverse order")
                range_match = lambda value, low=left, high=right: low <= value <= high
                if self.flags & IGNORECASE and self.flags & LOCALE:
                    predicates.append(_locale_range_predicate(left, right))
                elif self.flags & IGNORECASE:
                    predicates.append(
                        lambda value, predicate=range_match: _case_insensitive_match(
                            value, predicate, bool(self.flags & ASCII)
                        )
                    )
                else:
                    predicates.append(range_match)
                labels.append(f"{left}-{right}")
            elif isinstance(left, str):
                literal_match = self._literal_predicate(left)
                predicates.append(literal_match)
                labels.append(left)
            else:
                predicates.append(left[0])
                labels.append(left[1])

        raise RegexSyntaxError("Unclosed character class", pos=opening)

    def _class_item(self) -> str | tuple[_Predicate, str]:
        char = self.pattern[self.index]
        self.index += 1
        if char != "\\":
            return char
        if self.index >= len(self.pattern):
            raise RegexSyntaxError("Trailing escape")
        escaped = self.pattern[self.index]
        self.index += 1
        if escaped in "pP":
            predicate, label = self._unicode_property_escape(escaped)
            return predicate, label
        if escaped.lower() in ("d", "w", "s"):
            return _category_predicate(escaped, self.flags), "\\" + escaped
        if escaped == "b":
            return "\b"
        numeric = self._numeric_escape(escaped)
        if numeric is not None:
            return numeric
        if escaped.isdigit():
            return self._octal_escape(escaped)
        if escaped.isalpha() and escaped not in "nrtfvab":
            raise RegexSyntaxError(
                f"Unsupported escape \\{escaped} in character class",
                pos=self.index - 2,
            )
        literal = self._escaped_literal(escaped)
        return literal

    def _escape(self) -> object:
        escape_position = self.index - 1
        if self.index >= len(self.pattern):
            raise RegexSyntaxError("Trailing escape", pos=escape_position)
        escaped = self.pattern[self.index]
        self.index += 1
        if escaped in "pP":
            predicate, label = self._unicode_property_escape(escaped)
            return _Character(predicate, label)
        if escaped in "dDwWsS":
            return _Character(
                _category_predicate(escaped, self.flags), "\\" + escaped
            )
        if escaped in "bB":
            boundary = "word_boundary" if escaped == "b" else "not_word_boundary"
            if self.flags & LOCALE:
                boundary += "_locale"
            elif self.flags & ASCII:
                boundary += "_ascii"
            return _Assertion(boundary)
        if escaped == "A":
            return _Assertion("absolute_start")
        if escaped == "Z":
            return _Assertion("absolute_end")
        numeric = self._numeric_escape(escaped)
        if numeric is not None:
            return _Character(self._literal_predicate(numeric), "\\" + escaped, numeric)
        if escaped.isdigit():
            octal_start = self.index
            if escaped in "01234567":
                digits = escaped
                while (
                    len(digits) < 3
                    and (next_char := self._peek()) is not None
                    and next_char in "01234567"
                ):
                    digits += next_char
                    self.index += 1
                if escaped == "0" or len(digits) == 3:
                    literal = self._octal_value(digits)
                    return _Character(
                        self._literal_predicate(literal), "\\" + digits, literal
                    )
                self.index = octal_start
            digits = escaped
            while len(digits) < 2 and (next_char := self._peek()) is not None and next_char.isdigit():
                digits += next_char
                self.index += 1
            group = int(digits)
            if group not in self.closed_groups:
                raise RegexSyntaxError(
                    f"Backreference \\{group} refers to no prior capture group",
                    pos=escape_position,
                )
            self.backreferences.append(group)
            return _Backreference(group)
        if escaped.isalpha() and escaped not in "nrtfvab":
            raise RegexSyntaxError(
                f"Unsupported escape \\{escaped}", pos=escape_position
            )
        literal = self._escaped_literal(escaped)
        return _Character(
            self._literal_predicate(literal),
            "\\" + escaped,
            literal,
        )

    def _unicode_property_escape(self, escaped: str) -> tuple[_Predicate, str]:
        if self.bytes_pattern:
            raise RegexSyntaxError(
                f"Unicode properties are not supported in bytes patterns: \\{escaped}"
            )
        if self._peek() != "{":
            raise RegexSyntaxError(f"Missing {{ in \\{escaped} property escape")
        closing = self.pattern.find("}", self.index + 1)
        if closing < 0:
            raise RegexSyntaxError(f"Missing }} in \\{escaped} property escape")
        expression = self.pattern[self.index + 1 : closing]
        try:
            predicate = _unicode_category_predicate(expression, escaped == "P")
        except ValueError as error:
            raise RegexSyntaxError(str(error)) from None
        self.index = closing + 1
        return predicate, f"\\{escaped}{{{expression}}}"

    def _numeric_escape(self, escaped: str) -> str | None:
        if self.bytes_pattern and escaped in ("u", "U", "N"):
            raise RegexSyntaxError(
                f"Unicode escapes are not supported in bytes patterns: \\{escaped}"
            )
        widths = {"x": 2, "u": 4, "U": 8}
        if escaped in widths:
            width = widths[escaped]
            digits = self.pattern[self.index : self.index + width]
            if len(digits) != width or any(
                char not in "0123456789abcdefABCDEF" for char in digits
            ):
                raise RegexSyntaxError(f"Invalid \\{escaped} escape")
            self.index += width
            value = int(digits, 16)
            if value > 0x10FFFF:
                raise RegexSyntaxError("Unicode escape is out of range")
            return chr(value)
        if escaped == "N":
            if self._peek() != "{":
                raise RegexSyntaxError("Missing { in \\N character escape")
            closing = self.pattern.find("}", self.index + 1)
            if closing < 0:
                raise RegexSyntaxError("Missing } in \\N character escape")
            name = self.pattern[self.index + 1 : closing]
            try:
                value = unicodedata.lookup(name)
            except KeyError:
                raise RegexSyntaxError(f"Unknown Unicode character name {name!r}") from None
            self.index = closing + 1
            return value
        return None

    def _octal_escape(self, first: str) -> str:
        if first not in "01234567":
            raise RegexSyntaxError(f"Invalid escape \\{first} in character class")
        digits = first
        while (
            len(digits) < 3
            and (next_char := self._peek()) is not None
            and next_char in "01234567"
        ):
            digits += next_char
            self.index += 1
        return self._octal_value(digits)

    @staticmethod
    def _octal_value(digits: str) -> str:
        value = int(digits, 8)
        if value > 0xFF:
            raise RegexSyntaxError("Octal escape value outside of range 0-0o377")
        return chr(value)

    @staticmethod
    def _escaped_literal(char: str) -> str:
        return {
            "n": "\n",
            "r": "\r",
            "t": "\t",
            "f": "\f",
            "v": "\v",
            "a": "\a",
            "b": "\b",
        }.get(char, char)

    def _peek(self) -> str | None:
        return self.pattern[self.index] if self.index < len(self.pattern) else None


@dataclass()
class _State:
    epsilon: list[tuple[str, int, int | None]] = field(default_factory=list)
    transitions: list[tuple[_Predicate, str, int]] = field(default_factory=list)
    assertions: list[tuple[str, int]] = field(default_factory=list)
    backreferences: list[tuple[int, int]] = field(default_factory=list)


@dataclass(frozen=True)
class _Fragment:
    start: int
    end: int


@dataclass(frozen=True, repr=False)
class Match:
    """Represents a successful regular expression match.

    A Match object provides access to the matched text, captured groups,
    their positions, and replacement-template expansion.

    Groups can be accessed using either numerical indices or assigned
    names. Group 0 represents the entire match; groups starting at 1
    represent capturing groups.

    Match objects are returned by operations such as search(), match(),
    and fullmatch().
    """

    string: str | bytes
    _start: int
    _end: int
    captures: tuple[tuple[int, int] | None, ...] = ()
    lastindex: int | None = None
    groupindex: dict[str, int] = field(default_factory=dict)
    re: Regex | None = None
    pos: int = 0
    endpos: int | None = None
    groups_count: int | None = None

    @classmethod
    def __class_getitem__(cls, item: object) -> type[Match]:
        del item
        return cls

    def __post_init__(self) -> None:
        if self.endpos is None:
            object.__setattr__(self, "endpos", len(self.string))
        if self.groups_count is None:
            object.__setattr__(self, "groups_count", len(self.captures))

    def __repr__(self) -> str:
        return (
            f"<re.Match object; span={self.span()!r}, "
            f"match={self.group()!r}>"
        )

    @property
    def lastgroup(self) -> str | None:
        """Return the name of the last matched capturing group.

        Returns None if no capturing group participated in the match
        or if the last matched group has no assigned name.
        """
        if self.lastindex is None:
            return None
        return next(
            (name for name, index in self.groupindex.items() if index == self.lastindex),
            None,
        )
    def group(
        self,
        index: int | str = 0,
        second: int | str | None = None,
        *indices: int | str,
    ) -> str | bytes | None | tuple[str | bytes | None, ...]:
        """Return the matched text for one or more groups.

        Accepts group numbers or names. Group 0 returns the entire match.
        If multiple groups are supplied, returns a tuple of their values.
        Unmatched groups return None.

        Raises:
            IndexError: If a requested group does not exist.
        """
        selected = (index,) if second is None else (index, second, *indices)
        values = tuple(self._value(index) for index in selected)
        return values[0] if len(values) == 1 else values

    def groups(
        self, default: str | bytes | None = None
    ) -> tuple[str | bytes | None, ...]:
        """Return all captured groups as a tuple.

        Groups are returned in numerical order, excluding group 0.
        Unmatched groups are replaced with the supplied default value.

        Args:
            default: Value used for groups that did not participate
                in the match. Defaults to None.
        """
        return tuple(
            default if span is None or span[1] < 0 else self.string[span[0] : span[1]]
            for span in self.captures[: self.groups_count or 0]
        )

    def groupdict(
        self, default: str | bytes | None = None
    ) -> dict[str, str | bytes | None]:
        """Return a dictionary mapping group names to their values.

        Groups that did not participate in the match are replaced with the
        supplied default value.

        Args:
            default: Value used for groups that did not participate
                in the match. Defaults to None.

        Returns:
            A dictionary mapping group names to their values.
        """
        result: dict[str, str | bytes | None] = {}
        for name, index in self.groupindex.items():
            value = self._value(index)
            result[name] = value if value is not None else default
        return result

    def start(self, index: int | str = 0) -> int:
        """Return the starting position of the matched group.

        Args:
            index: Group number or name. Defaults to 0 (the entire match).

        Returns:
            The starting index, or -1 if the group did not participate.
        """
        return self.span(index)[0]

    def end(self, index: int | str = 0) -> int:
        """Return the ending position of the matched group.

        Args:
            index: Group number or name. Defaults to 0 (the entire match).

        Returns:
            The ending index, or -1 if the group did not participate.
        """
        return self.span(index)[1]

    @property
    def regs(self) -> tuple[tuple[int, int], ...]:
        """Return the spans of the entire match and all capturing groups.

        The first tuple represents the entire match (group 0).
        Subsequent tuples represent capturing groups in numerical order.

        Unmatched groups have a span of (-1, -1).
        """
        return (self.span(),) + tuple(
            self.span(index)
            for index in range(1, (self.groups_count or 0) + 1)
        )

    def span(self, index: int | str = 0) -> tuple[int, int]:
        """Return the start and end positions of a matched group.

        Args:
            index: Group number or name. Defaults to 0 (the entire match).

        Returns:
            A (start, end) tuple. Returns (-1, -1) if the group did
            not participate in the match.
        """
        if index == 0:
            return self._start, self._end
        span = self._capture(index)
        return (-1, -1) if span is None or span[1] < 0 else span

    def __getitem__(self, index: int | str) -> str | bytes | None:
        """Return a captured value using its group number or name.

        Supports square-bracket access, such as match[1] or
        match["name"]. Group 0 returns the entire match.

        Returns None if the requested group did not participate.

        Raises:
            IndexError: If the requested group does not exist.
        """
        return self._value(index)

    def expand(self, template: str | bytes) -> str | bytes:
        """Return the result of expanding a replacement template.

        Group references in the template are replaced with the
        corresponding captured text.

        Args:
            template: Replacement template containing group references.

        Returns:
            The expanded template, preserving the input type (str or bytes).
        """
        if isinstance(template, bytes):
            return _expand_template(template.decode("latin-1"), self).encode("latin-1")
        return _expand_template(template, self)

    def _value(self, index: int | str) -> str | bytes | None:
        if index == 0:
            return self.string[self._start : self._end]
        span = self._capture(index)
        if span is None or span[1] < 0:
            return None
        return self.string[span[0] : span[1]]

    def _capture(self, index: int | str) -> tuple[int, int] | None:
        if isinstance(index, str):
            try:
                index = self.groupindex[index]
            except KeyError as error:
                raise IndexError(f"no such group: {error.args[0]}") from None
        if index < 0 or index > (self.groups_count or 0):
            raise IndexError("no such group")
        if index == 0:
            return self._start, self._end
        return self.captures[index - 1]


def _parse_template(
    template: str, groupindex: dict[str, int], groups_count: int
) -> list[tuple[str, str | int]]:
    if not isinstance(template, str):
        raise TypeError("replacement must be a string")
    parts: list[tuple[str, str | int]] = []
    index = 0
    while index < len(template):
        char = template[index]
        if char != "\\":
            parts.append(("text", char))
            index += 1
            continue
        index += 1
        if index >= len(template):
            raise RegexSyntaxError("Trailing escape in replacement")
        escaped = template[index]
        if escaped == "\\":
            parts.append(("text", "\\"))
            index += 1
        elif escaped == "g":
            if index + 1 >= len(template) or template[index + 1] != "<":
                raise RegexSyntaxError("Expected '<' after \\g in replacement")
            end = template.find(">", index + 2)
            if end < 0:
                raise RegexSyntaxError("Unclosed group reference in replacement")
            reference = template[index + 2 : end]
            group: int | str
            if reference.isdigit():
                group = int(reference)
            else:
                group = reference
            if (
                isinstance(group, int)
                and not 0 <= group <= groups_count
                or isinstance(group, str)
                and group not in groupindex
            ):
                raise RegexSyntaxError(f"invalid group reference {reference!r}")
            parts.append(("group", group))
            index = end + 1
        elif escaped in "0123456789":
            if escaped == "0":
                end = index + 1
                while (
                    end < len(template)
                    and end < index + 3
                    and template[end] in "01234567"
                ):
                    end += 1
                parts.append(("text", chr(int(template[index:end], 8))))
                index = end
            else:
                end = index + 1
                while end < len(template) and end < index + 2 and template[end].isdigit():
                    end += 1
                if (
                    end < len(template)
                    and end == index + 2
                    and template[index] in "01234567"
                    and template[index + 1] in "01234567"
                    and template[end] in "01234567"
                ):
                    parts.append(("text", chr(int(template[index : end + 1], 8))))
                    index = end + 1
                    continue
                reference = template[index:end]
                group = int(reference)
                if group > groups_count:
                    raise RegexSyntaxError(
                        f"invalid group reference {reference!r}"
                    )
                parts.append(("group", group))
                index = end
        else:
            escapes = {
                "a": "\a",
                "b": "\b",
                "f": "\f",
                "n": "\n",
                "r": "\r",
                "t": "\t",
                "v": "\v",
            }
            if escaped.isalpha() and escaped not in escapes:
                raise RegexSyntaxError(
                    f"bad character in group name or number: {escaped!r}"
                )
            parts.append(("text", escapes.get(escaped, escaped)))
            index += 1
    return parts


def _render_template(
    parts: list[tuple[str, str | int]], match: Match
) -> str:
    output: list[str] = []
    for kind, value in parts:
        if kind == "text":
            if not isinstance(value, str):
                raise RuntimeError("Template text must be a string")
            output.append(value)
            continue
        group_value = match._value(value)
        if isinstance(group_value, bytes):
            group_value = group_value.decode("latin-1")
        output.append("" if group_value is None else group_value)
    return "".join(output)


def _expand_template(template: str, match: Match) -> str:
    return _render_template(
        _parse_template(template, match.groupindex, match.groups_count or 0), match
    )


_Thread = tuple[
    int,
    int,
    tuple[tuple[int, int] | None, ...],
    int | None,
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
    tuple[int, ...],
]
_StateKey = (
    int
    | tuple[int, tuple[tuple[int, int], ...]]
    | tuple[int, tuple[tuple[int, int], ...], bool]
    | tuple[
        int, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]
    ]
)
_DFAState = tuple[frozenset[int], bool, bool, bool, bool, str | None]


def _is_word_char(char: str) -> bool:
    return char == "_" or char.isalnum()


def _assertion_matches(
    kind: str, position: int, text: str, endpos: int | None = None
) -> bool:
    limit = len(text) if endpos is None else endpos
    if kind in ("start", "absolute_start"):
        return position == 0
    if kind == "start_multiline":
        return position == 0 or text[position - 1] == "\n"
    if kind == "absolute_end":
        return position == limit
    if kind == "end":
        return position == limit or (
            position == limit - 1 and text[position] == "\n"
        )
    if kind == "end_multiline":
        return position == limit or position < limit and text[position] == "\n"
    if kind in (
        "word_boundary",
        "not_word_boundary",
        "word_boundary_ascii",
        "not_word_boundary_ascii",
        "word_boundary_locale",
        "not_word_boundary_locale",
    ):
        ascii_only = kind.endswith("_ascii")
        locale_aware = kind.endswith("_locale")
        base_kind = kind.removesuffix("_ascii").removesuffix("_locale")
        if locale_aware:
            is_word = _locale_character_predicate(rb"\w")
        elif ascii_only:
            is_word = lambda char: char == "_" or char.isascii() and char.isalnum()
        else:
            is_word = _is_word_char
        left = position > 0 and is_word(text[position - 1])
        right = position < limit and is_word(text[position])
        boundary = left != right
        if base_kind == "not_word_boundary" and position == 0 and limit == 0:
            return False
        return boundary if base_kind == "word_boundary" else not boundary
    return False


@dataclass(frozen=True)
class _CharInstruction:
    predicate: _Predicate
    target: int


@dataclass(frozen=True)
class _AssertInstruction:
    kind: str
    target: int


@dataclass(frozen=True)
class _BackrefInstruction:
    group: int
    target: int


@dataclass(frozen=True)
class _SaveInstruction:
    group: int
    target: int
    is_start: bool


@dataclass()
class _SplitInstruction:
    branches: tuple[int, ...]
    repeat_id: int | None = None
    body_branch: int | None = None
    unbounded_repeat: bool = False


@dataclass(frozen=True)
class _RepeatStartInstruction:
    repeat_id: int
    target: int


@dataclass(frozen=True)
class _RepeatEndInstruction:
    repeat_id: int
    target: int


@dataclass(frozen=True)
class _LookInstruction:
    kind: str
    positive: bool
    entry: int
    program: tuple[_Instruction, ...]
    width: tuple[int, int | None] | None
    target: int


@dataclass(frozen=True)
class _GroupConditionInstruction:
    group: int
    yes: int
    no: int


@dataclass(frozen=True)
class _MatchInstruction:
    pass


_Instruction = (
    _CharInstruction
    | _AssertInstruction
    | _BackrefInstruction
    | _SaveInstruction
    | _SplitInstruction
    | _RepeatStartInstruction
    | _RepeatEndInstruction
    | _LookInstruction
    | _GroupConditionInstruction
    | _MatchInstruction
)


class Regex:
    """Compile and execute regular expressions.

    Regex objects represent compiled patterns and can be reused for
    searching, matching, finding multiple matches, and substitutions.

    The engine uses an ordered Thompson NFA for supported patterns
    and a bounded backtracking virtual machine for features requiring
    backtracking.

    Patterns may be strings or bytes. Matching behavior can be
    configured using flags and resource limits.

    Args:
        pattern: Regular expression pattern to compile.
        flags: Optional compilation flags.
        limits: Optional resource limits controlling compilation
            and matching complexity.

    Raises:
        TypeError: If the pattern, flags, or limits have invalid types.
        ValueError: If incompatible or unsupported flags are supplied.
        RegexSyntaxError: If the pattern contains invalid syntax.
        RegexComplexityError: If a configured complexity limit is exceeded.
    """

    def __init__(
        self, pattern: str | bytes, flags: int = 0, limits: RegexLimits | None = None
    ):
        if not isinstance(pattern, (str, bytes)):
            raise TypeError("pattern must be a string or bytes")
        if not isinstance(flags, int):
            raise TypeError("flags must be an integer")
        supported_flags = (
            IGNORECASE
            | MULTILINE
            | DOTALL
            | VERBOSE
            | ASCII
            | UNICODE
            | DEBUG
            | LOCALE
        )
        if flags & ~supported_flags:
            raise ValueError(f"Unsupported regex flags: {flags & ~supported_flags!r}")
        self._bytes_mode = isinstance(pattern, bytes)
        if self._bytes_mode and flags & UNICODE:
            raise ValueError("Unicode matching is not supported for bytes patterns")
        if flags & LOCALE and not self._bytes_mode:
            raise ValueError("LOCALE flag is only supported for bytes patterns")
        if flags & LOCALE and flags & (ASCII | UNICODE):
            raise ValueError("LOCALE is incompatible with ASCII and UNICODE flags")
        if flags & ASCII and flags & UNICODE:
            raise ValueError("ASCII and UNICODE flags are incompatible")
        self.limits = limits or RegexLimits()
        if not isinstance(self.limits, RegexLimits):
            raise TypeError("limits must be a RegexLimits instance")
        if len(pattern) > self.limits.pattern_length:
            raise RegexComplexityError(
                f"Patterns are limited to {self.limits.pattern_length} characters"
            )
        self.pattern = pattern
        pattern_text = (
            pattern.decode("latin-1") if isinstance(pattern, bytes) else pattern
        )
        parser_flags = flags
        if self._bytes_mode and not flags & LOCALE:
            parser_flags |= ASCII
        elif not self._bytes_mode and not parser_flags & ASCII:
            parser_flags |= UNICODE
        parser = _Parser(pattern_text, parser_flags, self.limits, self._bytes_mode)
        try:
            self._syntax = parser.parse()
        except RegexSyntaxError as error:
            error.pattern = pattern
            raise
        self.flags = parser.compiled_flags
        if self._bytes_mode and not flags & ASCII:
            self.flags &= ~ASCII
        self.groups = parser.group_count
        self.groupindex = dict(parser.groupindex)
        atomic_nodes = _atomic_nodes(self._syntax)
        self._atomic_group_indices = {
            id(node): self.groups + index + 1
            for index, node in enumerate(atomic_nodes)
        }
        self._capture_count = self.groups + len(atomic_nodes)
        self._has_conditionals = _contains_conditional(self._syntax)
        self._has_lookarounds = _contains_lookaround(self._syntax)
        self._dfa_lookarounds: dict[str, _Lookaround] = {}
        self._simple_backreference_tail = self._find_simple_backreference_tail(
            self._syntax
        )
        self._literal = (
            _literal_text(self._syntax)
            if self.groups == 0
            and not parser.has_scoped_flags
            and not self.flags & IGNORECASE
            else None
        )
        self._required_literals = (
            set() if parser.has_scoped_flags else _guaranteed_literals(self._syntax)
        )
        self._has_backreferences = bool(parser.backreferences)
        self._requires_vm = (
            self._has_backreferences
            or bool(atomic_nodes)
            or self._has_conditionals
            or _contains_variable_width_lookbehind(self._syntax)
            or _contains_nullable_alternative_repeat(self._syntax)
        )
        self._states: list[_State] = []
        fragment = self._compile(self._syntax)
        self._start = fragment.start
        self._accept = fragment.end
        self._first_char_predicate = _first_character_predicate(self._syntax)
        self._first_char_literals = (
            None
            if parser.has_scoped_flags
            or self.flags & LOCALE
            or _nullable(self._syntax)
            else _first_character_literals(self._syntax)
        )
        self._lazy_dfa_enabled = (
            self.groups == 0
            and not self._has_backreferences
            and not self._requires_vm
            and not any(state.assertions for state in self._states)
        )
        self._lazy_dfa_lock = RLock()
        self._lazy_dfa_subsets: list[frozenset[int]] = []
        self._lazy_dfa_indices: dict[frozenset[int], int] = {}
        self._lazy_dfa_transitions: dict[tuple[int, str], int] = {}
        self._lazy_dfa_characters: set[str] = set()
        if self._lazy_dfa_enabled:
            initial = self._lazy_dfa_closure({self._start})
            self._lazy_dfa_subsets.append(initial)
            self._lazy_dfa_indices[initial] = 0
        self._vm: list[_Instruction] | None = None
        self._vm_entry = 0
        if self._literal is None or self._requires_vm:
            self._vm_entry, self._vm = self._compile_vm()
        has_lazy_unbounded_repeat = self._vm is not None and any(
            isinstance(item, _SplitInstruction)
            and item.repeat_id is not None
            and item.body_branch == 1
            and item.unbounded_repeat
            for item in self._vm
        )
        unbounded_repeat_count = (
            sum(
                1
                for item in self._vm
                if isinstance(item, _SplitInstruction)
                and item.repeat_id is not None
                and item.unbounded_repeat
            )
            if self._vm is not None
            else 0
        )
        self._preserve_repeat_history = (
            self._capture_count > 0
            or (
                self._vm is not None
                and any(isinstance(item, _AssertInstruction) for item in self._vm)
                and has_lazy_unbounded_repeat
            )
            or (has_lazy_unbounded_repeat and unbounded_repeat_count > 1)
        )
        if flags & DEBUG:
            print(self.to_dot())

    @classmethod
    def __class_getitem__(cls, item: object) -> type[Regex]:
        del item
        return cls

    def __repr__(self) -> str:
        flags = self.flags & ~UNICODE
        flag_repr = "" if not flags else f", {RegexFlag(flags)!r}"
        return f"re.compile({self.pattern!r}{flag_repr})"

    @staticmethod
    def _find_simple_backreference_tail(
        node: object,
    ) -> tuple[int, _Predicate] | None:
        items = node.items if isinstance(node, _Sequence) else (node,)
        if (
            len(items) != 3
            or not isinstance(items[0], _Group)
            or not isinstance(items[1], _Backreference)
            or items[1].group != items[0].index
            or not isinstance(items[2], _Assertion)
            or items[2].kind != "end"
        ):
            return None
        repeated = items[0].node
        if (
            not isinstance(repeated, _Repeat)
            or repeated.minimum != 1
            or repeated.maximum is not None
            or not repeated.greedy
            or not isinstance(repeated.node, _Character)
        ):
            return None
        return items[0].index, repeated.node.predicate

    def _simple_backreference_match(
        self,
        text: str,
        start: int,
        limit: int,
        match_pos: int,
        match_string: str | bytes,
        *,
        fullmatch: bool = False,
    ) -> Match | None:
        if self._simple_backreference_tail is None:
            return None
        group, predicate = self._simple_backreference_tail
        ends = (limit,) if fullmatch else (
            (limit, limit - 1)
            if limit > 0 and text[limit - 1] == "\n"
            else (limit,)
        )
        for end in ends:
            length = end - start
            if length <= 0 or length % 2:
                continue
            capture_end = start + length // 2
            captured = text[start:capture_end]
            candidate = text[capture_end:end]
            equal = (
                all(
                    _case_insensitive_equal(
                        left,
                        right,
                        bool(self.flags & ASCII),
                        bool(self.flags & LOCALE),
                    )
                    for left, right in zip(captured, candidate)
                )
                if self.flags & IGNORECASE
                else captured == candidate
            )
            if all(predicate(char) for char in captured) and equal:
                captures: list[tuple[int, int] | None] = [None] * self.groups
                captures[group - 1] = (start, capture_end)
                return self._make_match(
                    text,
                    start,
                    end,
                    tuple(captures),
                    group,
                    match_pos,
                    limit,
                    match_string,
                )
        return None

    def __copy__(self) -> Regex:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Regex:
        memo[id(self)] = self
        return self

    def __reduce__(
        self,
    ) -> tuple[Callable[..., Regex], tuple[str | bytes, int, RegexLimits]]:
        return Regex, (self.pattern, self.flags, self.limits)

    def _prepare_text(self, text: str | bytes) -> tuple[str, str | bytes]:
        if self._bytes_mode:
            if not isinstance(text, (bytes, bytearray, memoryview)):
                raise TypeError("cannot use a bytes pattern on a string")
            byte_text = bytes(text)
            return byte_text.decode("latin-1"), byte_text
        if not isinstance(text, str):
            raise TypeError("cannot use a string pattern on a bytes-like object")
        return text, text

    def _lazy_dfa_closure(self, states: set[int]) -> frozenset[int]:
        reached = set(states)
        pending = list(states)
        while pending:
            state_id = pending.pop()
            for kind, target, _ in self._states[state_id].epsilon:
                if kind == "epsilon" and target not in reached:
                    reached.add(target)
                    pending.append(target)
        return frozenset(reached)

    def _lazy_dfa_fullmatch(self, text: str, pos: int, endpos: int) -> bool | None:
        state_id = 0
        for char in text[pos:endpos]:
            key = (state_id, char)
            with self._lazy_dfa_lock:
                target_id = self._lazy_dfa_transitions.get(key)
                if target_id is None:
                    if char not in self._lazy_dfa_characters:
                        if len(self._lazy_dfa_characters) >= self.limits.dfa_alphabet:
                            return None
                        self._lazy_dfa_characters.add(char)
                    sources = self._lazy_dfa_subsets[state_id]
                    destinations = {
                        target
                        for source in sources
                        for predicate, _, target in self._states[source].transitions
                        if predicate(char)
                    }
                    subset = self._lazy_dfa_closure(destinations)
                    target_id = self._lazy_dfa_indices.get(subset)
                    if target_id is None:
                        if len(self._lazy_dfa_subsets) >= self.limits.dfa_states:
                            return None
                        target_id = len(self._lazy_dfa_subsets)
                        self._lazy_dfa_indices[subset] = target_id
                        self._lazy_dfa_subsets.append(subset)
                    self._lazy_dfa_transitions[key] = target_id
            state_id = target_id
        return self._accept in self._lazy_dfa_subsets[state_id]

    def search(
        self, text: str | bytes, pos: int = 0, endpos: int | None = None
    ) -> Match | None:
        """Search the input for the first matching substring.

        The search begins at ``pos`` and stops at ``endpos``.
        If omitted, ``endpos`` defaults to the end of the input.

        Args:
            text: String or bytes input to search.
            pos: Starting position of the search. Defaults to 0.
            endpos: Optional ending position of the search.

        Returns:
            A Match object for the first match, or None if no match exists.

        Raises:
            TypeError: If the input type does not match the pattern type.
        """
        text, match_string = self._prepare_text(text)
        pos, endpos = _normalize_bounds(text, pos, endpos)
        return self._search_from(
            text, pos, endpos=endpos, match_string=match_string
        )

    def explain(
        self, text: str | bytes, pos: int = 0, endpos: int | None = None
    ) -> str:
        """Explain how the engine processes a search operation.

        Provides a textual trace describing the selected matching
        engine, search bounds, candidate positions, and matching
        decisions.

        This is intended for debugging and understanding why a
        pattern matched or failed to match.

        Args:
            text: String or bytes input to search.
            pos: Starting position of the search. Defaults to 0.
            endpos: Optional ending position of the search.

        Returns:
            A textual explanation of the search process.

        Raises:
            TypeError: If the input type does not match the pattern type.
        """
        text, match_string = self._prepare_text(text)
        pos, limit = _normalize_bounds(text, pos, endpos)
        events = [
            f"Pattern: {self.pattern!r}",
            f"Input length: {len(text)}",
            f"Search bounds: [{pos}, {limit})",
            "Matcher: "
            + (
                "backtracking VM"
                if self._requires_vm
                else "ordered Thompson NFA"
                if self._literal is None
                else "literal search"
            ),
        ]
        match = self._search_from(
            text,
            pos,
            endpos=limit,
            match_string=match_string,
            trace=events,
        )
        if match is None:
            events.append("Result: no match.")
        else:
            events.append(f"Result: matched span {match.span()}.")
            for name, value in match.groupdict().items():
                events.append(f"  Group {name!r}: {value!r}")
            if self.groups:
                for index, value in enumerate(match.groups(), 1):
                    events.append(f"  Group {index}: {value!r}")
        return "\n".join(events)

    def _make_match(
        self,
        text: str,
        start: int,
        end: int,
        captures: tuple[tuple[int, int] | None, ...] = (),
        lastindex: int | None = None,
        pos: int = 0,
        endpos: int | None = None,
        match_string: str | bytes | None = None,
    ) -> Match:
        return Match(
            text if match_string is None else match_string,
            start,
            end,
            captures,
            lastindex,
            self.groupindex,
            self,
            pos,
            len(text) if endpos is None else endpos,
            self.groups,
        )

    def _required_text_is_present(self, text: str, start: int, end: int) -> bool:
        if (
            self._literal is not None
            or self.flags & IGNORECASE
            or not self._required_literals
        ):
            return True
        return all(
            text.find(literal, start, end) >= 0
            for literal in self._required_literals
        )

    def _search_from(
        self,
        text: str,
        position: int,
        *,
        endpos: int | None = None,
        reject_empty_at: int | None = None,
        match_string: str | bytes | None = None,
        trace: list[str] | None = None,
    ) -> Match | None:
        limit = len(text) if endpos is None else endpos
        if position > limit or limit < 0:
            _record_explanation(trace, "Search bounds contain no candidate positions.")
            return None
        if not self._required_text_is_present(text, position, limit):
            if trace is not None:
                missing_literals = [
                    literal
                    for literal in self._required_literals
                    if text.find(literal, position, limit) < 0
                ]
                for literal in sorted(missing_literals):
                    _record_explanation(
                        trace,
                        f"Required literal {literal!r} is absent from the searched bounds.",
                    )
                if not missing_literals:
                    _record_explanation(
                        trace,
                        "A required literal is absent from the searched input; matching was skipped.",
                    )
            return None
        if self._literal is not None:
            if self._literal == "":
                start = position + (position == reject_empty_at)
                _record_explanation(
                    trace,
                    f"Empty literal matched at position {start}."
                    if start <= limit
                    else "The only empty match was rejected at the end of the search bounds.",
                )
                return (
                    None
                    if start > limit
                    else self._make_match(
                        text,
                        start,
                        start,
                        pos=position,
                        endpos=limit,
                        match_string=match_string,
                    )
                )
            start = text.find(self._literal, position, limit)
            _record_explanation(
                trace,
                f"Literal {self._literal!r} "
                + (
                    f"was found at position {start}."
                    if start >= 0
                    else "does not occur in the searched bounds."
                ),
            )
            return (
                None
                if start < 0
                else self._make_match(
                    text,
                    start,
                    start + len(self._literal),
                    pos=position,
                    endpos=limit,
                    match_string=match_string,
                )
            )
        if self._simple_backreference_tail is not None:
            for start in range(position, limit + 1):
                if trace is not None:
                    _record_explanation(trace, f"Trying candidate start {start}.")
                result = self._simple_backreference_match(
                    text, start, limit, position, match_string or text
                )
                if result is not None:
                    _record_explanation(trace, "The captured text equals the required backreference.")
                    return result
            _record_explanation(trace, "No candidate satisfied the backreference.")
            return None
        if self._requires_vm:
            budget = [0]
            first_char_candidates = self._first_char_literals
            if first_char_candidates is not None and self.flags & IGNORECASE:
                first_char_candidates = {
                    candidate
                    for literal in first_char_candidates
                    for candidate in _case_equivalent_characters(
                        literal, bool(self.flags & ASCII) or self._bytes_mode
                    )
                }
            for start in range(position, limit + 1):
                if (
                    first_char_candidates is not None
                    and (
                        start >= limit
                        or text[start] not in first_char_candidates
                    )
                ):
                    if trace is not None:
                        _record_explanation(
                            trace,
                            f"Candidate start {start} skipped because its first "
                            "character cannot match.",
                        )
                    continue
                if trace is not None:
                    _record_explanation(trace, f"Trying candidate start {start}.")
                result = self._run_vm(
                    text,
                    start,
                    False,
                    budget,
                    search_end=limit,
                    reject_empty=start == reject_empty_at,
                    match_pos=position,
                    match_string=match_string,
                    trace=trace,
                )
                if result is not None:
                    return result
            _record_explanation(trace, "All candidate starts failed.")
            return None
        if len(self._states) <= 256:
            _record_explanation(trace, "Running the ordered NFA from the search position.")
            return self._run_nfa(
                text,
                position,
                search=True,
                fullmatch=False,
                search_end=limit,
                match_pos=position,
                reject_empty_at=reject_empty_at,
                match_string=match_string,
                trace=trace,
            )
        for start in range(position, limit + 1):
            if trace is not None:
                _record_explanation(trace, f"Trying candidate start {start}.")
            result = self._run_nfa(
                text,
                start,
                search=False,
                fullmatch=False,
                search_end=limit,
                match_pos=position,
                reject_empty_at=start if start == reject_empty_at else None,
                match_string=match_string,
                trace=trace,
            )
            if result is not None:
                return result
        _record_explanation(trace, "All candidate starts failed.")
        return None

    def match(
        self, text: str | bytes, pos: int = 0, endpos: int | None = None
    ) -> Match | None:
        """Match the pattern at the specified starting position.

        Unlike search(), this method only attempts a match at ``pos``.
        It does not scan later positions in the input.

        Args:
            text: String or bytes input to match against.
            pos: Position at which matching begins. Defaults to 0.
            endpos: Optional ending position of the matching region.

        Returns:
            A Match object if the pattern matches at ``pos``,
            or None otherwise.

        Raises:
            TypeError: If the input type does not match the pattern type.
        """
        text, match_string = self._prepare_text(text)
        pos, limit = _normalize_bounds(text, pos, endpos)
        if pos > limit:
            return None
        if not self._required_text_is_present(text, pos, limit):
            return None
        if self._literal is not None:
            if not text.startswith(self._literal, pos, limit):
                return None
            return self._make_match(
                text,
                pos,
                pos + len(self._literal),
                pos=pos,
                endpos=limit,
                match_string=match_string,
            )
        if self._simple_backreference_tail is not None:
            return self._simple_backreference_match(
                text, pos, limit, pos, match_string, fullmatch=False
            )
        if self._requires_vm:
            return self._run_vm(
                text,
                pos,
                False,
                [0],
                search_end=limit,
                match_pos=pos,
                match_string=match_string,
            )
        return self._run_nfa(
            text,
            pos,
            search=False,
            fullmatch=False,
            search_end=limit,
            match_pos=pos,
            match_string=match_string,
        )

    def fullmatch(
        self, text: str | bytes, pos: int = 0, endpos: int | None = None
    ) -> Match | None:
        """Check whether the entire specified input region matches.

        The pattern must match from ``pos`` through ``endpos``.
        Unlike search(), this method does not accept a partial match.

        Args:
            text: String or bytes input to match against.
            pos: Starting position of the matching region. Defaults to 0.
            endpos: Optional ending position of the matching region.

        Returns:
            A Match object if the entire region matches, or None otherwise.

        Raises:
            TypeError: If the input type does not match the pattern type.
        """
        text, match_string = self._prepare_text(text)
        pos, limit = _normalize_bounds(text, pos, endpos)
        if pos > limit:
            return None
        if not self._required_text_is_present(text, pos, limit):
            return None
        if self._simple_backreference_tail is not None:
            return self._simple_backreference_match(
                text, pos, limit, pos, match_string, fullmatch=True
            )
        if self._lazy_dfa_enabled:
            matched = self._lazy_dfa_fullmatch(text, pos, limit)
            if matched is not None:
                return (
                    self._make_match(
                        text,
                        pos,
                        limit,
                        pos=pos,
                        endpos=limit,
                        match_string=match_string,
                    )
                    if matched
                    else None
                )
        if self._literal is not None:
            if text[pos:limit] != self._literal:
                return None
            return self._make_match(
                text, pos, limit, pos=pos, endpos=limit, match_string=match_string
            )
        if self._requires_vm:
            return self._run_vm(
                text,
                pos,
                True,
                [0],
                search_end=limit,
                match_pos=pos,
                match_string=match_string,
            )
        return self._run_nfa(
            text,
            pos,
            search=False,
            fullmatch=True,
            search_end=limit,
            match_pos=pos,
            match_string=match_string,
        )

    def finditer(
        self, text: str | bytes, pos: int = 0, endpos: int | None = None
    ) -> Iterator[Match]:
        """Return an iterator yielding Match objects for successive matches.

        Matches are found from left to right within the specified input
        region. Empty matches are handled without repeatedly yielding
        the same empty match at one position.

        Args:
            text: String or bytes input to search.
            pos: Starting position of the search. Defaults to 0.
            endpos: Optional ending position of the search.

        Returns:
            An iterator yielding Match objects.

        Raises:
            TypeError: If the input type does not match the pattern type.
        """
        text, match_string = self._prepare_text(text)
        pos, limit = _normalize_bounds(text, pos, endpos)
        position = pos
        reject_empty_at: int | None = None
        while position <= limit:
            match = self._search_from(
                text,
                position,
                endpos=limit,
                reject_empty_at=reject_empty_at,
                match_string=match_string,
            )
            if match is None:
                return
            yield match
            if match.start() == match.end():
                position = match.end()
                reject_empty_at = match.end()
            else:
                position = match.end()
                reject_empty_at = None

    def findall(
        self, text: str | bytes, pos: int = 0, endpos: int | None = None
    ) -> list[str | bytes] | list[tuple[str | bytes, ...]]:
        """Return all non-overlapping matches as a list.

        The returned values depend on the number of capturing groups:
        - No capturing groups: matched strings are returned.
        - One capturing group: the captured values are returned.
        - Multiple capturing groups: tuples of captured values are returned.

        Unmatched optional groups are represented by an empty string
        or empty bytes value, depending on the pattern type.

        Args:
            text: String or bytes input to search.
            pos: Starting position of the search. Defaults to 0.
            endpos: Optional ending position of the search.

        Returns:
            A list containing matched strings, bytes, or tuples of
            captured values.
        """
        empty = b"" if self._bytes_mode else ""
        if self.groups == 0:
            return [
                match.string[match.start() : match.end()]
                for match in self.finditer(text, pos, endpos)
            ]
        if self.groups == 1:
            return [
                value if value is not None else empty
                for match in self.finditer(text, pos, endpos)
                for value in match.groups()
            ]
        return [
            tuple(value if value is not None else empty for value in match.groups())
            for match in self.finditer(text, pos, endpos)
        ]

    def subn(
        self,
        replacement: str | bytes | Callable[[Match], str | bytes],
        text: str | bytes,
        count: int = 0,
        pos: int = 0,
        endpos: int | None = None,
    ) -> tuple[str | bytes, int]:
        """Replace matches and return the resulting input and replacement count.

        The replacement can be a string, bytes value, or callable.
        A callable receives each Match object and must return a value
        of the same type as the input.

        Replacement templates support capture-group references.

        Args:
            replacement: Replacement template or callable.
            text: String or bytes input to process.
            count: Maximum number of replacements. Zero means unlimited.
            pos: Starting position of the search. Defaults to 0.
            endpos: Optional ending position of the search.

        Returns:
            A tuple containing the resulting input and the number
            of substitutions performed.

        Raises:
            TypeError: If argument types are invalid or replacement
                results do not match the input type.
            ValueError: If count is negative.
        """
        _, source_text = self._prepare_text(text)
        if not isinstance(count, int):
            raise TypeError("count must be an integer")
        if count < 0:
            raise ValueError("count must be non-negative")
        if not isinstance(replacement, (str, bytes)) and not callable(replacement):
            raise TypeError("replacement must be a string, bytes, or callable")
        if self._bytes_mode != isinstance(replacement, bytes) and not callable(replacement):
            raise TypeError("replacement type must match the pattern type")
        template_parts = None
        if not callable(replacement):
            template = (
                replacement.decode("latin-1")
                if isinstance(replacement, bytes)
                else replacement
            )
            template_parts = _parse_template(template, self.groupindex, self.groups)
        pieces: list[str | bytes] = []
        cursor = 0
        substitutions = 0
        matches = iter(self.finditer(source_text, pos, endpos))
        while not count or substitutions < count:
            try:
                match = next(matches)
            except StopIteration:
                break
            pieces.append(source_text[cursor : match.start()])
            if callable(replacement):
                value = replacement(match)
            else:
                if template_parts is None:
                    raise RuntimeError("Replacement template was not parsed")
                expanded = _render_template(template_parts, match)
                value = (
                    expanded.encode("latin-1")
                    if isinstance(replacement, bytes)
                    else expanded
                )
            if not isinstance(value, type(source_text)):
                raise TypeError("replacement function must return the same type as the input")
            pieces.append(value)
            cursor = match.end()
            substitutions += 1
        pieces.append(source_text[cursor:])
        if isinstance(source_text, bytes):
            byte_pieces = [piece for piece in pieces if isinstance(piece, bytes)]
            if len(byte_pieces) != len(pieces):
                raise RuntimeError("Bytes substitution produced a non-bytes piece")
            return b"".join(byte_pieces), substitutions
        string_pieces = [piece for piece in pieces if isinstance(piece, str)]
        if len(string_pieces) != len(pieces):
            raise RuntimeError("String substitution produced a non-string piece")
        return "".join(string_pieces), substitutions

    def sub(
        self,
        replacement: str | bytes | Callable[[Match], str | bytes],
        text: str | bytes,
        count: int = 0,
        pos: int = 0,
        endpos: int | None = None,
    ) -> str | bytes:
        """Replace matches and return the resulting input.

        Accepts the same replacement templates and callable replacements
        as subn(), but returns only the resulting string or bytes value.

        Args:
            replacement: Replacement template or callable.
            text: String or bytes input to process.
            count: Maximum number of replacements. Zero means unlimited.
            pos: Starting position of the search. Defaults to 0.
            endpos: Optional ending position of the search.

        Returns:
            The resulting string or bytes value.
        """
        return self.subn(replacement, text, count, pos, endpos)[0]

    def split(
        self,
        text: str | bytes,
        maxsplit: int = 0,
        pos: int = 0,
        endpos: int | None = None,
    ) -> list[str | bytes | None]:
        """Split the input using occurrences of the compiled pattern.

        Capturing groups in the pattern are included in the resulting
        list. Unmatched optional capturing groups may produce None.

        Args:
            text: String or bytes input to split.
            maxsplit: Maximum number of splits. Zero means unlimited.
            pos: Starting position of the search. Defaults to 0.
            endpos: Optional ending position of the search.

        Returns:
            A list containing the resulting string or bytes segments
            and captured separator values.
        """
        _, source_text = self._prepare_text(text)
        if not isinstance(maxsplit, int):
            raise TypeError("maxsplit must be an integer")
        if maxsplit < 0:
            raise ValueError("maxsplit must be non-negative")
        pieces: list[str | bytes | None] = []
        cursor = 0
        splits = 0
        matches = iter(self.finditer(source_text, pos, endpos))
        while not maxsplit or splits < maxsplit:
            try:
                match = next(matches)
            except StopIteration:
                break
            pieces.append(source_text[cursor : match.start()])
            if self.groups:
                pieces.extend(match.groups())
            cursor = match.end()
            splits += 1
        pieces.append(source_text[cursor:])
        return pieces

    def _run_nfa(
        self,
        text: str,
        initial_start: int,
        *,
        search: bool,
        fullmatch: bool,
        reject_empty_at: int | None = None,
        program: tuple[_Instruction, ...] | None = None,
        entry: int | None = None,
        initial_captures: tuple[tuple[int, int] | None, ...] | None = None,
        initial_lastindex: int | None = None,
        end_position: int | None = None,
        search_end: int | None = None,
        match_pos: int | None = None,
        match_string: str | bytes | None = None,
        look_cache: dict[tuple[int, int], tuple[bool, ...]] | None = None,
        trace: list[str] | None = None,
    ) -> Match | None:
        if program is None:
            if self._vm is None:
                raise RuntimeError("NFA matcher program was not compiled")
            instructions = tuple(self._vm)
        else:
            instructions = program
        start_pc = entry if entry is not None else self._vm_entry
        active: list[_Thread] = []
        pending: Match | None = None
        position = initial_start
        limit = len(text) if search_end is None else search_end
        match_start_pos = initial_start if match_pos is None else match_pos
        first_char_predicate = self._first_char_predicate if program is None else None
        empty_captures = initial_captures or (None,) * self._capture_count
        if look_cache is None:
            look_cache = {}
        first_char_literals = self._first_char_literals if program is None else None
        first_char_candidates = first_char_literals if search else None
        if (
            search
            and first_char_literals is not None
            and self.flags & IGNORECASE
        ):
            first_char_candidates = {
                candidate
                for literal in first_char_literals
                for candidate in _case_equivalent_characters(
                    literal, bool(self.flags & ASCII) or self._bytes_mode
                )
            }

        while position <= limit:
            if trace is not None:
                _record_explanation(
                    trace,
                    f"At position {position}, {len(active)} consuming path(s) "
                    "are active.",
                )
            if (
                search
                and not active
                and pending is None
                and first_char_candidates is not None
            ):
                next_position = min(
                    (
                        found
                        for char in first_char_candidates
                        if (found := text.find(char, position, limit)) >= 0
                    ),
                    default=-1,
                )
                if next_position < 0:
                    _record_explanation(
                        trace,
                        "No remaining input character can start the pattern.",
                    )
                    return pending
                if trace is not None:
                    _record_explanation(
                        trace,
                        f"Skipped to possible first-character position {next_position}.",
                    )
                position = next_position
            seeds = list(active)
            if first_char_candidates is not None:
                can_start = (
                    position < limit and text[position] in first_char_candidates
                )
            else:
                can_start = (
                    first_char_predicate is None
                    or position < limit and first_char_predicate(text[position])
                )
            if (
                pending is None
                and (search or position == initial_start)
                and can_start
            ):
                seeds.append(
                    (
                        start_pc,
                        position,
                        empty_captures,
                        initial_lastindex,
                        (),
                        (),
                        (position,),
                    )
                )
            seeds = [
                (*thread[:6], (rank,))
                for rank, thread in enumerate(seeds)
            ]
            closure = self._closure_program(
                seeds,
                position,
                text,
                instructions,
                look_cache,
                limit,
                match_string,
                trace,
            )

            accepted = next(
                (
                    index
                    for index, (pc, _, _, _, _, _, _) in enumerate(closure)
                    if isinstance(instructions[pc], _MatchInstruction)
                    and (not fullmatch or position == limit)
                    and (end_position is None or position == end_position)
                ),
                None,
            )
            if accepted is not None:
                _, match_start, captures, lastindex, _, _, _ = closure[accepted]
                if reject_empty_at == match_start and position == match_start:
                    _record_explanation(
                        trace,
                        f"Empty match at {position} was rejected to advance iteration.",
                    )
                    closure.pop(accepted)
                else:
                    _record_explanation(
                        trace,
                        f"An accepting path was reached for candidate start {match_start}.",
                    )
                    pending = self._make_match(
                        text,
                        match_start,
                        position,
                        captures,
                        lastindex,
                        match_start_pos,
                        limit,
                        match_string,
                    )
                    closure = closure[:accepted]
                    if not closure:
                        _record_explanation(
                            trace, "No higher-priority path can extend this match."
                        )
                        return pending

            if (
                position == limit
                or (end_position is not None and position >= end_position)
            ):
                if pending is None:
                    _record_explanation(
                        trace,
                        f"Input ended at position {position} before an accepting path was reached.",
                    )
                return pending
            if not closure:
                if search and pending is None:
                    _record_explanation(
                        trace,
                        f"No NFA path survived at position {position}; trying the next start.",
                    )
                    position += 1
                    continue
                if pending is None:
                    _record_explanation(
                        trace,
                        f"No NFA path survived at position {position}.",
                    )
                return pending

            next_threads: list[_Thread] = []
            char = text[position]
            for rank, (
                pc,
                match_start,
                captures,
                lastindex,
                repeat_starts,
                _,
                _,
            ) in enumerate(closure):
                instruction = instructions[pc]
                if (
                    isinstance(instruction, _CharInstruction)
                    and instruction.predicate(char)
                ):
                    next_threads.append(
                        (
                            instruction.target,
                            match_start,
                            captures,
                            lastindex,
                            repeat_starts,
                            (),
                            (rank,),
                        )
                    )
            active = next_threads
            if not active:
                _record_explanation(
                    trace,
                    f"No consuming path accepts character {char!r} at position {position}.",
                )
            position += 1
            if not active and (not search or pending is not None):
                return pending
        return pending

    def _closure_program(
        self,
        seeds: list[_Thread],
        position: int,
        text: str,
        instructions: tuple[_Instruction, ...],
        look_cache: dict[tuple[int, int], tuple[bool, ...]],
        endpos: int,
        match_string: str | bytes | None,
        trace: list[str] | None = None,
    ) -> list[_Thread]:
        reached: set[_StateKey] = set()
        best_priorities: dict[_StateKey, tuple[int, ...]] = {}
        ordered: list[_Thread] = []
        sequence = count()
        pending: list[tuple[tuple[int, ...], int, _Thread]] = []

        def state_key(
            pc: int,
            repeat_starts: tuple[tuple[int, int], ...],
            empty_guards: tuple[tuple[int, int], ...],
        ) -> _StateKey:
            if not self._preserve_repeat_history:
                guards = tuple(
                    (repeat_id, guard_position)
                    for repeat_id, guard_position in empty_guards
                    if guard_position == position
                )
                instruction = instructions[pc]
                if isinstance(instruction, _RepeatEndInstruction):
                    return (
                        pc,
                        guards,
                        any(
                            started_id == instruction.repeat_id
                            and started_at == position
                            for started_id, started_at in repeat_starts
                        ),
                    )
                return (pc, guards) if guards else pc
            return (
                pc,
                tuple(
                    (repeat_id, start)
                    for repeat_id, start in repeat_starts
                    if start == position
                ),
                tuple(
                    (repeat_id, guard_position)
                    for repeat_id, guard_position in empty_guards
                    if guard_position == position
                ),
            )

        def push(
            pc: int,
            match_start: int,
            captures: tuple[tuple[int, int] | None, ...],
            lastindex: int | None,
            repeat_starts: tuple[tuple[int, int], ...],
            empty_guards: tuple[tuple[int, int], ...],
            priority: tuple[int, ...],
        ) -> None:
            key = state_key(pc, repeat_starts, empty_guards)
            current_priority = best_priorities.get(key)
            if current_priority is not None and current_priority <= priority:
                return
            best_priorities[key] = priority
            thread = (
                pc,
                match_start,
                captures,
                lastindex,
                repeat_starts,
                empty_guards,
                priority,
            )
            heapq.heappush(pending, (priority, next(sequence), thread))

        for thread in seeds:
            push(*thread[:6], thread[6])

        while pending:
            priority, _, thread = heapq.heappop(pending)
            (
                pc,
                match_start,
                captures,
                lastindex,
                repeat_starts,
                empty_guards,
                _,
            ) = thread
            instruction = instructions[pc]
            key = state_key(pc, repeat_starts, empty_guards)
            if best_priorities.get(key) != priority:
                continue
            if key in reached:
                continue
            reached.add(key)
            ordered.append(
                (
                    pc,
                    match_start,
                    captures,
                    lastindex,
                    repeat_starts,
                    empty_guards,
                    priority,
                )
            )
            if isinstance(instruction, _SplitInstruction):
                branches = list(enumerate(instruction.branches))
                if instruction.repeat_id is not None:
                    guarded = any(
                        repeat_id == instruction.repeat_id
                        and guard_position == position
                        for repeat_id, guard_position in empty_guards
                    )
                    if guarded:
                        if instruction.body_branch is None:
                            raise RuntimeError("Repeated split has no body branch")
                        branches = tuple(
                            (index, branch)
                            for index, branch in branches
                            if index != instruction.body_branch
                        )
                for index, branch in branches:
                    push(
                        branch,
                        match_start,
                        captures,
                        lastindex,
                        repeat_starts,
                        empty_guards,
                        priority + (index,),
                    )
            elif isinstance(instruction, _RepeatStartInstruction):
                starts = tuple(
                    (repeat_id, start)
                    for repeat_id, start in repeat_starts
                    if repeat_id != instruction.repeat_id
                ) + ((instruction.repeat_id, position),)
                push(
                    instruction.target,
                    match_start,
                    captures,
                    lastindex,
                    starts,
                    empty_guards,
                    priority,
                )
            elif isinstance(instruction, _RepeatEndInstruction):
                start = next(
                    (
                        start
                        for repeat_id, start in reversed(repeat_starts)
                        if repeat_id == instruction.repeat_id
                    ),
                    position,
                )
                starts = tuple(
                    (repeat_id, start_position)
                    for repeat_id, start_position in repeat_starts
                    if repeat_id != instruction.repeat_id
                )
                guards = empty_guards
                if start == position and not any(
                    repeat_id == instruction.repeat_id
                    and guard_position == position
                    for repeat_id, guard_position in guards
                ):
                    guards += ((instruction.repeat_id, position),)
                push(
                    instruction.target,
                    match_start,
                    captures,
                    lastindex,
                    starts,
                    guards,
                    priority,
                )
            elif isinstance(instruction, _SaveInstruction):
                updated = list(captures)
                current = updated[instruction.group - 1]
                updated[instruction.group - 1] = (
                    (position, -1)
                    if instruction.is_start
                    else (
                        current[0] if current is not None else position,
                        position,
                    )
                )
                push(
                    instruction.target,
                    match_start,
                    tuple(updated),
                    (
                        instruction.group
                        if not instruction.is_start
                        and instruction.group <= self.groups
                        else lastindex
                    ),
                    repeat_starts,
                    empty_guards,
                    priority,
                )
            elif isinstance(instruction, _AssertInstruction):
                matched_assertion = _assertion_matches(
                    instruction.kind, position, text, endpos
                )
                if not matched_assertion:
                    _record_explanation(
                        trace,
                        f"Assertion {instruction.kind.replace('_', ' ')} failed at position {position}.",
                    )
                if matched_assertion:
                    push(
                        instruction.target,
                        match_start,
                        captures,
                        lastindex,
                        repeat_starts,
                        empty_guards,
                        priority,
                    )
            elif isinstance(instruction, _LookInstruction):
                cached_result: bool | None = None
                cache_key = (id(instruction.program), instruction.entry)
                if (
                    instruction.kind == "ahead"
                    and not any(
                        isinstance(
                            child,
                            (_SaveInstruction, _LookInstruction, _BackrefInstruction),
                        )
                        for child in instruction.program
                    )
                ):
                    outcomes = look_cache.get(cache_key)
                    if outcomes is None:
                        outcomes = self._lookahead_results(
                            instruction.program, instruction.entry, text, endpos
                        )
                        look_cache[cache_key] = outcomes
                    cached_result = outcomes[position]

                matched: Match | None = None
                if cached_result is None:
                    assertion_starts = (position,)
                    if instruction.kind == "behind":
                        if instruction.width is None:
                            raise RuntimeError("Lookbehind width bounds were not compiled")
                        minimum_width, maximum_width = instruction.width
                        earliest_start = (
                            0
                            if maximum_width is None
                            else max(0, position - maximum_width)
                        )
                        latest_start = position - minimum_width
                        assertion_starts = (
                            range(earliest_start, latest_start + 1)
                            if earliest_start <= latest_start
                            else ()
                        )
                    for assertion_start in assertion_starts:
                        matched = self._run_nfa(
                            text,
                            assertion_start,
                            search=False,
                            fullmatch=False,
                            program=instruction.program,
                            entry=instruction.entry,
                            initial_captures=captures,
                            initial_lastindex=lastindex,
                            end_position=(
                                position if instruction.kind == "behind" else None
                            ),
                            search_end=endpos,
                            match_pos=match_start,
                            match_string=match_string,
                            look_cache=look_cache,
                            trace=trace,
                        )
                        if matched is not None:
                            break
                    success = matched is not None
                else:
                    success = cached_result

                if trace is not None:
                    _record_explanation(
                        trace,
                        f"{'Look' if instruction.positive else 'Negative look'}"
                        f"{instruction.kind} "
                        f"{'succeeded' if success else 'failed'} at position {position}.",
                    )
                if instruction.positive and success:
                    push(
                        instruction.target,
                        match_start,
                        matched.captures if matched is not None else captures,
                        matched.lastindex if matched is not None else lastindex,
                        repeat_starts,
                        empty_guards,
                        priority,
                    )
                elif not instruction.positive and not success:
                    push(
                        instruction.target,
                        match_start,
                        captures,
                        lastindex,
                        repeat_starts,
                        empty_guards,
                        priority,
                    )
            elif isinstance(instruction, _GroupConditionInstruction):
                capture = captures[instruction.group - 1]
                matched_group = capture is not None and capture[1] >= 0
                if trace is not None:
                    _record_explanation(
                        trace,
                        f"Conditional group {instruction.group} is "
                        f"{'set' if matched_group else 'unset'} at position {position}.",
                    )
                target = (
                    instruction.yes
                    if matched_group
                    else instruction.no
                )
                push(
                    target,
                    match_start,
                    captures,
                    lastindex,
                    repeat_starts,
                    empty_guards,
                    priority,
                )
        return ordered

    @staticmethod
    def _lookahead_results(
        instructions: tuple[_Instruction, ...], entry: int, text: str, endpos: int
    ) -> tuple[bool, ...]:
        reverse_epsilon: list[list[int]] = [
            [] for _ in range(len(instructions))
        ]
        for pc, instruction in enumerate(instructions):
            if isinstance(instruction, _SplitInstruction):
                targets = instruction.branches
            elif isinstance(
                instruction, (_AssertInstruction, _RepeatStartInstruction, _RepeatEndInstruction)
            ):
                targets = (instruction.target,)
            else:
                targets = ()
            for target in targets:
                reverse_epsilon[target].append(pc)

        outcomes = [False] * (endpos + 1)
        next_reachable: set[int] = set()
        for position in range(endpos, -1, -1):
            reachable: set[int] = set()
            pending: list[int] = []
            for pc, instruction in enumerate(instructions):
                if isinstance(instruction, _MatchInstruction):
                    reachable.add(pc)
                    pending.append(pc)
                elif (
                    isinstance(instruction, _CharInstruction)
                    and position < endpos
                    and instruction.predicate(text[position])
                    and instruction.target in next_reachable
                ):
                    reachable.add(pc)
                    pending.append(pc)

            while pending:
                target = pending.pop()
                for pc in reverse_epsilon[target]:
                    instruction = instructions[pc]
                    if (
                        isinstance(instruction, _AssertInstruction)
                        and not _assertion_matches(
                            instruction.kind, position, text, endpos
                        )
                    ):
                        continue
                    if pc not in reachable:
                        reachable.add(pc)
                        pending.append(pc)

            outcomes[position] = entry in reachable
            next_reachable = reachable
        return tuple(outcomes)

    def _compile(self, node: object) -> _Fragment:
        if isinstance(node, (_Empty, _GlobalFlags)):
            start, end = self._new_state(), self._new_state()
            self._states[start].epsilon.append(("epsilon", end, None))
            return _Fragment(start, end)

        if isinstance(node, _Character):
            start, end = self._new_state(), self._new_state()
            self._states[start].transitions.append((node.predicate, node.label, end))
            return _Fragment(start, end)

        if isinstance(node, _Assertion):
            start, end = self._new_state(), self._new_state()
            self._states[start].assertions.append((node.kind, end))
            return _Fragment(start, end)

        if isinstance(node, _Lookaround):
            start, end = self._new_state(), self._new_state()
            label = f"lookaround:{len(self._dfa_lookarounds)}"
            self._dfa_lookarounds[label] = node
            self._states[start].assertions.append((label, end))
            return _Fragment(start, end)

        if isinstance(node, _Backreference):
            start, end = self._new_state(), self._new_state()
            self._states[start].backreferences.append((node.group, end))
            return _Fragment(start, end)

        if isinstance(node, _Group):
            fragment = self._compile(node.node)
            start, end = self._new_state(), self._new_state()
            self._states[start].epsilon.append(("save_start", fragment.start, node.index))
            self._states[fragment.end].epsilon.append(("save_end", end, node.index))
            return _Fragment(start, end)

        if isinstance(node, _Atomic):
            return self._compile(node.node)

        if isinstance(node, _Conditional):
            return self._compile(_Alternative((node.yes, node.no)))

        if isinstance(node, _Sequence):
            fragments = [self._compile(item) for item in node.items]
            for left, right in zip(fragments, fragments[1:]):
                self._states[left.end].epsilon.append(("epsilon", right.start, None))
            return _Fragment(fragments[0].start, fragments[-1].end)

        if isinstance(node, _Alternative):
            start, end = self._new_state(), self._new_state()
            for option in node.options:
                fragment = self._compile(option)
                self._states[start].epsilon.append(("epsilon", fragment.start, None))
                self._states[fragment.end].epsilon.append(("epsilon", end, None))
            return _Fragment(start, end)

        if isinstance(node, _Repeat):
            return self._compile_repeat(node)

        raise TypeError(f"Unsupported syntax node: {type(node).__name__}")

    def _compile_repeat(self, node: _Repeat) -> _Fragment:
        if node.maximum is None and node.minimum == 1:
            body = self._compile(node.node)
            start, end, split = self._new_state(), self._new_state(), self._new_state()
            self._states[start].epsilon.append(("epsilon", body.start, None))
            choices = (
                [("epsilon", body.start, None), ("epsilon", end, None)]
                if node.greedy
                else [("epsilon", end, None), ("epsilon", body.start, None)]
            )
            self._states[split].epsilon.extend(choices)
            self._states[body.end].epsilon.append(("epsilon", split, None))
            return _Fragment(start, end)

        fragments = [self._compile(node.node) for _ in range(node.minimum)]
        start = self._new_state()
        cursor = start
        for fragment in fragments:
            self._states[cursor].epsilon.append(("epsilon", fragment.start, None))
            cursor = fragment.end

        end = self._new_state()
        if node.maximum is None:
            split = self._new_state()
            body = self._compile(node.node)
            choices = (
                [("epsilon", body.start, None), ("epsilon", end, None)]
                if node.greedy
                else [("epsilon", end, None), ("epsilon", body.start, None)]
            )
            self._states[split].epsilon.extend(choices)
            self._states[body.end].epsilon.append(("epsilon", split, None))
            self._states[cursor].epsilon.append(("epsilon", split, None))
            return _Fragment(start, end)

        optional_count = node.maximum - node.minimum
        for index in range(optional_count):
            split = self._new_state()
            next_cursor = end if index == optional_count - 1 else self._new_state()
            body = self._compile(node.node)
            choices = (
                [("epsilon", body.start, None), ("epsilon", end, None)]
                if node.greedy
                else [("epsilon", end, None), ("epsilon", body.start, None)]
            )
            self._states[cursor].epsilon.append(("epsilon", split, None))
            self._states[split].epsilon.extend(choices)
            self._states[body.end].epsilon.append(("epsilon", next_cursor, None))
            cursor = next_cursor
        if optional_count == 0:
            self._states[cursor].epsilon.append(("epsilon", end, None))
        return _Fragment(start, end)

    def _new_state(self) -> int:
        if len(self._states) >= self.limits.nfa_states:
            raise RegexComplexityError(
                f"Compiled patterns are limited to {self.limits.nfa_states} NFA states"
            )
        self._states.append(_State())
        return len(self._states) - 1

    def _compile_vm(self) -> tuple[int, list[_Instruction]]:
        instruction_count = 0
        repeat_count = 0

        def compile_program(root: object) -> tuple[int, list[_Instruction]]:
            nonlocal instruction_count
            instruction_count += 1
            if instruction_count > self.limits.nfa_states:
                raise RegexComplexityError(
                    f"Compiled patterns are limited to {self.limits.nfa_states} matcher instructions"
                )
            instructions: list[_Instruction] = [_MatchInstruction()]

            def emit(instruction: _Instruction) -> int:
                nonlocal instruction_count
                instruction_count += 1
                if instruction_count > self.limits.nfa_states:
                    raise RegexComplexityError(
                        f"Compiled patterns are limited to {self.limits.nfa_states} matcher instructions"
                    )
                instructions.append(instruction)
                return len(instructions) - 1

            def build(node: object, continuation: int) -> int:
                nonlocal repeat_count
                if isinstance(node, (_Empty, _GlobalFlags)):
                    return continuation
                if isinstance(node, _Character):
                    return emit(_CharInstruction(node.predicate, continuation))
                if isinstance(node, _Assertion):
                    return emit(_AssertInstruction(node.kind, continuation))
                if isinstance(node, _Lookaround):
                    entry, program = compile_program(node.node)
                    return emit(
                        _LookInstruction(
                            node.kind,
                            node.positive,
                            entry,
                            tuple(program),
                            _width_bounds(node.node) if node.kind == "behind" else None,
                            continuation,
                        )
                    )
                if isinstance(node, _Atomic):
                    group = self._atomic_group_indices[id(node)]
                    backreference = emit(_BackrefInstruction(group, continuation))
                    entry, program = compile_program(_Group(group, node.node))
                    return emit(
                        _LookInstruction(
                            "ahead", True, entry, tuple(program), None, backreference
                        )
                    )
                if isinstance(node, _Conditional):
                    if not isinstance(node.group, int):
                        raise RuntimeError("Conditional group reference was not resolved")
                    yes = build(node.yes, continuation)
                    no = build(node.no, continuation)
                    return emit(_GroupConditionInstruction(node.group, yes, no))
                if isinstance(node, _Backreference):
                    return emit(_BackrefInstruction(node.group, continuation))
                if isinstance(node, _Group):
                    after = emit(_SaveInstruction(node.index, continuation, False))
                    body = build(node.node, after)
                    return emit(_SaveInstruction(node.index, body, True))
                if isinstance(node, _Sequence):
                    current = continuation
                    for item in reversed(node.items):
                        current = build(item, current)
                    return current
                if isinstance(node, _Alternative):
                    branches = tuple(build(option, continuation) for option in node.options)
                    return emit(_SplitInstruction(branches))
                if isinstance(node, _Repeat):
                    current = continuation
                    if node.maximum is None:
                        repeat_id = repeat_count
                        repeat_count += 1
                        split = emit(_SplitInstruction(()))
                        repeat_end = emit(_RepeatEndInstruction(repeat_id, split))
                        body = build(node.node, repeat_end)
                        repeat_start = emit(
                            _RepeatStartInstruction(repeat_id, body)
                        )
                        choices = (
                            (repeat_start, current)
                            if node.greedy
                            else (current, repeat_start)
                        )
                        instruction = instructions[split]
                        if not isinstance(instruction, _SplitInstruction):
                            raise RuntimeError("Expected a split instruction")
                        instruction.branches = choices
                        instruction.repeat_id = repeat_id
                        instruction.body_branch = 0 if node.greedy else 1
                        instruction.unbounded_repeat = True
                        current = split if node.minimum == 0 else repeat_start
                        for _ in range(node.minimum - 1):
                            current = build(node.node, current)
                        return current
                    else:
                        optional_count = node.maximum - node.minimum
                        repeat_id = repeat_count if optional_count else None
                        if repeat_id is not None:
                            repeat_count += 1
                        for _ in range(node.maximum - node.minimum):
                            body_continuation = current
                            if repeat_id is not None:
                                body_continuation = emit(
                                    _RepeatEndInstruction(repeat_id, current)
                                )
                            body = build(node.node, body_continuation)
                            if repeat_id is not None:
                                body = emit(_RepeatStartInstruction(repeat_id, body))
                            branches = (
                                (body, continuation)
                                if node.greedy
                                else (continuation, body)
                            )
                            current = emit(
                                _SplitInstruction(
                                    branches,
                                    repeat_id,
                                    0 if node.greedy else 1,
                                )
                            )
                    for _ in range(node.minimum):
                        current = build(node.node, current)
                    return current
                raise TypeError(f"Unsupported syntax node: {type(node).__name__}")

            return build(root, 0), instructions

        return compile_program(self._syntax)

    def _run_vm(
        self,
        text: str,
        start: int,
        fullmatch: bool,
        budget: list[int],
        *,
        program: tuple[_Instruction, ...] | None = None,
        entry: int | None = None,
        initial_captures: tuple[tuple[int, int] | None, ...] | None = None,
        initial_lastindex: int | None = None,
        end_position: int | None = None,
        search_end: int | None = None,
        match_pos: int | None = None,
        match_string: str | bytes | None = None,
        reject_empty: bool = False,
        trace: list[str] | None = None,
    ) -> Match | None:
        if self._vm is None:
            raise RuntimeError("Backreference VM was not compiled")
        instructions = program if program is not None else tuple(self._vm)
        start_pc = entry if entry is not None else self._vm_entry
        limit = len(text) if search_end is None else search_end
        match_start_pos = start if match_pos is None else match_pos
        captures = initial_captures or (None,) * self._capture_count
        stack: list[
            tuple[
                int,
                int,
                tuple[tuple[int, int] | None, ...],
                int | None,
                dict[int, int],
                dict[int, int],
                set[tuple[int, int, tuple[tuple[int, int] | None, ...]]],
            ]
        ] = [(start_pc, start, captures, initial_lastindex, {}, {}, set())]

        while stack:
            (
                pc,
                position,
                current_captures,
                lastindex,
                repeat_starts,
                capture_starts,
                visited,
            ) = stack.pop()
            while True:
                budget[0] += 1
                if budget[0] > self.limits.backtrack_steps:
                    if trace is not None:
                        _record_explanation(
                            trace,
                            f"The backtracking step limit ({self.limits.backtrack_steps}) was reached.",
                        )
                    raise RegexMatchLimitError(
                        f"Backtracking matching exceeded {self.limits.backtrack_steps} steps"
                    )
                configuration = (pc, position, current_captures)
                if configuration in visited:
                    instruction = instructions[pc]
                    if not (
                        isinstance(instruction, _SplitInstruction)
                        and instruction.unbounded_repeat
                        and instruction.repeat_id is not None
                        and repeat_starts.get(instruction.repeat_id) == position
                    ):
                        break
                else:
                    visited.add(configuration)
                instruction = instructions[pc]
                if isinstance(instruction, _MatchInstruction):
                    if (
                        (not fullmatch or position == limit)
                        and (end_position is None or position == end_position)
                    ):
                        if reject_empty and position == start:
                            if trace is not None:
                                _record_explanation(
                                    trace,
                                    f"Empty match at position {position} was rejected to advance iteration.",
                                )
                            break
                        if trace is not None:
                            _record_explanation(
                                trace,
                                f"Accepting path found from candidate start {start} through {position}.",
                            )
                        return self._make_match(
                            text,
                            start,
                            position,
                            current_captures,
                            lastindex,
                            match_start_pos,
                            limit,
                            match_string,
                        )
                    if trace is not None:
                        _record_explanation(
                            trace,
                            f"Pattern ended at position {position}, but the required end condition failed.",
                        )
                    break
                elif isinstance(instruction, _CharInstruction):
                    if (
                        position >= limit
                        or not instruction.predicate(text[position])
                    ):
                        if trace is not None:
                            _record_explanation(
                                trace,
                                f"Character condition failed at position {position}"
                                + (
                                    " because the search bounds ended."
                                    if position >= limit
                                    else f" for input {text[position]!r}."
                                ),
                            )
                        break
                    position += 1
                    pc = instruction.target
                elif isinstance(instruction, _AssertInstruction):
                    if not _assertion_matches(
                        instruction.kind, position, text, limit
                    ):
                        if trace is not None:
                            _record_explanation(
                                trace,
                                f"Assertion {instruction.kind.replace('_', ' ')} failed at position {position}.",
                            )
                        break
                    pc = instruction.target
                elif isinstance(instruction, _BackrefInstruction):
                    span = current_captures[instruction.group - 1]
                    if span is None or span[1] < 0:
                        if trace is not None:
                            _record_explanation(
                                trace,
                                f"Backreference group {instruction.group} is unset at position {position}.",
                            )
                        break
                    value = text[span[0] : span[1]]
                    candidate = text[position : position + len(value)]
                    if position + len(value) > limit or (
                        not all(
                            _case_insensitive_equal(
                                left,
                                right,
                                bool(self.flags & ASCII),
                                bool(self.flags & LOCALE),
                            )
                            for left, right in zip(value, candidate)
                        )
                        if self.flags & IGNORECASE
                        else value != candidate
                    ):
                        if trace is not None:
                            _record_explanation(
                                trace,
                                f"Backreference group {instruction.group} did not match at position {position}.",
                            )
                        break
                    position += len(value)
                    pc = instruction.target
                elif isinstance(instruction, _SaveInstruction):
                    if instruction.is_start:
                        capture_starts = capture_starts.copy()
                        capture_starts[instruction.group] = position
                    else:
                        updated = list(current_captures)
                        capture_start = capture_starts.get(
                            instruction.group, position
                        )
                        updated[instruction.group - 1] = (
                            capture_start,
                            position,
                        )
                        current_captures = tuple(updated)
                        capture_starts = capture_starts.copy()
                        capture_starts.pop(instruction.group, None)
                        if instruction.group <= self.groups:
                            lastindex = instruction.group
                    pc = instruction.target
                elif isinstance(
                    instruction, (_RepeatStartInstruction, _RepeatEndInstruction)
                ):
                    if isinstance(instruction, _RepeatStartInstruction):
                        repeat_starts[instruction.repeat_id] = position
                        visited.clear()
                    pc = instruction.target
                elif isinstance(instruction, _SplitInstruction):
                    branches = instruction.branches

                    if len(branches) == 0:
                        raise RuntimeError("Split instruction has no branches")

                    if (
                        instruction.unbounded_repeat
                        and instruction.repeat_id in repeat_starts
                        and repeat_starts[instruction.repeat_id] == position
                    ):
                        if instruction.body_branch is None:
                            raise RuntimeError("Unbounded repeat has no body branch")

                        if trace is not None:
                            _record_explanation(
                                trace,
                                f"Repeat stopped at position {position} after an empty iteration.",
                            )

                        pc = branches[1 - instruction.body_branch]
                        continue

                    if trace is not None and len(branches) > 1:
                        _record_explanation(
                            trace,
                            f"Trying {len(branches)} prioritized alternative(s) at position {position}.",
                        )

                    for branch in reversed(branches[1:]):
                        stack.append(
                            (
                                branch,
                                position,
                                current_captures,
                                lastindex,
                                repeat_starts.copy(),
                                capture_starts.copy(),
                                visited.copy(),
                            )
                        )

                    pc = branches[0]
                elif isinstance(instruction, _LookInstruction):
                    matched = None
                    if instruction.kind == "behind":
                        if instruction.width is None:
                            raise RuntimeError("Lookbehind width bounds were not compiled")
                        minimum_width, maximum_width = instruction.width
                        earliest_start = (
                            0
                            if maximum_width is None
                            else max(0, position - maximum_width)
                        )
                        latest_start = position - minimum_width
                        assertion_starts = (
                            range(earliest_start, latest_start + 1)
                            if earliest_start <= latest_start
                            else ()
                        )
                        for assertion_start in assertion_starts:
                            matched = self._run_vm(
                                text,
                                assertion_start,
                                False,
                                budget,
                                program=instruction.program,
                                entry=instruction.entry,
                                initial_captures=current_captures,
                                initial_lastindex=lastindex,
                                end_position=position,
                                search_end=limit,
                                match_pos=match_start_pos,
                                match_string=match_string,
                                trace=trace,
                            )
                            if matched is not None:
                                break
                    else:
                        matched = self._run_vm(
                            text,
                            position,
                            False,
                            budget,
                            program=instruction.program,
                            entry=instruction.entry,
                            initial_captures=current_captures,
                            initial_lastindex=lastindex,
                            search_end=limit,
                            match_pos=match_start_pos,
                            match_string=match_string,
                            trace=trace,
                        )
                    success = matched is not None
                    if trace is not None:
                        _record_explanation(
                            trace,
                            f"{'Look' if instruction.positive else 'Negative look'}"
                            f"{instruction.kind} "
                            f"{'succeeded' if success else 'failed'} at position {position}.",
                        )
                    if success != instruction.positive:
                        break
                    if instruction.positive and matched is not None:
                        current_captures = matched.captures
                        lastindex = matched.lastindex
                    pc = instruction.target
                elif isinstance(instruction, _GroupConditionInstruction):
                    capture = current_captures[instruction.group - 1]
                    matched_group = capture is not None and capture[1] >= 0
                    if trace is not None:
                        _record_explanation(
                            trace,
                            f"Conditional group {instruction.group} is "
                            f"{'set' if matched_group else 'unset'} "
                            f"at position {position}.",
                        )
                    pc = (
                        instruction.yes
                        if matched_group
                        else instruction.no
                    )
                else:
                    raise RuntimeError(f"Unknown matcher instruction: {instruction!r}")
        return None

    def to_dot(self) -> str:
        """Export the compiled NFA as a Graphviz DOT description.

        The generated graph represents the compiled states and their
        connections, including epsilon transitions, assertions,
        consuming transitions, and backreferences.

        The DOT output can be rendered using Graphviz-compatible tools
        to help inspect the compiled pattern structure.

        Returns:
            A string containing the Graphviz DOT representation.
        """
        lines = ["digraph NFA {", "  rankdir=LR;", '  node [shape=circle];']
        lines.append(f'  start [shape=point]; start -> {self._start};')
        for state_id, state in enumerate(self._states):
            shape = "doublecircle" if state_id == self._accept else "circle"
            lines.append(f'  {state_id} [shape={shape}, label="{state_id}"];')
            for kind, target, group in state.epsilon:
                label = f"save {group} start" if kind == "save_start" else (
                    f"save {group} end" if kind == "save_end" else "ε"
                )
                lines.append(f"  {state_id} -> {target} [label={_dot_quote(label)}];")
            for kind, target in state.assertions:
                label = {
                    "start": "^",
                    "end": "$",
                    "absolute_start": r"\A",
                    "absolute_end": r"\Z",
                    "word_boundary": r"\b",
                    "not_word_boundary": r"\B",
                }.get(kind, kind.replace("_", " "))
                lines.append(f"  {state_id} -> {target} [label={_dot_quote(label)}];")
            for _, label, target in state.transitions:
                lines.append(f"  {state_id} -> {target} [label={_dot_quote(label)}];")
            for group, target in state.backreferences:
                lines.append(f'  {state_id} -> {target} [label="\\\\{group}"];')
        lines.append("}")
        return "\n".join(lines)

    def to_dfa(self, alphabet: str) -> DFA:
        """Determinize this pattern over an explicit finite alphabet."""
        if not isinstance(alphabet, str):
            raise TypeError("alphabet must be a string")
        if self._has_backreferences:
            raise ValueError("Backreference patterns cannot be converted to a DFA")
        if self._has_conditionals:
            raise ValueError("Conditional groups cannot be converted to a DFA")
        return DFA.from_regex(self, alphabet)


Pattern = Regex


class DFA:
    """Represent a deterministic finite automaton (DFA).

    A DFA processes an input string one character at a time using
    a transition table. After processing the entire input, the
    automaton accepts if its final state is an accepting state.

    The DFA operates over a caller-supplied alphabet. Characters
    outside that alphabet cause matching to fail.
    """

    def __init__(
        self,
        alphabet: tuple[str, ...],
        transitions: list[dict[str, int]],
        accepting: set[int],
        start: int = 0,
    ):
        """Initialize a DFA from its alphabet and transition table.

        Args:
            alphabet: Non-empty tuple of supported input characters.
            transitions: Transition table mapping characters to
                destination state indices for each state.
            accepting: Set of accepting state indices.
            start: Index of the initial state. Defaults to 0.

        Raises:
            ValueError: If the alphabet is empty.
        """
        if not alphabet:
            raise ValueError("The DFA alphabet cannot be empty")
        self.alphabet = alphabet
        self.transitions = transitions
        self.accepting = accepting
        self.start = start

    @classmethod
    def from_regex(cls, regex: Regex, alphabet: str) -> DFA:
        """Construct a DFA from a compiled regular expression.

        Converts the supported regular expression into a deterministic
        finite automaton over the supplied alphabet.

        The alphabet must contain single characters. Duplicate
        characters are removed while preserving their order.

        Backreferences, conditional groups, and unsupported
        lookarounds cannot be converted.

        Args:
            regex: Compiled Regex object to convert.
            alphabet: Characters supported by the resulting DFA.

        Returns:
            A DFA representing the supported pattern behavior
            over the supplied alphabet.

        Raises:
            ValueError: If the alphabet is empty or invalid, or if
                the pattern contains unsupported constructs.
            RegexComplexityError: If the alphabet or resulting
                automaton exceeds configured limits.
        """
        symbols = tuple(dict.fromkeys(alphabet))
        if not symbols:
            raise ValueError("The DFA alphabet cannot be empty")
        if any(len(symbol) != 1 for symbol in symbols):
            raise ValueError("The DFA alphabet must contain single characters")
        if len(symbols) > regex.limits.dfa_alphabet:
            raise RegexComplexityError(
                f"DFA alphabets are limited to {regex.limits.dfa_alphabet} characters"
            )
        if regex._has_backreferences:
            raise ValueError("Backreference patterns cannot be converted to a DFA")
        if regex._has_conditionals:
            raise ValueError("Conditional groups cannot be converted to a DFA")
        for lookaround in regex._dfa_lookarounds.values():
            if _single_character_predicate(lookaround.node) is None:
                raise ValueError(
                    "DFA conversion supports only single-character lookarounds"
                )

        def is_word(char: str, ascii_only: bool) -> bool:
            if regex.flags & LOCALE and not ascii_only:
                return _locale_character_predicate(rb"\w")(char)
            return (
                char == "_" or char.isascii() and char.isalnum()
                if ascii_only
                else _is_word_char(char)
            )

        def assertion_matches(
            kind: str,
            *,
            at_start: bool,
            at_end: bool,
            previous_newline: bool,
            previous_word: bool,
            previous_ascii_word: bool,
            previous_char: str | None,
            next_char: str | None,
        ) -> bool:
            lookaround = regex._dfa_lookarounds.get(kind)
            if lookaround is not None:
                predicate = _single_character_predicate(lookaround.node)
                if predicate is None:
                    raise RuntimeError("Unsupported lookaround reached DFA closure")
                char = next_char if lookaround.kind == "ahead" else previous_char
                matched = char is not None and predicate(char)
                return matched if lookaround.positive else not matched
            if kind in ("start", "absolute_start"):
                return at_start
            if kind == "start_multiline":
                return at_start or previous_newline
            if kind == "absolute_end":
                return at_end
            if kind == "end":
                return at_end or next_char == "\n"
            if kind == "end_multiline":
                return at_end or next_char == "\n"
            if kind in (
                "word_boundary",
                "not_word_boundary",
                "word_boundary_ascii",
                "not_word_boundary_ascii",
                "word_boundary_locale",
                "not_word_boundary_locale",
            ):
                ascii_only = kind.endswith("_ascii")
                left = (
                    previous_ascii_word
                    if ascii_only
                    else previous_word
                )
                right = next_char is not None and is_word(next_char, ascii_only)
                boundary = left != right
                positive = kind.startswith("word_boundary")
                if not positive and at_start and at_end:
                    return False
                return boundary if positive else not boundary
            return False

        def closure(
            states: frozenset[int],
            *,
            at_start: bool,
            at_end: bool,
            previous_newline: bool,
            previous_word: bool,
            previous_ascii_word: bool,
            previous_char: str | None,
            next_char: str | None,
        ) -> frozenset[int]:
            reached = set(states)
            pending = list(states)
            while pending:
                state_id = pending.pop()
                state = regex._states[state_id]
                for kind, target, _ in state.epsilon:
                    if kind in ("epsilon", "save_start", "save_end") and target not in reached:
                        reached.add(target)
                        pending.append(target)
                for kind, target in state.assertions:
                    if assertion_matches(
                        kind,
                        at_start=at_start,
                        at_end=at_end,
                        previous_newline=previous_newline,
                        previous_word=previous_word,
                        previous_ascii_word=previous_ascii_word,
                        previous_char=previous_char,
                        next_char=next_char,
                    ) and target not in reached:
                        reached.add(target)
                        pending.append(target)
            return frozenset(reached)

        initial: _DFAState = (
            frozenset({regex._start}),
            True,
            False,
            False,
            False,
            None,
        )
        states: list[_DFAState] = [initial]
        indices: dict[_DFAState, int] = {initial: 0}
        transitions: list[dict[str, int]] = []
        accepting: set[int] = set()
        cursor = 0

        while cursor < len(states):
            (
                subset,
                at_start,
                previous_newline,
                previous_word,
                previous_ascii_word,
                previous_char,
            ) = states[cursor]
            row: dict[str, int] = {}
            final_closure = closure(
                subset,
                at_start=at_start,
                at_end=True,
                previous_newline=previous_newline,
                previous_word=previous_word,
                previous_ascii_word=previous_ascii_word,
                previous_char=previous_char,
                next_char=None,
            )
            if regex._accept in final_closure:
                accepting.add(cursor)
            for symbol in symbols:
                active = closure(
                    subset,
                    at_start=at_start,
                    at_end=False,
                    previous_newline=previous_newline,
                    previous_word=previous_word,
                    previous_ascii_word=previous_ascii_word,
                    previous_char=previous_char,
                    next_char=symbol,
                )
                destinations = {
                    target
                    for state_id in active
                    for predicate, _, target in regex._states[state_id].transitions
                    if predicate(symbol)
                }
                target_state: _DFAState = (
                    frozenset(destinations),
                    False,
                    symbol == "\n",
                    is_word(symbol, False),
                    is_word(symbol, True),
                    symbol,
                )
                if target_state not in indices:
                    if len(states) >= regex.limits.dfa_states:
                        raise RegexComplexityError(
                            f"DFAs are limited to {regex.limits.dfa_states} states"
                        )
                    indices[target_state] = len(states)
                    states.append(target_state)
                row[symbol] = indices[target_state]
            transitions.append(row)
            cursor += 1
        return cls(symbols, transitions, accepting)

    def fullmatch(self, text: str) -> bool:
        """Check whether the entire input string is accepted.

        Processes every character using the DFA transition table.
        If a character is outside the alphabet or has no available
        transition, matching fails.

        Args:
            text: Input string to evaluate.

        Returns:
            True if the entire input reaches an accepting state,
            otherwise False.
        """
        state = self.start
        for char in text:
            if char not in self.transitions[state]:
                return False
            state = self.transitions[state][char]
        return state in self.accepting

    def minimize(self) -> DFA:
        """Return an equivalent DFA with fewer states where possible.

        Uses Hopcroft's partition-refinement algorithm to combine
        equivalent accepting and non-accepting states.

        The original DFA is not modified.

        Returns:
            A new minimized DFA with the same alphabet and
            equivalent accepted language.
        """
        all_states = set(range(len(self.transitions)))
        partitions = [block for block in (self.accepting, all_states - self.accepting) if block]
        work = [min(partitions, key=len)]

        while work:
            splitter = work.pop()
            for symbol in self.alphabet:
                predecessors = {
                    state
                    for state, row in enumerate(self.transitions)
                    if row[symbol] in splitter
                }
                for block in partitions.copy():
                    inside = block & predecessors
                    outside = block - predecessors
                    if not inside or not outside:
                        continue
                    partitions.remove(block)
                    partitions.extend((inside, outside))
                    if block in work:
                        work.remove(block)
                        work.extend((inside, outside))
                    else:
                        work.append(inside if len(inside) <= len(outside) else outside)

        state_to_block = {
            state: block_index
            for block_index, block in enumerate(partitions)
            for state in block
        }
        minimized_transitions = [
            {
                symbol: state_to_block[self.transitions[next(iter(block))][symbol]]
                for symbol in self.alphabet
            }
            for block in partitions
        ]
        minimized_accepting = {
            state_to_block[state] for state in self.accepting
        }
        return DFA(
            self.alphabet,
            minimized_transitions,
            minimized_accepting,
            state_to_block[self.start],
        )

    def to_dot(self) -> str:
        """Export the DFA as a Graphviz DOT description.

        The generated graph represents DFA states, transitions,
        the initial state, and accepting states.

        The output can be rendered using Graphviz-compatible tools.

        Returns:
            A string containing the Graphviz DOT representation.
        """
        lines = ["digraph DFA {", "  rankdir=LR;", '  node [shape=circle];']
        lines.append(f'  start [shape=point]; start -> {self.start};')
        for state_id, row in enumerate(self.transitions):
            shape = "doublecircle" if state_id in self.accepting else "circle"
            lines.append(f'  {state_id} [shape={shape}, label="{state_id}"];')
            for symbol, target in row.items():
                lines.append(f"  {state_id} -> {target} [label={_dot_quote(symbol)}];")
        lines.append("}")
        return "\n".join(lines)


@lru_cache(maxsize=128)
def _compile_cached(
    pattern: str | bytes, flags: int = 0, limits: RegexLimits = RegexLimits()
) -> Regex:
    return Regex(pattern, flags, limits)


_compile_lock = RLock()


def compile(
    pattern: str | bytes | Regex,
    flags: int = 0,
    limits: RegexLimits | None = None,
) -> Regex:
    """Compile a regular expression into a reusable Regex object.

    Compiled patterns can be reused for searching, matching,
    substitutions, and other regular expression operations.
    Repeated calls with the same pattern, flags, and limits
    reuse a bounded compilation cache.

    Args:
        pattern: Regular expression string, bytes, or compiled Regex.
        flags: Optional compilation flags.
        limits: Optional resource limits.

    Returns:
        A compiled Regex object.

    Raises:
        TypeError: If the pattern, flags, or limits have invalid types.
        ValueError: If incompatible arguments or flags are supplied.
        RegexSyntaxError: If the pattern contains invalid syntax.
        RegexComplexityError: If a configured complexity limit is exceeded.
    """
    if isinstance(pattern, Regex):
        if flags:
            raise ValueError("cannot process flags argument with a compiled pattern")
        if limits is not None:
            raise ValueError("cannot process limits argument with a compiled pattern")
        return pattern
    if not isinstance(pattern, (str, bytes)):
        raise TypeError("pattern must be a string or bytes")
    if not isinstance(flags, int):
        raise TypeError("flags must be an integer")
    if limits is not None and not isinstance(limits, RegexLimits):
        raise TypeError("limits must be a RegexLimits instance")
    if flags & DEBUG:
        return Regex(pattern, flags, limits)
    with _compile_lock:
        return _compile_cached(pattern, flags, limits or RegexLimits())


def purge() -> None:
    """Clear the cache of compiled patterns."""
    with _compile_lock:
        _compile_cached.cache_clear()


def search(
    pattern: str | bytes | Regex,
    string: str | bytes,
    flags: int = 0,
    pos: int = 0,
    endpos: int | None = None,
) -> Match | None:
    """Return the earliest match of *pattern* within *string*.

    Matching begins at *pos* and is limited to the end-exclusive range ending
    at *endpos*. *flags* are used when *pattern* is not already compiled.
    Return ``None`` if no match is found.
    """
    return compile(pattern, flags).search(string, pos, endpos)


def match(
    pattern: str | bytes | Regex,
    string: str | bytes,
    flags: int = 0,
    pos: int = 0,
    endpos: int | None = None,
) -> Match | None:
    """Match *pattern* at *pos* in *string*.

    Matching is limited to the end-exclusive range ending at *endpos*.
    *flags* are used when *pattern* is not already compiled. Return ``None``
    if the pattern does not match at *pos*.
    """
    return compile(pattern, flags).match(string, pos, endpos)


def fullmatch(
    pattern: str | bytes | Regex,
    string: str | bytes,
    flags: int = 0,
    pos: int = 0,
    endpos: int | None = None,
) -> Match | None:
    """Match *pattern* against the entire selected range of *string*.

    The range starts at *pos* and ends before *endpos*. *flags* are used when
    *pattern* is not already compiled. Return ``None`` unless the whole range
    matches.
    """
    return compile(pattern, flags).fullmatch(string, pos, endpos)


def finditer(
    pattern: str | bytes | Regex,
    string: str | bytes,
    flags: int = 0,
    pos: int = 0,
    endpos: int | None = None,
) -> Iterator[Match]:
    """Yield successive non-overlapping matches of *pattern* in *string*.

    Matching is limited to the end-exclusive range from *pos* to *endpos*.
    *flags* are used when *pattern* is not already compiled.
    """
    return compile(pattern, flags).finditer(string, pos, endpos)


def findall(
    pattern: str | bytes | Regex,
    string: str | bytes,
    flags: int = 0,
    pos: int = 0,
    endpos: int | None = None,
) -> list[str | bytes] | list[tuple[str | bytes, ...]]:
    """Return all non-overlapping matches of *pattern* in *string*.

    Matching is limited to the end-exclusive range from *pos* to *endpos*.
    *flags* are used when *pattern* is not already compiled. Results contain
    whole matches when there are no capturing groups, captured strings when
    there is one group, or tuples of captured strings when there are multiple
    groups; unmatched captures are returned as empty strings or bytes.
    """
    return compile(pattern, flags).findall(string, pos, endpos)


def sub(
    pattern: str | bytes | Regex,
    repl: str | bytes | Callable[[Match], str | bytes],
    string: str | bytes,
    count: int = 0,
    flags: int = 0,
    pos: int = 0,
    endpos: int | None = None,
) -> str | bytes:
    """Return *string* with matching portions replaced by *repl*.

    *repl* may be a replacement template or a callable receiving each
    :class:`Match`. Replace at most *count* matches, or all matches when
    *count* is zero. Matching is limited to the end-exclusive range from
    *pos* to *endpos*. *flags* are used when *pattern* is not already compiled.
    """
    return compile(pattern, flags).sub(repl, string, count, pos, endpos)


def subn(
    pattern: str | bytes | Regex,
    repl: str | bytes | Callable[[Match], str | bytes],
    string: str | bytes,
    count: int = 0,
    flags: int = 0,
    pos: int = 0,
    endpos: int | None = None,
) -> tuple[str | bytes, int]:
    """Return the result of replacing matches and the number of substitutions.

    *repl* may be a replacement template or a callable receiving each
    :class:`Match`. Replace at most *count* matches, or all matches when
    *count* is zero.

    Matching is limited to the end-exclusive range from *pos* to *endpos*.
    *flags* are used when *pattern* is not already compiled.

    Returns:
        A tuple containing the resulting string or bytes and the number
        of substitutions performed.
    """
    return compile(pattern, flags).subn(repl, string, count, pos, endpos)


def split(
    pattern: str | bytes | Regex,
    string: str | bytes,
    maxsplit: int = 0,
    flags: int = 0,
    pos: int = 0,
    endpos: int | None = None,
) -> list[str | bytes | None]:
    """Split *string* at matches of *pattern* and return the resulting pieces.

    Split at most *maxsplit* times, or at every match when *maxsplit* is zero.
    Matching is limited to the end-exclusive range from *pos* to *endpos*.
    Captured groups in the separator are included in the result. *flags* are
    used when *pattern* is not already compiled.
    """
    return compile(pattern, flags).split(string, maxsplit, pos, endpos)


_ESCAPED_PATTERN_CHARACTERS = frozenset(
    r"\.^$|?*+()[]{}#&~- " + "\t\n\v\f\r"
)


def escape(pattern: str | bytes) -> str | bytes:
    """Escape regex metacharacters and verbose-mode whitespace or comments.

    Preserve the input type: strings produce strings and bytes produce bytes.
    """
    if not isinstance(pattern, (str, bytes)):
        raise TypeError("pattern must be a string or bytes")
    if isinstance(pattern, bytes):
        escaped = bytearray()
        for value in pattern:
            if chr(value) in _ESCAPED_PATTERN_CHARACTERS:
                escaped.append(ord("\\"))
            escaped.append(value)
        return bytes(escaped)
    return "".join(
        "\\" + character
        if character in _ESCAPED_PATTERN_CHARACTERS
        else character
        for character in pattern
    )
