"""Check Resilex against recorded results in CPython's re_tests.py corpus."""

import sys
from typing import Any

from resilex.regex_engine import Regex, RegexSyntaxError

_EXPECTED_CASES = 403


def _match_variables(match):
    variables: dict[str, Any] = {
        "found": match.group(0),
        "groups": match.group(),
        "flags": match.re.flags,
    }
    for group_index in range(1, 100):
        try:
            value = match.group(group_index)
        except IndexError:
            value = "Error"
        variables[f"g{group_index}"] = "None" if value is None else value
    for name in match.re.groupindex:
        value = match.group(name)
        variables[name] = "None" if value is None else value
    return variables


def _evaluate_result(match, expression):
    return eval(expression, {"__builtins__": {}}, _match_variables(match))


def _error(message):
    print(f"{sys.argv[0]}: error: {message}", file=sys.stderr)
    raise SystemExit(2)


def _load_corpus(path):
    with open(path, encoding="utf-8") as corpus_file:
        source = corpus_file.read()
    namespace = {"__file__": path, "__name__": "__resilex_corpus__"}
    exec(compile(source, path, "exec"), namespace)
    return namespace


def main() -> int:
    usage = f"Usage: {sys.argv[0]} <CPython-v3.10.13-re_tests.py>"
    if sys.argv[1:] in (["-h"], ["--help"]):
        print(usage)
        return 0
    if len(sys.argv) != 2:
        print(usage, file=sys.stderr)
        return 2

    corpus = _load_corpus(sys.argv[1])
    cases = corpus["tests"]
    if len(cases) != _EXPECTED_CASES:
        _error(f"expected {_EXPECTED_CASES} corpus cases, found {len(cases)}")

    passed = 0
    failures: list[str] = []
    for number, case in enumerate(cases, start=1):
        if len(case) not in (3, 5):
            _error(f"case {number} has {len(case)} fields; expected 3 or 5")

        pattern, text, outcome = case[:3]
        if outcome == corpus["SYNTAX_ERROR"]:
            try:
                Regex(pattern)
            except RegexSyntaxError:
                passed += 1
            else:
                failures.append(
                    f"case {number}: Resilex accepted invalid pattern {pattern!r}"
                )
            continue

        actual = Regex(pattern).search(text)
        if outcome == corpus["FAIL"]:
            if actual is None:
                passed += 1
            else:
                failures.append(
                    f"case {number}: expected no match; Resilex matched"
                )
            continue

        if outcome != corpus["SUCCEED"]:
            _error(f"case {number} has unknown outcome {outcome!r}")
        if actual is None:
            failures.append(
                f"case {number}: expected a match; Resilex did not match"
            )
            continue

        expression, expected = case[3:]
        actual_result = _evaluate_result(actual, expression)
        if actual_result != expected:
            failures.append(
                f"case {number}: Resilex result {actual_result!r} "
                f"does not match corpus expectation {expected!r}"
            )
        else:
            passed += 1

    for failure in failures:
        print(failure)
    print(f"Recorded re_tests.py expected-results corpus: {passed}/{len(cases)} passed")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
