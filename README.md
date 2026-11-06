

# Resilex

**A regular-expression engine written in Python, with its own parser, compiler, and matching engines.**

Resilex is a standalone regular-expression engine designed to follow Python's `re` behavior where syntax overlaps, while providing additional extensions and tools for exploring how regex engines work.

Its installed runtime does not import or delegate matching to Python's `re` module.

For regular patterns, Resilex uses an ordered Thompson NFA with a linear-time guarantee for a fixed pattern. This prevents catastrophic backtracking on patterns such as `(a+)+b`.

Patterns containing backreferences or variable-width lookbehind use a separate bounded backtracking matcher and may reach the configured work limit.

> **Note:** Resilex prioritizes learning, experimentation, and bounded execution over speed. It is not a performance replacement for Python's built-in `re` module.

## Installation

Install Resilex from the project directory:

```bash
python -m pip install -e .
```

Resilex requires Python 3.10 or newer and has no third-party runtime dependencies.

Import it as `resilex`.

The package also provides two command-line tools:

- `resilex-visualize`
- `resilex-benchmark`

## Quick start

```python
from resilex import compile

pattern = compile(r"\d+")
match = pattern.search("item-42")

if match:
    print(match.group())  # 42

print(pattern.explain("item-42"))
```

## 200k Tests

NEEDS `hypothesis` TO WORK!

```python
python3 tests/twohundredktests.py <number of workers>
```

---

## At a glance

| Category | Details |
|---|---|
| Language | Python 3.10+ |
| Dependencies | No third-party runtime dependencies |
| Regular-pattern engine | Ordered Thompson NFA |
| Extended-pattern engine | Bounded backtracking |
| Main strength | Linear-time guarantee on the NFA path for a fixed pattern |
| Visualization | NFA and DFA diagrams |
| Compatibility goal | Python `re` behavior where syntax overlaps, with additional extensions |
| Main trade-off | Slower than Python's C-based `re` |

The parser, compiler, matchers, flags, syntax errors, escaping, and pattern representation are implemented by Resilex.

The installed runtime does not use Python's `re` module, even indirectly. `LOCALE` byte matching uses the platform's C locale functions through Python's standard-library `ctypes`.

---

## Features

### Pattern syntax

Resilex supports:

- **Literals and character classes:** `.`, `[abc]`, `[^abc]`, `[a-z]`
- **Character escapes:** `\d`, `\D`, `\w`, `\W`, `\s`, `\S`
- **Boundary and position escapes:** `\b`, `\B`, `\A`, `\Z`
- **Unicode and numeric escapes:** `\x41`, `\101`, `\u0041`, `\N{...}`
- **Unicode properties:** `\p{Letter}`, `\p{Script=Greek}`, `\p{scx=Hira}`, `\P{Number}`
- **Anchors:** `^`, `$`
- **Groups:** capturing, non-capturing, and named groups
- **Backreferences:** numbered (`\1`) and named (`(?P=name)`)
- **Alternation:** `a|b`
- **Quantifiers:** `*`, `+`, `?`, `{n}`, `{n,m}`, `{n,}`, `{,m}`
- **Quantifier modes:** greedy, lazy, and possessive
- **Atomic groups:** `(?>...)`
- **Conditionals:** `(?(1)yes|no)`
- **Lookarounds:** positive and negative lookahead and lookbehind
- **Inline flags:** `(?i)`, scoped flags such as `(?i:...)`
- **Comments:** `(?#...)`

Unicode General_Category and Script properties use bundled Unicode 15.0 data.

### Flags

| Flag | Short alias | Description |
|---|---|---|
| `IGNORECASE` | `I` | Case-insensitive matching |
| `MULTILINE` | `M` | Multiline matching |
| `DOTALL` | `S` | Dot matches newline characters |
| `VERBOSE` | `X` | Verbose pattern formatting |
| `ASCII` | `A` | ASCII character-class semantics |
| `UNICODE` | `U` | Unicode matching semantics |
| `LOCALE` | `L` | Locale-dependent byte matching |
| `DEBUG` | — | Display compiled NFA debugging information |
| `NOFLAG` | — | No flags enabled |

`LOCALE` is available for bytes patterns.

Case-insensitive matching includes special Unicode case pairs, such as `İ`/`i`, `ı`/`I`, and the Kelvin sign/`K`, as well as case-insensitive backreferences.

`LOCALE` byte matching uses the platform C runtime's locale character-classification and case-folding functions.

### Python-style API

Resilex provides module-level functions and corresponding compiled-pattern methods.

Supported functions include:

- `compile`
- `search`
- `match`
- `fullmatch`
- `finditer`
- `findall`
- `sub`
- `subn`
- `split`
- `escape`
- `purge`

Additional API features:

- `pos` and `endpos` matching arguments.
- Match objects with `group`, `groups`, `groupdict`, `start`, `end`, `span`, and `expand`.
- Match metadata including `lastindex`, `lastgroup`, `regs`, `string`, `re`, `pos`, and `endpos`.
- Numbered match indexing, such as `match[1]`.
- Compiled-pattern explanations through `.explain(text)`.
- Support for `str`, `bytes`, `bytearray`, and `memoryview`.
- Pickling and copying of compiled patterns.
- Shared compiled-pattern concurrency tests covering search, match, fullmatch, findall, and finditer.

The `.explain(text)` method shows the actual search path, including match spans or reasons candidate paths failed.

---

## Execution model and safety

Resilex uses two matching approaches depending on the pattern.

### Thompson NFA

Regular patterns are executed using an ordered Thompson NFA.

For a fixed pattern, this execution path has a linear-time guarantee with respect to input length.

Nested quantifiers such as `(a+)+b` do not trigger catastrophic backtracking on this path.

### Bounded backtracking

Patterns requiring backtracking, including those with backreferences and variable-width lookbehind, use a separate bounded backtracking matcher.

These patterns do not have the same linear-time guarantee and may raise `RegexMatchLimitError` when the configured work limit is reached.

### Configurable limits

`RegexLimits` controls resource limits during compilation, automaton construction, and bounded backtracking.

| Limit | Default |
|---|---:|
| Pattern length | 65,536 |
| Group nesting depth | 100 |
| Counted repeat bound | 49,999 |
| NFA states | 100,000 |
| DFA states | 10,000 |
| DFA alphabet size | 256 |
| Backtracking steps | 1,000,000 |

The backtracking-step limit applies to the bounded VM, not the NFA execution path.

Exceeding configured limits raises `RegexComplexityError` or `RegexMatchLimitError`, rather than allowing the relevant operation to continue without a bound.

Example:

```python
from resilex import RegexLimits, compile

limits = RegexLimits(backtrack_steps=100_000)

pattern = compile(
    r"(a+)+b",
    limits=limits
)

print(pattern.search("a" * 1000))
```

---

## Tools

### NFA visualizer

Generates Graphviz DOT representations of compiled NFA structures.

Tool: `resilex-visualize`

Example:

```bash
resilex-visualize '\d+' --output pattern.dot
```

Render the generated DOT file using Graphviz:

```bash
dot -Tsvg pattern.dot -o pattern.svg
```

The visualizer also supports DFA output for a specified finite alphabet:

```bash
resilex-visualize 'a|b' --format dfa --alphabet ab --output pattern.dot
```

### DFA conversion and minimization

Resilex supports:

- Subset construction.
- Hopcroft minimization.
- Explicit alphabet-based DFA construction.
- Anchors and single-character lookarounds within the supported DFA conversion boundaries.

### Benchmarking

The benchmark tool measures Resilex against pathological input patterns, including `(a+)+b`, across different input sizes.

Tool: `resilex-benchmark`

Check available options:

```bash
resilex-benchmark --help
```

---

## Testing and verification

Resilex includes a test suite covering supported syntax, matching behavior, limits, concurrency, and other engine functionality.

### Test suite

Run:

```bash
python -m pytest -q
```

The current test suite contains 64 tests.

The latest recorded run completed with:

- 64 tests passed.
- 10,282 subtests passed.

### CPython expected-results corpus

Resilex includes a comparison script for the recorded expected-results corpus from CPython v3.10.13:

- 403/403 corpus cases passed.
- Cases cover syntax errors, no-match results, and successful search results.
- Successful cases evaluate expected expressions against whole matches, capture groups, named groups, and flags.

To reproduce:

```bash
curl -fsSLo /tmp/re_tests.py \
  https://raw.githubusercontent.com/python/cpython/v3.10.13/Lib/test/re_tests.py

python tests/check_cpython_conformance.py /tmp/re_tests.py
```

This is a fixture-based check against recorded CPython expectations. It is not a live comparison against CPython's regex engine and does not run the complete CPython `test_re.py` suite.

### Additional verification

The project also includes:

- Installed-runtime checks verifying that the engine does not import Python's `re` module.
- An isolated runtime check using `python -S`.
- Seeded randomized search checks across supported syntax, flags, and bounds.
- Search-span cross-checks against the engine's full-match operation.
- Locale byte-matching tests covering all 256 byte values under available locales.
- Shared-pattern concurrency checks.

The project has been tested against Python 3.12.3.

---

## Reproducible clean-wheel verification

The following procedure builds both distribution formats and tests the installed wheel from a fresh virtual environment.

Requirements: Python 3.10 or newer.

```bash
python3.10 -m pip install build

python3.10 -m build \
  --sdist \
  --wheel \
  --outdir /tmp/resilex-dist

python3.10 -m venv /tmp/resilex-clean-venv

/tmp/resilex-clean-venv/bin/python -m pip install --no-deps \
  /tmp/resilex-dist/resilex-1.0.0-py3-none-any.whl

mkdir -p /tmp/resilex-validation

cp tests/test_regex_engine.py tests/check_cpython_conformance.py \
  /tmp/resilex-validation/

cd /tmp/resilex-validation
```

Verify the installed package:

```bash
/tmp/resilex-clean-venv/bin/python -c \
'import sys, resilex; p=resilex.__file__; assert p.startswith(sys.prefix); assert "re" not in sys.modules; assert resilex.compile(r"\p{Script=Greek}+").fullmatch("Αβ"); print(p)'
```

Run the tests:

```bash
/tmp/resilex-clean-venv/bin/python -m unittest -v
```

Run the CPython corpus check:

```bash
curl -fsSLo re_tests.py \
  https://raw.githubusercontent.com/python/cpython/v3.10.13/Lib/test/re_tests.py

/tmp/resilex-clean-venv/bin/python \
  check_cpython_conformance.py re_tests.py
```

Inspect the distribution archives:

```bash
python3.10 -m zipfile -l \
  /tmp/resilex-dist/resilex-1.0.0-py3-none-any.whl

tar -tzf /tmp/resilex-dist/resilex-1.0.0.tar.gz
```

Confirm that both archives contain:

- `LICENSE`
- `LICENSE-UNICODE.txt`
- `NOTICE`
- `resilex_unicode_data/LICENSE.txt`
- All three bundled Unicode data files.

For another machine, substitute its Python 3.10+ executable and repeat the procedure.

Results reported here apply only to the machines and interpreters actually tested.

---

## Limitations and compatibility boundaries

### Compatibility

- Resilex aims to follow Python `re` behavior where syntax overlaps, but it is not a complete replacement.
- Script and Script_Extensions properties use bundled Unicode 15.0 data. Newer Unicode scripts are not included.
- Variable-width lookbehind is a Resilex extension and is not accepted by Python's `re`.
- Variable-width lookbehind may require trying multiple candidate starting positions.
- DFA conversion supports only single-character lookarounds.
- Backreferences and conditionals are not supported by DFA conversion. Backreferences can describe non-regular languages, which cannot generally be represented by a DFA.
- Syntax errors are reported through Resilex's own `RegexSyntaxError` class, which does not inherit from Python's `re.error`.

### Performance and trade-offs

**Speed:** Resilex is implemented in Python and is substantially slower than the C-based `re` engine.

In one development-machine test:

| Operation | Approximate time |
|---|---:|
| Searching `\d+` over 1 MB | 1.8 seconds |
| Case-insensitive literal search over 1 MB | 6 seconds |

These figures are machine-dependent and should not be treated as universal benchmarks.

**Backtracking:** Backreference patterns and variable-width lookbehind use bounded backtracking and do not receive the NFA path's linear-time guarantee.

**Default limits:** Resilex has configurable resource limits that may be stricter than Python's `re`. See the `RegexLimits` table above.

---

## Unicode data attribution

Resilex bundles the following files from Unicode Character Database 15.0.0:

- `Scripts.txt`
- `ScriptExtensions.txt`
- `PropertyValueAliases.txt`

Source: [Unicode Character Database 15.0.0](https://www.unicode.org/Public/15.0.0/ucd/)

Copyright © 2022 Unicode, Inc.

These files are subject to the Unicode License V3. See:

- [`LICENSE-UNICODE.txt`](./LICENSE-UNICODE.txt)
- [`NOTICE`](./NOTICE)

Resilex's own source code is licensed under the MIT License.

See [`LICENSE`](./LICENSE).

The bundled Unicode data files are subject to their separate Unicode License V3 terms.

---

## Who is Resilex for?

**Good fit:**

- Learning how regular-expression engines work.
- Studying automata and matching algorithms.
- Experimenting with regular-expression syntax.
- Exploring NFA and DFA structures.
- Running supported untrusted patterns with bounded execution.

**Not a good fit:**

- High-volume text processing.
- Performance-critical production workloads.
- Applications requiring complete compatibility with Python's `re`.

---

## Development status

Resilex is currently in active development.

Although the project includes an extensive test suite and multiple verification procedures, bugs and compatibility differences may still exist.

Additional testing, fuzzing, fixes, and improvements will be carried out progressively.

Contributions, bug reports, and constructive feedback are welcome.

---

## Attribution

Resilex was originally created by **Minecrafterify**.

You are free to use, modify, distribute, and sell software based on Resilex under the MIT License.

**This licensing applies to all versions of the project, including earlier commits in the repository's Git history.**

If you use or build upon Resilex, please credit **Minecrafterify** as the original creator and link back to the original repository:

https://github.com/Minecrafterify/Resilex

For additional attribution information, see [ATTRIBUTION.md](ATTRIBUTION.md).


