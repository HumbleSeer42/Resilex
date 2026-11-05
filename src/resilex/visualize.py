"""Export an engine NFA or finite-alphabet DFA as Graphviz DOT."""

from __future__ import annotations

import argparse
from pathlib import Path

from resilex import compile


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Use Graphviz (dot -Tsvg input.dot -o output.svg) to render the DOT output.",
    )
    parser.add_argument("pattern", help="regular expression pattern to visualize")
    parser.add_argument(
        "--format",
        choices=("nfa", "dfa"),
        default="nfa",
        help="visualize the NFA (default) or a minimized DFA",
    )
    parser.add_argument(
        "--alphabet",
        help="finite character alphabet required when --format=dfa",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write Graphviz DOT to this file instead of printing it",
    )
    args = parser.parse_args()

    regex = compile(args.pattern)
    if args.format == "dfa":
        if args.alphabet is None:
            parser.error("--alphabet is required for DFA output")
        dot = regex.to_dfa(args.alphabet).minimize().to_dot()
    else:
        dot = regex.to_dot()

    if args.output is None:
        print(dot)
    else:
        args.output.write_text(dot + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
