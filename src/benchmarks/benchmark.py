"""Benchmark Thompson-NFA search on a pathological pattern."""

from __future__ import annotations

import math
import os
import sys
from timeit import timeit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resilex import compile


PATTERN = "(a+)+b"


def write_svg(rows: list[tuple[int, float]], output: str) -> None:
    width, height = 800, 440
    left, right, top, bottom = 80, 30, 55, 75
    plot_width, plot_height = width - left - right, height - top - bottom
    max_value = max((seconds for _, seconds in rows), default=0.00001)
    max_value = max(max_value, 0.00001)
    x = lambda index: left + plot_width * index / max(1, len(rows) - 1)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="400" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">Resilex Thompson NFA search</text>',
        f'<text x="400" y="{height - 20}" text-anchor="middle" font-family="sans-serif">Input length (a repeated, then c; no match)</text>',
        f'<text x="20" y="220" transform="rotate(-90 20 220)" text-anchor="middle" font-family="sans-serif">Seconds (log scale)</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#444"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#444"/>',
    ]

    def y_log(value: float) -> float:
        lo = -5.0
        hi = max(0.0, math.log10(max_value))
        current = math.log10(max(value, 1e-5))
        return top + plot_height * (1 - (current - lo) / max(hi - lo, 0.01))

    points = " ".join(
        f"{x(index):.1f},{y_log(seconds):.1f}"
        for index, (_, seconds) in enumerate(rows)
    )
    lines.extend(
        [
            f'<polyline points="{points}" fill="none" stroke="#1463c3" stroke-width="3"/>',
        ]
    )
    for index, (length, seconds) in enumerate(rows):
        cx = x(index)
        cy = y_log(seconds)
        lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="#1463c3"/>')
        lines.append(
            f'<text x="{cx:.1f}" y="{top + plot_height + 22}" text-anchor="middle" font-family="sans-serif" font-size="12">{length}</text>'
        )
        lines.append(
            f'<text x="{cx:.1f}" y="{cy - 9:.1f}" text-anchor="middle" font-family="sans-serif" font-size="10">{seconds:.3g}s</text>'
        )
    lines.extend(
        [
            '<line x1="550" y1="48" x2="580" y2="48" stroke="#1463c3" stroke-width="3"/>',
            '<text x="588" y="52" font-family="sans-serif" font-size="12">Resilex</text>',
            f'<text x="{left}" y="{height - 42}" font-family="sans-serif" font-size="11">Pattern: {PATTERN}</text>',
            "</svg>",
        ]
    )
    output_directory = os.path.dirname(output)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    with open(output, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines) + "\n")


def _parse_args(arguments: list[str]) -> tuple[list[int], str]:
    lengths = [8, 12, 16, 20, 24]
    output = "benchmarks/benchmark.svg"
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in ("-h", "--help"):
            print(
                f"Usage: {sys.argv[0]} [--lengths N [N ...]] [--output PATH]\n"
                "  --lengths  number of 'a' characters before the final c\n"
                "  --output   output SVG graph"
            )
            raise SystemExit(0)
        if argument == "--lengths":
            values = []
            index += 1
            while index < len(arguments) and not arguments[index].startswith("--"):
                try:
                    values.append(int(arguments[index]))
                except ValueError:
                    raise SystemExit(
                        f"{sys.argv[0]}: invalid length {arguments[index]!r}"
                    ) from None
                index += 1
            if not values:
                raise SystemExit(f"{sys.argv[0]}: --lengths requires at least one value")
            lengths = values
            continue
        if argument == "--output":
            index += 1
            if index == len(arguments) or arguments[index].startswith("--"):
                raise SystemExit(f"{sys.argv[0]}: --output requires a path")
            output = arguments[index]
            index += 1
            continue
        raise SystemExit(f"{sys.argv[0]}: unrecognized argument {argument!r}")
    return lengths, output


def main() -> None:
    lengths, output = _parse_args(sys.argv[1:])
    rows = []
    matcher = compile(PATTERN)
    for length in lengths:
        text = "a" * length + "c"
        nfa = timeit(lambda: matcher.search(text), number=3) / 3
        rows.append((length, nfa))
        print(f"{length:>5}  Resilex Thompson NFA {nfa:.6f}s")

    write_svg(rows, output)
    print(f"Graph written to {output}")


if __name__ == "__main__":
    main()
