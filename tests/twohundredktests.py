import re
import resilex

from concurrent.futures import ThreadPoolExecutor
from hypothesis import given, settings, strategies as st

import argparse

print(r"""
 ____           _ _           
|  _ \ ___  ___(_) | _____  __
| |_) / _ \/ __| | |/ _ \ \/ /
|  _ <  __/\__ \ | |  __/>  < 
|_| \_\___||___/_|_|\___/_/\_\

    Multithreaded 200k Test""")

# Parse Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument("workercount", help="Number of workers")
args = parser.parse_args()

# Configuration
WORKERS = int(args.workercount)
TOTAL_TESTS = 200_000
BATCH_SIZE = WORKERS

# Regex pattern generator
atom = st.sampled_from([
    "a", "b", ".", "[ab]", r"\w", r"\d", r"\b"
])

pattern = st.recursive(
    atom,
    lambda s: st.one_of(
        st.tuples(s, s).map(lambda t: t[0] + t[1]),
        st.tuples(s, s).map(
            lambda t: f"(?:{t[0]}|{t[1]})"
        ),
        st.tuples(
            s,
            st.sampled_from(["*", "+", "?", "*?", "{1,2}"])
        ).map(lambda t: f"(?:{t[0]}){t[1]}"),
    ),
    max_leaves=8,
)

text_strategy = st.text(
    alphabet="abc1 _\n",
    max_size=8
)

case_strategy = st.tuples(pattern, text_strategy)

batch_strategy = st.lists(
    case_strategy,
    min_size=BATCH_SIZE,
    max_size=BATCH_SIZE
)


def compare_case(case):
    p, s = case

    try:
        ref = re.compile(p)
    except re.error:
        return True

    a = ref.search(s)
    b = resilex.compile(p).search(s)

    expected = (a and (a.span(), a.groups()))
    actual = (b and (b.span(), b.groups()))

    assert expected == actual, (
        f"\nPattern: {p!r}"
        f"\nString: {s!r}"
        f"\nPython re: {expected}"
        f"\nResilex: {actual}"
    )

    return True


@settings(
    max_examples=TOTAL_TESTS // BATCH_SIZE,
    deadline=None
)
@given(batch_strategy)
def test_multithreaded(batch):
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        results = executor.map(compare_case, batch)

        for result in results:
            assert result


if __name__ == "__main__":
    test_multithreaded()

    print(f"All {TOTAL_TESTS:,} concurrent comparisons passed!")
    print(f"Worker threads: {WORKERS}")
