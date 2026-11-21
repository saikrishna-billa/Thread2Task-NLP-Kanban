
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)

from discourse_engine import analyze_discourse


def test_noncommittal_suggestions():
    examples = [
        "We should consider changing the database.",
        "Maybe we could redesign the homepage.",
        "It might be useful to add notifications.",
    ]

    for sentence in examples:
        assert analyze_discourse(sentence)["intent"] == "SUGGESTION"


def test_passive_assignment():
    result = analyze_discourse(
        "Alex has been assigned to test the release."
    )

    assert result["intent"] == "ASSIGNMENT"


def test_explicit_commitment():
    result = analyze_discourse(
        "I'll fix the authentication bug."
    )

    assert result["intent"] == "COMMITMENT"


def test_explicit_request():
    result = analyze_discourse(
        "Please review the pull request."
    )

    assert result["intent"] == "REQUEST"


def test_unknown_statement():
    result = analyze_discourse(
        "The server crashed last night."
    )

    assert result["intent"] == "UNKNOWN"


def test_empty_input():
    assert analyze_discourse("")["intent"] == "UNKNOWN"
