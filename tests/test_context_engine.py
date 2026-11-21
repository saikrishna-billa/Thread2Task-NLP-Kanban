
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)

from context_engine import consolidate_tasks


def make_message(
    mid,
    speaker,
    text,
    intent="UNKNOWN",
    decision="IGNORE",
    probability=0.1,
):
    return {
        "message_id": mid,
        "speaker": speaker,
        "timestamp": None,
        "text": text,
        "discourse_intent": intent,
        "decision": decision,
        "action_probability": probability,
    }


def test_commitment_links_to_previous_task():
    messages = [
        make_message(
            "m0001", "Sarah",
            "Alex, finish the API by Friday.",
            "UNKNOWN", "CREATE_CANDIDATE", 0.99
        ),
        make_message(
            "m0002", "Alex",
            "I'll handle it.",
            "COMMITMENT", "CREATE_CANDIDATE", 0.99
        ),
    ]

    tasks = consolidate_tasks(messages)

    assert len(tasks) == 1
    assert tasks[0]["source_message_ids"] == [
        "m0001", "m0002"
    ]
    assert tasks[0]["commitment_speaker"] == "Alex"
    assert tasks[0]["context_linked"] is True


def test_suggestion_not_promoted():
    messages = [
        make_message(
            "m0001", "David",
            "We should consider redesigning the UI.",
            "SUGGESTION", "REVIEW", 0.99
        )
    ]

    assert consolidate_tasks(messages) == []


def test_review_assignment_preserved():
    messages = [
        make_message(
            "m0001", "Sarah",
            "Alex has been assigned to test the release.",
            "ASSIGNMENT", "REVIEW", 0.04
        )
    ]

    tasks = consolidate_tasks(messages)

    assert len(tasks) == 1
    assert tasks[0]["needs_review"] is True


def test_unresolved_commitment_not_invented():
    messages = [
        make_message(
            "m0001", "Alex",
            "I'll handle it.",
            "COMMITMENT", "CREATE_CANDIDATE", 0.99
        )
    ]

    assert consolidate_tasks(messages) == []


def test_distinct_requests():
    messages = [
        make_message(
            "m0001", "Sarah",
            "Prepare the slides.",
            "REQUEST", "CREATE_CANDIDATE", 0.99
        ),
        make_message(
            "m0002", "David",
            "Review the pull request.",
            "REQUEST", "CREATE_CANDIDATE", 0.98
        ),
    ]

    tasks = consolidate_tasks(messages)

    assert len(tasks) == 2


def test_unrelated_message_breaks_link():
    messages = [
        make_message(
            "m0001", "Sarah",
            "Complete the report.",
            "REQUEST", "CREATE_CANDIDATE", 0.99
        ),
        make_message(
            "m0002", "David",
            "The weather is nice.",
            "UNKNOWN", "IGNORE", 0.01
        ),
        make_message(
            "m0003", "Alex",
            "I'll handle it.",
            "COMMITMENT", "CREATE_CANDIDATE", 0.99
        ),
    ]

    tasks = consolidate_tasks(messages)

    assert len(tasks) == 1
    assert tasks[0]["source_message_ids"] == ["m0001"]
