
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from parser_v2 import parse_transcript
from run_pipeline import run_thread2task


class FakeActionPipeline:
    """
    Deterministic NLP outputs for integration testing.
    Real execution uses the trained DistilBERT model.
    """

    def process(self, transcript, known_speakers=None):
        messages = parse_transcript(
            transcript,
            known_speakers=known_speakers,
        )

        results = []

        for message in messages:
            text = message["text"]

            if "finish the login API" in text:
                intent = "UNKNOWN"
                decision = "CREATE_CANDIDATE"

            elif "I'll handle it" in text:
                intent = "COMMITMENT"
                decision = "CREATE_CANDIDATE"

            elif "should consider" in text:
                intent = "SUGGESTION"
                decision = "REVIEW"

            elif "Please prepare" in text:
                intent = "REQUEST"
                decision = "CREATE_CANDIDATE"

            elif "has been assigned" in text:
                intent = "ASSIGNMENT"
                decision = "REVIEW"

            elif "started working" in text:
                intent = "UNKNOWN"
                decision = "CREATE_CANDIDATE"

            elif "is complete" in text:
                intent = "UNKNOWN"
                decision = "CREATE_CANDIDATE"

            else:
                intent = "UNKNOWN"
                decision = "IGNORE"

            results.append({
                **message,
                "action_probability": (
                    0.99
                    if decision == "CREATE_CANDIDATE"
                    else 0.04
                ),
                "discourse_intent": intent,
                "discourse_evidence": "Test fixture",
                "decision": decision,
            })

        return results


TRANSCRIPT = """
[2026-10-30 09:00] Sarah: Alex, finish the login API by Friday.
[2026-10-30 09:05] Alex: I'll handle it.
[2026-10-30 09:10] David: We should consider redesigning the dashboard.
[2026-10-30 09:20] David: Please prepare the testing report.
[2026-10-30 09:30] Sarah: Alex has been assigned to test the release.
[2026-10-30 10:00] Alex: I've started working on the login API.
[2026-10-30 11:00] Alex: The login API is complete.
"""


def get_result():
    return run_thread2task(
        TRANSCRIPT,
        action_pipeline=FakeActionPipeline(),
    )


def test_correct_number_of_tasks():
    result = get_result()

    assert result["metadata"]["total_messages"] == 7
    assert result["metadata"]["final_tasks"] == 3


def test_login_task_completed():
    result = get_result()

    login_tasks = [
        task for task in result["tasks"]
        if task["description"] == "Finish the login API"
    ]

    assert len(login_tasks) == 1

    task = login_tasks[0]

    assert task["assignee"] == "Alex"
    assert task["due_date"] == "2026-10-30"
    assert task["status"] == "DONE"
    assert len(task["state_history"]) == 2


def test_no_duplicate_progress_tasks():
    result = get_result()

    assert not any(
        "started working" in (
            task["description_raw"].lower()
        )
        for task in result["tasks"]
    )


def test_suggestion_not_created():
    result = get_result()

    assert not any(
        "redesigning the dashboard" in (
            task["description_raw"].lower()
        )
        for task in result["tasks"]
    )


def test_review_assignment_preserved():
    result = get_result()

    matching = [
        task for task in result["tasks"]
        if task["description"] == "Test the release"
    ]

    assert len(matching) == 1
    assert matching[0]["assignee"] == "Alex"
    assert matching[0]["needs_review"] is True


def test_provenance_is_preserved():
    result = get_result()

    login_task = next(
        task for task in result["tasks"]
        if task["description"] == "Finish the login API"
    )

    assert login_task["source_message_ids"] == [
        "m0001",
        "m0002",
        "m0006",
        "m0007",
    ]


def test_unrelated_status_not_created():
    result = run_thread2task(
        """
[2026-10-30 09:00] Sarah: The server crashed yesterday.
""",
        action_pipeline=FakeActionPipeline(),
    )

    assert result["tasks"] == []


def test_empty_input():
    result = run_thread2task(
        "",
        action_pipeline=FakeActionPipeline(),
    )

    assert result["metadata"]["total_messages"] == 0
    assert result["tasks"] == []
