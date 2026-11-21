
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from task_extractor import enrich_task, extract_deadline


def make_task(
    text,
    speaker="Sarah",
    timestamp="2026-10-30T09:00:00",
    commitment_speaker=None,
    needs_review=False,
):
    return {
        "task_id": "t0001",
        "description_raw": text,
        "origin_speaker": speaker,
        "origin_timestamp": timestamp,
        "commitment_speaker": commitment_speaker,
        "needs_review": needs_review,
        "source_message_ids": ["m0001"],
        "evidence": [],
        "status": "TODO",
    }


def test_direct_request():
    task = make_task(
        "Alex, finish the login API by Friday.",
        commitment_speaker="Alex",
    )

    result = enrich_task(task)

    assert result["description"] == "Finish the login API"
    assert result["assignee"] == "Alex"
    assert result["due_date"] == "2026-10-30"
    assert result["needs_review"] is False


def test_unassigned_request():
    task = make_task(
        "Please prepare the testing report.",
        speaker="David",
    )

    result = enrich_task(task)

    assert result["description"] == "Prepare the testing report"
    assert result["assignee"] is None
    assert result["due_date"] is None


def test_passive_assignment():
    task = make_task(
        "Alex has been assigned to test the release.",
        needs_review=True,
    )

    result = enrich_task(task)

    assert result["description"] == "Test the release"
    assert result["assignee"] == "Alex"
    assert result["needs_review"] is True


def test_first_person_commitment():
    task = make_task(
        "I'll prepare the presentation tomorrow.",
        speaker="David",
    )

    result = enrich_task(task)

    assert result["description"] == "Prepare the presentation"
    assert result["assignee"] == "David"
    assert result["due_date"] == "2026-10-31"


def test_next_weekday():
    result = extract_deadline(
        "Submit the report next Monday.",
        "2026-10-30T09:00:00",
    )

    assert result["due_date"] == "2026-11-02"


def test_missing_reference_date():
    task = make_task(
        "Alex, submit the report tomorrow.",
        timestamp=None,
    )

    result = enrich_task(task)

    assert result["due_date"] is None
    assert result["deadline_text"] is not None
    assert result["needs_review"] is True


def test_conflicting_assignees():
    task = make_task(
        "Sarah, update the dashboard.",
        commitment_speaker="Alex",
    )

    result = enrich_task(task)

    assert result["assignee"] == "Sarah"
    assert result["assignee_conflict"] is True
    assert result["needs_review"] is True


def test_explicit_iso_deadline():
    task = make_task(
        "Alex, submit the report by 2026-11-05.",
    )

    result = enrich_task(task)

    assert result["due_date"] == "2026-11-05"
    assert result["description"] == "Submit the report"
