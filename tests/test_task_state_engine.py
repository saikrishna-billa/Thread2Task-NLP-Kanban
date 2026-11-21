
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from task_state_engine import (
    detect_event,
    track_task_states,
)


def make_task(
    task_id="t0001",
    description="Finish the login API",
    assignee="Alex",
):
    return {
        "task_id": task_id,
        "description": description,
        "assignee": assignee,
        "status": "TODO",
        "origin_timestamp": "2026-10-30T09:00:00",
        "source_message_ids": ["m0001"],
    }


def make_message(
    mid,
    speaker,
    text,
    timestamp="2026-10-30T10:00:00",
):
    return {
        "message_id": mid,
        "speaker": speaker,
        "timestamp": timestamp,
        "text": text,
    }


def test_detect_started():
    event = detect_event(
        "I've started working on the login API."
    )
    assert event["event_type"] == "STARTED"


def test_started_updates_status():
    messages = [
        make_message(
            "m0002",
            "Alex",
            "I've started working on the login API.",
        )
    ]

    result = track_task_states(
        [make_task()], messages
    )

    task = result["tasks"][0]

    assert task["status"] == "IN_PROGRESS"
    assert len(task["state_history"]) == 1


def test_completed_updates_status():
    messages = [
        make_message(
            "m0002",
            "Alex",
            "The login API is complete.",
        )
    ]

    result = track_task_states(
        [make_task()], messages
    )

    assert result["tasks"][0]["status"] == "DONE"
    assert result["unmatched_events"] == []


def test_reassignment():
    messages = [
        make_message(
            "m0002",
            "Sarah",
            "David will handle the login API instead.",
        )
    ]

    result = track_task_states(
        [make_task()], messages
    )

    task = result["tasks"][0]

    assert task["assignee"] == "David"
    assert task["status"] == "TODO"


def test_cancellation():
    messages = [
        make_message(
            "m0002",
            "Sarah",
            "We no longer need the login API.",
        )
    ]

    result = track_task_states(
        [make_task()], messages
    )

    assert result["tasks"][0]["status"] == "CANCELLED"


def test_unrelated_update_not_applied():
    messages = [
        make_message(
            "m0002",
            "Alex",
            "The payment gateway is complete.",
        )
    ]

    result = track_task_states(
        [make_task()], messages
    )

    assert result["tasks"][0]["status"] == "TODO"
    assert len(result["unmatched_events"]) == 1
    assert (
        result["unmatched_events"][0]["reason"]
        == "NO_MATCH"
    )


def test_ambiguous_task_not_updated():
    tasks = [
        make_task("t0001", "Fix login API"),
        make_task("t0002", "Improve payment API"),
    ]

    messages = [
        make_message(
            "m0002",
            "Alex",
            "The API is complete.",
        )
    ]

    result = track_task_states(tasks, messages)

    assert all(
        task["status"] == "TODO"
        for task in result["tasks"]
    )

    assert (
        result["unmatched_events"][0]["reason"]
        == "AMBIGUOUS_MATCH"
    )


def test_terminal_task_not_reopened():
    messages = [
        make_message(
            "m0002",
            "Alex",
            "The login API is complete.",
        ),
        make_message(
            "m0003",
            "Alex",
            "I've started working on the login API.",
            "2026-10-30T11:00:00",
        ),
    ]

    result = track_task_states(
        [make_task()], messages
    )

    assert result["tasks"][0]["status"] == "DONE"
    assert (
        result["unmatched_events"][0]["reason"]
        == "TERMINAL_TASK"
    )


def test_provenance_preserved():
    messages = [
        make_message(
            "m0002",
            "Alex",
            "The login API is complete.",
        )
    ]

    result = track_task_states(
        [make_task()], messages
    )

    task = result["tasks"][0]

    assert task["source_message_ids"] == [
        "m0001", "m0002"
    ]

    assert task["state_history"][0][
        "message_id"
    ] == "m0002"
