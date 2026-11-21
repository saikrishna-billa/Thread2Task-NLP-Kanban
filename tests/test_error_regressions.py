
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from action_pipeline import is_meeting_management
from context_engine import consolidate_tasks
from task_state_engine import track_task_states


def make_candidate(mid, speaker, text, intent):
    return {
        "message_id": mid,
        "speaker": speaker,
        "timestamp": "2026-10-30T09:00:00",
        "text": text,
        "action_probability": 0.995,
        "discourse_intent": intent,
        "decision": "CREATE_CANDIDATE",
    }


def test_greeting_is_filtered():
    text = (
        "Good morning team. "
        "Let's review our sprint priorities."
    )

    assert is_meeting_management(text) is True


def test_real_task_not_filtered():
    assert is_meeting_management(
        "Alex, finish the authentication API by Monday."
    ) is False


def test_documentation_commitment_is_linked():
    messages = [
        make_candidate(
            "m0005",
            "Sarah",
            "Emma will update the project documentation by Tuesday.",
            "UNKNOWN",
        ),
        make_candidate(
            "m0006",
            "Emma",
            "I'll update the documentation.",
            "COMMITMENT",
        ),
    ]

    tasks = consolidate_tasks(messages)

    assert len(tasks) == 1
    assert tasks[0]["source_message_ids"] == [
        "m0005", "m0006"
    ]
    assert tasks[0]["commitment_speaker"] == "Emma"


def test_release_reassignment():
    tasks = [
        {
            "task_id": "t0001",
            "description": "Prepare the testing report",
            "assignee": None,
            "status": "TODO",
            "origin_timestamp": "2026-10-30T09:05:00",
            "source_message_ids": ["m0004"],
        },
        {
            "task_id": "t0002",
            "description": "Test the release",
            "assignee": "David",
            "status": "TODO",
            "origin_timestamp": "2026-10-30T09:16:00",
            "source_message_ids": ["m0009"],
        },
    ]

    messages = [{
        "message_id": "m0014",
        "speaker": "Sarah",
        "timestamp": "2026-10-30T11:30:00",
        "text": (
            "Emma will handle the release testing instead."
        ),
    }]

    result = track_task_states(tasks, messages)

    assert result["unmatched_events"] == []
    assert result["tasks"][0]["assignee"] is None
    assert result["tasks"][1]["assignee"] == "Emma"


def test_unrelated_topic_not_reassigned():
    tasks = [{
        "task_id": "t0001",
        "description": "Test the release",
        "assignee": "David",
        "status": "TODO",
        "origin_timestamp": "2026-10-30T09:00:00",
        "source_message_ids": ["m0001"],
    }]

    messages = [{
        "message_id": "m0002",
        "speaker": "Sarah",
        "timestamp": "2026-10-30T10:00:00",
        "text": "Emma will handle the database migration instead.",
    }]

    result = track_task_states(tasks, messages)

    assert result["tasks"][0]["assignee"] == "David"
    assert len(result["unmatched_events"]) == 1
