
from task_state_engine import detect_event


def test_imperative_cancellation():
    event = detect_event(
        "Cancel the survey. Research already covered it."
    )

    assert event is not None
    assert event["event_type"] == "CANCELLED"
    assert event["topic"] == "the survey"


def test_ongoing_work():
    event = detect_event(
        "I'm still working on the login API."
    )

    assert event is not None
    assert event["event_type"] == "STARTED"
    assert event["topic"] == "the login API"


def test_negated_completion_not_completed():
    event = detect_event(
        "I haven't finished the incident report."
    )

    assert event is None


def test_hypothetical_not_state_update():
    event = detect_event(
        "We might replace that dependency eventually."
    )

    assert event is None



from task_state_engine import track_task_states


def test_reassignment_detection():
    text = (
        "Correction: Meera will handle "
        "the onboarding tutorial, not Rohan."
    )

    event = detect_event(text)

    assert event is not None
    assert event["event_type"] == "REASSIGNED"
    assert event["new_assignee"] == "Meera"
    assert event["topic"] == "the onboarding tutorial"


def test_reassignment_updates_existing_task():
    tasks = [{
        "task_id": "t0001",
        "description": "Create the onboarding tutorial",
        "assignee": "Rohan",
        "status": "TODO",
        "origin_timestamp": "2026-11-10 11:00",
        "source_message_ids": ["m0001"],
    }]

    messages = [{
        "message_id": "m0002",
        "speaker": "Asha",
        "timestamp": "2026-11-10 11:05",
        "text": (
            "Correction: Meera will handle "
            "the onboarding tutorial, not Rohan."
        ),
    }]

    result = track_task_states(tasks, messages)

    assert len(result["tasks"]) == 1
    assert result["tasks"][0]["assignee"] == "Meera"
    assert result["tasks"][0]["status"] == "TODO"
    assert len(result["tasks"][0]["state_history"]) == 1
    assert result["unmatched_events"] == []
