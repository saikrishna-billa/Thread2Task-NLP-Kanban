
import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from parser_v2 import parse_transcript


ROOT = Path(__file__).resolve().parent

TASKS_PATH = ROOT / "data/processed/structured_tasks.json"
OUTPUT_PATH = ROOT / "data/processed/state_tracking_results.json"

STOPWORDS = {
    "the", "a", "an", "to", "for", "of", "in",
    "on", "at", "by", "with", "and", "or", "is",
    "are", "was", "were", "be", "been", "it",
    "this", "that", "please", "will", "should",
    "i", "we", "you", "he", "she", "they",
    "finish", "complete", "completed", "done",
    "start", "started", "working", "work",
    "prepare", "update", "handle",
      # Common, but often too generic alone
    "changes", "change",
}

EVENT_PATTERNS = [
    
    (
        "CANCELLED",
        re.compile(
            r"^cancel\s+(?P<topic>[^.!?]+)"
            r"(?:[.!?].*)?$",
            re.IGNORECASE,
        ),
    ),
    (
        "STARTED",
        re.compile(
            r"^(?:i['’]m|i\s+am)\s+"
            r"(?:still\s+)?working\s+on\s+"
            r"(?P<topic>.+?)[.!]?$",
            re.IGNORECASE,
        ),
    ),
    
    (
        "REASSIGNED",
        re.compile(
            r"^(?:correction\s*:\s*)?"
            r"(?P<person>[A-Za-z][\w.-]*)\s+"
            r"will\s+(?:handle|take\s+over|work\s+on)\s+"
            r"(?P<topic>.+?)"
            r"(?:\s*,?\s+not\s+[A-Za-z][\w.-]*)"
            r"[.!]?$",
            re.IGNORECASE,
        ),
    ),


    (
        "REASSIGNED",
        re.compile(
            r"^(?P<person>[A-Za-z][\w.-]*)\s+"
            r"will\s+(?:handle|take\s+over|work\s+on)"
            r"\s+(?P<topic>.+?)\s+instead[.!]?$",
            re.IGNORECASE,
        ),
    ),
    (
        "CANCELLED",
        re.compile(
            r"^(?:we|i)\s+no\s+longer\s+need\s+"
            r"(?P<topic>.+?)[.!]?$",
            re.IGNORECASE,
        ),
    ),
    (
        "COMPLETED",
        re.compile(
            r"^(?P<topic>.+?)\s+"
            r"(?:is|has\s+been)\s+"
            r"(?:complete|completed|done|finished)[.!]?$",
            re.IGNORECASE,
        ),
    ),
    (
        "COMPLETED",
        re.compile(
            r"^(?:i['’]ve|i\s+have|we['’]ve|we\s+have)"
            r"\s+(?:finished|completed)\s+"
            r"(?P<topic>.+?)[.!]?$",
            re.IGNORECASE,
        ),
    ),
    (
        "STARTED",
        re.compile(
            r"^(?:i['’]ve|i\s+have|we['’]ve|we\s+have)"
            r"\s+started\s+(?:working\s+on\s+)?"
            r"(?P<topic>.+?)[.!]?$",
            re.IGNORECASE,
        ),
    ),
]



def normalize_tokens(text):
    """
    Normalize task-related words for matching.

    1. Convert morphological variants to base forms.
    2. Remove generic stopwords.
    3. Preserve meaningful task-related keywords.
    """

    words = re.findall(
        r"[a-z0-9]+",
        (text or "").lower()
    )

    lemma_overrides = {
        "testing": "test",
        "tested": "test",
        "tests": "test",
        "reports": "report",
        "releases": "release",
        "updates": "update",
        "updated": "update",
        "working": "work",
        "finished": "finish",
        "completed": "complete",
    }

    protected_words = {
        "test",
        "report",
        "release",
    }

    normalized = set()

    for word in words:

        lemma = lemma_overrides.get(
            word, word
        )

        if len(lemma) <= 1:
            continue

        if (
            lemma not in STOPWORDS
            or lemma in protected_words
        ):
            normalized.add(lemma)

    return normalized



def detect_event(text):
    """
    Detect supported update constructions.

    Returns None when the sentence is not recognized.
    """
    text = (text or "").strip()

    for event_type, pattern in EVENT_PATTERNS:
        match = pattern.match(text)

        if match:
            return {
                "event_type": event_type,
                "topic": match.group("topic").strip(),
                "new_assignee": (
                    match.groupdict().get("person")
                ),
            }

    return None


def parse_time(value):
    if not value:
        return None

    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def task_match_score(task, topic):
    """
    Compare the update's topic with the task description.

    Return 0 if there is no distinctive lexical overlap.
    """
    task_tokens = normalize_tokens(
        task.get("description", "")
        or task.get("description_raw", "")
    )

    topic_tokens = normalize_tokens(topic)

    if not task_tokens or not topic_tokens:
        return 0

    overlap = task_tokens & topic_tokens

    if not overlap:
        return 0

    # Count matching words. Ties are considered ambiguous.
    return len(overlap)


def match_task(tasks, topic, event_timestamp=None):
    """
    Find one unambiguous task match.

    An update must not refer to a task created later.
    """
    candidates = []
    update_time = parse_time(event_timestamp)

    for index, task in enumerate(tasks):
        created_time = parse_time(
            task.get("origin_timestamp")
        )

        if (
            update_time is not None
            and created_time is not None
            and update_time < created_time
        ):
            continue

        score = task_match_score(task, topic)

        if score > 0:
            candidates.append((score, index))

    if not candidates:
        return None, "NO_MATCH"

    candidates.sort(reverse=True)

    highest_score = candidates[0][0]

    winners = [
        index for score, index in candidates
        if score == highest_score
    ]

    if len(winners) != 1:
        return None, "AMBIGUOUS_MATCH"

    return winners[0], None


def apply_event(task, event, message):
    """
    Apply a valid update and record its provenance.

    A terminal task is not reopened automatically.
    """
    event_type = event["event_type"]
    previous_status = task.get("status", "TODO")
    previous_assignee = task.get("assignee")

    if previous_status in {"DONE", "CANCELLED"}:
        return False, "TERMINAL_TASK"

    if event_type == "STARTED":
        task["status"] = "IN_PROGRESS"

    elif event_type == "COMPLETED":
        task["status"] = "DONE"

    elif event_type == "CANCELLED":
        task["status"] = "CANCELLED"

    elif event_type == "REASSIGNED":
        new_assignee = event.get("new_assignee")

        if not new_assignee:
            return False, "MISSING_ASSIGNEE"

        task["assignee"] = new_assignee
        task["assignee_source"] = "CONVERSATION_UPDATE"

    else:
        return False, "UNKNOWN_EVENT"

    task.setdefault("state_history", []).append({
        "event_type": event_type,
        "message_id": message["message_id"],
        "timestamp": message.get("timestamp"),
        "speaker": message.get("speaker"),
        "text": message["text"],
        "previous_status": previous_status,
        "new_status": task["status"],
        "previous_assignee": previous_assignee,
        "new_assignee": task.get("assignee"),
    })

    task.setdefault("source_message_ids", [])

    if message["message_id"] not in task["source_message_ids"]:
        task["source_message_ids"].append(
            message["message_id"]
        )

    return True, None


def track_task_states(tasks, messages):
    """
    Update existing tasks using chronological conversation messages.

    Output includes updated tasks and updates requiring review.
    Does not create new tasks.
    """
    updated_tasks = deepcopy(tasks)
    unmatched_events = []

    # Originating evidence is already represented in the task.
    original_message_ids = set()

    for task in updated_tasks:
        task.setdefault("state_history", [])

        original_message_ids.update(
            task.get("source_message_ids", [])
        )

    for message in messages:
        if message["message_id"] in original_message_ids:
            continue

        event = detect_event(message.get("text", ""))

        if event is None:
            continue

        index, match_error = match_task(
            updated_tasks,
            event["topic"],
            message.get("timestamp"),
        )

        if match_error:
            unmatched_events.append({
                "message_id": message["message_id"],
                "text": message["text"],
                "event_type": event["event_type"],
                "reason": match_error,
            })
            continue

        success, update_error = apply_event(
            updated_tasks[index],
            event,
            message,
        )

        if not success:
            unmatched_events.append({
                "message_id": message["message_id"],
                "text": message["text"],
                "event_type": event["event_type"],
                "reason": update_error,
            })

    return {
        "tasks": updated_tasks,
        "unmatched_events": unmatched_events,
    }


def main():
    if not TASKS_PATH.exists():
        raise FileNotFoundError(
            f"Missing structured tasks: {TASKS_PATH}"
        )

    with open(TASKS_PATH, encoding="utf-8") as file:
        tasks = json.load(file)

    # Diagnostic conversation using the existing 3 tasks.
    # m0001..m0007 correspond to the prior sample thread.
    transcript = """
[2026-10-30 09:00] Sarah: Alex, finish the login API by Friday.
[2026-10-30 09:05] Alex: I'll handle it.
[2026-10-30 09:10] David: We should consider redesigning the dashboard.
[2026-10-30 09:15] Sarah: The server crashed yesterday.
[2026-10-30 09:20] David: Please prepare the testing report.
[2026-10-30 09:25] Alex: The API is currently working.
[2026-10-30 09:30] Sarah: Alex has been assigned to test the release.
[2026-10-30 10:00] Alex: I've started working on the login API.
[2026-10-30 11:00] Alex: The login API is complete.
"""

    messages = parse_transcript(transcript)

    result = track_task_states(tasks, messages)

    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    with open(
        OUTPUT_PATH, "w", encoding="utf-8"
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nTASK STATE TRACKING RESULTS")
    print("=" * 60)

    for task in result["tasks"]:
        print(
            task["task_id"],
            "|", task.get("description"),
            "|", task.get("status"),
            "|", task.get("assignee"),
            "| events:",
            len(task["state_history"]),
        )

    print(
        "\nUnmatched events:",
        len(result["unmatched_events"]),
    )
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
