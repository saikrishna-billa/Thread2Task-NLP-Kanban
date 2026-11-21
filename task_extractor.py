
import json
import re
from copy import deepcopy
from datetime import date, datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parent

INPUT_PATH = ROOT / "data/processed/contextual_tasks.json"
OUTPUT_PATH = ROOT / "data/processed/structured_tasks.json"

WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

DAY_NAMES = "|".join(WEEKDAYS)

DATE_PATTERN = re.compile(
    rf"\b(?:by|on|before|due)\s+"
    rf"(?:next\s+)?(?:{DAY_NAMES})\b"
    rf"|\bnext\s+(?:{DAY_NAMES})\b"
    r"|\b(?:by|on|before|due)\s+tomorrow\b"
    r"|\btomorrow\b"
    r"|\b(?:by|on|before|due)\s+"
    r"\d{4}-\d{2}-\d{2}\b",
    re.IGNORECASE,
)

PASSIVE_ASSIGNMENT = re.compile(
    r"^(?P<person>[A-Za-z][\w.-]*)\s+"
    r"(?:(?:has|have|had)\s+been|is|was|were)"
    r"\s+assigned\s+to\s+"
    r"(?P<action>.+)$",
    re.IGNORECASE,
)

ACTIVE_ASSIGNMENT = re.compile(
    r"^(?P<person>[A-Za-z][\w.-]*)\s+"
    r"(?:will|shall)\s+(?P<action>.+)$",
    re.IGNORECASE,
)

DECISION_ASSIGNMENT = re.compile(
    r"^we\s+(?:agreed|decided)\s+that\s+"
    r"(?P<person>[A-Za-z][\w.-]*)\s+"
    r"will\s+(?P<action>.+)$",
    re.IGNORECASE,
)

DIRECT_REQUEST = re.compile(
    r"^(?P<person>[A-Za-z][\w.-]*)\s*,\s*"
    r"(?P<action>.+)$"
)

FIRST_PERSON = re.compile(
    r"^(?:i\s+will|i['’]ll|i\s+promise\s+to"
    r"|i\s+am\s+going\s+to|i['’]m\s+going\s+to)"
    r"\s+(?P<action>.+)$",
    re.IGNORECASE,
)

REQUEST_PREFIX = re.compile(
    r"^(?:please\s+|could\s+you\s+|can\s+you\s+)",
    re.IGNORECASE,
)


def get_reference_date(timestamp):
    """Return the message-local date, if available."""
    if not timestamp:
        return None

    try:
        return datetime.fromisoformat(
            timestamp.replace("Z", "+00:00")
        ).date()
    except (ValueError, TypeError):
        return None


def extract_deadline(text, timestamp):
    """
    Extract a limited set of deadlines.

    Supported:
    - tomorrow
    - by Friday / on Monday
    - next Friday
    - by 2026-11-05

    Never use the computer's current date as a fallback.
    """

    match = DATE_PATTERN.search(text or "")

    if not match:
        return {
            "due_date": None,
            "deadline_text": None,
            "deadline_confidence": "NOT_FOUND",
        }

    phrase = match.group(0)
    lower = phrase.lower()
    reference = get_reference_date(timestamp)

    iso_match = re.search(
        r"\d{4}-\d{2}-\d{2}",
        phrase,
    )

    if iso_match:
        try:
            due = date.fromisoformat(
                iso_match.group(0)
            )

            return {
                "due_date": due.isoformat(),
                "deadline_text": phrase,
                "deadline_confidence": "EXPLICIT_DATE",
            }
        except ValueError:
            pass

    if reference is None:
        return {
            "due_date": None,
            "deadline_text": phrase,
            "deadline_confidence": "NEEDS_REFERENCE_DATE",
        }

    if "tomorrow" in lower:
        due = reference + timedelta(days=1)

        return {
            "due_date": due.isoformat(),
            "deadline_text": phrase,
            "deadline_confidence": "RELATIVE_DATE",
        }

    for day_name, target_day in WEEKDAYS.items():
        if re.search(
            rf"\b{day_name}\b", lower
        ):
            if "next" in lower:
                # Next calendar week's named weekday.
                days_to_next_week = (
                    7 - reference.weekday() + target_day
                )
                due = reference + timedelta(
                    days=days_to_next_week
                )
            else:
                # Nearest occurrence, including today.
                offset = (
                    target_day - reference.weekday()
                ) % 7

                due = reference + timedelta(
                    days=offset
                )

            return {
                "due_date": due.isoformat(),
                "deadline_text": phrase,
                "deadline_confidence": "RELATIVE_DATE",
            }

    return {
        "due_date": None,
        "deadline_text": phrase,
        "deadline_confidence": "UNRESOLVED",
    }


def clean_action(text):
    """Remove recognized deadline expressions and punctuation."""

    cleaned = DATE_PATTERN.sub("", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip(" .,;:-")

    if not cleaned:
        return None

    return cleaned[0].upper() + cleaned[1:]


def extract_description_and_assignee(task):
    """
    Extract an action phrase and explicit responsibility.

    The rules are intentionally conservative:
    the sender of a request is not automatically
    the recipient of that request.
    """

    text = (task.get("description_raw") or "").strip()

    origin_speaker = task.get("origin_speaker")
    commitment_speaker = task.get("commitment_speaker")

    assignee = None
    assignee_source = None
    action = text

    # "We agreed that Sarah will update the API."
    match = DECISION_ASSIGNMENT.match(action)

    if match:
        assignee = match.group("person")
        assignee_source = "EXPLICIT_DECISION"
        action = match.group("action")

    else:
        # "Alex has been assigned to test the release."
        match = PASSIVE_ASSIGNMENT.match(action)

        if match:
            assignee = match.group("person")
            assignee_source = "PASSIVE_ASSIGNMENT"
            action = match.group("action")

        else:
            # "Sarah will prepare the slides."
            match = ACTIVE_ASSIGNMENT.match(action)

            if match:
                assignee = match.group("person")
                assignee_source = "EXPLICIT_FUTURE_ASSIGNMENT"
                action = match.group("action")

            else:
                # "Alex, finish the login API by Friday."
                match = DIRECT_REQUEST.match(action)

                if match:
                    assignee = match.group("person")
                    assignee_source = "DIRECT_ADDRESS"
                    action = match.group("action")

                else:
                    # "I'll prepare the presentation."
                    match = FIRST_PERSON.match(action)

                    if match:
                        if origin_speaker not in (
                            None, "Unknown"
                        ):
                            assignee = origin_speaker
                            assignee_source = (
                                "SPEAKER_COMMITMENT"
                            )

                        action = match.group("action")

    action = REQUEST_PREFIX.sub("", action)
    description = clean_action(action)

    conflict = False

    if commitment_speaker:
        if assignee is None:
            assignee = commitment_speaker
            assignee_source = "LINKED_COMMITMENT"

        elif assignee.casefold() != (
            commitment_speaker.casefold()
        ):
            # Do not silently resolve conflicting identities.
            conflict = True

    return {
        "description": description,
        "assignee": assignee,
        "assignee_source": assignee_source,
        "assignee_conflict": conflict,
    }


def enrich_task(task):
    """Add normalized attributes to one candidate task."""

    result = deepcopy(task)

    attributes = extract_description_and_assignee(
        result
    )

    deadline = extract_deadline(
        result.get("description_raw", ""),
        result.get("origin_timestamp"),
    )

    result.update(attributes)
    result.update(deadline)

    review_reasons = []

    if result.get("needs_review"):
        review_reasons.append(
            "Upstream candidate requires review"
        )

    if attributes["assignee_conflict"]:
        review_reasons.append(
            "Explicit assignee conflicts with commitment speaker"
        )

    if (
        deadline["deadline_text"]
        and deadline["due_date"] is None
    ):
        review_reasons.append(
            "Deadline could not be normalized"
        )

    if not attributes["description"]:
        review_reasons.append(
            "Could not extract a task description"
        )

    result["needs_review"] = bool(review_reasons)
    result["review_reasons"] = review_reasons

    return result


def enrich_tasks(tasks):
    return [enrich_task(task) for task in tasks]


def main():
    with open(
        INPUT_PATH, "r", encoding="utf-8"
    ) as file:
        tasks = json.load(file)

    structured = enrich_tasks(tasks)

    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            structured,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nSTRUCTURED TASK EXTRACTION")
    print("=" * 65)

    for task in structured:
        print("\nTask:", task["task_id"])
        print("Description:", task["description"])
        print("Assignee:", task["assignee"])
        print("Due date:", task["due_date"])
        print("Deadline text:", task["deadline_text"])
        print("Assignee source:", task["assignee_source"])
        print("Needs review:", task["needs_review"])

    print("\nTotal tasks:", len(structured))
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
