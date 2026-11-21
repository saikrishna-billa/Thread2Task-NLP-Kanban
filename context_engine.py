
import re
import json
from pathlib import Path


# These phrases are context-dependent commitments.
ACKNOWLEDGMENT_PATTERN = re.compile(
    r"^\s*(?:"
    r"i['’]ll\s+(?:handle|do|take\s+care\s+of)\s+it"
    r"|i\s+will\s+(?:handle|do)\s+it"
    r"|i['’]m\s+on\s+it"
    r"|got\s+it"
    r"|consider\s+it\s+done"
    r")\s*[.!]?\s*$",
    re.IGNORECASE,
)


def is_contextual_commitment(text):
    """Identify commitments without an explicit task object."""
    return bool(ACKNOWLEDGMENT_PATTERN.match(text))


def is_task_evidence(message):
    """
    Determine whether an explicit message can anchor a task.

    REVIEW assignments are retained with a review flag.
    REVIEW suggestions are not promoted to tasks.
    """
    decision = message["decision"]
    intent = message["discourse_intent"]

    if decision == "CREATE_CANDIDATE":
        return True

    if decision == "REVIEW" and intent == "ASSIGNMENT":
        return True

    return False

def is_matching_commitment(text, speaker, task):
    """
    Check whether a first-person commitment confirms
    the immediately preceding explicit assignment.

    Requires matching speaker identity and overlapping
    meaningful task-object words.
    """
    match = re.match(
        r"^\s*(?:i['’]ll|i\s+will|i['’]m\s+going\s+to)"
        r"\s+(?P<action>.+?)[.!]?\s*$",
        text or "",
        flags=re.IGNORECASE,
    )

    if not match or not task or not speaker:
        return False

    anchor = task.get("description_raw", "")

    # We only link when the prior assignment explicitly
    # names the current speaker.
    escaped = re.escape(speaker)

    named_assignee = (
        re.match(
            rf"^\s*{escaped}\s*,",
            anchor,
            flags=re.IGNORECASE,
        )
        or re.match(
            rf"^\s*{escaped}\s+"
            r"(?:will|shall|has been assigned to|"
            r"was assigned to|is assigned to)\b",
            anchor,
            flags=re.IGNORECASE,
        )
    )

    if not named_assignee:
        return False

    stopwords = {
        "the", "a", "an", "to", "by", "on", "at",
        "for", "of", "and", "will", "i", "we",
        "update", "prepare", "finish", "complete",
        "make", "do", "work", "handle",
        "tomorrow", "today", "monday", "tuesday",
        "wednesday", "thursday", "friday",
        "saturday", "sunday", "next",
    }

    def content_words(value):
        return {
            word
            for word in re.findall(
                r"[a-z0-9]+", value.lower()
            )
            if len(word) > 2 and word not in stopwords
        }

    commitment_words = content_words(
        match.group("action")
    )

    anchor_words = content_words(anchor)

    return bool(
        commitment_words
        and commitment_words & anchor_words
    )

def consolidate_tasks(messages):
    """
    Link contextual commitments to preceding task candidates.

    This is an initial, conservative conversation-memory layer.
    It does not perform full neural coreference resolution.
    """
    tasks = []
    last_task = None
    last_task_index = None

    for index, message in enumerate(messages):
        text = message["text"]
        speaker = message["speaker"]

        # Resolve simple anaphoric commitments.
        if is_contextual_commitment(text):
            if (
                last_task is not None
                and last_task_index is not None
                and index - last_task_index <= 2
            ):
                last_task["source_message_ids"].append(
                    message["message_id"]
                )

                last_task["evidence"].append({
                    "message_id": message["message_id"],
                    "speaker": speaker,
                    "text": text,
                    "relation": "CONTEXTUAL_COMMITMENT",
                })

                last_task["commitment_speaker"] = speaker
                last_task["context_linked"] = True

                # Keep the latest evidence position.
                last_task_index = index

            # Don't create an invented task when
            # the antecedent is unavailable.
            continue

        if (
            last_task is not None
            and last_task_index is not None
            and index - last_task_index == 1
            and is_matching_commitment(
                text, speaker, last_task
            )
        ):
            last_task["source_message_ids"].append(
                message["message_id"]
            )

            last_task["evidence"].append({
                "message_id": message["message_id"],
                "speaker": speaker,
                "text": text,
                "relation": "MATCHING_COMMITMENT",
            })

            last_task["commitment_speaker"] = speaker
            last_task["context_linked"] = True
            last_task_index = index
            continue
        if not is_task_evidence(message):
            # Break simple context linking on unrelated
            # messages rather than carrying stale tasks.
            last_task = None
            last_task_index = None
            continue

        needs_review = (
            message["decision"] == "REVIEW"
        )

        task = {
            "task_id": f"t{len(tasks) + 1:04d}",
            "description_raw": text,
            "source_message_ids": [
                message["message_id"]
            ],
            "origin_speaker": speaker,
            "origin_timestamp": message["timestamp"],
            "discourse_intent": message[
                "discourse_intent"
            ],
            "action_probability": message[
                "action_probability"
            ],
            "needs_review": needs_review,
            "context_linked": False,
            "commitment_speaker": None,
            "assignee": None,
            "due_date": None,
            "status": "TODO",
            "evidence": [{
                "message_id": message["message_id"],
                "speaker": speaker,
                "text": text,
                "relation": "TASK_ANCHOR",
            }],
        }

        tasks.append(task)
        last_task = task
        last_task_index = index

    return tasks


if __name__ == "__main__":
    root = Path(__file__).resolve().parent

    input_path = (
        root / "data/processed/pipeline_diagnostic.json"
    )

    output_path = (
        root / "data/processed/contextual_tasks.json"
    )

    with open(
        input_path, "r", encoding="utf-8"
    ) as f:
        messages = json.load(f)

    tasks = consolidate_tasks(messages)

    with open(
        output_path, "w", encoding="utf-8"
    ) as f:
        json.dump(
            tasks, f, indent=2, ensure_ascii=False
        )

    print("\nCONTEXTUAL TASK RESULTS")
    print("=" * 65)

    for task in tasks:
        print("\nTask ID:", task["task_id"])
        print("Description:", task["description_raw"])
        print(
            "Sources:",
            task["source_message_ids"]
        )
        print(
            "Commitment speaker:",
            task["commitment_speaker"]
        )
        print(
            "Needs review:",
            task["needs_review"]
        )

    print("\nTotal task candidates:", len(tasks))
    print("Saved:", output_path)
