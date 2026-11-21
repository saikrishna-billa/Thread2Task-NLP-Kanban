
import argparse
import json
from pathlib import Path

from parser_v2 import parse_transcript
from context_engine import consolidate_tasks
from task_extractor import enrich_tasks
from task_state_engine import (
    detect_event,
    track_task_states,
)


ROOT = Path(__file__).resolve().parent

OUTPUT_PATH = (
    ROOT / "data/processed/final_tasks.json"
)

DEMO_TRANSCRIPT = """
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


def get_message_fields(analyzed_message):
    """
    Restore the original message representation
    expected by the state-tracking engine.
    """
    return {
        "message_id": analyzed_message["message_id"],
        "speaker": analyzed_message["speaker"],
        "timestamp": analyzed_message["timestamp"],
        "text": analyzed_message["text"],
    }


def run_thread2task(
    transcript,
    known_speakers=None,
    action_pipeline=None,
):
    """
    Run all Thread2Task modules in one execution.

    An optional action_pipeline argument allows
    deterministic integration testing without
    loading the transformer model.
    """

    if action_pipeline is None:
        # Lazy import keeps lightweight tests fast.
        from action_pipeline import ActionPipeline
        action_pipeline = ActionPipeline()

    # --------------------------------------------------
    # Stage 1: Initial NLP analysis
    # --------------------------------------------------
    analyzed_messages = action_pipeline.process(
        transcript,
        known_speakers=known_speakers,
    )

    # --------------------------------------------------
    # Stage 2: Separate task anchors from updates
    # --------------------------------------------------
    creation_messages = []
    state_update_ids = []

    for message in analyzed_messages:
        event = detect_event(message["text"])

        if event is not None:
            state_update_ids.append(
                message["message_id"]
            )
        else:
            creation_messages.append(message)

    # --------------------------------------------------
    # Stage 3: Contextual task consolidation
    # --------------------------------------------------
    task_candidates = consolidate_tasks(
        creation_messages
    )

    # --------------------------------------------------
    # Stage 4: Task attribute extraction
    # --------------------------------------------------
    structured_tasks = enrich_tasks(
        task_candidates
    )

    # --------------------------------------------------
    # Stage 5: Task-state tracking
    # --------------------------------------------------
    all_messages = [
        get_message_fields(message)
        for message in analyzed_messages
    ]

    state_result = track_task_states(
        structured_tasks,
        all_messages,
    )

    # --------------------------------------------------
    # Stage 6: Assemble final output
    # --------------------------------------------------
    final_result = {
        "metadata": {
            "total_messages": len(
                analyzed_messages
            ),
            "task_candidates": len(
                task_candidates
            ),
            "recognized_state_update_messages": len(
                state_update_ids
            ),
            "final_tasks": len(
                state_result["tasks"]
            ),
            "unmatched_state_events": len(
                state_result["unmatched_events"]
            ),
        },
        "tasks": state_result["tasks"],
        "unmatched_events": state_result[
            "unmatched_events"
        ],
        "state_update_message_ids": state_update_ids,
        "analyzed_messages": analyzed_messages,
    }

    return final_result


def save_results(result, output_path=OUTPUT_PATH):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


def print_summary(result):
    print("\n" + "=" * 70)
    print("THREAD2TASK - END-TO-END RESULTS")
    print("=" * 70)

    metadata = result["metadata"]

    print(
        "Conversation messages:",
        metadata["total_messages"]
    )
    print(
        "Task candidates:",
        metadata["task_candidates"]
    )
    print(
        "State-update messages:",
        metadata["recognized_state_update_messages"]
    )
    print(
        "Final tasks:",
        metadata["final_tasks"]
    )
    print(
        "Unmatched state events:",
        metadata["unmatched_state_events"]
    )

    for task in result["tasks"]:
        print("\n" + "-" * 60)

        print("Task ID:", task["task_id"])
        print(
            "Description:",
            task.get("description")
        )
        print(
            "Assignee:",
            task.get("assignee")
        )
        print(
            "Deadline:",
            task.get("due_date")
        )
        print(
            "Status:",
            task.get("status")
        )
        print(
            "Needs review:",
            task.get("needs_review")
        )
        print(
            "Source IDs:",
            task.get("source_message_ids")
        )
        print(
            "State events:",
            len(task.get("state_history", []))
        )

    if result["unmatched_events"]:
        print("\nUNMATCHED STATE EVENTS")

        for event in result["unmatched_events"]:
            print(
                event["message_id"],
                "|",
                event["reason"],
                "|",
                event["text"],
            )


def main():
    parser = argparse.ArgumentParser(
        description="Thread2Task end-to-end NLP pipeline"
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional UTF-8 transcript file",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_PATH),
    )

    parser.add_argument(
        "--speakers",
        nargs="*",
        default=None,
        help="Known speakers for untimestamped input",
    )

    args = parser.parse_args()

    if args.input:
        input_path = Path(args.input)
        transcript = input_path.read_text(
            encoding="utf-8"
        )
    else:
        transcript = DEMO_TRANSCRIPT

    # Avoid loading DistilBERT for empty input.
    if not parse_transcript(
        transcript,
        known_speakers=args.speakers,
    ):
        print("No conversation messages found.")
        return

    result = run_thread2task(
        transcript,
        known_speakers=args.speakers,
    )

    output_path = save_results(
        result,
        args.output,
    )

    print_summary(result)

    print(
        "\nSaved final output:",
        output_path
    )


if __name__ == "__main__":
    main()
