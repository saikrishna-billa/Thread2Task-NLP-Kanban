
import json
import hashlib
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path

import streamlit as st

from run_pipeline import (
    run_thread2task,
    DEMO_TRANSCRIPT,
)


ROOT = Path(__file__).resolve().parent

AI_OUTPUT_PATH = (
    ROOT / "data/processed/final_tasks.json"
)

BOARD_PATH = (
    ROOT / "data/processed/kanban_board.json"
)

STATUSES = [
    "TODO",
    "IN_PROGRESS",
    "DONE",
    "CANCELLED",
]


# ---------------------------------------------------
# Page configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Thread2Task | NLP Kanban",
    page_icon="📋",
    layout="wide",
)

st.title("Thread2Task")
st.caption(
    "Automated Action-Item Extraction "
    "& Interactive Kanban Generation"
)


# ---------------------------------------------------
# File operations
# ---------------------------------------------------
def load_json(path):
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def transcript_fingerprint(transcript):
    return hashlib.sha256(
        transcript.encode("utf-8")
    ).hexdigest()


def initialize_board():
    """
    Load the saved board first.

    If no saved board exists, display tasks from
    the latest NLP result without modifying it.
    """

    if "board_tasks" in st.session_state:
        return

    saved_board = load_json(BOARD_PATH)

    if saved_board is not None:
        st.session_state.board_tasks = deepcopy(
            saved_board.get("tasks", [])
        )
        st.session_state.board_metadata = (
            saved_board.get("metadata", {})
        )
        st.session_state.board_source = "Saved board"

    else:
        ai_result = load_json(AI_OUTPUT_PATH)

        if ai_result is not None:
            st.session_state.board_tasks = deepcopy(
                ai_result.get("tasks", [])
            )
            st.session_state.board_metadata = (
                ai_result.get("metadata", {})
            )
            st.session_state.board_source = (
                "Latest NLP output"
            )
        else:
            st.session_state.board_tasks = []
            st.session_state.board_metadata = {}
            st.session_state.board_source = (
                "Empty board"
            )

    st.session_state.unsaved_changes = False
    st.session_state.widget_epoch = 0


def save_board():
    """
    Save user-edited tasks separately from
    the raw NLP output.
    """

    payload = {
        "format_version": 1,
        "saved_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "metadata": (
            st.session_state.board_metadata
        ),
        "tasks": st.session_state.board_tasks,
    }

    write_json(BOARD_PATH, payload)

    st.session_state.unsaved_changes = False
    st.session_state.board_source = "Saved board"


def replace_board(tasks, metadata, source):
    st.session_state.board_tasks = deepcopy(
        tasks
    )
    st.session_state.board_metadata = deepcopy(
        metadata
    )
    st.session_state.board_source = source
    st.session_state.unsaved_changes = True

    # Reset task editor widgets after loading
    # a different set of tasks.
    st.session_state.widget_epoch += 1


def valid_due_date(value):
    """
    Accept YYYY-MM-DD or an empty value.
    """
    value = value.strip()

    if not value:
        return True

    try:
        date.fromisoformat(value)
        return True
    except ValueError:
        return False


initialize_board()


# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
with st.sidebar:
    st.header("Board controls")

    st.write(
        "Current source:",
        st.session_state.board_source,
    )

    if st.session_state.unsaved_changes:
        st.warning("Board has unsaved changes.")

    if st.button(
        "Save board",
        use_container_width=True,
        type="primary",
    ):
        save_board()
        st.success("Board saved successfully.")

    if BOARD_PATH.exists():
        if st.button(
            "Reload saved board",
            use_container_width=True,
        ):
            saved = load_json(BOARD_PATH)

            replace_board(
                saved.get("tasks", []),
                saved.get("metadata", {}),
                "Saved board",
            )

            st.session_state.unsaved_changes = False
            st.rerun()

    st.divider()

    st.subheader("Storage")

    st.caption(
        "AI output: data/processed/final_tasks.json"
    )
    st.caption(
        "Editable board: "
        "data/processed/kanban_board.json"
    )

    if st.session_state.board_tasks:
        board_json = json.dumps(
            {
                "metadata": (
                    st.session_state.board_metadata
                ),
                "tasks": (
                    st.session_state.board_tasks
                ),
            },
            indent=2,
            ensure_ascii=False,
        )

        st.download_button(
            "Download board JSON",
            data=board_json,
            file_name="thread2task_board.json",
            mime="application/json",
            use_container_width=True,
        )

    st.divider()

    st.caption(
        "Model: Fine-tuned DistilBERT "
        "+ discourse and context analysis"
    )


# ---------------------------------------------------
# Conversation input
# ---------------------------------------------------
st.subheader("1. Process a conversation")

with st.expander(
    "Conversation input",
    expanded=False,
):
    transcript = st.text_area(
        "Paste a timestamped conversation",
        value=DEMO_TRANSCRIPT.strip(),
        height=250,
        key="transcript_input",
    )

    known_speakers_text = st.text_input(
        "Known speakers (optional, comma-separated)",
        placeholder="Sarah, Alex, David",
    )

    replace_confirmed = st.checkbox(
        "Replace the current board with newly "
        "extracted tasks (existing edits will be "
        "discarded unless saved/exported)"
    )

    run_disabled = (
        not transcript.strip()
        or (
            bool(st.session_state.board_tasks)
            and not replace_confirmed
        )
    )

    if st.button(
        "Extract tasks using NLP",
        type="primary",
        disabled=run_disabled,
    ):
        known_speakers = [
            name.strip()
            for name in known_speakers_text.split(",")
            if name.strip()
        ]

        with st.spinner(
            "Running the Thread2Task NLP pipeline..."
        ):
            try:
                result = run_thread2task(
                    transcript,
                    known_speakers=(
                        known_speakers or None
                    ),
                )

                # Store the original inference result.
                # User edits are stored elsewhere.
                write_json(AI_OUTPUT_PATH, result)

                metadata = deepcopy(
                    result.get("metadata", {})
                )

                metadata["transcript_sha256"] = (
                    transcript_fingerprint(transcript)
                )

                replace_board(
                    result.get("tasks", []),
                    metadata,
                    "New NLP extraction",
                )

                st.rerun()

            except Exception as exc:
                st.error(
                    "Task extraction failed: "
                    f"{type(exc).__name__}: {exc}"
                )

    if run_disabled and st.session_state.board_tasks:
        st.caption(
            "Confirm replacement to run extraction. "
            "Save your existing board first if needed."
        )


# ---------------------------------------------------
# Summary metrics
# ---------------------------------------------------
tasks = st.session_state.board_tasks

st.subheader("2. Task overview")

counts = {
    status: sum(
        task.get("status", "TODO") == status
        for task in tasks
    )
    for status in STATUSES
}

review_count = sum(
    bool(task.get("needs_review"))
    for task in tasks
)

metric_columns = st.columns(5)

metric_columns[0].metric(
    "Total tasks", len(tasks)
)
metric_columns[1].metric(
    "TODO", counts["TODO"]
)
metric_columns[2].metric(
    "In progress", counts["IN_PROGRESS"]
)
metric_columns[3].metric(
    "Done", counts["DONE"]
)
metric_columns[4].metric(
    "Needs review", review_count
)


# ---------------------------------------------------
# Task editing
# ---------------------------------------------------
def update_task(task_id, new_values):
    for task in st.session_state.board_tasks:
        if task["task_id"] == task_id:
            modified_fields = []

            for field, value in new_values.items():
                if task.get(field) != value:
                    task[field] = value
                    modified_fields.append(field)

            if modified_fields:
                task.setdefault(
                    "manual_edit_history", []
                ).append({
                    "edited_at": (
                        datetime.now().isoformat(
                            timespec="seconds"
                        )
                    ),
                    "fields": modified_fields,
                })

                task["manually_edited"] = True
                st.session_state.unsaved_changes = True

            break


def render_task_card(task):
    task_id = task["task_id"]
    epoch = st.session_state.widget_epoch

    description = (
        task.get("description")
        or task.get("description_raw")
        or "Untitled task"
    )

    assignee = task.get("assignee") or "Unassigned"

    with st.container(border=True):
        st.markdown(f"**{description}**")

        st.caption(
            f"ID: {task_id} | "
            f"Assignee: {assignee}"
        )

        if task.get("due_date"):
            st.caption(
                f"Due: {task['due_date']}"
            )

        if task.get("needs_review"):
            st.warning("AI review required")

        with st.expander(
            "Edit task",
            expanded=False,
        ):
            with st.form(
                key=f"form_{epoch}_{task_id}",
            ):
                new_description = st.text_input(
                    "Task description",
                    value=description,
                )

                new_assignee = st.text_input(
                    "Assignee",
                    value=(
                        task.get("assignee") or ""
                    ),
                    help=(
                        "Leave empty if the responsible "
                        "person is unknown."
                    ),
                )

                new_due_date = st.text_input(
                    "Due date (YYYY-MM-DD)",
                    value=(
                        task.get("due_date") or ""
                    ),
                )

                current_status = task.get(
                    "status", "TODO"
                )

                if current_status not in STATUSES:
                    current_status = "TODO"

                new_status = st.selectbox(
                    "Task status",
                    options=STATUSES,
                    index=STATUSES.index(
                        current_status
                    ),
                )

                new_review = st.checkbox(
                    "Needs review",
                    value=bool(
                        task.get("needs_review")
                    ),
                )

                submitted = st.form_submit_button(
                    "Apply changes",
                    type="primary",
                )

                if submitted:
                    if not new_description.strip():
                        st.error(
                            "Description cannot be empty."
                        )

                    elif not valid_due_date(
                        new_due_date
                    ):
                        st.error(
                            "Use YYYY-MM-DD for the "
                            "due date, or leave it empty."
                        )

                    else:
                        update_task(
                            task_id,
                            {
                                "description": (
                                    new_description.strip()
                                ),
                                "assignee": (
                                    new_assignee.strip()
                                    or None
                                ),
                                "due_date": (
                                    new_due_date.strip()
                                    or None
                                ),
                                "status": new_status,
                                "needs_review": new_review,
                            },
                        )

                        st.session_state.widget_epoch += 1
                        st.rerun()

        with st.expander(
            "Evidence and task history",
            expanded=False,
        ):
            source_ids = task.get(
                "source_message_ids", []
            )

            st.caption(
                "Source messages: "
                + ", ".join(source_ids)
            )

            evidence = task.get("evidence", [])

            if evidence:
                st.markdown("**Original evidence**")

                for item in evidence:
                    st.write(
                        f"**{item.get('speaker', 'Unknown')}** "
                        f"({item.get('message_id', '-')})"
                    )
                    st.caption(
                        item.get("text", "")
                    )
                    st.caption(
                        "Relation: "
                        + item.get(
                            "relation", "UNKNOWN"
                        )
                    )

            history = task.get(
                "state_history", []
            )

            if history:
                st.markdown("**State-change history**")

                for event in history:
                    st.write(
                        f"**{event.get('event_type')}** "
                        f"— {event.get('speaker', 'Unknown')}"
                    )
                    st.caption(
                        event.get("text", "")
                    )
                    st.caption(
                        "Timestamp: "
                        + str(
                            event.get("timestamp")
                        )
                    )

            if task.get("review_reasons"):
                st.markdown("**Review reasons**")

                for reason in task["review_reasons"]:
                    st.write("- " + reason)


# ---------------------------------------------------
# Kanban board
# ---------------------------------------------------
st.subheader("3. Interactive Kanban board")

if not tasks:
    st.info(
        "No tasks available. Paste a conversation "
        "and run extraction to create your first board."
    )

else:
    columns = st.columns(3)

    column_config = [
        ("TODO", "TODO"),
        ("IN_PROGRESS", "IN PROGRESS"),
        ("DONE", "DONE"),
    ]

    for column, (status, title) in zip(
        columns, column_config
    ):
        with column:
            column_tasks = [
                task for task in tasks
                if task.get("status", "TODO") == status
            ]

            st.markdown(
                f"### {title} ({len(column_tasks)})"
            )

            for task in column_tasks:
                render_task_card(task)

            if not column_tasks:
                st.caption("No tasks in this column.")

    cancelled_tasks = [
        task for task in tasks
        if task.get("status") == "CANCELLED"
    ]

    if cancelled_tasks:
        with st.expander(
            f"Cancelled tasks ({len(cancelled_tasks)})"
        ):
            for task in cancelled_tasks:
                render_task_card(task)


st.divider()

st.caption(
    "Thread2Task | NLP-based action extraction, "
    "contextual linking and task-state tracking"
)
