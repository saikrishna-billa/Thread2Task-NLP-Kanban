
import sys
from pathlib import Path

# Import parser_v2.py from the project root
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from parser_v2 import parse_transcript


def test_multiline_conversation():
    transcript = """
[2026-10-28 09:00] Sarah: Alex, update the API
by Friday.

[2026-10-28 09:15] Alex: I'll handle it.

[2026-10-28 10:00] David: I'll test it.
"""

    messages = parse_transcript(transcript)

    assert len(messages) == 3
    assert messages[0]["speaker"] == "Sarah"
    assert messages[0]["text"] == (
        "Alex, update the API\nby Friday."
    )
    assert messages[0]["timestamp"] == "2026-10-28T09:00:00"
    assert messages[2]["message_id"] == "m0003"


def test_without_timestamps():
    transcript = """
Sarah: Complete the report.
Alex: I'll prepare it.
"""

    messages = parse_transcript(transcript)

    assert len(messages) == 2
    assert messages[0]["speaker"] == "Sarah"
    assert messages[0]["timestamp"] is None
    assert messages[1]["speaker"] == "Alex"


def test_empty_conversation():
    assert parse_transcript("") == []
    assert parse_transcript("   ") == []
    assert parse_transcript("\n\n\n") == []


def test_long_conversation():
    transcript = "\n".join(
        f"Person{i % 5}: Message number {i}"
        for i in range(1000)
    )

    messages = parse_transcript(transcript)

    assert len(messages) == 1000
    assert messages[0]["message_id"] == "m0001"
    assert messages[-1]["message_id"] == "m1000"


def test_colon_in_continuation():
    transcript = """
[2026-10-28 09:00] Sarah: Please investigate the error.
Error: database connection timeout
This happened during deployment.

[2026-10-28 09:15] Alex: I'll check the logs.
"""

    messages = parse_transcript(transcript)

    assert len(messages) == 2
    assert messages[0]["speaker"] == "Sarah"
    assert "Error: database connection timeout" in messages[0]["text"]
    assert "This happened during deployment." in messages[0]["text"]
    assert messages[1]["speaker"] == "Alex"


def test_known_speakers():
    transcript = """
Sarah: Please investigate the issue.
Error: database timeout
Alex: I'll investigate it.
"""

    messages = parse_transcript(
        transcript,
        known_speakers=["Sarah", "Alex"]
    )

    assert len(messages) == 2
    assert messages[0]["speaker"] == "Sarah"
    assert "Error: database timeout" in messages[0]["text"]
    assert messages[1]["speaker"] == "Alex"


def test_timestamp_normalization():
    transcript = """
[2026-10-28 09:00:30] Sarah: Start development.
[2026-10-29 14:20] Alex: Development is underway.
"""

    messages = parse_transcript(transcript)

    assert len(messages) == 2
    assert messages[0]["timestamp"] == "2026-10-28T09:00:30"
    assert messages[1]["timestamp"] == "2026-10-29T14:20:00"


def test_message_ids_and_text():
    transcript = """
Sarah: Update the authentication API.
Alex: I'll complete the API by tomorrow.
David: I'll test the authentication API.
"""

    messages = parse_transcript(transcript)

    assert len(messages) == 3

    assert [m["message_id"] for m in messages] == [
        "m0001", "m0002", "m0003"
    ]

    assert messages[0]["text"] == (
        "Update the authentication API."
    )
    assert messages[1]["text"] == (
        "I'll complete the API by tomorrow."
    )
    assert messages[2]["text"] == (
        "I'll test the authentication API."
    )
