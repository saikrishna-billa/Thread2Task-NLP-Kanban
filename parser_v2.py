
import re
from datetime import datetime
from typing import List, Dict, Any, Optional


# -------------------------------------------------------
# Timestamped message headers
# -------------------------------------------------------
TIMESTAMPED_PATTERN = re.compile(
    r"^\s*"
    r"(?:"
    r"\[(?P<bracket_ts>[^\]]+)\]"
    r"|"
    r"(?P<plain_ts>\d{4}-\d{2}-\d{2}\s+"
    r"\d{2}:\d{2}(?::\d{2})?)"
    r")"
    r"\s+"
    r"(?P<speaker>[A-Za-z][A-Za-z0-9_. -]{0,79})"
    r":\s*(?P<text>.*)$"
)


# -------------------------------------------------------
# Untimestamped speaker headers
# -------------------------------------------------------
SPEAKER_PATTERN = re.compile(
    r"^\s*"
    r"(?P<speaker>[A-Za-z][A-Za-z0-9_. -]{0,79})"
    r":\s*(?P<text>.*)$"
)


# -------------------------------------------------------
# Timestamp normalization
# -------------------------------------------------------
def normalize_timestamp(
    raw_timestamp: Optional[str]
) -> Optional[str]:

    if not raw_timestamp:
        return None

    raw_timestamp = raw_timestamp.strip()

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(raw_timestamp, fmt)
            return dt.isoformat(timespec="seconds")
        except ValueError:
            continue

    return None


# -------------------------------------------------------
# Main conversation parser
# -------------------------------------------------------
def parse_transcript(
    transcript_text: str,
    known_speakers: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Convert raw conversational transcripts into structured
    message dictionaries.

    Supports:
    - Timestamped conversations
    - Untimestamped conversations
    - Known-speaker filtering
    - Multiline messages
    - Stable sequential message IDs
    - ISO timestamp normalization

    Original text is not modified outside whitespace cleanup.
    """

    if not transcript_text or not transcript_text.strip():
        return []

    lines = transcript_text.splitlines()

    messages = []
    current_message = None

    known_speaker_set = (
        {speaker.casefold() for speaker in known_speakers}
        if known_speakers is not None
        else None
    )

    # Detect whether valid timestamped headers exist.
    timestamped_mode = False

    for line in lines:
        match = TIMESTAMPED_PATTERN.match(line)

        if match:
            raw_timestamp = (
                match.group("bracket_ts")
                or match.group("plain_ts")
            )

            if normalize_timestamp(raw_timestamp) is not None:
                timestamped_mode = True
                break

    def save_current():
        nonlocal current_message

        if current_message is None:
            return

        current_message["text"] = (
            current_message["text"].strip()
        )

        if current_message["text"]:
            current_message["message_id"] = (
                f"m{len(messages) + 1:04d}"
            )
            messages.append(current_message)

        current_message = None

    for raw_line in lines:

        line = raw_line.strip()

        if not line:
            continue

        match = (
            TIMESTAMPED_PATTERN.match(line)
            if timestamped_mode
            else SPEAKER_PATTERN.match(line)
        )

        is_header = False
        timestamp = None
        speaker = None

        if match:
            speaker = match.group("speaker").strip()

            if timestamped_mode:
                raw_timestamp = (
                    match.group("bracket_ts")
                    or match.group("plain_ts")
                )

                timestamp = normalize_timestamp(raw_timestamp)

                is_header = timestamp is not None

            else:
                is_header = (
                    known_speaker_set is None
                    or speaker.casefold() in known_speaker_set
                )

        if is_header:
            save_current()

            current_message = {
                "message_id": None,
                "speaker": speaker,
                "timestamp": timestamp,
                "text": match.group("text").strip(),
            }

        elif current_message is not None:
            current_message["text"] += "\n" + line

        else:
            # Preserve content preceding the first
            # recognizable message header.
            current_message = {
                "message_id": None,
                "speaker": "Unknown",
                "timestamp": None,
                "text": line,
            }

    save_current()

    return messages


# -------------------------------------------------------
# Standalone verification
# -------------------------------------------------------
if __name__ == "__main__":

    sample = """
[2026-10-28 09:00] Sarah: Please investigate the error.
Error: database connection timeout
This happened during deployment.

[2026-10-28 09:15] Alex: I'll check the logs.
"""

    from pprint import pprint

    parsed = parse_transcript(sample)

    pprint(parsed, sort_dicts=False)

    print("\nTotal messages:", len(parsed))
