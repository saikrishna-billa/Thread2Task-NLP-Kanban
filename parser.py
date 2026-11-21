import re
from typing import List, Dict, Any

def parse_transcript(transcript_text: str) -> List[Dict[str, Any]]:
    """
    Parses unformatted raw transcript text into structured message dictionaries.
    Extracts timestamps, speakers, and message text.
    """
    lines = [line.strip() for line in transcript_text.strip().split("\n") if line.strip()]
    parsed_messages = []

    # Regex for timestamps: [10:15 AM], [2026-10-15 09:00:00], 10/18/2026 16:00:00
    timestamp_pattern = r'^(?:\[([^\]]+)\]|(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})|(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}))\s*'
    # Regex for speaker: "Sarah:", "Alex:"
    speaker_pattern = r'^([A-Za-z0-9_]+):\s*(.*)$'

    for line in lines:
        timestamp = "N/A"
        body = line

        # 1. Match Timestamp
        ts_match = re.match(timestamp_pattern, body)
        if ts_match:
            timestamp = ts_match.group(1) or ts_match.group(2) or ts_match.group(3)
            body = body[ts_match.end():].strip()

        # 2. Match Speaker
        spk_match = re.match(speaker_pattern, body)
        if spk_match:
            speaker = spk_match.group(1).strip()
            text = spk_match.group(2).strip()
        else:
            speaker = "System"
            text = body.strip()

        if text:
            parsed_messages.append({
                "timestamp": timestamp,
                "speaker": speaker,
                "text": text
            })

    return parsed_messages


def chunk_messages_sliding_window(
    messages: List[Dict[str, Any]], 
    window_size: int = 15, 
    step_size: int = 10
) -> List[List[Dict[str, Any]]]:
    """
    Slices long transcripts (100–1000 lines) into overlapping message chunks
    to fit within Transformer token limits (512–1024 tokens per pass).
    """
    if len(messages) <= window_size:
        return [messages]

    chunks = []
    i = 0
    while i < len(messages):
        chunk = messages[i : i + window_size]
        chunks.append(chunk)
        if i + window_size >= len(messages):
            break
        i += step_size

    return chunks