
import re
from typing import Dict, Any


# Explicit task assignments, including passive voice.
ASSIGNMENT_PATTERNS = [
    r"\b(?:has|have|had|was|were|is|are)\s+been\s+assigned\s+to\b",
    r"\b(?:was|were|is|are)\s+assigned\s+to\b",
    r"\b(?:has|have)\s+assigned\s+\w+\s+to\b",
]

# Non-committal proposals that should not automatically
# generate Kanban cards.
SUGGESTION_PATTERNS = [
    r"\bshould\s+consider\b",
    r"\bmaybe\s+(?:we|they|i)\s+could\b",
    r"\bmight\s+be\s+(?:useful|worthwhile|good)\s+to\b",
    r"\bperhaps\s+(?:we|they|i)\s+could\b",
    r"\bwhat\s+if\s+we\b",
]

# Explicit commitments or requests.
COMMITMENT_PATTERNS = [
    r"\bi\s+will\b",
    r"\bi['’]ll\b",
    r"\bi\s+promise\s+to\b",
    r"\bi['’]m\s+going\s+to\b",
    r"\bi\s+am\s+going\s+to\b",
]

REQUEST_PATTERNS = [
    r"\bplease\s+\w+",
    r"\bcould\s+you\s+\w+",
    r"\bcan\s+you\s+\w+",
]


def matches_any(text: str, patterns: list) -> bool:
    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in patterns
    )


def analyze_discourse(text: str) -> Dict[str, Any]:
    """
    Identify explicit conversational-intent cues.

    Returns an interpretable signal rather than a final
    task-extraction decision.

    UNKNOWN means the patterns alone are insufficient.
    """

    if not text or not text.strip():
        return {
            "intent": "UNKNOWN",
            "evidence": "Empty message",
        }

    if matches_any(text, ASSIGNMENT_PATTERNS):
        return {
            "intent": "ASSIGNMENT",
            "evidence": "Explicit assignment construction",
        }

    if matches_any(text, SUGGESTION_PATTERNS):
        return {
            "intent": "SUGGESTION",
            "evidence": "Non-committal proposal language",
        }

    if matches_any(text, COMMITMENT_PATTERNS):
        return {
            "intent": "COMMITMENT",
            "evidence": "First-person commitment language",
        }

    if matches_any(text, REQUEST_PATTERNS):
        return {
            "intent": "REQUEST",
            "evidence": "Request construction",
        }

    return {
        "intent": "UNKNOWN",
        "evidence": "No explicit discourse cue matched",
    }
