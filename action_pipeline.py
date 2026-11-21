
from pathlib import Path
import json
import re
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from parser_v2 import parse_transcript
from discourse_engine import analyze_discourse


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "action_classifier"

def is_meeting_management(text):
    """
    Detect obvious conversational opening/meeting-control
    messages that should not create deliverable tasks.

    Conservative heuristic; not a general intent classifier.
    """
    text = (text or "").strip()

    greeting = re.match(
        r"^(?:good morning|good afternoon|good evening|"
        r"hello everyone|hi everyone|hey team)\b",
        text,
        flags=re.IGNORECASE,
    )

    meeting_control = re.search(
        r"\b(?:let['’]s|let us)\s+"
        r"(?:review|discuss|go over)\s+"
        r"(?:our\s+)?(?:sprint\s+priorities|agenda|"
        r"meeting\s+agenda)\b",
        text,
        flags=re.IGNORECASE,
    )

    return bool(greeting or meeting_control)


class ActionPipeline:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Loading action classifier...")
        print("Device:", self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR
        )

        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                MODEL_DIR
            ).to(self.device)
        )

        self.model.eval()

        # Obtain the actual class index from saved config.
        self.action_id = self.model.config.label2id[
            "ACTION_ITEM"
        ]

    def predict_probabilities(self, texts, batch_size=8):
        if not texts:
            return []

        probabilities = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[
                start:start + batch_size
            ]

            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=96,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**encoded).logits
                probs = torch.softmax(
                    logits, dim=-1
                )[:, self.action_id]

            probabilities.extend(
                probs.cpu().tolist()
            )

        return probabilities

    def decide(self, probability, intent):
        """
        Preliminary candidate triage.

        CREATE_CANDIDATE:
            Strong evidence for a potential task.

        REVIEW:
            Conflicting or ambiguous evidence.

        IGNORE:
            No current task candidate.

        This is not final Kanban task-state logic.
        """

        if intent == "SUGGESTION":
            return "REVIEW"

        if intent in {
            "ASSIGNMENT",
            "COMMITMENT",
            "REQUEST",
        }:
            if probability >= self.threshold:
                return "CREATE_CANDIDATE"

            return "REVIEW"

        if probability >= self.threshold:
            return "CREATE_CANDIDATE"

        return "IGNORE"

    def process(self, transcript, known_speakers=None):
        messages = parse_transcript(
            transcript,
            known_speakers=known_speakers,
        )

        if not messages:
            return []

        texts = [m["text"] for m in messages]
        probabilities = self.predict_probabilities(texts)

        results = []

        for index, (message, probability) in enumerate(
            zip(messages, probabilities)
        ):
            discourse = analyze_discourse(
                message["text"]
            )

            intent = discourse["intent"]

            decision = self.decide(
                probability, intent
            )

            # If this is a meeting-control message, force IGNORE.
            if is_meeting_management(message["text"]):
                decision = "IGNORE"
                
            
            # Preserve preceding message as evidence
            # for later contextual interpretation.
            previous_message = (
                messages[index - 1]
                if index > 0
                else None
            )

            result = {
                "message_id": message["message_id"],
                "speaker": message["speaker"],
                "timestamp": message["timestamp"],
                "text": message["text"],
                "action_probability": round(
                    float(probability), 4
                ),
                "discourse_intent": intent,
                "discourse_evidence": discourse[
                    "evidence"
                ],
                "decision": decision,
                "previous_message_id": (
                    previous_message["message_id"]
                    if previous_message
                    else None
                ),
                "previous_message_text": (
                    previous_message["text"]
                    if previous_message
                    else None
                ),
            }

            results.append(result)

        return results


if __name__ == "__main__":
    transcript = """
[2026-10-30 09:00] Sarah: Alex, finish the login API by Friday.
[2026-10-30 09:05] Alex: I'll handle it.
[2026-10-30 09:10] David: We should consider redesigning the dashboard.
[2026-10-30 09:15] Sarah: The server crashed yesterday.
[2026-10-30 09:20] David: Please prepare the testing report.
[2026-10-30 09:25] Alex: The API is currently working.
[2026-10-30 09:30] Sarah: Alex has been assigned to test the release.
"""

    pipeline = ActionPipeline()

    results = pipeline.process(transcript)

    for result in results:
        print("\n" + "=" * 65)
        print("Message:", result["message_id"])
        print("Speaker:", result["speaker"])
        print("Text:", result["text"])
        print(
            "Action probability:",
            result["action_probability"]
        )
        print("Discourse:", result["discourse_intent"])
        print("Decision:", result["decision"])

    output_path = ROOT / "data" / "processed" / (
        "pipeline_diagnostic.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nSaved results to:", output_path)
