
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "action_classifier"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
).to(device)
model.eval()

# These are diagnostic labels, not a formal benchmark.
# 1 = actionable, 0 = not actionable.

examples = [
    # Clear requests
    ("Alex, send the report by Friday.", 1, "Request"),
    ("Please review the pull request.", 1, "Request"),
    ("Could you update the dashboard?", 1, "Indirect request"),

    # Commitments and assignments
    ("I'll fix the authentication bug.", 1, "Commitment"),
    ("Sarah will prepare the slides.", 1, "Assignment"),
    ("I can take care of the documentation.", 1, "Commitment"),

    # Non-committal ideas
    ("We should consider changing the database.", 0, "Suggestion"),
    ("Maybe we could redesign the homepage.", 0, "Suggestion"),
    ("It might be useful to add notifications.", 0, "Suggestion"),

    # Status and history
    ("I fixed the authentication bug yesterday.", 0, "Status"),
    ("The report was submitted last night.", 0, "Status"),
    ("Alex is currently testing the API.", 0, "Status"),

    # Discussion and facts
    ("The deadline is Friday.", 0, "Information"),
    ("We discussed the dashboard in our meeting.", 0, "Discussion"),
    ("There is a bug in the login module.", 0, "Information"),

    # Explicit decisions involving future work
    ("We agreed that Sarah will update the API.", 1, "Decision"),
    ("Alex has been assigned to test the release.", 1, "Assignment"),
    ("I promise to send the files tonight.", 1, "Commitment"),
]

sentences = [item[0] for item in examples]

inputs = tokenizer(
    sentences,
    padding=True,
    truncation=True,
    max_length=96,
    return_tensors="pt",
).to(device)

with torch.no_grad():
    logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=1)[:, 1]
    probabilities = probabilities.cpu().tolist()

correct = 0
errors = []

for (sentence, expected, category), probability in zip(
    examples, probabilities
):
    predicted = int(probability >= 0.5)

    if predicted == expected:
        correct += 1
        status = "PASS"
    else:
        status = "FAIL"
        errors.append((category, sentence, expected, probability))

    print(f"\n[{status}] {category}")
    print(f"Text: {sentence}")
    print(f"Expected: {'ACTION_ITEM' if expected else 'NONE'}")
    print(f"Predicted: {'ACTION_ITEM' if predicted else 'NONE'}")
    print(f"P(Action): {probability:.4f}")

print("\n" + "=" * 65)
print("HARD EXAMPLE DIAGNOSTICS")
print("=" * 65)
print(f"Correct: {correct}/{len(examples)}")
print(f"Errors: {len(errors)}")

print("\nMISCLASSIFICATIONS")
for category, sentence, expected, probability in errors:
    print(
        f"[{category}] {sentence} | "
        f"Expected={expected} | P(Action)={probability:.4f}"
    )
