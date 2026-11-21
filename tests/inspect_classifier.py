
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "action_classifier"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
).to(device)

model.eval()

# Handwritten diagnostic examples, not a benchmark.
examples = [
    ("Alex, finish the API by Friday.", 1),
    ("I'll prepare the presentation tomorrow.", 1),
    ("Sarah needs to submit the report.", 1),
    ("Please fix the login bug.", 1),
    ("I'll handle it.", 1),
    ("We discussed the API yesterday.", 0),
    ("The server crashed last night.", 0),
    ("Good morning everyone.", 0),
    ("The presentation looks great.", 0),
    ("We should consider improving performance.", 0),
]

sentences = [text for text, _ in examples]

inputs = tokenizer(
    sentences,
    padding=True,
    truncation=True,
    max_length=96,
    return_tensors="pt",
).to(device)

with torch.no_grad():
    outputs = model(**inputs)
    probabilities = torch.softmax(
        outputs.logits, dim=-1
    )[:, 1].cpu().tolist()

print("\nCLASSIFIER PREDICTIONS")
print("=" * 90)

for (sentence, expected), probability in zip(
    examples, probabilities
):
    predicted = int(probability >= 0.5)

    expected_label = (
        "ACTION_ITEM" if expected else "NONE"
    )
    predicted_label = (
        "ACTION_ITEM" if predicted else "NONE"
    )

    print(f"\nSentence: {sentence}")
    print(f"Expected: {expected_label}")
    print(f"Predicted: {predicted_label}")
    print(f"P(Action): {probability:.4f}")

print("\nDiagnostic inspection complete.")
