
from pathlib import Path
import pandas as pd

BASE = Path("data/processed")

train = pd.read_csv(BASE / "train.csv")
val = pd.read_csv(BASE / "val.csv")

def sample_examples(df, split_name, positive_count, negative_count):
    positive = df[df["label"] == "ACTION_ITEM"]
    negative = df[df["label"] == "NONE"]

    selected = pd.concat([
        positive.sample(n=positive_count, random_state=42),
        negative.sample(n=negative_count, random_state=42)
    ], ignore_index=True)

    selected["source_split"] = split_name

    return selected

# 80 training examples, 40 validation examples
review = pd.concat([
    sample_examples(train, "train", 40, 40),
    sample_examples(val, "val", 20, 20)
], ignore_index=True)

# Shuffle so annotations aren't grouped by class
review = review.sample(frac=1, random_state=123)
review = review.reset_index(drop=True)

review.insert(0, "review_id", range(1, len(review) + 1))

# Empty columns for manual human annotations
review["human_label"] = ""
review["review_notes"] = ""

review = review[
    [
        "review_id",
        "source_split",
        "sentence",
        "label",
        "human_label",
        "review_notes"
    ]
]

output_path = BASE / "annotation_review.csv"
review.to_csv(output_path, index=False)

print("Annotation review dataset created.")
print("Total examples:", len(review))
print("Saved at:", output_path)
print("\nSource labels:")
print(review["label"].value_counts())
