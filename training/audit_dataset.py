
from pathlib import Path
import pandas as pd

DATA_PATH = Path(
    "data/raw/action_item_source/data/sample_labeled_gemini.csv"
)

df = pd.read_csv(DATA_PATH)

print("=" * 65)
print("THREAD2TASK — DATASET QUALITY AUDIT")
print("=" * 65)

# 1. General information
print("\n1. DATASET OVERVIEW")
print("Total sentences:", len(df))
print("Unique meetings:", df["meeting_id"].nunique())
print("Missing meeting IDs:", df["meeting_id"].isna().sum())

# 2. Existing dataset splits
print("\n2. EXISTING SPLITS")
print(df["split"].value_counts(dropna=False))
print("\nLabels per split:")
print(pd.crosstab(df["split"], df["label"]))

# 3. Check meeting-level leakage
print("\n3. MEETING-LEVEL LEAKAGE")

valid = df.dropna(subset=["meeting_id"]).copy()

meeting_splits = valid.groupby("meeting_id")["split"].nunique()
leaking_meetings = meeting_splits[meeting_splits > 1]

print("Meetings present in multiple splits:",
      len(leaking_meetings))

if len(leaking_meetings) > 0:
    print("Example overlapping meeting IDs:")
    print(leaking_meetings.head(10))

# 4. Duplicate sentences
print("\n4. DUPLICATE ANALYSIS")

df["clean_sentence"] = (
    df["sentence"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
)

print("Duplicate sentences:",
      df["clean_sentence"].duplicated().sum())

# Conflicting annotations for identical text
conflicts = (
    df.groupby("clean_sentence")["label"]
    .nunique()
)

print("Sentences with conflicting labels:",
      (conflicts > 1).sum())

# 5. Sentence lengths
print("\n5. SENTENCE LENGTH ANALYSIS")

df["word_count"] = (
    df["clean_sentence"].str.split().str.len()
)

print(df["word_count"].describe())

print("\nVery short sentences (<= 3 words):",
      (df["word_count"] <= 3).sum())

# 6. Print action examples for inspection
print("\n6. RANDOM ACTION-ITEM EXAMPLES")

action_examples = df[df["label"] == "ACTION_ITEM"]

print(
    action_examples.sample(
        n=min(15, len(action_examples)),
        random_state=42
    )[["sentence", "label"]].to_string(index=False)
)

# 7. Save a summary-friendly report
output = Path("data/processed/dataset_audit_summary.csv")

audit_summary = pd.crosstab(
    df["split"],
    df["label"]
)

audit_summary.to_csv(output)

print("\nAudit summary saved to:", output)
