
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SOURCE = Path(
    "data/raw/action_item_source/data/sample_labeled_gemini.csv"
)
OUTPUT = Path("data/processed")
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SOURCE)

# Preserve original row identifiers for traceability
df["source_row"] = range(len(df))

# Normalize text for duplicate detection
df["text_key"] = (
    df["sentence"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

# Remove empty text and invalid labels
df = df[
    df["text_key"].ne("")
    & df["label"].isin(["ACTION_ITEM", "NONE"])
].copy()

# Exclude all conflicting duplicate labels
conflict_count = df.groupby("text_key")["label"].transform("nunique")
df = df[conflict_count == 1].copy()

# Remove exact normalized duplicates globally
df = df.drop_duplicates(subset=["text_key"]).copy()

# Give records with missing meeting IDs unique groups
df["group"] = df["meeting_id"].apply(
    lambda value: f"meeting_{int(value)}"
    if pd.notna(value) else None
)

for idx in df.index[df["group"].isna()]:
    df.at[idx, "group"] = f"unknown_{idx}"

# Split entire meetings rather than individual sentences
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, temp_idx = next(
    splitter.split(df, groups=df["group"])
)

train = df.iloc[train_idx].copy()
temp = df.iloc[temp_idx].copy()

# Divide remaining meetings into validation and test
splitter2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

val_idx, test_idx = next(
    splitter2.split(temp, groups=temp["group"])
)

val = temp.iloc[val_idx].copy()
test = temp.iloc[test_idx].copy()

datasets = {
    "train": train,
    "val": val,
    "test": test
}

# Verify no meeting overlaps
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    assert set(datasets[a]["group"]).isdisjoint(
        set(datasets[b]["group"])
    )

# Verify no identical normalized text crosses splits
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    assert set(datasets[a]["text_key"]).isdisjoint(
        set(datasets[b]["text_key"])
    )

print("\nTHREAD2TASK — PREPARED DATASET")
print("=" * 60)

for name, subset in datasets.items():
    subset = subset.copy()
    subset["binary_label"] = (
        subset["label"] == "ACTION_ITEM"
    ).astype(int)

    subset[
        ["source_row", "meeting_id", "sentence",
         "label", "binary_label"]
    ].to_csv(OUTPUT / f"{name}.csv", index=False)

    positives = subset["binary_label"].sum()

    print(
        f"{name:5s}: {len(subset):5d} examples | "
        f"{positives:4d} action items | "
        f"{positives / len(subset):.1%} positive"
    )

print("\nLeakage checks passed.")
print("Warning: Original labels remain weak/pseudo-labels.")
