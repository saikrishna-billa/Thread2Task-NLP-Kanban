
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

path = "data/processed/annotation_review.csv"
df = pd.read_csv(path)

df["human_label"] = (
    df["human_label"].fillna("").str.strip().str.upper()
)

valid_labels = {"ACTION_ITEM", "NONE", "UNCERTAIN", ""}
assert set(df["human_label"]).issubset(valid_labels), \
    "Unexpected annotation labels found"

reviewed = df[df["human_label"] != ""].copy()
certain = reviewed[
    reviewed["human_label"].isin(["ACTION_ITEM", "NONE"])
]

print("\nANNOTATION SUMMARY")
print("=" * 50)
print("Total rows:", len(df))
print("Reviewed:", len(reviewed))
print("Unreviewed:", len(df) - len(reviewed))
print("\nHuman labels:")
print(reviewed["human_label"].value_counts())

print("\nCONFUSION MATRIX — ORIGINAL VS HUMAN")
print(pd.crosstab(
    certain["human_label"],
    certain["label"],
    rownames=["Human"],
    colnames=["Original"],
    dropna=False
))

if len(certain):
    agreement = (
        certain["human_label"] == certain["label"]
    ).mean()
    print(f"\nLabel agreement: {agreement:.2%}")

print("\nDISAGREEMENTS")
disagreements = certain[
    certain["human_label"] != certain["label"]
]

print(disagreements[
    ["sentence", "label", "human_label", "review_notes"]
].to_string(index=False))

print("\nUNCERTAIN EXAMPLES")
print(reviewed[
    reviewed["human_label"] == "UNCERTAIN"
][["sentence", "review_notes"]].to_string(index=False))
