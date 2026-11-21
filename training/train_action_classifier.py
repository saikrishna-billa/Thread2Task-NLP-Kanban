
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data/processed/train.csv"
VAL_PATH = ROOT / "data/processed/val.csv"
MODEL_DIR = ROOT / "models/action_classifier"

MODEL_NAME = "distilbert-base-uncased"

LABEL_TO_ID = {
    "NONE": 0,
    "ACTION_ITEM": 1,
}

ID_TO_LABEL = {
    0: "NONE",
    1: "ACTION_ITEM",
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(path):
    df = pd.read_csv(path)

    required = {"sentence", "label"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing required columns in {path}"
        )

    df = df.dropna(subset=["sentence", "label"]).copy()

    df["sentence"] = df["sentence"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    df = df[
        (df["sentence"] != "")
        & df["label"].isin(LABEL_TO_ID)
    ].copy()

    df["label_id"] = df["label"].map(LABEL_TO_ID)

    return df.reset_index(drop=True)


def sample_data(df, limit, seed=42):
    if limit is None or limit >= len(df):
        return df.reset_index(drop=True)

    selected, _ = train_test_split(
        df,
        train_size=limit,
        random_state=seed,
        stratify=df["label_id"],
    )

    return selected.reset_index(drop=True)


class ActionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=96):
        self.sentences = dataframe["sentence"].tolist()
        self.labels = dataframe["label_id"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.sentences[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded[
                "attention_mask"
            ].squeeze(0),
            "labels": torch.tensor(
                self.labels[idx],
                dtype=torch.long,
            ),
        }


def calculate_metrics(y_true, y_pred):
    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            pos_label=1,
            zero_division=0,
        )
    )

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).tolist(),
    }


def evaluate(model, loader, device):
    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch[
                "attention_mask"
            ].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            predictions = torch.argmax(
                outputs.logits,
                dim=1,
            )

            all_true.extend(
                batch["labels"].tolist()
            )
            all_pred.extend(
                predictions.cpu().tolist()
            )

    return calculate_metrics(all_true, all_pred)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs", type=int, default=1
    )
    parser.add_argument(
        "--batch_size", type=int, default=8
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=160,
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
    )

    args = parser.parse_args()

    set_seed(42)

    torch.set_num_threads(
        min(4, torch.get_num_threads())
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Loading datasets...")

    train_df = load_data(TRAIN_PATH)
    val_df = load_data(VAL_PATH)

    train_df = sample_data(
        train_df, args.max_train_samples
    )
    val_df = sample_data(
        val_df, args.max_val_samples
    )

    print("Training examples:", len(train_df))
    print("Validation examples:", len(val_df))

    print("\nTraining label distribution:")
    print(train_df["label"].value_counts())

    if train_df["label_id"].nunique() != 2:
        raise ValueError(
            "Training data must contain both classes."
        )

    print("\nLoading tokenizer and model...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    ).to(device)

    train_dataset = ActionDataset(
        train_df, tokenizer
    )
    val_dataset = ActionDataset(
        val_df, tokenizer
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    counts = np.bincount(
        train_df["label_id"].to_numpy(),
        minlength=2,
    )

    class_weights = (
        counts.sum() / (2.0 * counts)
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )

    print(
        "Class weights:",
        class_weights.cpu().tolist(),
    )

    loss_function = torch.nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_f1 = -1.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        print(
            f"\nEpoch {epoch + 1}/{args.epochs}"
        )

        for batch_idx, batch in enumerate(
            train_loader, start=1
        ):
            input_ids = batch["input_ids"].to(
                device
            )

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            labels = batch["labels"].to(
                device
            )

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = loss_function(
                outputs.logits,
                labels,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            running_loss += loss.item()

            if (
                batch_idx % 10 == 0
                or batch_idx == len(train_loader)
            ):
                print(
                    f"Batch {batch_idx}/"
                    f"{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}",
                    flush=True,
                )

        avg_loss = (
            running_loss / len(train_loader)
        )

        metrics = evaluate(
            model, val_loader, device
        )

        print(
            f"\nAverage training loss: "
            f"{avg_loss:.4f}"
        )

        print(
            "Validation metrics:",
            json.dumps(metrics, indent=2),
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]

            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)

            with open(
                MODEL_DIR / "metrics.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {
                        "note": (
                            "Metrics are measured "
                            "against weak validation labels."
                        ),
                        "model": MODEL_NAME,
                        "train_samples": len(train_df),
                        "val_samples": len(val_df),
                        "epochs_completed": epoch + 1,
                        "validation_metrics": metrics,
                    },
                    f,
                    indent=2,
                )

            print("Best model saved:", MODEL_DIR)

    print("\nTraining completed.")
    print("Best weak-label validation F1:", best_f1)


if __name__ == "__main__":
    main()
