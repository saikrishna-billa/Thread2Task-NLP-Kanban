
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from action_pipeline import ActionPipeline
from run_pipeline import run_thread2task


GOLD_PATH = ROOT / "data/evaluation/heldout/thread2task_gold_ai_draft.json"

REPORT_PATH = ROOT / "data/evaluation/heldout/evaluation_report_ai_draft.json"

STOP = {
    "the", "a", "an", "to", "by", "on", "at",
    "for", "of", "and", "please", "will",
}

VERB_FORMS = {
    "preparing": "prepare",
    "prepared": "prepare",
    "finishing": "finish",
    "finished": "finish",
    "testing": "test",
    "tested": "test",
    "updating": "update",
    "updated": "update",
    "drafting": "draft",
    "drafted": "draft",
    "writing": "write",
    "written": "write",
    "reviewing": "review",
    "reviewed": "review",
    "verifying": "verify",
    "verified": "verify",
}


def normalize(text):
    words = re.findall(r"[a-z0-9]+", (text or "").lower())

    return {
        VERB_FORMS.get(word, word)
        for word in words
        if word not in STOP
    }


def similarity(a, b):
    left = normalize(a)
    right = normalize(b)

    if not left or not right:
        return 0.0

    overlap = len(left & right)

    return (2.0 * overlap) / (len(left) + len(right))


def align_tasks(gold, predicted, threshold=0.50):
    """
    Find a one-to-one task alignment maximizing:
    1. Number of accepted matches
    2. Total similarity

    Designed for small conversation-level task sets.
    """

    scores = [
        [
            similarity(
                g["description"],
                p.get("description") or p.get(
                    "description_raw", ""
                ),
            )
            for p in predicted
        ]
        for g in gold
    ]

    @lru_cache(None)
    def solve(i, used_mask):
        if i == len(gold):
            return (0, 0.0, ())

        best = solve(i + 1, used_mask)
        best = (best[0], best[1], best[2])

        for j in range(len(predicted)):
            if used_mask & (1 << j):
                continue

            score = scores[i][j]

            if score < threshold:
                continue

            count, total, pairs = solve(
                i + 1, used_mask | (1 << j)
            )

            option = (
                count + 1,
                total + score,
                ((i, j),) + pairs,
            )

            if option[:2] > best[:2]:
                best = option

        return best

    _, _, pairs = solve(0, 0)

    pairs = list(pairs)

    matched_gold = {i for i, _ in pairs}
    matched_pred = {j for _, j in pairs}

    return {
        "pairs": pairs,
        "scores": scores,
        "unmatched_gold": [
            i for i in range(len(gold))
            if i not in matched_gold
        ],
        "unmatched_pred": [
            j for j in range(len(predicted))
            if j not in matched_pred
        ],
    }


def compute_prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def same_assignee(a, b):
    return (a or "").strip().casefold() == (
        b or ""
    ).strip().casefold()


def evaluate_case(case, pipeline):
    result = run_thread2task(
        case["transcript"],
        action_pipeline=pipeline,
    )

    gold = case["gold_tasks"]
    predicted = result["tasks"]

    alignment = align_tasks(gold, predicted)

    pairs = alignment["pairs"]

    comparisons = []

    for i, j in pairs:
        g = gold[i]
        p = predicted[j]

        comparisons.append({
            "gold_description": g["description"],
            "predicted_description": p.get("description"),
            "matching_similarity": round(
                alignment["scores"][i][j], 4
            ),
            "assignee_correct": same_assignee(
                g.get("assignee"),
                p.get("assignee"),
            ),
            "deadline_correct": (
                g.get("due_date") == p.get("due_date")
            ),
            "status_correct": (
                g.get("status") == p.get("status")
            ),
            "gold": g,
            "predicted": {
                "task_id": p.get("task_id"),
                "assignee": p.get("assignee"),
                "due_date": p.get("due_date"),
                "status": p.get("status"),
                "needs_review": p.get("needs_review"),
                "source_message_ids": p.get(
                    "source_message_ids", []
                ),
            },
        })

    tp = len(pairs)
    fp = len(alignment["unmatched_pred"])
    fn = len(alignment["unmatched_gold"])

    return {
        "conversation_id": case["conversation_id"],
        "category": case["category"],
        "gold_count": len(gold),
        "predicted_count": len(predicted),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "comparisons": comparisons,
        "missed_gold_tasks": [
            gold[i]
            for i in alignment["unmatched_gold"]
        ],
        "extra_predicted_tasks": [
            {
                "description": predicted[j].get("description"),
                "task_id": predicted[j].get("task_id"),
            }
            for j in alignment["unmatched_pred"]
        ],
        "unmatched_events": result.get(
            "unmatched_events", []
        ),
    }


def main():
    cases = json.loads(
        GOLD_PATH.read_text(encoding="utf-8")
    )

    print("Loading saved DistilBERT model...")
    pipeline = ActionPipeline()

    results = []

    for case in cases:
        print("Evaluating:", case["conversation_id"])
        results.append(evaluate_case(case, pipeline))

    tp = sum(r["tp"] for r in results)
    fp = sum(r["fp"] for r in results)
    fn = sum(r["fn"] for r in results)

    comparisons = [
        c
        for result in results
        for c in result["comparisons"]
    ]

    attribute_metrics = {}

    for name in [
        "assignee_correct",
        "deadline_correct",
        "status_correct",
    ]:
        correct = sum(bool(c[name]) for c in comparisons)

        attribute_metrics[name.replace("_correct", "_accuracy")] = (
            round(correct / len(comparisons), 4)
            if comparisons else None
        )

    report = {
        "evaluation_type": "starter_conversation_benchmark",
        "gold_annotation_status": "REQUIRES_HUMAN_REVIEW",
        "matching_method": "normalized token F1 >= 0.50",
        "total_conversations": len(cases),
        "total_gold_tasks": sum(
            len(c["gold_tasks"]) for c in cases
        ),
        "total_predicted_tasks": sum(
            r["predicted_count"] for r in results
        ),
        "task_detection": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            **compute_prf(tp, fp, fn),
        },
        "matched_task_attributes": {
            "matched_tasks": len(comparisons),
            **attribute_metrics,
        },
        "per_conversation": results,
    }

    REPORT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("THREAD2TASK END-TO-END EVALUATION")
    print("=" * 65)

    print("Conversations:", report["total_conversations"])
    print("Gold tasks:", report["total_gold_tasks"])
    print("Predicted tasks:", report["total_predicted_tasks"])

    print("\nTASK DETECTION:")
    print(json.dumps(
        report["task_detection"], indent=2
    ))

    print("\nMATCHED TASK ATTRIBUTES:")
    print(json.dumps(
        report["matched_task_attributes"], indent=2
    ))

    print("\nPER-CONVERSATION RESULTS:")

    for r in results:
        print(
            r["conversation_id"],
            f"TP={r['tp']}",
            f"FP={r['fp']}",
            f"FN={r['fn']}",
            f"unmatched_events={len(r['unmatched_events'])}",
        )

    print("\nSaved report:", REPORT_PATH)
    print(
        "NOTE: Ground-truth annotations and automatic task "
        "matches require manual verification."
    )


if __name__ == "__main__":
    main()
