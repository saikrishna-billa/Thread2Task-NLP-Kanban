import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any

from parser import parse_transcript
from nlp_engine import process_messages

# Diagnostic Benchmark Test Dataset
EVALUATION_DATASET = [
    {
        "id": "LONG-01",
        "description": "Multi-turn incident with implicit tasks & temporal drift",
        "raw_transcript": """[2026-10-15 09:00:00] Marcus: We noticed high latency on the user database cluster since yesterday.
[2026-10-15 09:05:00] Elena: I am running diagnostic queries right now.
[2026-10-15 09:12:00] Marcus: Elena, if the query pool exhausts connections, we need the replica nodes re-indexed before 4 PM today.
[2026-10-15 09:20:00] Elena: Understood. I will also have Jordan inspect the connection pool limits tomorrow morning.
[2026-10-15 09:30:00] Jordan: Sounds good, I will check the pool configuration on Friday at 10 AM.""",
        "ground_truth": [
            {
                "task_trigger": "re-indexed",
                "expected_owner": "Elena",
                "expected_deadline": "2026-10-15 16:00:00",
                "expected_antecedent": "the replica nodes",
                "is_implicit": True
            },
            {
                "task_trigger": "inspect",
                "expected_owner": "Jordan",
                "expected_deadline": "2026-10-16 09:00:00",
                "expected_antecedent": "the connection pool limits",
                "is_implicit": False
            }
        ]
    }
]

def run_diagnostic_evaluation():
    print("=" * 80)
    print("      THREAD2TASK SYSTEM DIAGNOSTIC & EVALUATION ENGINE")
    print("=" * 80)

    total_tp, total_fp, total_fn = 0, 0, 0
    correct_owners, correct_deadlines, correct_coref = 0, 0, 0
    total_slots = 0

    error_log = []
    start_time = time.time()

    for item in EVALUATION_DATASET:
        test_id = item["id"]
        raw_text = item["raw_transcript"]
        ground_truth = item["ground_truth"]
        
        parsed = parse_transcript(raw_text)
        result = process_messages(parsed, enable_coref=True, enable_zero_shot=True)
        predicted_tasks = result["tasks"]

        total_slots += len(ground_truth)

        # Match predicted tasks against ground truth tasks
        matched_gt_indices = set()
        for pred in predicted_tasks:
            pred_desc = pred["description"].lower()
            matched = False

            for gt_idx, gt in enumerate(ground_truth):
                if gt["task_trigger"].lower() in pred_desc:
                    matched = True
                    matched_gt_indices.add(gt_idx)

                    # 1. Owner Evaluation
                    owner_pass = pred["owner"].lower() == gt["expected_owner"].lower()
                    if owner_pass:
                        correct_owners += 1
                    else:
                        error_log.append({
                            "test_id": test_id,
                            "error_type": "Owner Extraction Error",
                            "expected": gt["expected_owner"],
                            "predicted": pred["owner"],
                            "context": pred["description"]
                        })

                    # 2. Temporal Evaluation
                    gt_date_str = gt["expected_deadline"].split()[0] if gt["expected_deadline"] != "Unscheduled" else "Unscheduled"
                    pred_date_str = pred["deadline"].split()[0] if pred["deadline"] != "Unscheduled" else "Unscheduled"
                    deadline_pass = (gt_date_str == pred_date_str)
                    if deadline_pass:
                        correct_deadlines += 1
                    else:
                        error_log.append({
                            "test_id": test_id,
                            "error_type": "Temporal Resolution Error",
                            "expected": gt["expected_deadline"],
                            "predicted": pred["deadline"],
                            "context": pred["description"]
                        })

                    # 3. Coreference Evaluation
                    coref_pass = gt["expected_antecedent"].lower() in pred_desc
                    if coref_pass:
                        correct_coref += 1
                    else:
                        error_log.append({
                            "test_id": test_id,
                            "error_type": "Coreference Disambiguation Error",
                            "expected": gt["expected_antecedent"],
                            "predicted": pred["description"],
                            "context": pred["description"]
                        })

            if matched:
                total_tp += 1
            else:
                total_fp += 1
                error_log.append({
                    "test_id": test_id,
                    "error_type": "False Positive Task Extraction",
                    "expected": "Non-actionable message",
                    "predicted": pred["description"],
                    "context": pred["description"]
                })

        total_fn += (len(ground_truth) - len(matched_gt_indices))

    # Calculate Macro Metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    owner_accuracy = (correct_owners / total_slots) * 100 if total_slots > 0 else 0.0
    deadline_accuracy = (correct_deadlines / total_slots) * 100 if total_slots > 0 else 0.0
    coref_accuracy = (correct_coref / total_slots) * 100 if total_slots > 0 else 0.0

    latency = time.time() - start_time

    print("\n--- QUANTITATIVE PERFORMANCE METRICS ---")
    print(f"  * Task Extraction Precision : {precision:.2f}")
    print(f"  * Task Extraction Recall    : {recall:.2f}")
    print(f"  * Task Extraction F1-Score  : {f1:.2f}")
    print(f"  * Owner Slot Accuracy       : {owner_accuracy:.1f}%")
    print(f"  * Temporal Slot Accuracy    : {deadline_accuracy:.1f}%")
    print(f"  * Coreference Resolution    : {coref_accuracy:.1f}%")
    print(f"  * Total Processing Time     : {latency:.2f} seconds")

    if error_log:
        print("\n--- SYSTEM DIAGNOSTIC ERROR MATRIX ---")
        df_errors = pd.DataFrame(error_log)
        print(df_errors.to_string(index=False))

if __name__ == "__main__":
    run_diagnostic_evaluation()