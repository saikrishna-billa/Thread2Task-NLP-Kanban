import pandas as pd
from benchmark_data import BENCHMARK_CASES
from parser import parse_transcript
from nlp_engine import process_messages

def evaluate_pipeline():
    """
    Runs automated evaluation across the benchmark test cases 
    and returns metrics summary dataframes.
    """
    total_expected_tasks = 0
    total_extracted_tasks = 0
    correct_owner_matches = 0
    correct_coref_matches = 0
    correct_risk_matches = 0
    
    detailed_results = []

    for case in BENCHMARK_CASES:
        transcript = case["transcript"]
        expected_tasks = case["expected_tasks"]
        total_expected_tasks += len(expected_tasks)

        # Run pipeline
        parsed_msgs = parse_transcript(transcript)
        pipeline_output = process_messages(parsed_msgs, enable_coref=True, enable_zero_shot=True)
        extracted_tasks = pipeline_output["tasks"]
        total_extracted_tasks += len(extracted_tasks)

        for exp in expected_tasks:
            # Find matching extracted task by owner or keywords
            match = None
            for ext in extracted_tasks:
                if ext["owner"].lower() == exp["expected_owner"].lower() or exp["expected_owner"].lower() in ext["description"].lower():
                    match = ext
                    break

            if match:
                # 1. Owner Resolution Match
                owner_correct = match["owner"].lower() == exp["expected_owner"].lower()
                if owner_correct:
                    correct_owner_matches += 1

                # 2. Coreference Disambiguation Match (Keyword overlap in resolved text)
                exp_keywords = [w.lower() for w in exp["expected_resolved_text"].split() if len(w) > 3]
                matched_keywords = sum(1 for kw in exp_keywords if kw in match["description"].lower())
                coref_correct = matched_keywords >= max(1, int(len(exp_keywords) * 0.5))
                if coref_correct:
                    correct_coref_matches += 1

                # 3. Risk Level Match
                risk_correct = match.get("risk_level") == exp["expected_risk"]
                if risk_correct:
                    correct_risk_matches += 1

                detailed_results.append({
                    "case_id": case["id"],
                    "domain": case["domain"],
                    "expected_owner": exp["expected_owner"],
                    "extracted_owner": match["owner"],
                    "owner_match": "✅ Pass" if owner_correct else "❌ Fail",
                    "extracted_description": match["description"],
                    "coref_match": "✅ Pass" if coref_correct else "❌ Fail",
                    "expected_risk": exp["expected_risk"],
                    "predicted_risk": match.get("risk_level", "N/A"),
                    "risk_match": "✅ Pass" if risk_correct else "❌ Fail"
                })

    # Precision, Recall, F1 Calculation
    precision = (correct_owner_matches / total_extracted_tasks) if total_extracted_tasks > 0 else 0.0
    recall = (correct_owner_matches / total_expected_tasks) if total_expected_tasks > 0 else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    coref_accuracy = (correct_coref_matches / total_expected_tasks) if total_expected_tasks > 0 else 0.0
    risk_accuracy = (correct_risk_matches / total_expected_tasks) if total_expected_tasks > 0 else 0.0

    summary_metrics = {
        "Total Ground Truth Tasks": total_expected_tasks,
        "Total Extracted Tasks": total_extracted_tasks,
        "Task Precision": round(precision, 2),
        "Task Recall": round(recall, 2),
        "Task F1-Score": round(f1_score, 2),
        "Coreference Accuracy": f"{round(coref_accuracy * 100, 1)}%",
        "Owner Assignment Accuracy": f"{round(precision * 100, 1)}%",
        "Zero-Shot Risk Accuracy": f"{round(risk_accuracy * 100, 1)}%"
    }

    return summary_metrics, pd.DataFrame(detailed_results)

if __name__ == "__main__":
    metrics, df_details = evaluate_pipeline()
    print("\n--- PERFORMANCE EVALUATION SUMMARY METRICS ---")
    for key, val in metrics.items():
        print(f"  * {key}: {val}")
    print("\n--- DETAILED CASE-BY-CASE BREAKDOWN ---")
    print(df_details[["case_id", "expected_owner", "extracted_owner", "owner_match", "coref_match", "risk_match"]])