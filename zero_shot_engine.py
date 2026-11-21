from transformers import pipeline

_zero_shot_classifier = None

# Taxonomic Label Sets
RISK_LABELS = [
    "Critical System Outage",
    "Moderate Operational Risk",
    "Low Risk Routine Task"
]

CATEGORY_LABELS = [
    "Software Bug Fix",
    "Infrastructure Update",
    "Feature Request",
    "Administrative Task"
]

TONE_LABELS = [
    "Technical Blame / Frustration",
    "Constructive Collaboration",
    "Routine Directive"
]

def get_zero_shot_classifier():
    global _zero_shot_classifier
    if _zero_shot_classifier is None:
        _zero_shot_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1  # CPU execution
        )
    return _zero_shot_classifier

def evaluate_task_aspects(text: str, thread_context: str = "") -> dict:
    """
    Evaluates multi-dimensional aspects (Risk Level, Task Category, Tone)
    for a given task description using Zero-Shot Natural Language Inference.
    Optionally accepts thread_context to improve classification accuracy on complex threads.
    """
    if not text.strip():
        return {
            "risk_level": "Low Risk Routine Task",
            "risk_confidence": 0.0,
            "category": "Administrative Task",
            "aspect_tone": "Routine Directive"
        }

    # Prepend opening thread context if provided to preserve high-risk indicators
    eval_input = f"Context: {thread_context} | Task: {text}" if thread_context else text

    try:
        classifier = get_zero_shot_classifier()

        # 1. Risk Evaluation
        risk_res = classifier(eval_input, candidate_labels=RISK_LABELS)
        top_risk = risk_res["labels"][0]
        risk_conf = round(risk_res["scores"][0], 2)

        # 2. Category Classification
        cat_res = classifier(text, candidate_labels=CATEGORY_LABELS)
        top_cat = cat_res["labels"][0]

        # 3. Tone / Interpersonal Aspect
        tone_res = classifier(eval_input, candidate_labels=TONE_LABELS)
        top_tone = tone_res["labels"][0]

        return {
            "risk_level": top_risk,
            "risk_confidence": risk_conf,
            "category": top_cat,
            "aspect_tone": top_tone
        }
    except Exception as e:
        print(f"[Zero-Shot Warning] Classification skipped: {e}")
        return {
            "risk_level": "Moderate Operational Risk",
            "risk_confidence": 0.50,
            "category": "Administrative Task",
            "aspect_tone": "Routine Directive"
        }