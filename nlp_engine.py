import re
import spacy
import dateparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List, Dict, Any
from datetime import datetime

# Module imports
from coref_engine import GlobalAssetStore, resolve_coreferences_chunk
from zero_shot_engine import evaluate_task_aspects
from parser import chunk_messages_sliding_window

nlp = spacy.load("en_core_web_sm")
analyzer = SentimentIntensityAnalyzer()

URGENCY_KEYWORDS = {"asap", "urgent", "critical", "error", "errors", "500", "production", "failing", "losing", "immediately", "outage", "latency"}

NON_PERSON_WORDS = {
    "understood", "thanks", "thank", "please", "sure", "guys", "team", "hi", "hello",
    "also", "yes", "no", "ok", "okay", "sounds", "great", "perfect", "critical", "urgent",
    "regarding", "incident", "we", "i", "they", "you", "it", "this", "that", "however"
}

def extract_assignee(doc, text: str, default_speaker: str) -> str:
    causative_match = re.search(
        r'\b(?:have|ask|get|want)\s+([A-Z][a-z]+)\s+(?:to\s+)?(?:inspect|check|run|fix|update|review|do|handle|work|test|flush|restart|patch|re-index)\b', 
        text
    )
    if causative_match:
        candidate = causative_match.group(1).strip()
        if candidate.lower() not in NON_PERSON_WORDS:
            return candidate

    direct_address = re.match(r'^(?:[A-Za-z0-9_]+:\s*)?([A-Z][a-z]+)\s*,\s*', text)
    if direct_address:
        candidate = direct_address.group(1).strip()
        if candidate.lower() not in NON_PERSON_WORDS:
            return candidate

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            cleaned = ent.text.strip()
            if cleaned.lower() not in NON_PERSON_WORDS and len(cleaned) > 1:
                return cleaned

    first_token = doc[0] if len(doc) > 0 else None
    if first_token and first_token.text.strip(",:.").isalpha():
        potential_name = first_token.text.strip(",:.")
        if len(potential_name) > 1 and potential_name[0].isupper():
            if potential_name.lower() not in NON_PERSON_WORDS:
                return potential_name

    return default_speaker


def extract_deadline(text: str, message_timestamp_str: str = "N/A") -> str:
    base_date = None
    if message_timestamp_str and message_timestamp_str != "N/A":
        base_date = dateparser.parse(message_timestamp_str)
    if not base_date:
        base_date = datetime.now()

    doc = nlp(text)
    time_candidates = []

    for ent in doc.ents:
        if ent.label_ in ["DATE", "TIME"]:
            time_candidates.append(ent.text)

    if not time_candidates:
        time_patterns = [
            r'\b(?:before|by|at|on|after|until|around)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b',
            r'\b(?:today|tomorrow|yesterday|tonight|friday|monday|tuesday|wednesday|thursday|saturday|sunday)\b(?:\s+(?:morning|afternoon|evening|night|\d{1,2}(?::\d{2})?\s*(?:am|pm)?))?',
            r'\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\s*(?:today|tomorrow)?\b'
        ]
        for pattern in time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            time_candidates.extend(matches)

    for candidate in time_candidates:
        parsed_date = dateparser.parse(
            candidate,
            settings={
                'PREFER_DATES_FROM': 'future',
                'RELATIVE_BASE': base_date
            }
        )
        if parsed_date:
            return parsed_date.strftime("%Y-%m-%d %H:%M:%S")

    return "Unscheduled"


def calculate_urgency_and_sentiment(text: str):
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    words = [token.text.lower() for token in nlp(text)]
    urgency_hits = sum(1 for w in words if w in URGENCY_KEYWORDS)
    
    if urgency_hits >= 2 or compound <= -0.5:
        urgency = "High"
    elif urgency_hits == 1 or compound < 0:
        urgency = "Medium"
    else:
        urgency = "Low"
        
    return compound, urgency


def is_action_sentence(doc, text: str) -> bool:
    text_lower = text.lower().strip()
    non_task_prefixes = ["we noticed", "i noticed", "i am running", "i'm running", "i am checking", "looking into", "sounds good"]
    if any(text_lower.startswith(prefix) for prefix in non_task_prefixes):
        return False

    directive_triggers = [
        "we need", "need the", "need to", "can you", "please", "must", "make sure",
        "have jordan", "have alex", "have elena", "have marcus", "have david", "have sarah",
        "audit", "flush", "restart", "patch", "re-index", "re-indexed", "inspect", "fix", "update"
    ]
    return any(trig in text_lower for trig in directive_triggers)


def process_messages(messages: List[Dict[str, Any]], enable_coref: bool = True, enable_zero_shot: bool = True) -> Dict[str, Any]:
    """
    Processes arbitrarily long threads (100–1000 lines) using sliding window chunking,
    rolling coreference state, and deduplication.
    """
    if not messages:
        return {"tasks": [], "participant_metrics": []}

    # Slice into 15-message overlapping windows
    chunks = chunk_messages_sliding_window(messages, window_size=15, step_size=10)
    
    tasks = []
    seen_task_signatures = set()
    participant_scores = {}
    asset_store = GlobalAssetStore()

    for chunk in chunks:
        chunk_lines = [f"{m['speaker']}: {m['text']}" for m in chunk]
        
        # Step 1: Resolve coreferences within window using global asset store
        if enable_coref:
            resolved_lines = resolve_coreferences_chunk(chunk_lines, asset_store)
            for i, line in enumerate(resolved_lines):
                clean_text = re.sub(r'^[A-Za-z0-9_]+:\s*', '', line)
                chunk[i]["text"] = clean_text

        # Step 2: Extract tasks & fill metadata slots
        for msg in chunk:
            speaker = msg["speaker"]
            text = msg["text"]
            msg_timestamp = msg.get("timestamp", "N/A")

            doc = nlp(text)
            compound_score, urgency_level = calculate_urgency_and_sentiment(text)

            if speaker not in participant_scores:
                participant_scores[speaker] = {"scores": [], "urgency": "Low"}
            participant_scores[speaker]["scores"].append(compound_score)
            if urgency_level == "High":
                participant_scores[speaker]["urgency"] = "High"

            if is_action_sentence(doc, text):
                assignee = extract_assignee(doc, text, speaker)
                deadline = extract_deadline(text, message_timestamp_str=msg_timestamp)
                
                # Signature for deduplication across overlapping windows
                sig = f"{assignee.lower()}_{text.lower()[:30]}"
                if sig in seen_task_signatures:
                    continue
                seen_task_signatures.add(sig)

                status = "In Progress" if any(kw in text.lower() for kw in ["looking into", "handle", "working"]) else "To-Do"

                aspects = evaluate_task_aspects(text, thread_context=chunk_lines[0]) if enable_zero_shot else {
                    "risk_level": "Low Risk Routine Task",
                    "risk_confidence": 1.0,
                    "category": "Administrative Task",
                    "aspect_tone": "Routine Directive"
                }

                tasks.append({
                    "task_id": len(tasks) + 1,
                    "description": text,
                    "owner": assignee,
                    "status": status,
                    "deadline": deadline,
                    "urgency": urgency_level,
                    "speaker": speaker,
                    "category": aspects["category"],
                    "risk_level": aspects["risk_level"],
                    "risk_confidence": aspects["risk_confidence"],
                    "aspect_tone": aspects["aspect_tone"]
                })

    participant_metrics = []
    for person, data in participant_scores.items():
        avg_score = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0.0
        tone = "Urgent/Stressed" if avg_score < -0.2 else ("Constructive" if avg_score >= 0 else "Neutral")
        participant_metrics.append({
            "name": person,
            "avg_sentiment": round(avg_score, 2),
            "dominant_tone": tone,
            "urgency": data["urgency"]
        })

    return {
        "tasks": tasks,
        "participant_metrics": participant_metrics
    }