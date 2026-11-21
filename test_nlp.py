import json
from parser import parse_transcript
from nlp_engine import process_messages

sample_transcript = """
[10:15 AM] Sarah: Guys, the payment gateway is throwing 500 errors on production! We need this fixed ASAP.
[10:18 AM] Alex: I am looking into the backend logs now. Looks like the Stripe API key expired.
[10:22 AM] Sarah: Alex, can you update the API keys by 3 PM today? We are losing transactions.
[10:25 AM] Alex: Sure, I will handle the key update before 3 PM.
"""

if __name__ == "__main__":
    messages = parse_transcript(sample_transcript)
    result = process_messages(messages)
    
    print("--- EXTRACTED STRUCTURED JSON SCHEMAS ---")
    print(json.dumps(result, indent=2))