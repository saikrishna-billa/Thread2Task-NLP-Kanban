from coref_engine import resolve_coreferences

sample_transcript = """
Sarah: Guys, the Stripe API key expired on production and the SSL certificate is invalid!
Alex: I see both problems in the logs. I can check them now.
Sarah: Alex, please update it by 3 PM today and renew this before 5 PM.
"""

if __name__ == "__main__":
    print("--- ORIGINAL TRANSCRIPT ---")
    print(sample_transcript.strip())
    
    resolved = resolve_coreferences(sample_transcript)
    
    print("\n--- RESOLVED TRANSCRIPT ---")
    print(resolved.strip())