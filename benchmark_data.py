BENCHMARK_CASES = [
    {
        "id": "CASE-01",
        "title": "DevOps IT Outage & Multi-Asset Incident",
        "domain": "DevOps & Site Reliability",
        "transcript": """[11:02 AM] Marcus: Critical incident! The OAuth token service is throwing 500 errors and the Stripe webhook listener has stalled.
[11:05 AM] Elena: I checked Datadog metrics. The Redis cache cluster is also out of memory!
[11:08 AM] Marcus: Elena, can you flush it immediately before 11:30 AM today?
[11:12 AM] Marcus: Jordan, please restart them before 12 PM today or we lose enterprise sync.
[11:15 AM] Jordan: On it. I will patch this before 1 PM today.
[11:20 AM] Marcus: Thanks team. Elena, please review that once Jordan finishes.""",
        "expected_tasks": [
            {
                "line_index": 2,
                "expected_owner": "Elena",
                "expected_resolved_text": "can you flush the Redis cache cluster immediately",
                "expected_risk": "Critical System Outage",
                "expected_category": "Software Bug Fix"
            },
            {
                "line_index": 3,
                "expected_owner": "Jordan",
                "expected_resolved_text": "restart the OAuth token service and the Stripe webhook listener",
                "expected_risk": "Critical System Outage",
                "expected_category": "Infrastructure Update"
            },
            {
                "line_index": 4,
                "expected_owner": "Jordan",
                "expected_resolved_text": "patch the Redis cache cluster",
                "expected_risk": "Critical System Outage",
                "expected_category": "Infrastructure Update"
            }
        ]
    },
    {
        "id": "CASE-02",
        "title": "Regulatory Compliance & Legal Sprint",
        "domain": "Product & Legal",
        "transcript": """[02:15 PM] Rachel: Team, the GDPR privacy policy consent banner and the iOS app analytics tracker require urgent legal compliance sign-off.
[02:18 PM] Liam: I am reviewing the draft specs in Figma right now.
[02:22 PM] Rachel: Liam, please audit these before 4 PM tomorrow.
[02:26 PM] Liam: Understood. Should I also check the Q4 marketing campaign deck?
[02:30 PM] Rachel: Yes, please edit that before 6 PM on Friday.
[02:35 PM] Liam: Perfect, I will approve it once the legal team signs off.""",
        "expected_tasks": [
            {
                "line_index": 2,
                "expected_owner": "Liam",
                "expected_resolved_text": "audit the GDPR privacy policy consent banner and the iOS app analytics tracker",
                "expected_risk": "Moderate Operational Risk",
                "expected_category": "Administrative Task"
            },
            {
                "line_index": 4,
                "expected_owner": "Liam",
                "expected_resolved_text": "edit the Q4 marketing campaign deck",
                "expected_risk": "Low Risk Routine Task",
                "expected_category": "Administrative Task"
            }
        ]
    },
    {
        "id": "CASE-03",
        "title": "Enterprise Customer Churn Escalation",
        "domain": "Billing & Customer Success",
        "transcript": """[04:45 PM] Kevin: Urgent escalation! Client Acme Corp is threatening to terminate their $50k contract because the database migration script failed and their custom report export crashed!
[04:48 PM] Maya: I am looking into their account logs right now.
[04:52 PM] Kevin: Maya, can you fix this before 6 PM today?
[04:55 PM] Kevin: Also, Daniel must inspect them before 8 PM tonight to prevent contract cancellation.
[05:00 PM] Maya: I will process it right now and email the client CEO tonight.""",
        "expected_tasks": [
            {
                "line_index": 2,
                "expected_owner": "Maya",
                "expected_resolved_text": "fix the database migration script",
                "expected_risk": "Critical System Outage",
                "expected_category": "Software Bug Fix"
            },
            {
                "line_index": 3,
                "expected_owner": "Daniel",
                "expected_resolved_text": "inspect the database migration script and the custom report export",
                "expected_risk": "Critical System Outage",
                "expected_category": "Software Bug Fix"
            }
        ]
    }
]