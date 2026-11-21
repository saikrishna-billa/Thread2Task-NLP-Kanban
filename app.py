import streamlit as st
import pandas as pd
import plotly.express as px
import json
from parser import parse_transcript
from nlp_engine import process_messages

# Page Configuration
st.set_page_config(
    page_title="Thread2Task - Automated Kanban Generator",
    page_icon="📋",
    layout="wide"
)

# Custom Styling for Kanban Cards
st.markdown("""
    <style>
    .kanban-card {
        background-color: #1e222a;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 12px;
        border-left: 5px solid #4CAF50;
    }
    .urgency-High { border-left-color: #FF4B4B !important; }
    .urgency-Medium { border-left-color: #FFA500 !important; }
    .urgency-Low { border-left-color: #00C853 !important; }
    .owner-badge {
        background-color: #2e3545;
        color: #64B5F6;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: bold;
    }
    .deadline-badge {
        color: #B0BEC5;
        font-size: 0.8em;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📋 Thread2Task: Conversational Thread to Kanban Generator")
st.markdown("Extract actionable tasks, map assignees, normalize temporal deadlines, and monitor team tone from multi-turn chat threads.")

# Sidebar Controls
st.sidebar.header("Configuration")
sample_choice = st.sidebar.selectbox(
    "Load Preset Sample Thread",
    ["IT Incident Response", "Project Planning"]
)

# Sample Datasets
SAMPLE_THREADS = {
    "IT Incident Response": """[10:15 AM] Sarah: Guys, the payment gateway is throwing 500 errors on production! We need this fixed ASAP.
[10:18 AM] Alex: I am looking into the backend logs now. Looks like the Stripe API key expired.
[10:22 AM] Sarah: Alex, can you update the API keys by 3 PM today? We are losing transactions.
[10:25 AM] Alex: Sure, I will handle the key update before 3 PM.
[10:30 AM] David: I already updated the staging environment keys yesterday, so staging is fine.
[10:35 AM] Sarah: Thanks David. Please run the full regression test suite on prod once Alex finishes.""",
    
    "Project Planning": """[09:00 AM] Mark: We need to finalize the Q3 marketing launch plan by tomorrow.
[09:05 AM] Elena: I can draft the social media graphic assets before 5 PM today.
[09:12 AM] Mark: Great, Elena. John, please review the ad budget spreadsheet by Friday.
[09:20 AM] John: I will review the budget and update the slides tonight."""
}

default_text = SAMPLE_THREADS[sample_choice]

# Main Text Input
raw_input = st.text_area("Paste Conversational Thread Transcript Below:", value=default_text, height=180)

if st.button("🚀 Process Thread & Generate Board", type="primary"):
    if not raw_input.strip():
        st.warning("Please enter or paste a valid transcript thread.")
    else:
        parsed_msgs = parse_transcript(raw_input)
        results = process_messages(parsed_msgs)
        st.session_state["results"] = results
        st.session_state["parsed_msgs"] = parsed_msgs

# Render Board & Analytics if processed
if "results" in st.session_state:
    results = st.session_state["results"]
    tasks = results["tasks"]
    metrics = results["participant_metrics"]

    st.markdown("---")
    
    # Summary Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Messages Parsed", len(st.session_state.get("parsed_msgs", [])))
    m2.metric("Extracted Tasks", len(tasks))
    high_urgency_count = sum(1 for t in tasks if t["urgency"] == "High")
    m3.metric("High Urgency Items", high_urgency_count)
    m4.metric("Active Participants", len(metrics))

    st.subheader("📌 Interactive Kanban Task Board")
    
    # Kanban Columns
    col_todo, col_in_prog, col_done = st.columns(3)

    todo_tasks = [t for t in tasks if t["status"] == "To-Do"]
    in_prog_tasks = [t for t in tasks if t["status"] == "In Progress"]
    done_tasks = [t for t in tasks if t["status"] == "Completed"]

    def render_task_card(container, task):
        urgency_cls = f"urgency-{task['urgency']}"
        container.markdown(f"""
            <div class="kanban-card {urgency_cls}">
                <b>Task #{task['task_id']}</b><br>
                <span>{task['description']}</span><br><br>
                <span class="owner-badge">👤 {task['owner']}</span>
                <span class="deadline-badge">⏱️ {task['deadline']}</span><br>
                <small style="color:#888;">Urgency: <b>{task['urgency']}</b> | Speaker: {task['speaker']}</small>
            </div>
        """, unsafe_allow_html=True)

    with col_todo:
        st.markdown(f"### 🔴 To-Do ({len(todo_tasks)})")
        for t in todo_tasks:
            render_task_card(st, t)

    with col_in_prog:
        st.markdown(f"### 🟡 In Progress ({len(in_prog_tasks)})")
        for t in in_prog_tasks:
            render_task_card(st, t)

    with col_done:
        st.markdown(f"### 🟢 Completed ({len(done_tasks)})")
        for t in done_tasks:
            render_task_card(st, t)

    st.markdown("---")
    
    # Participant Sentiment & Urgency Analytics Section
    st.subheader("📊 Team Sentiment & Participant Urgency Analysis")
    
    df_metrics = pd.DataFrame(metrics)
    if not df_metrics.empty:
        c1, c2 = st.columns([2, 1])
        
        with c1:
            fig = px.bar(
                df_metrics,
                x="name",
                y="avg_sentiment",
                color="dominant_tone",
                title="Speaker Sentiment Score (-1.0 Negative / Stressed to +1.0 Positive / Constructive)",
                labels={"name": "Participant", "avg_sentiment": "Compound Polarity"},
                range_y=[-1, 1],
                text_auto=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.markdown("#### Participant Breakdown")
            st.dataframe(df_metrics, hide_index=True, use_container_width=True)

    # Structured Data Export Section
    st.markdown("---")
    st.subheader("💾 Structured JSON Export")
    st.json(results)
    
    st.download_button(
        label="📥 Download Structured JSON",
        data=json.dumps(results, indent=2),
        file_name="extracted_tasks.json",
        mime="application/json"
    )