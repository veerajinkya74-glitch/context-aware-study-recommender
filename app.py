import streamlit as st
import sqlite3
import pandas as pd
from db_setup import init_db, DB_NAME
from model import recommend_next, train_and_save

st.set_page_config(page_title="Context-Aware Study Recommender", layout="wide")

# Ensure DB exists on first run
try:
    conn = sqlite3.connect(DB_NAME)
    pd.read_sql("SELECT 1 FROM items LIMIT 1", conn)
    conn.close()
except Exception:
    init_db()
    train_and_save()

st.title("Context-Aware Study & Coding Recommender")
st.caption("Hybrid Collaborative + Content-Based Engine with Time-Window & Error-Recovery Context")

# Sidebar: Learner Context Controls
st.sidebar.header("Your Current Study Context")
user_id = st.sidebar.number_input("Learner ID", min_value=1, max_value=15, value=1)
available_minutes = st.sidebar.slider("Available Time Window (Minutes)", 15, 90, 30, step=5)
preferred_type = st.sidebar.selectbox("Resource Type", ["All", "Coding Problem", "Tutorial", "Dataset"])
focus_weak = st.sidebar.checkbox("Prioritize Weak Topics & Error Recovery", value=True)

if st.sidebar.button("Reset Demo Database"):
    init_db()
    train_and_save()
    st.sidebar.success("Database & model reset!")

# Get Recommendations
recs, weak_topics = recommend_next(
    user_id=user_id,
    available_minutes=available_minutes,
    preferred_type=preferred_type,
    focus_weak_topics=focus_weak
)

# Display Detected Learner Profile
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader(f"Top Picks for Your {available_minutes}-Minute Session")
    if recs.empty:
        st.warning("No unsolved items fit this exact time window and filter. Try increasing your available minutes.")
    else:
        for _, row in recs.iterrows():
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"### {row['title']}  `{row['item_type']}`")
                c1.markdown(f"**Topic:** {row['topic']} | **Difficulty:** {row['difficulty']} | **Est. Time:** {row['est_minutes']} mins")
                c1.info(f"**Why:** {row['why_recommended']}")
                c2.metric("Match Score", f"{row['final_score']:.2f}")

                # Live Feedback Loop Buttons
                b1, b2, _ = st.columns(3)
                if b1.button("Mark Solved", key=f"solve_{row['item_id']}"):
                    conn = sqlite3.connect(DB_NAME)
                    conn.execute(
                        "INSERT INTO interactions (user_id, item_id, status, time_spent, rating) VALUES (?, ?, 'solved', ?, 5)",
                        (user_id, int(row["item_id"]), int(row["est_minutes"]))
                    )
                    conn.commit()
                    conn.close()
                    train_and_save()
                    st.rerun()

                if b2.button("Log Failed Attempt", key=f"fail_{row['item_id']}"):
                    conn = sqlite3.connect(DB_NAME)
                    conn.execute(
                        "INSERT INTO interactions (user_id, item_id, status, time_spent, rating) VALUES (?, ?, 'failed_logic', ?, 2)",
                        (user_id, int(row["item_id"]), int(row["est_minutes"]))
                    )
                    conn.commit()
                    conn.close()
                    train_and_save()
                    st.rerun()

with col2:
    st.subheader("Learner Diagnostics")
    if weak_topics:
        st.error(f"**Detected Weak Topics:** {', '.join(weak_topics)}")
    else:
        st.success("No recent failed attempts detected!")

    conn = sqlite3.connect(DB_NAME)
    history = pd.read_sql(f"""
        SELECT i.title, i.topic, l.status, l.time_spent
        FROM interactions l
        JOIN items i ON l.item_id = i.item_id
        WHERE l.user_id = {user_id}
        ORDER BY l.interaction_id DESC
    """, conn)
    conn.close()
    st.write("**Recent Activity Log:**")
    st.dataframe(history, use_container_width=True, hide_index=True)