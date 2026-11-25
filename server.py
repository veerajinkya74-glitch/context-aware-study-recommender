import os
import sqlite3
import pandas as pd
from flask import Flask, render_template, request, jsonify
from db_setup import ingest_csv_to_db, DB_NAME, CSV_PATH
from model import recommend_next, train_and_save, compute_topic_mastery

app = Flask(__name__)

if not os.path.exists(DB_NAME):
    ingest_csv_to_db()
    train_and_save()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/users", methods=["GET", "POST"])
def api_users():
    conn = sqlite3.connect(DB_NAME)
    if request.method == "POST":
        data = request.json
        name = data.get("name", "New Learner").strip() or "New Learner"
        skill_level = data.get("skill_level", "Intermediate")
        topics = ",".join(data.get("target_topics", ["Arrays & Hashing"]))
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (name, skill_level, target_topics) VALUES (?, ?, ?)",
            (name, skill_level, topics)
        )
        new_id = cur.lastrowid
        conn.commit()
        conn.close()
        return jsonify({"user_id": new_id, "name": name})

    users_df = pd.read_sql("SELECT user_id, name, skill_level, target_topics FROM users ORDER BY user_id ASC", conn)
    conn.close()
    return jsonify(users_df.to_dict(orient="records"))

@app.route("/api/recommend", methods=["GET"])
def api_recommend():
    user_id = int(request.args.get("user_id", 1))
    minutes = int(request.args.get("minutes", 30))
    item_type = request.args.get("type", "All")
    roadmap = request.args.get("roadmap", "All")
    focus_weak = request.args.get("focus_weak", "true").lower() == "true"

    recs_df, weak_topics = recommend_next(
        user_id=user_id,
        available_minutes=minutes,
        preferred_type=item_type,
        roadmap=roadmap,
        focus_weak_topics=focus_weak,
        top_n=6
    )

    conn = sqlite3.connect(DB_NAME)
    history_df = pd.read_sql(f"""
        SELECT i.title, i.topic, i.difficulty, l.status, l.time_spent
        FROM interactions l
        JOIN items i ON l.item_id = i.item_id
        WHERE l.user_id = {user_id}
        ORDER BY l.interaction_id DESC
        LIMIT 12
    """, conn)
    stats_df = pd.read_sql(f"""
        SELECT 
            COUNT(*) as total_attempts,
            SUM(CASE WHEN status = 'solved' THEN 1 ELSE 0 END) as solved_count,
            COALESCE(SUM(time_spent), 0) as total_minutes
        FROM interactions WHERE user_id = {user_id}
    """, conn)
    conn.close()

    return jsonify({
        "recommendations": recs_df.to_dict(orient="records"),
        "weak_topics": list(weak_topics),
        "history": history_df.to_dict(orient="records"),
        "stats": stats_df.to_dict(orient="records")[0],
        "mastery": compute_topic_mastery(user_id)
    })

@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    data = request.json
    conn = sqlite3.connect(DB_NAME)
    conn.execute(
        "INSERT INTO interactions (user_id, item_id, status, time_spent, rating) VALUES (?, ?, ?, ?, ?)",
        (
            int(data["user_id"]),
            int(data["item_id"]),
            data["status"],
            max(1, int(data["time_spent"])),
            5 if data["status"] == "solved" else 2
        )
    )
    conn.commit()
    conn.close()
    train_and_save()
    return jsonify({"status": "ok"})

@app.route("/api/upload-csv", methods=["POST"])
def api_upload_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Please upload a .csv file"}), 400
    file.save(CSV_PATH)
    ingest_csv_to_db(CSV_PATH)
    train_and_save()
    return jsonify({"status": "ingested"})

@app.route("/api/reset", methods=["POST"])
def api_reset():
    if os.path.exists(CSV_PATH):
        os.remove(CSV_PATH)
    ingest_csv_to_db()
    train_and_save()
    return jsonify({"status": "reset"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)