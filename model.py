import sqlite3
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

DB_NAME = "study_rec.db"
MODEL_PATH = "recommender_artifacts.joblib"

SKILL_DIFF_MAP = {
    "Beginner": "Easy",
    "Intermediate": "Medium",
    "Advanced": "Hard"
}

def train_and_save():
    conn = sqlite3.connect(DB_NAME)
    items_df = pd.read_sql("SELECT * FROM items", conn)
    inter_df = pd.read_sql("SELECT * FROM interactions", conn)
    conn.close()

    items_df["combined_features"] = (
        items_df["topic"].fillna("") + " " +
        items_df["difficulty"].fillna("") + " " +
        items_df["item_type"].fillna("") + " " +
        items_df["roadmap"].fillna("") + " " +
        items_df["tags"].fillna("")
    )
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(items_df["combined_features"])
    content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    user_item = inter_df.pivot_table(
        index="user_id", columns="item_id", values="rating", aggfunc="mean"
    ).fillna(0)

    n_components = max(1, min(4, user_item.shape[1] - 1, user_item.shape[0] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    latent_matrix = svd.fit_transform(user_item)
    reconstructed = np.dot(latent_matrix, svd.components_)
    collab_preds = pd.DataFrame(
        reconstructed, index=user_item.index, columns=user_item.columns
    )

    artifacts = {
        "items_df": items_df,
        "content_sim": content_sim,
        "collab_preds": collab_preds
    }
    joblib.dump(artifacts, MODEL_PATH)
    return artifacts


def compute_topic_mastery(user_id):
    conn = sqlite3.connect(DB_NAME)
    items_df = pd.read_sql("SELECT item_id, topic FROM items", conn)
    logs_df = pd.read_sql(f"SELECT item_id, status FROM interactions WHERE user_id = {int(user_id)}", conn)
    conn.close()

    core_topics = ["Arrays & Hashing", "Dynamic Programming", "Graphs", "SQL", "Sliding Window", "Data Analytics", "Machine Learning"]
    if logs_df.empty:
        return {t: 25 for t in core_topics}

    merged = logs_df.merge(items_df, on="item_id", how="inner")
    mastery = {}
    for topic in core_topics:
        t_logs = merged[merged["topic"] == topic]
        if t_logs.empty:
            mastery[topic] = 25
        else:
            solved = (t_logs["status"] == "solved").sum()
            total = len(t_logs)
            # Smoothed mastery score (20% baseline + up to 80% from solve ratio)
            score = int(round(20 + 80 * (solved / total)))
            mastery[topic] = min(100, max(10, score))
    return mastery


def recommend_next(user_id, available_minutes, preferred_type="All", roadmap="All", focus_weak_topics=True, top_n=6):
    try:
        artifacts = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        artifacts = train_and_save()

    items_df = artifacts["items_df"].copy()
    content_sim = artifacts["content_sim"]
    collab_preds = artifacts["collab_preds"]

    conn = sqlite3.connect(DB_NAME)
    user_logs = pd.read_sql(f"SELECT * FROM interactions WHERE user_id = {int(user_id)}", conn)
    user_info = pd.read_sql(f"SELECT * FROM users WHERE user_id = {int(user_id)}", conn)
    conn.close()

    # Extract cold-start onboarding preferences
    target_topics = set()
    preferred_diff = "Medium"
    if not user_info.empty:
        raw_targets = str(user_info.iloc[0]["target_topics"] or "")
        target_topics = {t.strip() for t in raw_targets.split(",") if t.strip()}
        preferred_diff = SKILL_DIFF_MAP.get(user_info.iloc[0]["skill_level"], "Medium")

    is_cold_start = user_logs.empty
    solved_ids = set(user_logs[user_logs["status"] == "solved"]["item_id"])
    failed_ids = list(user_logs[user_logs["status"].str.contains("failed", na=False)]["item_id"])
    weak_topics = set(items_df[items_df["item_id"].isin(failed_ids)]["topic"])

    # Collaborative Score
    if user_id in collab_preds.index:
        items_df["collab_score"] = items_df["item_id"].map(collab_preds.loc[user_id]).fillna(2.5)
    else:
        items_df["collab_score"] = 2.5

    max_c = float(items_df["collab_score"].max())
    items_df["collab_norm"] = items_df["collab_score"] / (max_c if max_c > 0 else 1.0)

    # Content Recovery Score
    if failed_ids:
        failed_indices = items_df[items_df["item_id"].isin(failed_ids)].index.tolist()
        items_df["recovery_score"] = content_sim[failed_indices].mean(axis=0) if failed_indices else 0.5
    else:
        items_df["recovery_score"] = 0.5

    # Context Filtering: Unsolved + Time Window + Format + Roadmap
    candidates = items_df[
        (~items_df["item_id"].isin(solved_ids)) &
        (items_df["est_minutes"] <= available_minutes)
    ].copy()

    if preferred_type != "All":
        candidates = candidates[candidates["item_type"] == preferred_type]
    if roadmap != "All":
        candidates = candidates[candidates["roadmap"] == roadmap]

    if candidates.empty:
        return candidates, weak_topics

    # Boosts (Explicit float dtype for pandas 3.0)
    candidates["weak_topic_boost"] = np.where(
        focus_weak_topics & candidates["topic"].isin(weak_topics), 0.35, 0.0
    ).astype(float)

    candidates["tutorial_remediation_boost"] = np.where(
        (candidates["item_type"] == "Tutorial") & candidates["topic"].isin(weak_topics), 0.25, 0.0
    ).astype(float)

    # Cold-start onboarding boost for newly registered users
    candidates["onboarding_boost"] = np.where(
        candidates["topic"].isin(target_topics), 0.30 if is_cold_start else 0.12, 0.0
    ).astype(float) + np.where(
        candidates["difficulty"] == preferred_diff, 0.15 if is_cold_start else 0.05, 0.0
    ).astype(float)

    candidates["final_score"] = (
        0.35 * candidates["collab_norm"].astype(float) +
        0.30 * candidates["recovery_score"].astype(float) +
        candidates["weak_topic_boost"] +
        candidates["tutorial_remediation_boost"] +
        candidates["onboarding_boost"]
    )

    def explain(row):
        reasons = []
        if is_cold_start and row["topic"] in target_topics:
            reasons.append(f"Matches onboarding goal ({row['topic']} • {row['difficulty']})")
        if row["topic"] in weak_topics:
            reasons.append(f"Targets weak area ({row['topic']})")
        if row["item_type"] == "Tutorial" and row["topic"] in weak_topics:
            reasons.append("Visual guide recommended after recent error")
        reasons.append(f"Fits {available_minutes}-min window")
        return " • ".join(reasons)

    candidates["why_recommended"] = candidates.apply(explain, axis=1)

    if "combined_features" in candidates.columns:
        candidates = candidates.drop(columns=["combined_features"])

    return candidates.sort_values("final_score", ascending=False).head(top_n), weak_topics