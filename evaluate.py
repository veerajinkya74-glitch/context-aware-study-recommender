import sqlite3
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

DB_NAME = "study_rec.db"

def precision_at_k(recommended, relevant, k):
    if k == 0:
        return 0.0
    rec_k = recommended[:k]
    hits = len(set(rec_k) & set(relevant))
    return hits / k

def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0
    rec_k = recommended[:k]
    hits = len(set(rec_k) & set(relevant))
    return hits / len(relevant)

def ndcg_at_k(recommended, relevant, k):
    rec_k = recommended[:k]
    dcg = sum(1.0 / np.log2(idx + 2) for idx, item in enumerate(rec_k) if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(idx + 2) for idx in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0

def run_evaluation(k=5):
    conn = sqlite3.connect(DB_NAME)
    items_df = pd.read_sql("SELECT * FROM items", conn)
    inter_df = pd.read_sql("SELECT * FROM interactions", conn)
    conn.close()

    # Hold out 20% of positive ('solved') interactions per user as ground-truth test items
    train_rows = []
    test_ground_truth = {}

    for uid, group in inter_df.groupby("user_id"):
        solved = group[group["status"] == "solved"]
        if len(solved) >= 2:
            test_sample = solved.sample(frac=0.3, random_state=42)
            test_ground_truth[uid] = set(test_sample["item_id"])
            train_rows.append(group.drop(test_sample.index))
        else:
            train_rows.append(group)

    train_df = pd.concat(train_rows)

    # Train Content TF-IDF
    items_df["combined"] = (
        items_df["topic"].fillna("") + " " +
        items_df["difficulty"].fillna("") + " " +
        items_df["tags"].fillna("")
    )
    tfidf = TfidfVectorizer(stop_words="english")
    content_sim = cosine_similarity(tfidf.fit_transform(items_df["combined"]))

    # Train Collaborative SVD on train split only
    user_item = train_df.pivot_table(index="user_id", columns="item_id", values="rating", aggfunc="mean").fillna(0)
    n_comp = max(1, min(4, user_item.shape[1] - 1, user_item.shape[0] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    reconstructed = np.dot(svd.fit_transform(user_item), svd.components_)
    collab_preds = pd.DataFrame(reconstructed, index=user_item.index, columns=user_item.columns)

    precisions, recalls, ndcgs = [], [], []
    all_recommended_items = set()

    for uid, relevant_items in test_ground_truth.items():
        user_train = train_df[train_df["user_id"] == uid]
        seen_train = set(user_train[user_train["status"] == "solved"]["item_id"])
        failed_ids = list(user_train[user_train["status"].str.contains("failed", na=False)]["item_id"])
        weak_topics = set(items_df[items_df["item_id"].isin(failed_ids)]["topic"])

        candidates = items_df[~items_df["item_id"].isin(seen_train)].copy()
        candidates["collab"] = candidates["item_id"].map(collab_preds.loc[uid]).fillna(2.5)
        max_c = float(candidates["collab"].max()) or 1.0
        candidates["collab_norm"] = candidates["collab"] / max_c

        if failed_ids:
            f_idx = items_df[items_df["item_id"].isin(failed_ids)].index.tolist()
            candidates["recovery"] = content_sim[f_idx].mean(axis=0)[candidates.index]
        else:
            candidates["recovery"] = 0.5

        candidates["weak_boost"] = np.where(candidates["topic"].isin(weak_topics), 0.35, 0.0)
        candidates["score"] = 0.40 * candidates["collab_norm"] + 0.35 * candidates["recovery"] + candidates["weak_boost"]

        ranked_ids = candidates.sort_values("score", ascending=False)["item_id"].tolist()
        all_recommended_items.update(ranked_ids[:k])

        precisions.append(precision_at_k(ranked_ids, relevant_items, k))
        recalls.append(recall_at_k(ranked_ids, relevant_items, k))
        ndcgs.append(ndcg_at_k(ranked_ids, relevant_items, k))

    coverage = len(all_recommended_items) / len(items_df)

    print(f"\n=== Offline Evaluation Results (Top-{k}) ===")
    print(f"Evaluated Users  : {len(test_ground_truth)}")
    print(f"Precision@{k}      : {np.mean(precisions):.4f}")
    print(f"Recall@{k}         : {np.mean(recalls):.4f}")
    print(f"NDCG@{k}           : {np.mean(ndcgs):.4f}")
    print(f"Catalog Coverage : {coverage * 100:.1f}%\n")

if __name__ == "__main__":
    run_evaluation(k=5)