import os
import sqlite3
import random
import pandas as pd

DB_NAME = "study_rec.db"
CSV_PATH = "problems_catalog.csv"

def create_default_csv():
    rows = [
        # SDE Blind 75 / DSA Track
        ("Two Sum", "Coding Problem", "Arrays & Hashing", "Easy", 15, "arrays hash-map lookup two-pointers", "SDE Blind 75", "https://leetcode.com/problems/two-sum/"),
        ("Best Time to Buy and Sell Stock", "Coding Problem", "Arrays & Hashing", "Easy", 15, "arrays greedy sliding-window", "SDE Blind 75", "https://leetcode.com/problems/best-time-to-buy-and-sell-stock/"),
        ("Product of Array Except Self", "Coding Problem", "Arrays & Hashing", "Medium", 25, "arrays prefix-sum suffix", "SDE Blind 75", "https://leetcode.com/problems/product-of-array-except-self/"),
        ("Group Anagrams", "Coding Problem", "Arrays & Hashing", "Medium", 25, "strings hash-map sorting", "SDE Blind 75", "https://leetcode.com/problems/group-anagrams/"),
        ("Subarray Sum Equals K", "Coding Problem", "Arrays & Hashing", "Medium", 30, "arrays prefix-sum hash-map", "SDE Blind 75", "https://leetcode.com/problems/subarray-sum-equals-k/"),
        ("3Sum", "Coding Problem", "Two Pointers", "Medium", 30, "arrays two-pointers sorting", "SDE Blind 75", "https://leetcode.com/problems/3sum/"),
        ("Container With Most Water", "Coding Problem", "Two Pointers", "Medium", 25, "arrays two-pointers greedy", "SDE Blind 75", "https://leetcode.com/problems/container-with-most-water/"),
        ("Trapping Rain Water", "Coding Problem", "Two Pointers", "Hard", 45, "two-pointers dynamic-programming stack", "SDE Blind 75", "https://leetcode.com/problems/trapping-rain-water/"),
        ("Longest Substring Without Repeating Characters", "Coding Problem", "Sliding Window", "Medium", 25, "strings sliding-window hash-set", "SDE Blind 75", "https://leetcode.com/problems/longest-substring-without-repeating-characters/"),
        ("Minimum Window Substring", "Coding Problem", "Sliding Window", "Hard", 50, "strings sliding-window hash-map", "SDE Blind 75", "https://leetcode.com/problems/minimum-window-substring/"),
        ("Valid Parentheses", "Coding Problem", "Stacks", "Easy", 15, "stack strings parsing", "SDE Blind 75", "https://leetcode.com/problems/valid-parentheses/"),
        ("Daily Temperatures", "Coding Problem", "Stacks", "Medium", 30, "monotonic-stack arrays", "SDE Blind 75", "https://leetcode.com/problems/daily-temperatures/"),
        ("Climbing Stairs", "Coding Problem", "Dynamic Programming", "Easy", 15, "dp fibonacci memoization", "SDE Blind 75", "https://leetcode.com/problems/climbing-stairs/"),
        ("House Robber", "Coding Problem", "Dynamic Programming", "Medium", 25, "dp 1d-array state-transition", "SDE Blind 75", "https://leetcode.com/problems/house-robber/"),
        ("Coin Change", "Coding Problem", "Dynamic Programming", "Medium", 35, "dp unbounded-knapsack bottom-up", "SDE Blind 75", "https://leetcode.com/problems/coin-change/"),
        ("Longest Increasing Subsequence", "Coding Problem", "Dynamic Programming", "Medium", 35, "dp binary-search subsequence", "SDE Blind 75", "https://leetcode.com/problems/longest-increasing-subsequence/"),
        ("Edit Distance", "Coding Problem", "Dynamic Programming", "Hard", 55, "dp 2d-matrix strings levenshtein", "SDE Blind 75", "https://leetcode.com/problems/edit-distance/"),
        ("Number of Islands", "Coding Problem", "Graphs", "Medium", 30, "bfs dfs grid connected-components", "SDE Blind 75", "https://leetcode.com/problems/number-of-islands/"),
        ("Clone Graph", "Coding Problem", "Graphs", "Medium", 30, "bfs dfs hash-map graph", "SDE Blind 75", "https://leetcode.com/problems/clone-graph/"),
        ("Course Schedule", "Coding Problem", "Graphs", "Medium", 40, "topological-sort dfs cycle-detection", "SDE Blind 75", "https://leetcode.com/problems/course-schedule/"),
        ("Word Ladder", "Coding Problem", "Graphs", "Hard", 55, "bfs shortest-path strings", "SDE Blind 75", "https://leetcode.com/problems/word-ladder/"),
        ("Binary Tree Level Order Traversal", "Coding Problem", "Trees", "Medium", 25, "tree bfs queue", "SDE Blind 75", "https://leetcode.com/problems/binary-tree-level-order-traversal/"),
        ("Lowest Common Ancestor of a BST", "Coding Problem", "Trees", "Medium", 20, "bst recursion tree", "SDE Blind 75", "https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/"),
        # Data Analyst & SQL Track
        ("Second Highest Salary", "Coding Problem", "SQL", "Easy", 15, "sql subquery limit offset null", "Data Analyst Track", "https://leetcode.com/problems/second-highest-salary/"),
        ("Consecutive Numbers", "Coding Problem", "SQL", "Medium", 25, "sql window-functions lead lag", "Data Analyst Track", "https://leetcode.com/problems/consecutive-numbers/"),
        ("Department Top Three Salaries", "Coding Problem", "SQL", "Hard", 40, "sql dense-rank window-functions joins", "Data Analyst Track", "https://leetcode.com/problems/department-top-three-salaries/"),
        ("SQL Window Functions Cheat Sheet & Labs", "Tutorial", "SQL", "Medium", 20, "sql rank dense-rank partition-by", "Data Analyst Track", "https://mode.com/sql-tutorial/sql-window-functions/"),
        ("Telco Customer Churn Prediction (Kaggle)", "Dataset", "Data Analytics", "Medium", 45, "pandas eda classification imbalanced-data", "Data Analyst Track", "https://www.kaggle.com/datasets/blastchar/telco-customer-churn"),
        ("Store Sales Time Series Forecasting (Kaggle)", "Dataset", "Data Analytics", "Medium", 50, "pandas time-series rolling-features", "Data Analyst Track", "https://www.kaggle.com/c/store-sales-time-series-forecasting"),
        # ML Engineering Track & Visual Guides
        ("Credit Card Fraud Detection (Kaggle)", "Dataset", "Machine Learning", "Hard", 60, "anomaly-detection precision-recall smote", "ML Engineering", "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"),
        ("Titanic Survival Feature Engineering Lab", "Dataset", "Machine Learning", "Easy", 30, "pandas seaborn classification baseline", "ML Engineering", "https://www.kaggle.com/c/titanic"),
        ("Dynamic Programming State Transition Visual Guide", "Tutorial", "Dynamic Programming", "Medium", 20, "dp state recurrence base-case", "SDE Blind 75", "https://neetcode.io/"),
        ("Graph Traversal: BFS vs DFS Patterns", "Tutorial", "Graphs", "Medium", 20, "bfs dfs shortest-path topological-sort", "SDE Blind 75", "https://visualgo.net/en/dfsbfs"),
        ("Sliding Window & Two Pointer Masterclass", "Tutorial", "Sliding Window", "Easy", 15, "sliding-window two-pointers strings", "SDE Blind 75", "https://leetcode.com/explore/"),
    ]
    df = pd.DataFrame(rows, columns=["title", "item_type", "topic", "difficulty", "est_minutes", "tags", "roadmap", "url"])
    df.to_csv(CSV_PATH, index=False)

def ingest_csv_to_db(csv_path=CSV_PATH):
    if not os.path.exists(csv_path):
        create_default_csv()

    df = pd.read_csv(csv_path)
    col_map = {c.lower().strip(): c for c in df.columns}
    if "related_topics" in col_map and "topic" not in col_map:
        df["topic"] = df[col_map["related_topics"]].fillna("General").apply(lambda x: str(x).split(",")[0].strip())
    if "item_type" not in df.columns:
        df["item_type"] = "Coding Problem"
    if "est_minutes" not in df.columns:
        diff_time = {"Easy": 15, "Medium": 30, "Hard": 50}
        df["est_minutes"] = df["difficulty"].map(diff_time).fillna(30).astype(int)
    if "tags" not in df.columns:
        df["tags"] = df["topic"].str.lower() + " " + df["difficulty"].str.lower()
    if "roadmap" not in df.columns:
        df["roadmap"] = "SDE Blind 75"
    if "url" not in df.columns:
        df["url"] = "https://leetcode.com/problemset/"

    df = df[["title", "item_type", "topic", "difficulty", "est_minutes", "tags", "roadmap", "url"]].dropna(subset=["title"]).head(250)
    df.insert(0, "item_id", range(1, len(df) + 1))

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.executescript("""
    DROP TABLE IF EXISTS items;
    DROP TABLE IF EXISTS users;
    DROP TABLE IF EXISTS interactions;

    CREATE TABLE items (
        item_id INTEGER PRIMARY KEY,
        title TEXT,
        item_type TEXT,
        topic TEXT,
        difficulty TEXT,
        est_minutes INTEGER,
        tags TEXT,
        roadmap TEXT,
        url TEXT
    );

    CREATE TABLE users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        skill_level TEXT,
        target_topics TEXT
    );

    CREATE TABLE interactions (
        interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        item_id INTEGER,
        status TEXT,
        time_spent INTEGER,
        rating INTEGER
    );
    """)

    df.to_sql("items", conn, if_exists="append", index=False)

    # Seed initial learner profiles
    users_seed = [
        ("Ajinkya (Demo Profile)", "Intermediate", "Dynamic Programming,SQL,Machine Learning"),
        ("Peer Learner #2", "Beginner", "Arrays & Hashing,Two Pointers"),
        ("Peer Learner #3", "Advanced", "Graphs,Dynamic Programming"),
        ("Peer Learner #4", "Intermediate", "SQL,Data Analytics"),
        ("Peer Learner #5", "Intermediate", "Machine Learning,Data Analytics"),
    ]
    cursor.executemany("INSERT INTO users (name, skill_level, target_topics) VALUES (?, ?, ?)", users_seed)

    # Seed multi-user interactions
    random.seed(42)
    max_id = len(df)
    interactions_data = []
    for uid in range(1, 6):
        sampled = random.sample(range(1, max_id + 1), min(10, max_id))
        for iid in sampled:
            status = random.choices(["solved", "failed_logic", "failed_tle"], weights=[0.6, 0.25, 0.15])[0]
            rating = 5 if status == "solved" else 2
            interactions_data.append((uid, iid, status, random.randint(10, 50), rating))

    # Give User 1 specific strengths & weak areas
    interactions_data.extend([
        (1, 1, "solved", 12, 5),
        (1, 2, "solved", 14, 5),
        (1, 15, "failed_logic", 35, 2),  # Coin Change (DP)
        (1, 24, "failed_tle", 18, 2),    # Second Highest Salary (SQL)
    ])

    cursor.executemany(
        "INSERT INTO interactions (user_id, item_id, status, time_spent, rating) VALUES (?, ?, ?, ?, ?)",
        interactions_data
    )
    conn.commit()
    conn.close()
    print(f"Database initialized with {len(df)} items and user profiles.")

if __name__ == "__main__":
    ingest_csv_to_db()