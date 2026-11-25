# NextStep AI — Context-Aware Study & Coding Problem Recommender

An end-to-end hybrid recommendation system that suggests the next coding problem, visual tutorial, or Kaggle dataset based on a learner's **weak topics**, **recent error patterns**, **available time window**, and **curated interview roadmap**.

---

## Key Features
* **Hybrid ML Engine (`TruncatedSVD` + `TF-IDF`):** Blends collaborative filtering across peer learners with content-based cosine similarity to recover from failed attempts.
* **Context-Aware Ranking:** Filters candidates strictly by available session duration (`15m` quick drills to `90m` deep dives) and boosts weak topics alongside targeted remediation guides.
* **Cold-Start Onboarding:** Registers new learners with target topics and skill level (`Beginner`, `Intermediate`, `Advanced`) so recommendations work immediately with zero prior history.
* **Curated Study Roadmaps:** Supports filtering across **SDE Blind 75 / DSA**, **Data Analyst (SQL & Pandas)**, and **ML Engineering** tracks.
* **Interactive Analytics & Pomodoro Tracking:** Live **Chart.js** Skill Mastery Radar, per-card countdown timers that record actual solving duration, and CSV ingestion for external LeetCode/Kaggle datasets.

---

## System Architecture

| Layer | Technology | Role |
| :--- | :--- | :--- |
| **Data & Ingestion** | Python, Pandas, SQLite | Ingests `problems_catalog.csv` and logs user interactions (`solved`, `failed_logic`, `failed_tle`). |
| **Recommendation Engine** | Scikit-learn (`TfidfVectorizer`, `TruncatedSVD`), NumPy, `joblib` | Computes hybrid scores, weak-topic boosts, cold-start matching, and topic mastery percentages. |
| **Backend API** | Flask | Exposes REST endpoints (`/api/recommend`, `/api/feedback`, `/api/users`, `/api/upload-csv`). |
| **Frontend UI** | HTML5, Tailwind CSS, JavaScript, Chart.js | Glassmorphic dashboard with live Pomodoro timers, radar chart, and onboarding modal. |

---

## Offline Evaluation Metrics (Top-5)
Evaluated on a held-out test split of positive interactions (`evaluate.py`):

| Metric | Score |
| :--- | :--- |
| **Precision@5** | `0.1200` |
| **Recall@5** | `0.4000` |
| **NDCG@5** | `0.2649` |
| **Catalog Coverage** | `38.2%` |

---

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt