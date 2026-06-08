# 🌫️ Tharparkar AQI Prediction Pipeline

> **72-hour Air Quality Index forecasting for District Tharparkar, Pakistan — through cloud-native ML pipeline.**

[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=github-actions&logoColor=white)](https://github.com/ShankerLal25150/aqi_prediction_Thar/actions)
[![Model Registry](https://img.shields.io/badge/Model%20Registry-AWS%20S3-FF9900?logo=amazon-s3&logoColor=white)](#)
[![Feature Store](https://img.shields.io/badge/Feature%20Store-MongoDB%20Atlas-47A248?logo=mongodb&logoColor=white)](#)
[![Dashboard](https://img.shields.io/badge/Live%20Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://tharparkar-aqi.streamlit.app/)

---

## What This Project Does

Tharparkar is home to one of Pakistan's largest active coal mining operations. Coal extraction and combustion generate a continuous stream of airborne pollutants — particulate matter (PM2.5, PM10), sulfur dioxide, and nitrogen oxides — that interact with the desert region's wind patterns and temperature inversions in ways that make air quality difficult to predict without a data-driven approach.

This pipeline ingests **live meteorological and pollutant data** every hour via Open-Meteo and OpenAQ APIs, engineers lag-based and rolling-window features, and serves predictions through two trained models — **XGBoost** and **Random Forest** — selected for their ability to capture the non-linear relationships between weather variables and pollutant dispersion. The result is a dashboard that displays AQI forecasts for the next 72 hours, refreshed automatically without any manual intervention.

### End-to-End Pipeline Flow

```
Hourly Cron (GitHub Actions)
        │
        ▼
 API Ingestion Layer ──► Raw data stored as Parquet (PyArrow)
        │
        ▼
 Feature Engineering ──► Lag features, rolling means, wind vectors
        │
        ▼
 MongoDB Atlas (Feature Store) ──► Live features for inference
        │
        ▼
 Daily Retraining (GitHub Actions) ──► XGBoost + Random Forest
        │
        ▼
 AWS S3 (Model Registry) ──► Versioned .joblib artifacts
        │
        ▼
 Streamlit Dashboard ──► Fetches models from S3 on boot → serves 72hr forecast
```

---

## Live Links

| Service | URL |
|---|---|
| 📊 **Live Dashboard** | [tharparkar-aqi.streamlit.app](https://tharparkar-aqi.streamlit.app/) |
| ⚙️ **CI/CD Pipeline** | [github.com/ShankerLal25150/aqi_prediction_Thar/actions](https://github.com/ShankerLal25150/aqi_prediction_Thar/actions) |


---

## Technical Stack

| Layer | Technology |
|---|---|
| **Machine Learning** | XGBoost, Random Forest, Scikit-Learn |
| **Data Processing** | Pandas, PyArrow (Parquet) |
| **Cloud Storage** | AWS S3 (Model Registry) |
| **Feature Store** | MongoDB Atlas |
| **Frontend** | Streamlit Community Cloud |
| **Automation** | GitHub Actions (hourly + daily workflows) |

---

## Challenges & Solutions

The core requirement of this project was zero local dependency — the entire pipeline had to ingest data, retrain models, and serve predictions without any manual steps. Achieving that required solving a set of interconnected architectural problems across cloud storage, CI/CD synchronization, and secrets management. Below is a detailed account of each challenge and how it was resolved.

---

### 1. Migrating from Version Control to a True Model Registry

**The Challenge:**

Initially, the pipeline saved serialized ML models (`.joblib` files) and preprocessing artifacts (`scaler.joblib`) directly into the root of the GitHub repository. While this made it trivially easy for the Streamlit frontend to access them via relative paths, it was a clear **MLOps anti-pattern**. Git is a version control system designed for text-based source code — not binary blobs. Storing trained model artifacts in the repo had three concrete problems:

- **Repository bloat:** Each daily retraining cycle committed new binary files, inflating the repository size over time with no practical benefit.
- **Tight coupling:** The inference layer was coupled to the repository's state, meaning a bad training run or accidental deletion could break the live dashboard.
- **No true versioning:** Git doesn't diff binaries meaningfully. Storing models in Git gave the illusion of versioning without the ability to compare, roll back, or audit model artifacts properly.

**The Solution:**

I refactored the storage architecture to decouple model artifacts from source code entirely. I provisioned a dedicated **AWS S3 bucket** to serve as a cloud model registry. After each training cycle, `model_training.py` uses `boto3` to push the freshly trained `.joblib` files and the fitted scaler to S3 with consistent, predictable object keys. The Streamlit frontend was rewritten to act as a lightweight cloud client — on startup, it fetches the latest model artifacts directly from S3 into temporary in-memory buffers, so no file ever touches the deployment machine's disk. All binary files were removed from Git tracking via `.gitignore`.

---

### 2. CI/CD Asynchrony and Git Synchronization Conflicts

**The Challenge:**

Because two GitHub Actions workflows run autonomously — one hourly for data ingestion, one daily for retraining — the cloud runner was frequently committing new data and execution logs directly to `main`. This created a divergence problem: my local branch was perpetually behind the remote. Attempting to push local changes (frontend updates, notebook fixes) was consistently rejected with `fetch first` errors.

The naive fix — `git pull` followed by a merge — would have generated a noisy stream of merge commits (e.g., *"Merge branch 'main' of..."*) interleaved with the automated CI commits, making the history unreadable and auditing nearly impossible. In a pipeline where every commit to `main` can trigger a workflow, a polluted history is a real operational hazard: it obscures whether a given repository state was produced by automated ingestion, a model retrain, or a deliberate manual code change — which matters enormously when you're trying to trace the root cause of a bad prediction or a failed pipeline run.

**The Solution:**

I adopted `git pull --rebase` as the standard synchronization strategy for this repo. Rather than creating a new merge commit that sits on top of both histories, rebasing re-applies my local commits *on top of* whatever the remote has accumulated — preserving a linear, chronological commit history. The full workflow became:

```bash
# Protect unstaged local changes before syncing
git stash

# Replay cloud-automated commits first, then re-apply local work on top
git pull origin main --rebase

# Restore local changes cleanly
git stash pop

# Push with a now-linear history
git push origin main
```

This approach is specifically well-suited to automated CI/CD environments because it keeps the commit graph linear and deterministic. Each entry in `git log` maps to exactly one discrete action — either an automated pipeline run or a deliberate manual change — which makes debugging pipeline failures and auditing model versions significantly more reliable.

---

### 3. Secure Credential Injection Across Environments

**The Challenge:**

Transitioning to AWS S3 and MongoDB Atlas meant that both the training pipeline (running in a GitHub Actions cloud runner) and the inference pipeline (running in Streamlit Community Cloud) required programmatic credentials: AWS IAM access keys and a MongoDB Atlas connection URI. Hardcoding these values directly into Python scripts was not an option in a public repository — exposed credentials in a public repo are indexed by automated scrapers within minutes, routinely leading to unauthorized cloud resource usage and data exposure.

The non-trivial part wasn't knowing that hardcoding was wrong — it was ensuring that *two completely separate execution environments* could both authenticate to the same cloud resources without sharing credentials through the codebase.

**The Solution:**

I implemented environment variable injection natively within each platform's secrets management system, keeping the credential stores entirely separate from the code:

- **GitHub Actions:** `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `MONGODB_URI` were stored as **Repository Secrets** (Settings → Secrets and Variables → Actions). The workflows reference them as `${{ secrets.AWS_ACCESS_KEY_ID }}` — injected into the runner's environment at execution time, never written to the filesystem or exposed in logs.

- **Streamlit Community Cloud:** The same credentials were stored in the app's **Secrets** configuration, accessed via `st.secrets["AWS_ACCESS_KEY_ID"]` at runtime.

The `boto3` and `pymongo` clients in both environments pick up credentials through the standard environment variable lookup chain, meaning the application code itself contains zero hardcoded values. The same scripts run identically in local development (reading from a `.env` file excluded from Git), the CI/CD runner, and any future environment — with no code changes required.

---

## Project Structure

```
tharparkar-aqi/
│
├── .github/
│   └── workflows/
│       ├── hourly_ingestion.yml     # Fetches live met + pollutant data
│       └── daily_retrain.yml        # Retrains models, pushes to S3
│
├── data/
│   └── raw/                         # Parquet-formatted ingested data
│
├── src/
│   ├── ingest.py                    # API clients (Open-Meteo, OpenAQ)
│   ├── features.py                  # Lag/rolling feature engineering
│   ├── model_training.py            # XGBoost + RF training, S3 upload
│   └── inference.py                 # MongoDB Atlas feature retrieval
│
├── app.py                           # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Setup & Deployment

This pipeline is designed for fully cloud-hosted execution. To deploy your own instance:

1. **Fork the repository** and connect it to your GitHub Actions.
2. **Provision an AWS S3 bucket** and an IAM user with `s3:PutObject` and `s3:GetObject` permissions.
3. **Create a MongoDB Atlas cluster** and obtain the connection URI.
4. **Inject secrets** into GitHub Repository Secrets and Streamlit Secrets as described in Challenge 3 above.
5. **Enable the two workflows** — ingestion will begin on the next hourly tick; the first model will be trained within 24 hours.
6. **Deploy `app.py`** to Streamlit Community Cloud pointed at your fork.

*No local environment is required for the pipeline to operate after initial deployment.*
