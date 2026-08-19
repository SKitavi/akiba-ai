# AkibaAI

AkibaAI is an offline-first credit-risk assessment demonstration for African
SACCOs. It turns synthetic mobile-money transaction evidence into behavioral
features, produces an XGBoost risk score, explains the score with SHAP, and lets
a human loan officer record a separate decision.

> AkibaAI is an educational MVP that uses synthetic data. It is not a production
> lending system and must not be used to make real credit decisions.

## Start here

Read the [complete system guide](docs/SYSTEM_OVERVIEW.md) for a plain-language
tour of the problem, architecture, assessment workflow, model, explanations,
database, analytics, security boundaries, deployment, testing, and limitations.

Additional reference documents:

- [Synthetic data schema](docs/data_schema.md)
- [Contabo VPS deployment and operations](docs/VPS_DEPLOYMENT.md)

## What the application demonstrates

1. Select a synthetic member or enter a member ID.
2. Provide transaction evidence from demo data, CSV, supported SMS text, or a
   receipt image processed with local OCR.
3. Validate and normalize the evidence into one canonical transaction format.
4. Aggregate the transactions into 32 behavioral features.
5. Run the bundled XGBoost model locally.
6. Explain the score with traceable SHAP contributions in English or Kiswahili.
7. Save the assessment to SQLite.
8. Record a separate human decision: approve, review, or decline.
9. Review persistent history and operational analytics.

No cloud model, generative-AI API, or internet connection is required for the
assessment pipeline.

## Run locally

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
streamlit run src/ui/app.py
```

Open `http://localhost:8501`.

To load 12 idempotent synthetic dashboard records:

```bash
python -m src.storage.seed_dashboard_demo
```

You can also load or reset demo assessments from **Settings**. The default demo
Settings key is `CMU#AB39`; set `SETTINGS_ACCESS_KEY` to override it.

## Run the tests

```bash
pytest -q
```

The repository currently has 157 automated tests across ingestion, feature
engineering, modeling, explanations, persistence, analytics, and the Streamlit
workflow.

## Deploy the VPS demo

The included Docker Compose setup stores SQLite in a persistent named volume:

```bash
cp .env.example .env
docker compose up -d --build
```

Open `http://YOUR_VPS_IP:8501`. Follow the complete
[VPS deployment guide](docs/VPS_DEPLOYMENT.md) for updates, firewall setup,
backups, and troubleshooting.

## Project boundaries

Included in the MVP:

- Synthetic M-Pesa and MTN MoMo-style data
- CSV, SMS, receipt OCR, and demonstration input paths
- Canonical transaction validation
- 32 behavioral features and local XGBoost scoring
- Local SHAP explanations and deterministic English/Kiswahili narratives
- Atomic SQLite persistence, history, and analytics
- A separate human decision workflow
- Streamlit UI and Docker-based VPS demo deployment

Not included:

- Real customer data or a consented pilot
- Validated lending policy or automatic eligibility bands
- Production authentication, authorization, encryption, or audit governance
- Fairness validation, regulatory approval, or real-world model monitoring
- A production multi-user database

## Team roles

| Role | Owner |
|---|---|
| PM / Architecture | Team Lead |
| Data Engineer | Swafiyah |
| ML Engineer | Sharon |
| Explainability Engineer | Joshua |
| Frontend / Dashboard Engineer | Umutoni |
