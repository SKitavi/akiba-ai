# AkibaAI

AkibaAI is an **offline-first credit risk scoring simulation** for African SACCOs, built as a 7-day MVP sprint with a target demo date of **Aug 17, 2026**.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/ui/app.py
```

## Synthetic Data Disclaimer

> This project uses **synthetic data only**, not real customer data.

## Scope

### In Scope (MVP)
- Synthetic transaction and repayment behavior generation
- Offline SMS/receipt ingestion stubs
- Feature engineering stubs for risk scoring
- Local SQLite storage and model pipeline scaffolding
- Explainability narrative scaffolding and Streamlit dashboard skeleton

### Out of Scope (Stretch Goals)
- SQLCipher encryption
- ONNX quantization
- Native Android app
- Third language support beyond English/Kiswahili
- Real consented data pilot

## Team & Roles

| Role | Owner |
|---|---|
| PM / Architecture | Team Lead |
| Data Engineer | Swafiyah |
| ML Engineer | Sharon |
| Explainability Engineer | Joshua |
| Frontend / Dashboard Engineer | Umutoni |
