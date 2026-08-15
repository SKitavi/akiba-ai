# AkibaAI Project Guide

This document is the practical handbook for understanding, running, extending, and evaluating the AkibaAI repository.

## 1. Executive summary

AkibaAI is an **offline-first credit-risk scoring simulation for African SACCOs**. It uses synthetic M-Pesa and MTN MoMo transaction histories to demonstrate how alternative financial behavior could be converted into a probability-of-default score.

The project currently provides a working backend pipeline for:

- Generating synthetic applicants and mobile-money transactions
- Producing provider-style transaction SMS messages
- Parsing SMS messages and OCR receipt text
- Aggregating transactions into applicant-level behavioral features
- Training and evaluating an XGBoost classifier
- Scoring new applicant feature rows
- Persisting features, scores, and decisions in SQLite

The SHAP explainability and English/Kiswahili narrative layers are implemented as reusable backend modules. The Streamlit application now provides the complete synthetic officer workflow from evidence validation through a separately recorded human decision. This is an MVP simulation, not a production lending platform, and it uses no real customer data.

## 2. The problem the project is exploring

Many SACCO applicants and informal workers have limited conventional credit histories. AkibaAI explores whether mobile-money activity can provide alternative signals such as:

- Frequency and regularity of transactions
- Stability of inflows and outflows
- Net cash flow relative to estimated income
- Frequency of low-wallet-balance events
- Balance direction over time
- Mix of productive payments, cash withdrawals, airtime, and peer transfers

The intended output is a model risk score between `0` and `1`, where a larger value means the model estimates greater default risk. It is not independently probability-calibrated.

This repository only demonstrates the engineering and modeling concept. It does not establish that these signals are valid, fair, lawful, or sufficiently predictive for real lending.

## 3. System architecture

```text
Synthetic applicant profiles
            |
            v
Synthetic M-Pesa / MTN MoMo transactions and SMS messages
            |
            +----------------------+
            |                      |
            v                      v
    Structured CSV logs     SMS / receipt OCR parser
            |                      |
            +----------+-----------+
                       |
                       v
          Applicant-level feature engineering
                       |
                       v
              XGBoost model training
                       |
             +---------+---------+
             |                   |
             v                   v
       Evaluation report    Applicant risk score
                                 |
                          +------+------+
                          |             |
                          v             v
                    SQLite storage   SHAP/narrative
                                      (implemented)
                                            |
                                            v
                                    Streamlit dashboard
                                      (implemented)
```

Structured synthetic records and supported M-Pesa/MTN SMS formats now pass through a typed normalization boundary before feature engineering. Records missing required financial fields are rejected with structured reasons rather than receiving fabricated values. Generic OCR receipts may still require explicit caller-supplied transaction context.

## 4. Repository map

```text
akiba-ai/
|-- README.md                         Short project introduction
|-- PROJECT_GUIDE.md                  This handbook
|-- requirements.txt                  Pinned Python dependencies
|-- Akiba_AI_credit_scoring_model.ipynb
|                                      Companion Colab/notebook walkthrough
|-- data/
|   |-- raw/                          Generated CSV files; ignored by Git
|   `-- samples/                      Sample receipt PNG files
|-- docs/
|   |-- data_schema.md                Synthetic dataset specification
|   `-- BACKEND_INTEGRATION.md        Complete backend architecture and APIs
|-- models/                           Generated model/evaluation artifacts
|-- reports/                          Generated reports
|-- src/
|   |-- application/                  Applicant assessment orchestration
|   |-- data_gen/                     Synthetic data and labels
|   |-- domain/                       Canonical transaction domain models
|   |-- ingestion/                    Parsing, OCR, normalization, validation
|   |-- features/                     Applicant-level feature engineering
|   |-- model/                        Training, orchestration, prediction
|   |-- eval/                         Classification metrics
|   |-- storage/                      SQLite schema and persistence helpers
|   |-- xai/                          SHAP and localized narratives
|   `-- ui/                           Modular Streamlit officer workstation
`-- tests/                            Automated unit tests
```

## 5. Environment setup

The repository is Python-based. The dependency file pins Faker, NumPy, pandas, pytesseract, Pillow, scikit-learn, XGBoost, SHAP, Streamlit, matplotlib, pytest, Black, and Flake8.

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The activation command shown in the original README, `source .venv/bin/activate`, is for Linux/macOS. Use the PowerShell command above on Windows.

### Linux or macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Optional OCR system dependency

`pytesseract` is a Python wrapper. Real OCR also requires the Tesseract executable to be installed and available on the system path. If it is unavailable or extraction fails, normal application ingestion raises `OCRExtractionError`; it never substitutes believable financial data.

Only the explicitly synthetic OCR stub and the module's developer demonstration opt into provider-specific mock receipt text. Pillow is pinned directly because receipt-image generation imports it as a supported project dependency.

## 6. Running the project

Run all commands from the repository root.

### Generate the raw synthetic datasets

```powershell
python -m src.data_gen.generate_synthetic_data
```

This writes:

- `data/raw/synthetic_momo_sms_logs.csv`
- `data/raw/synthetic_applicants_labeled.csv`

### Run the complete training pipeline

```powershell
python -m src.model.run_training
```

Optional parameters:

```powershell
python -m src.model.run_training --num-applicants 500 --out models/xgb_custom.json
```

The default run generates data in memory, creates features, makes a stratified 80/20 train/test split, performs cross-validation, trains XGBoost, evaluates it, and writes:

- `models/xgb_v1.json` — XGBoost model
- `models/xgb_v1.meta.json` — training metadata
- `models/xgb_v1.eval.json` — metadata plus holdout metrics

These JSON artifacts are ignored by the current `.gitignore` rules.

### Launch the dashboard

```powershell
python -m streamlit run src/ui/app.py
```

The dashboard provides Overview, New Assessment, and current-session History views. Train the default local model first. The UI then supports synthetic, CSV, SMS, and receipt evidence; canonical validation; financial review; cached model scoring; SHAP narratives; local assessment persistence; and a separate officer decision. See `docs/UI_IMPLEMENTATION.md`.

### Run tests

```powershell
python -m pytest -q
```

### Run formatting and lint checks

```powershell
python -m black --check src tests
python -m flake8 src tests
```

To apply Black formatting:

```powershell
python -m black src tests
```

## 7. Synthetic data generation

The main implementation is `src/data_gen/generate_synthetic_data.py`.

### Global assumptions

- Random seed: `42`
- Default cohort: `250` applicants
- Observation period: January 1 through June 30, 2026
- Target synthetic default rate: approximately `20%`
- Personas: `informal_trader` and `farmer`
- Providers: `M-Pesa` and `MTN_MoMo`

### Persona behavior

Informal traders receive a higher transaction frequency, approximately 30–75 transactions per month. Farmers receive approximately 6–22 transactions per month to represent more seasonal or lumpy activity.

### Transaction types

The generator supports:

- `CASH_IN`
- `P2P_RECEIVE`
- `CASH_OUT`
- `P2P_SEND`
- `BUY_GOODS_TILL` for M-Pesa merchants
- `MOMOPAY_MERCHANT` for MTN MoMo merchants
- `PAYBILL` for M-Pesa utilities
- `UTILITY` for MTN MoMo utilities
- `AIRTIME`

Each event records the applicant, timestamp, provider, transaction type, amount, post-transaction balance, and a provider-formatted SMS body.

### Synthetic label construction

The generator aggregates each applicant's:

- Net flow: inflows minus outflows
- Transaction count
- Number of balances below `1500`
- Income-normalized net flow
- Farmer/negative-flow interaction

It combines these values with random noise into a synthetic risk score. Applicants above the 80th percentile are assigned `default_label = 1`, yielding approximately a 20% default rate.

These are generated ground-truth labels, not observed repayment outcomes.

## 8. Data schemas

### Transaction log

`synthetic_momo_sms_logs.csv` contains one row per transaction:

| Field | Meaning |
|---|---|
| `applicant_id` | Synthetic borrower identifier |
| `timestamp` | Transaction date and time |
| `provider` | `M-Pesa` or `MTN_MoMo` |
| `sms_text` | Provider-style raw message |
| `tx_type` | Normalized transaction category |
| `amount` | Transaction value |
| `post_balance` | Wallet balance after the transaction |

### Applicant labels

`synthetic_applicants_labeled.csv` contains one row per applicant:

| Field | Meaning |
|---|---|
| `applicant_id` | Synthetic borrower identifier |
| `persona` | `informal_trader` or `farmer` |
| `provider` | Primary mobile-money provider |
| `avg_monthly_income` | Synthetic baseline income |
| `net_flow` | Total simulated inflows minus outflows |
| `tx_count` | Number of observed transactions |
| `low_balance_events` | Number of balances below `1500` |
| `default_label` | `0` healthy, `1` default/high risk |

The use of income-normalized ratios is intended to reduce distortion between KES and RWF without requiring currency conversion. It does not completely solve cross-country comparability.

## 9. SMS parsing and OCR

The central parser is `parse_transaction_text()` in `src/ingestion/ocr_parser.py`. `parse_sms_message()` in `src/ingestion/sms_parser.py` is a thin wrapper around it.

The parser returns:

```python
{
    "tx_id": str | None,
    "amount": float | None,
    "fee": float,
    "balance": float | None,
    "counterparty": str,
    "provider": str | None,
    "raw_text": str,
}
```

It recognizes M-Pesa and MTN MoMo currency/provider markers and uses regular expressions for IDs, amounts, fees, balances, and counterparties.

The parser now extracts timestamps and transaction types from supported provider wording. `src/ingestion/normalization.py` then validates applicant association, provider, timestamp, canonical type, positive amount, and non-negative post-balance before creating an immutable `NormalizedTransaction`. Batch results expose valid records, indexed rejections, warnings, and processing counts.

The structured synthetic CSV path is the guaranteed demo path. A generic receipt that does not identify its transaction type is rejected unless the caller supplies that missing context explicitly. The real OCR path raises an extraction error when Tesseract is unavailable; only the explicitly synthetic OCR stub may request mock receipt text.

## 10. Feature engineering

`build_feature_table()` in `src/features/build_features.py` accepts transaction events and returns one model-ready row per applicant.

Required event columns are:

```text
applicant_id, timestamp, tx_type, amount, post_balance
```

It optionally uses applicant metadata containing `avg_monthly_income`. Without it, observed inflows become the income proxy.

### The 32 model features

Velocity and activity:

- `tx_per_month`
- `peak_week_tx`
- `active_months`
- `days_since_last_inflow`

Net cash-flow stability:

- `net_flow_total`
- `net_flow_mean`
- `net_flow_std`
- `net_flow_cv`
- `net_flow_ratio`
- `negative_net_months`

Inflow behavior:

- `inflow_total`
- `inflow_mean`
- `inflow_std`
- `inflow_cv`
- `inflow_per_month`
- `inflow_regularity`

Outflow behavior:

- `outflow_total`
- `outflow_mean`
- `outflow_std`
- `outflow_cv`
- `outflow_per_month`
- `productive_ratio`
- `inflow_outflow_ratio`

Balance health:

- `low_balance_events`
- `low_balance_rate`
- `min_balance`
- `mean_balance`
- `balance_trend_slope`

Transaction mix:

- `airtime_ratio`
- `cashout_ratio`
- `p2p_send_ratio`
- `p2p_receive_ratio`

The `applicant_id` identifier is retained in the feature table but excluded from model inputs.

### Notable implementation detail

`days_since_last_tx` is calculated internally but is always zero because the observation end is defined as the applicant's latest transaction. It is not included in the final feature list. `days_since_last_inflow` is the useful recency measure that is retained.

## 11. Model training

`src/model/train.py` trains `xgboost.XGBClassifier` using conservative baseline parameters:

- 200 trees
- Maximum depth 4
- Learning rate 0.05
- Row and feature subsampling of 0.8
- Class weighting through `scale_pos_weight = 4`
- Fixed random seed 42

`train_model()` validates the required columns, replaces infinity with missing values, median-imputes missing data, runs five-fold stratified cross-validation, fits the final model, and saves its metadata.

`src/model/run_training.py` adds a separate 20% holdout set around that process. Cross-validation occurs only on the 80% training partition, and the saved model is then scored on the holdout partition.

`src/model/loader.py` centralizes artifact loading. It resolves an explicit path before `MODEL_PATH` and the documented default, loads optional sidecar metadata, exposes the model version, and validates stored feature names/counts against the canonical 32-feature schema. Legacy artifacts without metadata receive version `unknown`; no version is guessed from a filename.

## 12. Prediction

`score_applicant()` in `src/model/predict.py` loads a saved XGBoost artifact and validates that all 32 model features are present.

For one feature row it returns that applicant's positive-class probability. If multiple rows are passed, it returns the mean predicted probability, which is described as an ensembling convenience.

The value is the raw XGBoost positive-class output. The public scoring and assessment APIs describe it as a model risk score because no calibration algorithm such as isotonic regression or Platt scaling has been implemented and validated.

## 13. Evaluation

`generate_classification_report()` in `src/eval/metrics.py` reports:

- Sample count and observed positive rate
- Accuracy at a configurable threshold, default `0.5`
- Majority-class baseline accuracy
- Whether model accuracy beats that baseline
- ROC-AUC
- Precision
- Recall
- F1 score
- True positives, false positives, true negatives, and false negatives

In lending, accuracy alone is not sufficient. With an approximately 20% positive class, a model that predicts every applicant as healthy achieves about 80% accuracy. ROC-AUC, recall, false-negative rate, calibration, and the operational cost of each error type matter more.

The custom ROC-AUC implementation returns `0.0` if a test set contains only one class. Production evaluation should flag AUC as undefined in that situation rather than interpreting zero as model performance.

## 14. SQLite persistence

`src/storage/schema.sql` defines four append-only tables:

- `features`: applicant ID, JSON feature payload, timestamp
- `scores`: applicant ID, risk score, model version, timestamp
- `explanations`: model version, structured SHAP payload, language, narrative payload, timestamp
- `decisions`: applicant ID, decision label, rationale, timestamp

`src/storage/db.py` provides configuration-aware connection, schema initialization, and insert helpers. `src/storage/assessment_store.py` writes features, score, explanation, and narrative in one SQLite transaction, rolling back the complete operation if any required write fails.

Human decisions are stored through a separate validated operation supporting `APPROVE`, `REVIEW`, and `DECLINE`. These are values selected by an authorized person; the model never chooses or derives them.

The current schema is intentionally minimal. It has no applicant table, foreign keys, uniqueness constraints, model registry, consent/audit records, ingestion records, or update/version policies. It is not encrypted; SQLCipher is explicitly outside the current MVP scope.

## 15. Explainability and narratives

`src/xai/shap_explainer.py` now provides structured, typed explanations for one applicant. It validates the canonical 32-feature schema, ignores identifier columns, computes raw-margin Tree SHAP values, and returns all contributions plus deterministically ranked factors that increase or reduce the model output.

`src/xai/narratives.py` converts those ranked factors into structured English or Kiswahili explanations. It contains human-readable labels for every model feature, centralized language resources, an unknown-feature fallback, and a reusable responsible-AI disclaimer. Generation is deterministic and offline; it uses no LLM or translation service.

The model risk score is kept separate from SHAP contributions. The score is the XGBoost positive-class output and is not independently calibrated. The SHAP base value and contributions are additive in raw-margin/log-odds space, not probability percentage points.

Positive contributions move this model toward greater estimated risk, while negative contributions move it toward lower estimated risk. SHAP explains model behavior, not causation, and the synthetic MVP output should support rather than replace human review. See `docs/SHAP_NARRATIVES_IMPLEMENTATION.md` for the complete API and integration guide.

## 16. Streamlit dashboard

`src/ui/app.py` configures a compact task shell and dispatches Overview, New Assessment, and History views. `src/ui/state.py` owns the session contract, `src/ui/services.py` adapts frozen backend APIs, `src/ui/components.py` contains reusable display primitives, and `src/ui/theme.css` centralizes the Console visual system.

The New Assessment view guides an officer through applicant selection, synthetic/CSV/SMS/receipt evidence, canonical validation, financial feature review, model scoring, localized explanation, assessment saving, and a separately confirmed human decision. Model loading uses `st.cache_resource`, but scoring and persistence remain explicit operations. Rerun guard flags prevent duplicate database writes.

The Model Risk Score uses a neutral 0–1 scale. There are no invented risk bands or lending thresholds. Increasing and reducing factors retain backend ranking, English/Kiswahili text comes from the narrative backend, and the officer decision starts unselected.

Because the frozen backend has no historical read API, the History view honestly displays only records saved in the current browser session. See `docs/UI_IMPLEMENTATION.md` for the full architecture, state contract, visual tokens, and limitations.

## 17. Automated tests

The repository currently has 143 passing tests covering:

- `test_data_generation.py`: phone/ID formats, dataset schema, repeatability, default rate
- `test_sms_parser.py`: M-Pesa, MTN MoMo, and receipt parsing
- `test_features.py`: feature values, bounds, edge cases, missing columns
- `test_storage.py`: schema initialization and SQLite inserts
- `test_model.py`: training artifacts, metadata, AUC sanity, prediction validation
- `test_normalization.py`: canonical records, rejection reasons, SMS-to-feature integration, OCR safety
- `test_model_loader.py`: model artifacts, metadata/version handling, schema compatibility
- `test_assessment_service.py`: typed English/Kiswahili assessment orchestration and errors
- `test_assessment_store.py`: atomic persistence, rollback, explanation payloads, human decisions
- `test_backend_integration.py`: complete synthetic backend golden path
- `test_ui.py`: navigation, validation, model/error states, localization, persistence, and rerun safety
- XAI/narrative tests: SHAP semantics, ranking, localization, and integration

The full suite passes under the repository's Python 3.10 virtual environment. The remaining warnings are third-party matplotlib/Pyparsing deprecation warnings.

There are currently no tests for:

- The training orchestrator as a complete command
- Evaluation edge cases such as empty inputs
- Database migrations and concurrency

## 18. The most important modeling limitation: synthetic target leakage

The synthetic default label is constructed using net flow, transaction count, low-balance events, and income-normalized behavior. The feature table then provides the model with those same values or close derivatives.

Therefore, strong validation performance mainly shows that XGBoost can reconstruct the rule used to generate the labels. It does **not** demonstrate that the model predicts real loan defaults.

This is acceptable for validating that the software pipeline operates, but reported metrics must not be presented as evidence of real-world creditworthiness performance.

For meaningful validation, the model would need consented historical data with outcomes that occur after the feature observation window, along with strict temporal splitting and leakage review.

## 19. Other important limitations and risks

### Statistical and modeling risks

- Only 250 applicants are generated by default.
- Applicant behavior follows hand-written assumptions rather than measured distributions.
- Labels are deterministic derivatives of the modeled inputs plus noise.
- No hyperparameter selection protocol is documented.
- Raw XGBoost scores are not probability-calibrated.
- No threshold-selection strategy reflects SACCO costs or risk appetite.
- There is no out-of-time, geographic, provider, or persona validation.
- There is no drift detection or ongoing performance monitoring.

### Fairness and responsible-lending risks

- Persona and transaction patterns may encode socioeconomic status or location proxies.
- Different transaction frequency does not necessarily mean different repayment ability.
- Low balances may reflect timing or wallet preference rather than financial distress.
- Applicants with cash-heavy livelihoods may appear artificially inactive.
- Explanations can make a weak model seem more trustworthy than it is.
- Real deployment would require adverse-action practices, appeal mechanisms, human oversight, and jurisdiction-specific legal review.

### Privacy and security risks

- The database is unencrypted.
- No authentication or authorization is implemented.
- No consent, retention, deletion, or data-minimization process exists.
- Raw SMS messages can contain phone numbers and counterparties.
- Logs and explanations could expose sensitive financial information.

### Engineering gaps

- Persisted-history retrieval is unavailable because the frozen backend exposes no supported read API; the UI uses current-session history.
- Generic OCR receipts may omit transaction type and therefore require explicit context.
- Configuration reads environment variables but does not parse `.env` files automatically.
- There is no model registry, caching policy, or database migration framework.
- Generated model artifacts, metadata, and evaluation files remain local under the documented `.gitignore` policy.

## 20. Recommended implementation order

### Phase 1: Make the existing backend reproducible

1. Create and document the supported Python version.
2. Install dependencies and run all tests.
3. Run the training command and record the resulting metrics.
4. Resolve remaining legacy repository-wide formatting and lint findings separately.

### Phase 2: Extend ingestion coverage if policy requires it

1. Add confidence metadata if OCR/SMS acceptance policy later requires it.
2. Extend only explicitly supported provider message formats.
3. Define any additional provider-specific acceptance rules before extending formats.
4. Continue collecting explicit caller context for receipts that omit transaction type.

### Phase 3: Govern scoring and explanations

1. Validate the integrated UI workflow with intended SACCO stakeholders.
2. Define model artifact promotion and rollback practices.
3. Define score bands and thresholds only when an explicit SACCO policy exists.
4. Continue labeling raw model outputs as risk scores unless calibration is implemented.

### Phase 4: Extend the integrated dashboard only through supported APIs

1. Add a supported backend history/read contract.
2. Add authenticated officer identity and authorization.
3. Add audit, consent, retention, and migration policies.
4. Preserve the current separation between model output and human judgment.

### Phase 5: Prepare for any real pilot

1. Conduct privacy, security, and legal reviews.
2. Obtain explicit consent and define data retention.
3. Create real outcome labels with a temporal definition of default.
4. Remove target leakage and use out-of-time validation.
5. Test calibration, fairness, stability, and subgroup performance.
6. Establish human review, appeals, monitoring, and rollback procedures.
7. Encrypt data at rest and implement access control and audit logging.

## 21. Suggested definition of done for the MVP

The MVP can be considered integrated when:

- A clean environment can install and pass all automated tests.
- A user can provide supported SMS/receipt or synthetic CSV input.
- Invalid records receive understandable validation messages.
- The system produces a complete applicant feature row.
- A versioned model produces a risk score.
- SHAP drivers and an English/Kiswahili narrative are shown.
- A human can record approve/review/decline with a rationale.
- Features, scores, model version, explanation, and decision are persisted.
- The workflow operates without network access.
- Every screen states that the data/model are synthetic and non-production.

## 22. Quick reference

| Task | Command or file |
|---|---|
| Install | `python -m pip install -r requirements.txt` |
| Generate CSV data | `python -m src.data_gen.generate_synthetic_data` |
| Train and evaluate | `python -m src.model.run_training` |
| Run UI | `python -m streamlit run src/ui/app.py` |
| Run tests | `python -m pytest -q` |
| Generate data code | `src/data_gen/generate_synthetic_data.py` |
| Parse SMS/OCR | `src/ingestion/ocr_parser.py` |
| Normalize transactions | `src/ingestion/normalization.py` |
| Build features | `src/features/build_features.py` |
| Train model | `src/model/train.py` |
| Load model bundle | `src/model/loader.py` |
| Score applicant | `src/model/predict.py` |
| Assess applicant | `src/application/assessment.py` |
| Evaluate model | `src/eval/metrics.py` |
| SQLite helpers | `src/storage/db.py` |
| Persist assessment/decision | `src/storage/assessment_store.py` |
| Database schema | `src/storage/schema.sql` |
| Explain one score | `src/xai/shap_explainer.py` |
| Generate localized narrative | `src/xai/narratives.py` |
| Dashboard architecture | `docs/UI_IMPLEMENTATION.md` |

## 23. Final perspective

AkibaAI now has an integrated educational backend and a professional Streamlit consumer: canonical normalization, feature engineering, validated model loading, scoring, SHAP, localized narratives, typed assessment results, atomic persistence, and separate human decisions. The most important unfinished work is the governance, security, persisted-history API, real-data validation, and operating controls required beyond an academic MVP.

Treat the current model as a software-pipeline demonstration. The synthetic evaluation cannot support real credit decisions, and the code needs substantial validation, governance, privacy, security, fairness, and operational work before any real-data pilot.
