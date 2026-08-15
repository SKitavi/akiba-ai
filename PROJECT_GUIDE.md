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

The SHAP explainability and English/Kiswahili narrative layers are implemented as reusable backend modules. The Streamlit user interface remains a placeholder. This is an MVP simulation, not a production lending platform, and it uses no real customer data.

## 2. The problem the project is exploring

Many SACCO applicants and informal workers have limited conventional credit histories. AkibaAI explores whether mobile-money activity can provide alternative signals such as:

- Frequency and regularity of transactions
- Stability of inflows and outflows
- Net cash flow relative to estimated income
- Frequency of low-wallet-balance events
- Balance direction over time
- Mix of productive payments, cash withdrawals, airtime, and peer transfers

The intended output is a risk score between `0` and `1`, where a larger value means a higher estimated probability of default.

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
                                      (unfinished)
```

The implemented training path operates directly on the structured synthetic transaction data. The OCR/SMS parser is tested separately but is not yet connected to feature engineering as a complete production ingestion path.

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
|   `-- architecture.png              Current placeholder architecture image
|-- models/                           Generated model/evaluation artifacts
|-- reports/                          Generated reports
|-- src/
|   |-- data_gen/                     Synthetic data and labels
|   |-- ingestion/                    SMS parsing, OCR, receipt generation
|   |-- features/                     Applicant-level feature engineering
|   |-- model/                        Training, orchestration, prediction
|   |-- eval/                         Classification metrics
|   |-- storage/                      SQLite schema and persistence helpers
|   |-- xai/                          SHAP and localized narratives
|   `-- ui/                           Streamlit placeholder
`-- tests/                            Automated unit tests
```

## 5. Environment setup

The repository is Python-based. The dependency file pins Faker, NumPy, pandas, pytesseract, scikit-learn, XGBoost, SHAP, Streamlit, matplotlib, pytest, Black, and Flake8.

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

`pytesseract` is a Python wrapper. Real OCR also requires the Tesseract executable to be installed and available on the system path. If it is unavailable, AkibaAI deliberately falls back to provider-specific mock receipt text.

Pillow is used by the receipt-image code but is not explicitly pinned in `requirements.txt`. It may arrive as a transitive dependency; it should be added explicitly if receipt image generation is treated as a supported feature.

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

The current dashboard only renders the application title and a TODO notice. It does not yet load a model, ingest transactions, calculate scores, or display explanations.

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

Important integration gap: this result does not include all fields required by feature engineering. In particular, the parser does not consistently return `applicant_id`, `timestamp`, `tx_type`, `amount`, and `post_balance` under the feature builder's names. A normalization stage still needs to infer transaction type and timestamp, attach an applicant, rename `balance` to `post_balance`, validate the record, and handle parsing failures.

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

## 12. Prediction

`score_applicant()` in `src/model/predict.py` loads a saved XGBoost artifact and validates that all 32 model features are present.

For one feature row it returns that applicant's positive-class probability. If multiple rows are passed, it returns the mean predicted probability, which is described as an ensembling convenience.

The code calls this a calibrated probability of default, but no calibration algorithm such as isotonic regression or Platt scaling is implemented. The value is currently the raw XGBoost class probability and should be described as a model risk score until calibration is added and validated.

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

`src/storage/schema.sql` defines three append-only tables:

- `features`: applicant ID, JSON feature payload, timestamp
- `scores`: applicant ID, risk score, model version, timestamp
- `decisions`: applicant ID, decision label, rationale, timestamp

`src/storage/db.py` provides connection, schema initialization, and insert helpers.

The current schema is intentionally minimal. It has no applicant table, foreign keys, uniqueness constraints, model registry, consent/audit records, ingestion records, or update/version policies. It is not encrypted; SQLCipher is explicitly outside the current MVP scope.

## 15. Explainability and narratives

`src/xai/shap_explainer.py` now provides structured, typed explanations for one applicant. It validates the canonical 32-feature schema, ignores identifier columns, computes raw-margin Tree SHAP values, and returns all contributions plus deterministically ranked factors that increase or reduce the model output.

`src/xai/narratives.py` converts those ranked factors into structured English or Kiswahili explanations. It contains human-readable labels for every model feature, centralized language resources, an unknown-feature fallback, and a reusable responsible-AI disclaimer. Generation is deterministic and offline; it uses no LLM or translation service.

The model risk score is kept separate from SHAP contributions. The score is the XGBoost positive-class output and is not independently calibrated. The SHAP base value and contributions are additive in raw-margin/log-odds space, not probability percentage points.

Positive contributions move this model toward greater estimated risk, while negative contributions move it toward lower estimated risk. SHAP explains model behavior, not causation, and the synthetic MVP output should support rather than replace human review. See `docs/SHAP_NARRATIVES_IMPLEMENTATION.md` for the complete API and integration guide.

## 16. Streamlit dashboard

`src/ui/app.py` is currently a shell. A useful MVP dashboard would need at least:

- Applicant selection or creation
- CSV/SMS/receipt ingestion
- Parsing and validation feedback
- Feature summary
- Model artifact/version selection
- Risk score with a documented threshold
- Top positive and negative explanation factors
- English/Kiswahili narrative selection
- Human approve/review/decline action
- SQLite persistence and audit history
- Explicit synthetic-data and non-production warnings

## 17. Automated tests

The repository contains 47 unit tests across five areas:

- `test_data_generation.py`: phone/ID formats, dataset schema, repeatability, default rate
- `test_sms_parser.py`: M-Pesa, MTN MoMo, and receipt parsing
- `test_features.py`: feature values, bounds, edge cases, missing columns
- `test_storage.py`: schema initialization and SQLite inserts
- `test_model.py`: training artifacts, metadata, AUC sanity, prediction validation

At the time this guide was created, all source and test files passed Python bytecode compilation. The full test suite could not be executed in the existing environment because `pytest` had not yet been installed. Run `python -m pip install -r requirements.txt` followed by `python -m pytest -q` to obtain the actual test result.

There are currently no tests for:

- End-to-end OCR-to-feature normalization
- The training orchestrator as a complete command
- Evaluation edge cases such as empty inputs
- Streamlit interactions
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

- Parsed SMS/OCR output does not directly match the feature schema.
- The UI, model, explanations, and database are not integrated.
- There is no central configuration loader for `.env` values.
- `DB_PATH` and `MODEL_PATH` in `.env.example` are not consumed by the current application.
- The example `MODEL_PATH=./artifacts/model.json` disagrees with the implemented default under `models/`.
- Receipt-image generation relies on Pillow without declaring it directly.
- The architecture image is presently not useful documentation.
- Model metadata and evaluation JSON files are unintentionally ignored along with model JSON artifacts, despite the `.gitignore` comment suggesting otherwise.

## 20. Recommended implementation order

### Phase 1: Make the existing backend reproducible

1. Create and document the supported Python version.
2. Install dependencies and run all tests.
3. Add Pillow explicitly.
4. Run the training command and record the resulting metrics.
5. Fix lint, type, and test failures before extending the system.
6. Align `.env.example`, README paths, and `.gitignore` behavior.

### Phase 2: Close the ingestion gap

1. Define a canonical normalized transaction model.
2. Extend parsing to extract timestamp and transaction type.
3. Add applicant association and validation rules.
4. Record parse confidence, warnings, and rejected records.
5. Add end-to-end tests from raw SMS and receipt images to feature rows.

### Phase 3: Implement scoring and explanations

1. Add a model loader that also validates metadata/version.
2. Integrate the implemented SHAP explanation and localized narratives into the UI.
3. Define score bands and thresholds only when an explicit SACCO policy exists.
4. Clearly label raw model scores unless calibration is implemented.

### Phase 4: Build the integrated dashboard

1. Add ingestion and applicant-selection pages.
2. Connect feature building and prediction.
3. Show transparent model and data validation errors.
4. Display explanations and allow human decisions.
5. Persist the complete audit trail in SQLite.
6. Add Streamlit and end-to-end UI tests.

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
| Build features | `src/features/build_features.py` |
| Train model | `src/model/train.py` |
| Score applicant | `src/model/predict.py` |
| Evaluate model | `src/eval/metrics.py` |
| SQLite helpers | `src/storage/db.py` |
| Database schema | `src/storage/schema.sql` |
| Explain one score | `src/xai/shap_explainer.py` |
| Generate localized narrative | `src/xai/narratives.py` |
| Dashboard TODO | `src/ui/app.py` |

## 23. Final perspective

AkibaAI has a solid educational backend for demonstrating an offline alternative-credit pipeline. Its strongest implemented areas are synthetic data generation, feature engineering, basic XGBoost training, evaluation, SHAP explainability, localized narratives, and unit-test coverage. Its most important unfinished work is integration: turning parsed inputs into normalized transactions, connecting the reusable scoring/explanation APIs to persistence, and providing a usable dashboard.

Treat the current model as a software-pipeline demonstration. The synthetic evaluation cannot support real credit decisions, and the code needs substantial validation, governance, privacy, security, fairness, and operational work before any real-data pilot.
