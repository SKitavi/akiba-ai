# AkibaAI Backend Integration

This document describes the complete Streamlit-independent backend workflow implemented for AkibaAI.

## 1. Purpose

Before this integration, AkibaAI contained capable but disconnected modules for parsing, feature engineering, XGBoost scoring, SHAP, narratives, and SQLite. A UI would have needed to understand and coordinate every low-level component itself.

The integrated backend now gives future interfaces clear boundaries:

```text
collect input
    |
    v
normalize and validate
    |
    v
call assessment service
    |
    v
render typed result
    |
    v
persist assessment / record separate human decision
```

No backend module imports Streamlit, and the golden path requires no internet connection.

## 2. Architecture

```text
Structured CSV record or parsed SMS/OCR fields
                       |
                       v
          Transaction normalizer
          | valid       | rejected
          v             v
NormalizedTransaction  reason/code
          |
          v
   build_feature_table()
          |
          v
      32-feature row
          |
          +-------------------------+
          |                         |
          v                         v
 load_model_bundle()          validated metadata
          |                   and model version
          +------------+------------+
                       |
                       v
              assess_applicant()
                       |
          +------------+------------+
          |                         |
          v                         v
  XGBoost risk score         Tree SHAP explanation
                                      |
                                      v
                         English/Kiswahili narrative
                                      |
                                      v
                              AssessmentResult
                                      |
                       +--------------+--------------+
                       |                             |
                       v                             v
             persist_assessment()        record_human_decision()
                       |                             |
                       v                             v
        features/scores/explanations              decisions
```

## 3. Canonical transaction model

### Why it exists

The parser historically returned fields such as `balance`, `tx_id`, and `raw_text`, while feature engineering requires `applicant_id`, `timestamp`, `tx_type`, `amount`, and `post_balance`.

Passing parser dictionaries directly into feature engineering would make every caller invent its own renaming, validation, and missing-data behavior. That is especially unsafe for financial values.

`src/domain/transactions.py` now defines the single boundary used across ingestion and feature engineering.

### `NormalizedTransaction`

```python
@dataclass(frozen=True)
class NormalizedTransaction:
    applicant_id: str
    timestamp: datetime
    provider: TransactionProvider
    tx_type: TransactionType
    amount: float
    post_balance: float
    transaction_id: str | None = None
    raw_text: str | None = None
```

The dataclass is frozen so downstream code cannot accidentally change validated financial data.

### Canonical providers

- `M-Pesa`
- `MTN_MoMo`

Common spelling variants such as `mpesa`, `M-PESA`, and `mtn momo` normalize to these values. Unknown providers are rejected.

### Canonical transaction types

- `CASH_IN`
- `P2P_RECEIVE`
- `CASH_OUT`
- `P2P_SEND`
- `BUY_GOODS_TILL`
- `MOMOPAY_MERCHANT`
- `PAYBILL`
- `UTILITY`
- `AIRTIME`

The same centralized constants define inflow, outflow, productive-outflow, and feature-input categories. Feature engineering imports them rather than maintaining a second list.

### Feature DataFrame boundary

`transactions_to_dataframe()` converts domain records to the canonical DataFrame. It contains at least:

```text
applicant_id
timestamp
tx_type
amount
post_balance
```

It also retains provider, transaction ID, and raw text for traceability. Feature engineering ignores those extra columns.

## 4. Normalization

The implementation is in `src/ingestion/normalization.py`.

### Parser versus normalizer

The parser extracts fields from text. It does not decide whether a record is safe for modeling.

The normalizer:

- Associates the applicant.
- Canonicalizes the provider and transaction type.
- Parses supported timestamp formats.
- Requires a positive finite amount.
- Requires a non-negative finite post-transaction balance.
- Maps parser `balance` to canonical `post_balance`.
- Detects conflicting balance fields.
- Preserves transaction ID and raw text.
- Rejects missing or unsupported information.

### Explicit context

`TransactionContext` allows a caller to supply a value that the source genuinely omits:

```python
context = TransactionContext(applicant_id="APP_0001")
```

Context is not fabricated by the backend. It must come from an explicit application action, such as a loan officer selecting the applicant whose messages are being imported.

If both the record and context contain different applicant IDs, normalization rejects the record.

### Parsed SMS example

Input:

```text
AAA11BBB22 Confirmed. You have received Ksh2,500.00 from 0712345678
on 1/1/26 at 10:00 AM. New M-PESA balance is Ksh5,000.00.
```

Parser output, simplified:

```python
{
    "tx_id": "AAA11BBB22",
    "provider": "M-Pesa",
    "timestamp": "2026-01-01 10:00:00",
    "tx_type": "P2P_RECEIVE",
    "amount": 2500.0,
    "balance": 5000.0,
    "raw_text": "...",
}
```

Normalization call:

```python
parsed = parse_sms_message(message)
transaction = normalize_transaction(
    parsed,
    context=TransactionContext(applicant_id="APP_0001"),
)
```

Canonical result:

```python
NormalizedTransaction(
    applicant_id="APP_0001",
    timestamp=datetime(2026, 1, 1, 10, 0),
    provider=TransactionProvider.MPESA,
    tx_type=TransactionType.P2P_RECEIVE,
    amount=2500.0,
    post_balance=5000.0,
    transaction_id="AAA11BBB22",
    raw_text="...",
)
```

### Structured CSV golden path

The generated synthetic DataFrame already includes every required field:

```python
normalization = normalize_transactions(synthetic_transaction_frame)

print(normalization.processed_count)
print(normalization.valid_count)
print(normalization.rejected_count)

feature_frame = normalization.to_dataframe()
```

This is the guaranteed demonstration path.

### Rejected records

Batch normalization does not abort on the first invalid row. It returns:

- `valid_transactions`
- `rejected_transactions`
- `warnings`
- `processed_count`
- `valid_count`
- `rejected_count`

Each rejection includes the source index, a stable code, and an understandable message. Examples include:

- `missing_applicant`
- `missing_timestamp`
- `invalid_timestamp`
- `unsupported_provider`
- `unsupported_transaction_type`
- `invalid_amount`
- `invalid_post_balance`
- `balance_conflict`

### OCR safety

The normal OCR function no longer silently returns a believable mock transaction when OCR fails. It raises `OCRExtractionError`.

The explicitly named synthetic OCR stub may opt into mock fallback for fixture/demo generation. This prevents missing OCR software from becoming invented financial evidence.

Generic receipts may provide a timestamp and amounts but no reliable transaction type. Such a receipt remains rejected unless the caller explicitly supplies the type through `TransactionContext`.

## 5. Model bundle and configuration

The implementation is in `src/model/loader.py`.

### Path precedence

Model path resolution uses:

```text
explicit function argument
        |
        v
MODEL_PATH environment variable
        |
        v
<project>/models/xgb_v1.json
```

Database path resolution similarly uses:

```text
explicit function argument
        |
        v
DB_PATH environment variable
        |
        v
<project>/akiba_ai.db
```

The backend reads environment variables. It does not automatically parse a `.env` file because no dotenv dependency is currently required.

### `ModelBundle`

`load_model_bundle()` returns:

```python
ModelBundle(
    model=...,
    model_path=...,
    model_version=...,
    metadata=...,
    feature_names=...,
    schema_verified=...,
)
```

The loader:

- Requires an existing model file.
- Converts invalid/corrupt XGBoost artifacts into a clear backend error.
- Reads `<model>.meta.json` when present.
- Validates metadata structure.
- Validates feature count and ordering against `FEATURE_COLUMNS`.
- Validates booster feature names when stored in the artifact.
- Exposes immutable metadata to callers.

If a legacy model has no metadata, the version is `unknown`. The loader never guesses a version from a filename.

The UI may cache a `ModelBundle`, but caching is intentionally not global backend state.

## 6. Assessment service

The implementation is in `src/application/assessment.py`.

### `AssessmentResult`

```python
@dataclass(frozen=True)
class AssessmentResult:
    applicant_id: str
    model_version: str
    risk_score: float
    features: Mapping[str, float]
    explanation: PredictionExplanation
    narrative: RiskNarrative
```

It contains no lending decision.

### `assess_applicant()`

The service:

1. Requires at least one `NormalizedTransaction`.
2. Requires all transactions to belong to exactly one applicant.
3. Converts canonical transactions to a feature DataFrame.
4. Calls the existing `build_feature_table()` implementation.
5. Selects exactly one applicant feature row.
6. Reuses the loaded XGBoost model.
7. Calls the existing `explain_prediction()` API.
8. Calls the existing `generate_risk_narrative()` API.
9. Returns an immutable result.

Example:

```python
bundle = load_model_bundle("models/xgb_v1.json")

result = assess_applicant(
    transactions=normalization.valid_transactions,
    model_bundle=bundle,
    applicant_id="APP_0001",
    applicants_df=applicants,
    language="sw",
    top_n=5,
)
```

The service supports English (`en`) and Kiswahili (`sw`). Unsupported languages fail clearly.

### Why orchestration is separate

Feature engineering, scoring, SHAP, and translation remain in their established modules. The application service coordinates them; it does not copy their logic.

This makes Streamlit a consumer of business logic instead of the owner of it. Tests and future interfaces can call exactly the same operation.

## 7. SHAP and narrative integration

The assessment service reuses the already implemented XAI APIs:

```python
explanation = explain_prediction(
    model=model_bundle.model,
    features=applicant_features,
    top_n=top_n,
)

narrative = generate_risk_narrative(
    explanation,
    language=language,
)
```

It does not recompute contribution directions, rankings, labels, or translations.

The result preserves the distinction between:

- XGBoost model risk score in `[0, 1]`
- SHAP base/contributions in raw-margin log-odds space
- deterministic human-readable narrative

No calibration claim or causal claim is added.

## 8. Persistence

### Tables

The SQLite schema contains:

- `features`
- `scores`
- `explanations`
- `decisions`

The new `explanations` table stores:

- Applicant ID
- Model version
- Structured SHAP payload
- Narrative language
- Structured narrative payload
- Creation timestamp

### Atomic assessment persistence

`persist_assessment()` writes features, score, explanation, and narrative inside one SQLite transaction.

```python
with get_connection("akiba_ai.db") as connection:
    initialize_schema(connection, schema_path)
    stored = persist_assessment(connection, result)
```

If any required insert fails, all earlier inserts in that assessment are rolled back. The function returns the three inserted row IDs only after the transaction succeeds.

### Human decision boundary

`record_human_decision()` is separate from `persist_assessment()`:

```python
record_human_decision(
    connection,
    applicant_id=result.applicant_id,
    decision=HumanDecision.REVIEW,
    rationale="Supporting documents requested.",
)
```

Controlled values are:

- `APPROVE`
- `REVIEW`
- `DECLINE`

These are storage values for a human-selected decision. The backend never maps a risk score to one of them.

## 9. Error contract

### Normalization

`normalize_transaction()` raises `TransactionNormalizationError` with a stable code for one record. `normalize_transactions()` collects these into `RejectedTransaction` objects.

### Model loading

- `ModelArtifactNotFoundError`
- `ModelArtifactError`
- `ModelMetadataError`
- `ModelSchemaError`

### Assessment

- `NoTransactionsError`
- `ApplicantScopeError`
- `AssessmentInputError`
- `FeatureEngineeringError`
- `AssessmentExecutionError`

### Persistence

- `AssessmentPersistenceError`
- `ValueError` for unsupported human-decision values

Low-level exceptions are chained as causes, preserving diagnostic detail without making raw library tracebacks the normal application contract.

## 10. Complete backend example

```python
from pathlib import Path

from src.application.assessment import assess_applicant
from src.ingestion.normalization import normalize_transactions
from src.model.loader import load_model_bundle
from src.storage.assessment_store import persist_assessment
from src.storage.db import get_connection, initialize_schema


normalization = normalize_transactions(structured_transactions)
if normalization.rejected_count:
    for rejected in normalization.rejected_transactions:
        print(rejected.record_index, rejected.code, rejected.message)

bundle = load_model_bundle("models/xgb_v1.json")
applicant_transactions = tuple(
    transaction
    for transaction in normalization.valid_transactions
    if transaction.applicant_id == "APP_0001"
)

result = assess_applicant(
    transactions=applicant_transactions,
    model_bundle=bundle,
    applicant_id="APP_0001",
    applicants_df=applicants,
    language="en",
)

connection = get_connection("akiba_ai.db")
try:
    initialize_schema(connection, Path("src/storage/schema.sql"))
    persist_assessment(connection, result)
finally:
    connection.close()
```

## 11. Testing

The backend integration is covered by unit, boundary, and real integration tests.

### Normalization tests

- Valid structured and parsed records
- Applicant context and conflicts
- Provider/type canonicalization
- Missing and malformed timestamps
- Invalid amounts and balances
- Unsupported transaction types
- Rejected-record counts and ordering
- SMS to normalized records to feature builder
- Explicit versus silent OCR fallback

### Model loader tests

- Valid artifact and metadata
- Missing and corrupt artifacts
- Metadata present/absent/corrupt
- Explicit `unknown` legacy version
- Feature-count/order mismatch
- Model usability by scoring and SHAP
- Configuration precedence

### Assessment tests

- English and Kiswahili results
- Applicant scope
- Model version and 32-feature payload
- Empty/mixed/invalid transactions
- Missing applicant metadata
- Model/SHAP and feature error wrapping
- Absence of automatic decisions

### Persistence tests

- Schema initialization
- Complete assessment payloads
- Model version and narrative language
- Forced partial failure rollback
- Valid and invalid human decisions
- Score/decision separation

### Golden-path test

`tests/test_backend_integration.py` uses the actual implementations to:

```text
generate 30 synthetic applicants
→ normalize all structured transactions
→ build feature rows
→ train and save XGBoost
→ load and validate ModelBundle
→ assess one applicant in English and Kiswahili
→ persist the assessment
→ verify no automatic decision
→ record an explicit human REVIEW decision
```

## 12. Offline and responsible-AI properties

The backend workflow contains no:

- Streamlit imports
- OpenAI/Gemini/LLM clients
- Translation APIs
- Remote databases
- Analytics/network calls

The implementation preserves these boundaries:

- Synthetic data is not represented as real evidence.
- OCR failure does not silently become mock financial data.
- Model output is called a risk score, not a calibrated probability.
- SHAP is model explanation, not proof of causality.
- No lending threshold is invented.
- Human review and decision persistence remain explicit and separate.

## 13. Limitations

- The model and validation evidence remain synthetic.
- Synthetic target leakage remains by design.
- Generic receipts can omit a reliable transaction type.
- Only explicitly recognized provider SMS wording can infer transaction type safely.
- OCR quality/confidence scoring is not implemented.
- `.env` files are not parsed automatically; environment variables are supported.
- Legacy models without metadata expose version `unknown`.
- The SQLite database is not encrypted.
- The schema has no foreign keys, applicant master table, migrations, authentication, consent records, or retention controls.
- Model caching is left to the application/UI boundary.
- The Streamlit shell does not yet invoke these services.
- Real-world calibration, fairness, drift, and outcome validation are not implemented.

## 14. Future Streamlit integration

The Streamlit phase can rely on the backend to:

1. Normalize records and return accepted/rejected counts.
2. Validate/load a reusable model bundle.
3. Return the complete applicant assessment.
4. Persist assessment output atomically.
5. Record a separate human-selected decision.

The UI should be limited to:

```text
collect input
→ show normalization feedback
→ call assess_applicant()
→ render AssessmentResult
→ request an explicit human decision
→ call persistence functions
```

It must not duplicate feature engineering, model loading, scoring, SHAP ranking, localization, SQL, or decision policy.

### UI handoff contract

The UI should use the actual backend APIs in this order:

```python
from pathlib import Path

from src.application.assessment import assess_applicant
from src.ingestion.normalization import normalize_transactions
from src.model.loader import load_model_bundle
from src.storage.assessment_store import (
    HumanDecision,
    persist_assessment,
    record_human_decision,
)
from src.storage.db import get_connection, initialize_schema


normalization = normalize_transactions(records)
applicant_transactions = tuple(
    transaction
    for transaction in normalization.valid_transactions
    if transaction.applicant_id == applicant_id
)

bundle = load_model_bundle()
result = assess_applicant(
    transactions=applicant_transactions,
    model_bundle=bundle,
    applicant_id=applicant_id,
    applicants_df=applicants,
    language="en",
)

connection = get_connection()
initialize_schema(connection, Path("src/storage/schema.sql"))
stored = persist_assessment(connection, result)

# Run only after an authorized person makes an explicit choice in the UI.
decision_id = record_human_decision(
    connection,
    applicant_id=result.applicant_id,
    decision=HumanDecision.REVIEW,
    rationale=human_entered_rationale,
)
```

Before assessment, Streamlit should display `processed_count`, `valid_count`,
`rejected_count`, and each structured rejection. It should load the bundle once
and reuse it across reruns where practical. The human-decision call must remain a
separate UI action after the assessment is displayed.
