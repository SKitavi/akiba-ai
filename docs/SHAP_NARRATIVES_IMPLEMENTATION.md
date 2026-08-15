# SHAP Explainability and Localized Narratives

This document explains the implementation of GitHub issues #12 and #13 and how future application code should consume the new XAI APIs.

## 1. Problem being solved

An XGBoost risk score alone does not tell a SACCO loan officer which model inputs moved the result upward or downward. A number such as `0.63` is difficult to review responsibly without knowing whether the model relied on wallet balances, cash-flow stability, inflow regularity, or another signal.

AkibaAI now produces two complementary outputs:

1. A structured SHAP explanation containing traceable feature contributions.
2. A deterministic English or Kiswahili narrative suitable for a human reviewer.

The implementation does not make a lending decision, invent risk bands, or claim that a model feature caused a real-world outcome.

## 2. Architecture

```text
Canonical applicant feature row
              |
              v
      XGBoost classifier
         |             |
         |             v
         |      Tree SHAP explainer
         |             |
         v             v
    Risk score    Raw-margin contributions
                         |
               +---------+---------+
               |                   |
               v                   v
       Increasing-risk      Reducing-risk
           factors             factors
               \                   /
                +--------+--------+
                         |
                         v
              Narrative generator
                  |             |
                  v             v
               English      Kiswahili
```

The XAI modules contain no Streamlit code. They return immutable dataclasses so a dashboard, report generator, test, or future API can choose how to display the result.

## 3. SHAP implementation

### Explainer choice

`explain_prediction()` uses `shap.TreeExplainer` with `model_output="raw"`. Tree SHAP is designed for tree ensembles and is an appropriate efficient explainer for the repository's `xgboost.XGBClassifier`.

The implementation targets the pinned versions:

- XGBoost `3.0.3`
- SHAP `0.48.0`

For one row and the current binary model, SHAP 0.48 returns:

- `values` with shape `(1, 32)`
- one base value
- one contribution per canonical feature

The code validates this contract explicitly. An unexpected shape raises `RuntimeError` rather than being silently reshaped or misinterpreted.

### What a SHAP value means

SHAP begins with the model's expected or baseline output and assigns each feature a contribution for one prediction.

For AkibaAI's binary XGBoost model:

- A positive SHAP value moves the raw model output toward higher estimated risk.
- A negative SHAP value moves the raw model output toward lower estimated risk.
- A zero SHAP value is neutral for that prediction.

The additive relationship is:

```text
base raw margin + sum(feature SHAP values) = applicant raw model margin
```

### Output space

The base value and contributions use XGBoost's raw margin, which is log-odds for the binary logistic objective. They are not percentage-point changes in probability.

The ordinary applicant `risk_score` is calculated separately through the same prediction helper used by `score_applicant()`. It is XGBoost's positive-class output in `[0, 1]`. The repository does not currently include a separate calibration stage, so the score must not be described as a calibrated probability.

Keeping these values separate prevents a common technical error: presenting a raw-margin SHAP value such as `0.4` as a 40-percentage-point increase in probability.

### Model behavior is not causality

SHAP faithfully explains how this fitted model used a feature. It cannot establish that the feature caused financial distress, default, repayment, or any other real-world outcome.

For example, a positive contribution from `low_balance_rate` means frequent low wallet balances moved this model's output toward higher estimated risk for that applicant. It does not prove that low balances caused a default.

### Feature validation

`prepare_model_features()` in `src/model/predict.py` is shared by prediction and explanation. It:

- Requires a pandas DataFrame with at least one row.
- Rejects duplicate columns.
- Requires every name from the canonical `FEATURE_COLUMNS` list.
- Selects features in canonical training order.
- Excludes extra fields such as `applicant_id`.
- Rejects non-numeric values with a clear error.
- Replaces infinity and missing values with zero, preserving the established prediction-time behavior.

`explain_prediction()` additionally requires exactly one applicant row and compares stored XGBoost feature names with `FEATURE_COLUMNS` when the model artifact contains them. This detects accidental model/schema drift.

### Ranking

Every feature remains available in `PredictionExplanation.contributions` in canonical order. Directional lists contain only the strongest non-zero factors:

- `increasing_risk_factors`
- `reducing_risk_factors`

Each direction is sorted first by descending absolute SHAP magnitude and then by feature name. The feature-name secondary key makes ties deterministic across runs. `top_n=0` returns empty directional lists, and a large `top_n` returns only the factors that actually exist.

## 4. Applicant data flow

For one applicant, the flow is:

1. Feature engineering creates a DataFrame row containing the canonical 32 features and optionally `applicant_id`.
2. `prepare_model_features()` validates the row, removes the identifier, applies the prediction-time missing-value policy, and orders the columns.
3. `predict_risk_score()` obtains the XGBoost positive-class score.
4. `TreeExplainer` computes the raw-margin base value and 32 feature contributions.
5. `explain_prediction()` assigns each contribution a direction and ranks both directional groups.
6. `generate_risk_narrative()` resolves localized feature labels and direction-specific wording.
7. The caller displays the summary, factor sections, and responsible-AI disclaimer.

No step sends data over the network.

## 5. Public classes and functions

### `prepare_model_features(applicant_features)`

Location: `src/model/predict.py`

Accepts one or more applicant feature rows and returns a numeric DataFrame containing only `FEATURE_COLUMNS` in canonical order. It centralizes prediction and explanation validation so the two pathways cannot gradually adopt conflicting schemas.

### `predict_risk_score(model, applicant_features)`

Location: `src/model/predict.py`

Accepts a fitted `XGBClassifier` and feature rows. It returns the mean positive-class model score. `score_applicant()` now loads the saved model and delegates to this helper.

### `ContributionDirection`

Location: `src/xai/shap_explainer.py`

A string enum with:

- `INCREASES_RISK`
- `REDUCES_RISK`
- `NEUTRAL`

An enum prevents inconsistent direction strings from spreading across SHAP, narrative, test, and UI code.

### `FeatureContribution`

Location: `src/xai/shap_explainer.py`

An immutable dataclass containing:

- `feature_name`
- `feature_value`
- `shap_value`
- `direction`
- computed `absolute_importance`

The technical name is retained for auditability even when the UI uses a localized label.

### `PredictionExplanation`

Location: `src/xai/shap_explainer.py`

An immutable dataclass containing:

- `risk_score`
- `base_value`
- `output_space`
- all `contributions`
- ranked `increasing_risk_factors`
- ranked `reducing_risk_factors`

The structured representation is preferable to a plot-only API because callers can render it as text, cards, tables, charts, JSON-like data, or tests.

### `explain_prediction(model, features, top_n=5)`

Location: `src/xai/shap_explainer.py`

The main issue #12 interface. It validates one applicant, calculates the score and SHAP result, creates typed contributions, and returns ranked factors. It raises clear exceptions for missing features, multiple rows, invalid `top_n`, unfitted models, schema drift, and incompatible SHAP shapes.

### `compute_shap_values(model, features_df)`

Location: `src/xai/shap_explainer.py`

A compatibility function for the original stub API. It returns the SHAP values array. New application code should prefer `explain_prediction()` because the raw array lacks the base value, risk score, directions, values, and rankings.

### `NarrativeLanguage`

Location: `src/xai/narratives.py`

A string enum containing the supported codes `en` and `sw`. Adding another language requires adding an enum member, labels, and centralized text resources rather than redesigning the business logic.

### `NarrativeFactor`

Location: `src/xai/narratives.py`

An immutable UI-ready factor containing the technical feature name, localized label, original feature value, SHAP value, direction, and rendered sentence.

### `RiskNarrative`

Location: `src/xai/narratives.py`

An immutable structured result containing:

- selected language
- risk score
- localized summary
- increasing-risk narrative factors
- reducing-risk narrative factors
- localized disclaimer

Sections remain separate so Streamlit does not need to parse a long text string.

### `get_feature_label(feature_name, language="en")`

Location: `src/xai/narratives.py`

Returns the localized human-readable label for one feature. Every canonical model feature has both English and Kiswahili labels. Unknown identifiers receive a readable underscore-to-space fallback while their technical name remains available for traceability.

### `generate_risk_narrative(explanation, language="en")`

Location: `src/xai/narratives.py`

The main issue #13 interface. It accepts `PredictionExplanation`, preserves its rankings, validates factor directions, and returns `RiskNarrative`. It never adds thresholds, decisions, or fabricated factors.

### `build_narrative(explanation, language="en")`

Location: `src/xai/narratives.py`

A compatibility helper for the original mapping-based stub. It treats mapping values as SHAP contributions and returns a compact string plus the disclaimer. New code should use the structured interface.

## 6. Narrative architecture

### Human-readable labels

`FEATURE_LABELS` contains an English and Kiswahili label for every canonical model feature. Examples:

| Technical name | English | Kiswahili |
|---|---|---|
| `negative_net_months` | Months with negative net cash flow | Miezi yenye mtiririko hasi wa fedha |
| `low_balance_rate` | Frequency of low wallet balances | Marudio ya salio dogo la pochi |
| `inflow_regularity` | Regularity of incoming funds | Uthabiti wa fedha zinazoingia |

### Direction-specific wording

The narrative direction comes directly from SHAP:

- Positive: the feature pushed the model output toward higher estimated risk.
- Negative: the feature pushed the model output toward lower estimated risk.

The wording is intentionally neutral. It avoids labels such as “irresponsible,” guarantees such as “will repay,” and causal statements such as “caused default.”

### Language selection

All user-facing sentences are centralized by `NarrativeLanguage`. Unsupported languages fail clearly rather than silently falling back and mixing languages.

Unknown future feature names use a readable fallback label. This allows schema changes to remain inspectable, but production language review should add a deliberate translation before such a feature is released.

### Why deterministic templates

Rule-based templates were selected because AkibaAI is offline-first and explanations affect a sensitive lending workflow. Compared with an LLM or remote translation service, deterministic templates are:

- Reproducible
- Testable
- Auditable
- Free of network dependencies
- Resistant to invented facts
- Consistent across applicants with equivalent contributions

The tradeoff is less linguistic variety, which is desirable for an academic credit-risk MVP.

## 7. Tests

### SHAP tests

`tests/test_shap_explainer.py` protects against:

- Missing or reordered model features
- Identifier leakage into XGBoost
- Multiple-applicant misuse
- Incorrect contribution count
- Wrong positive/negative separation
- Non-deterministic magnitude ties
- Invalid and boundary `top_n` values
- Zero-contribution fabrication
- Non-finite preprocessing drift
- Saved-model schema drift
- Score disagreement with `score_applicant()`
- Unexpected SHAP output shapes

Exact SHAP floats are not asserted because they can vary with model internals. Tests assert stable structure, semantics, ordering, and score consistency instead.

### Narrative tests

`tests/test_narratives.py` protects against:

- Missing feature labels
- English/Kiswahili mixing
- Incorrect directional wording
- Unsupported language fallback
- Unknown-feature crashes
- Non-deterministic output
- Fabricated factors when one direction is empty
- Missing responsible-AI context
- Regression to `NotImplementedError`
- SHAP-to-narrative integration failures

The integration test fits an XGBoost model, explains one applicant with real Tree SHAP, and generates both localized narratives.

## 8. Limitations

- Training data and outcomes are synthetic.
- The synthetic target is derived from signals also present in the feature table, creating target leakage by design.
- Evaluation shows that the pipeline learns the synthetic rule; it does not validate real credit performance.
- The raw XGBoost score is not independently probability-calibrated.
- SHAP explains model behavior, not real-world causality.
- Only English and Kiswahili are supported.
- Kiswahili labels and wording should receive review from a domain-fluent SACCO practitioner before a pilot.
- No score-band or approve/decline policy is implemented.
- The current UI does not yet consume these APIs.
- Explanations are not yet persisted in SQLite.

## 9. Future Streamlit integration

The Streamlit issue should load the model once, build or select one applicant feature row, and call the two public functions. It should not recompute SHAP directions, rankings, labels, translations, or disclaimers itself.

```python
from pathlib import Path

import pandas as pd
import xgboost as xgb

from src.xai.narratives import generate_risk_narrative
from src.xai.shap_explainer import explain_prediction


def explain_for_ui(
    model_path: Path,
    applicant_features: pd.DataFrame,
    language: str,
):
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    explanation = explain_prediction(
        model=model,
        features=applicant_features,
        top_n=5,
    )
    return generate_risk_narrative(explanation, language=language)
```

The UI can then render:

```python
narrative = explain_for_ui(model_path, applicant_row, language="sw")

st.metric("Risk score", f"{narrative.risk_score:.3f}")
st.write(narrative.summary)

for factor in narrative.increasing_risk_factors:
    st.warning(factor.text)

for factor in narrative.reducing_risk_factors:
    st.success(factor.text)

st.caption(narrative.disclaimer)
```

The UI should label the number as a model risk score, display both directional sections even when one is empty, show the disclaimer, retain the model version for audit, and leave the final lending decision to an authorized human reviewer.
