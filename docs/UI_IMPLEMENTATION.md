# AkibaAI Streamlit UI implementation

## Purpose and status

The Streamlit application is the officer-facing consumer of AkibaAI's frozen
backend. It provides a complete synthetic MVP workflow from evidence intake to
an independently recorded human decision. It does not contain scoring,
feature-engineering, SHAP, narrative, or persistence business logic.

Run it from the repository root after training the local model:

```powershell
$env:PYTHONUTF8 = "1"  # needed on Windows consoles that use CP1252
python -m src.model.run_training
python -m streamlit run src/ui/app.py
```

The training command is deliberately separate. If the artifact is missing, the
UI shows setup guidance and never trains implicitly.

## Design direction

The primary reference was the local AkibaAI **Console** design direction. Its
compact branch-workstation layout became the basis for the top task navigation,
cool steel canvas, institutional navy actions, restrained borders, small corner
radii, dense financial rows, and monospaced numeric output. The Ledger and
Branch explorations informed hierarchy and tone but were not selected because
they were less efficient for repeated desktop assessment work.

The DBOS frontend was used as a quality benchmark for shell discipline,
constrained content width, compact headers, consistent states, accessible native
controls, and responsive stacking. Its branding, gradients, authentication,
domain concepts, and application code were not copied.

### Visual tokens

| Role | Token |
|---|---|
| Primary ink | `#111C26` |
| Institutional navy | `#0F2C44` |
| Navy hover | `#163C5C` |
| Link/focus | `#1F5C8B` |
| Canvas | `#EEF1F4` |
| Surface/subtle surface | `#FFFFFF` / `#F7F9FB` |
| Border | `#D3DAE1` |
| Valid / attention / failure | `#2F7D62` / `#7A5A1E` / `#A4452F` |

Inter/system UI is used for interface text and Roboto Mono/Consolas for figures.
Controls use 5–6 px radii and borders rather than decorative shadows. Color is
always accompanied by words, counts, or direction labels.

## Screen and component structure

```text
src/ui/
|-- app.py                       page configuration and route dispatch
|-- state.py                     documented session-state contract
|-- services.py                  thin adapters to backend APIs
|-- components.py                shared shell and financial UI components
|-- theme.css                    centralized tokens and responsive styling
`-- views/
    |-- overview.py              honest current-session operational state
    |-- new_assessment.py        five-step assessment workflow
    `-- history.py               current-session saved assessments
```

The application uses three task-level destinations: Overview, New Assessment,
and History. The five assessment steps are Applicant, Transactions, Validation,
Assessment, and Decision.

## Backend integration

The main execution path is:

```text
synthetic / CSV / SMS / receipt
  -> normalize_transactions(...)
  -> build_feature_table(...) preview
  -> load_model_bundle(...) via st.cache_resource
  -> assess_applicant(...)
  -> generate_risk_narrative(...) when language changes
  -> persist_assessment(...)
  -> record_human_decision(...)
```

The UI adapters call the canonical data generator for demo records, the real SMS
and receipt parsers, the real OCR extractor, the typed application assessment
service, and the established SQLite helpers. Receipt OCR failures are surfaced;
no believable fallback transaction is fabricated. Database connections are
closed deterministically after each explicit write.

The model cache key includes the resolved artifact path. This reuses one
validated model per configured artifact while still respecting `MODEL_PATH`
changes between environments.

## Session state

`state.py` initializes the complete state contract:

- `route` selects the top-level task.
- `assessment_step` selects the current workflow step.
- `applicant_id` and `applicants_df` retain applicant scope and optional demo
  metadata.
- `source_key` and `source_records` retain the selected evidence boundary.
- `normalization_result` and `feature_preview` retain validated backend output.
- `assessment_result` is the typed backend `AssessmentResult`.
- `narrative_language` records `en` or `sw`.
- `assessment_saved` and `persisted_assessment` guard and identify the assessment
  write.
- `decision_value`, `decision_rationale`, `decision_saved`, and `decision_id`
  keep the human action separate and prevent rerun writes.
- `last_error` provides recoverable workflow feedback.
- `session_history` contains only assessments saved during the browser session.

Widget keys are also centralized so starting another assessment clears stale
form values without deleting session history.

Streamlit reruns the script after every interaction. Persistence therefore occurs
only inside explicit Save Assessment and Record Decision button branches. The two
saved flags make later reruns no-ops. Automated tests verify one features, score,
and explanation row and one independently recorded decision row.

## Input and validation behavior

The demo path uses the repository's deterministic synthetic generator. CSV rows,
provider SMS messages, and receipt images pass through thin adapters and the
canonical normalizer. Validation reports processed, valid, rejected, and warning
counts. Every rejected row shows the backend error code, understandable reason,
and a suggested correction. If zero rows survive, assessment progression is
disabled.

The financial preview exposes useful grouped behavior and all 32 model inputs on
demand. Amounts are labelled as provider wallet units because the application
does not infer or convert currency.

## Assessment and responsible use

The score is shown neutrally on a 0–1 line with three decimal places. It is called
the **Model Risk Score**, not a calibrated probability. No low/medium/high bands,
approval thresholds, traffic-light colors, or recommendation are invented.

Increasing and reducing factors remain in backend-ranked order. The primary UI
shows localized labels and narrative text; raw feature names, values, SHAP values,
and direction are available under technical details. English and Kiswahili use
the backend narrative generator, not UI translation.

The Officer Decision panel is visually and technically separate. Approve, Review,
and Decline start unselected, and the score never preselects or styles an answer.
Recording requires an explicit action and supports an officer rationale.

## Error, empty, and loading states

- Missing model: expected artifact path and documented training command.
- OCR/input failure: calm action-needed panel without fabricated data.
- Rejected evidence: row-level reason and correction.
- Zero valid evidence: blocked progression.
- Assessment failure: validated evidence remains available for retry.
- Database failure: user-facing failure with optional technical details.
- Empty overview/history: no fake analytics or persisted-history query.
- Long operations: specific validating, scoring, and saving messages.

## Accessibility and responsiveness

Native Streamlit buttons, radios, inputs, uploaders, and tables preserve keyboard
behavior and labels. Focus outlines use the link token, status does not rely on
color, body contrast is restrained but readable, and tablet touch targets expand
to 44 px. The 1440 px desktop canvas preserves operational density; Streamlit
columns stack at narrower widths and tables retain horizontal overflow.

## Testing

`tests/test_ui.py` uses `streamlit.testing.v1.AppTest` and a small temporary model.
It covers application load, navigation, demo validation, zero-valid blocking,
missing-model guidance, model score and version, directional explanations,
English/Kiswahili switching, unselected decisions, explicit confirmation,
database writes, and rerun idempotency. Adapter tests cover empty CSV and SMS
input. The complete backend suite remains the regression boundary.

## Known limitations

- The frozen backend exposes no supported read API for saved assessments, so the
  History page intentionally shows current-session records only.
- Authentication, authorization, encryption, audit identity, migrations, consent,
  and production governance are outside this synthetic MVP.
- Receipt quality and local Tesseract installation determine OCR success.
- On Windows consoles using CP1252, the frozen training CLI can fail while
  printing a Unicode evaluation symbol after it has written the model. Set
  `$env:PYTHONUTF8 = "1"` before training so the full evaluation report completes.
- The synthetic model has no approved SACCO policy thresholds and must not be used
  for real lending decisions.
