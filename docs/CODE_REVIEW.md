# Engineering review

Standing review of the codebase, ordered by impact. Findings concern code
structure and statistical method; no measurement values are reproduced here.

---

## 1. Training/serving skew — critical

`extract_chip_features_from_unified` (fitting) and `compute_real_time_features`
(inference) emit dictionaries with **the same keys computed from different data**.

| Feature | At fit time | At inference |
|---|---|---|
| `last_baking_resistance` | `resistance_data[-1]` — end of run | `first_10_resistances[-1]` — 10th sample |
| `baking_time` | `t[-1] - t[0]` — full duration | `t[9] - t[0]` |
| `baking_change`, `baking_pct_change` | full run | first 10 points |

Measured across a representative wafer set, the inference-time values of
`last_baking_resistance` and `baking_time` come out at roughly 0.7x and 0.1x
their fit-time counterparts. The model is fitted on end-of-anneal state and
queried with early-anneal state, so it is evaluated outside its training
distribution on every live prediction.

Nothing else in this document matters as much. It also has a specific
consequence for the model-selection story: a candidate that appears to lose on
paper may only be losing because it leans on a feature that means something
different at inference. **Comparisons between model families are not fully
trustworthy until this is fixed.**

**Fix.** One module, `src/abaa/features.py`, exposing
`compute_features(times, resistances, initial_resistance, meta) -> dict`,
called from both paths, always over the first *N* points. Fitting then
truncates each historical run to its first *N* points before extraction.
Assert at load time that `set(feature_dict) == set(model.metadata["features"])`
and fail loudly on mismatch.

## 2. Model selection: what the candidate set does and does not cover

The design intent is correct, and the candidate set is defensible. Lasso is
L1-penalised linear regression, so `{Lasso, RandomForest}` already spans
"linear with feature selection" against "arbitrary nonlinear". Where the
underlying relationship is linear, the Lasso branch wins on cross-validation
and the system reports it. Where it is not — and the evidence so far is that
curvature appears on some chips with no pattern predicting which — the forest
captures it. That is a sound reason to keep both rather than hard-coding a
formula.

Two things are genuinely missing, and neither is a linear model.

- **The null baseline.** No candidate answers "does this beat predicting the
  training mean?" Without it, a reported RMSE has no reference point: a model
  can look reasonable in absolute terms while learning nothing. This is the
  one baseline that must be in every comparison, and it costs one line.
- **A record of the selection.** Model metadata should carry chip type,
  resistance range, temperature, bias amplitude, *n*, and the CV score of
  *every* candidate — not just the winner. "Lasso won on chip A, the forest
  won on chip B" is the evidence that per-context fitting is necessary rather
  than decorative, and right now that evidence lives in the operator's memory
  instead of in the artifact.

A one-parameter rule forced through the origin (`rise = k x |drop|`, no
intercept) is worth carrying in `tools/benchmark_models.py` as a physical
reference, since it encodes "zero drop implies zero rise" in a way a fitted
intercept does not. It is a sanity check, not a production candidate.

## 2b. Possible target leakage in the historical feature set

The June 2025 feature-importance chart lists `cooled_std_resistance` among the
model inputs. If that quantity is computed from the **cooled** measurement,
it is derived from the same measurement the model is predicting, and any
performance figure obtained with it in the feature set is inflated.

The current extractor in `abaa_gui.py` does not appear to compute it, so this
may already be resolved — but the reported RMSE of 44 Ohm predates that
extractor. **Confirm which feature set produced the headline number, and
re-run it on the current one if they differ.** Any feature whose name begins
`cooled_` should be treated as a target, not an input.

The same chart lists `baking_max_resistance`, `late_to_mid_ratio`,
`pre_avg_resistance`, `mid_to_early_ratio`, `baking_std_resistance`,
`late_slope` and `max_accel`, none of which the current extractor produces.
The feature set has drifted between versions without a record. Pin it: an
explicit `FEATURES` list, versioned, stored in the model metadata.

## 3. Asymmetric loss

Annealing raises R_n irreversibly. Under-predicting the final resistance
causes overshoot and destroys the target; over-predicting causes an early
stop, which is recoverable by annealing again. The consequences are not
symmetric, so the objective should not be either.

The existing "conservative model" (a fixed scale factor on the standard
prediction) is the right instinct implemented ad hoc. The principled form is
quantile regression — `GradientBoostingRegressor(loss="quantile", alpha=...)`
or `QuantileRegressor` — which estimates the margin from the data rather than
applying a hand-tuned constant, and lets the safety level be stated as a
number ("wrong on the unsafe side 20% of the time") rather than a scale
factor whose meaning depends on the dataset.

Report both: RMSE for accuracy, overshoot rate for safety. A single symmetric
error figure describes this system poorly.

## 4. Statistical methodology

- **Selection on the test set.** `train_rf_model` picks between candidates
  with `if test_rmse < best_score`, then reports that same `test_rmse` as
  model performance. The reported number is optimistically biased. Select on
  a CV score computed inside the training fold; touch the test set once.
- **Unstable splits.** `random_state = int(time.time()) % 10000` draws a
  different split on every retrain. On a small wafer the reported RMSE then
  swings by more than the difference between the models being compared —
  which means the *selection itself* is partly noise.
- **Grid size vs sample size.** The RF grid is 3x3x3x2x3 = 162 configurations
  at 3-fold, on a few dozen rows. That is aggressive model selection on very
  little data. Two or three configurations per family is plenty.
- **Suggested replacement:** nested CV, or `RepeatedKFold(n_splits=5,
  n_repeats=20)` with a fixed seed, reporting mean +/- std per candidate.

## 5. Feature engineering

- `initial_drop_pct_5x` and `initial_drop_pct_10x` are exact multiples of
  `initial_drop_pct`. Both pipelines begin with `StandardScaler`, and scaling
  maps all three columns to **bit-identical** values. The intended "5x weight"
  does not exist. For the random forest the three clones split importance
  three ways, *reducing* the apparent importance of the feature the comments
  call most critical. Same for `jump_absolute_weighted_3x`. Delete all four.
- `jump_pct` and `initial_drop_pct` are assigned from the same expression —
  a duplicate column under two names.
- **Feature selection by substring.** `[col for col in df.columns if 'change'
  in col.lower()]` silently drops `baking_change` and `baking_pct_change`
  from the feature set. The model's inputs depend on naming convention. Use
  an explicit `FEATURES = [...]` list.
- `temperature`, `voltage`, `frequency` are regex-parsed from filenames with
  a silent `except: pass` fallback to constants. Within a single wafer run
  they are usually constant, so they contribute nothing — while being exactly
  the variables the per-context selection story depends on. Carry them as
  real columns in the CSV metadata so cross-condition fits become possible.
- `StandardScaler` in front of a random forest is a no-op for tree splits.
  Harmless, but a reviewer will read it as pipeline-by-pattern.

## 6. Dead and orphaned code

- `optimize_feature_subset` (65 lines) is never called, and its default
  `target_col="resistance_rise"` names a column the extractor does not
  produce. It would raise if invoked.
- `ConservativeRegressor` is defined in `abaa_gui.py` but only reachable via
  `tools/inspect_model.py` when unpickling historical models — despite the
  asymmetric-loss behaviour it implements being one of the more important
  ideas in the project. Either promote it to a first-class candidate in the
  selector or replace it with a quantile model (see finding 3).
- `check_hardware_connection(self)` is defined at module level with a `self`
  parameter; the live copy is the method on `ABAAController`.
- 15 bare `except:` clauses. These swallow `KeyboardInterrupt` and
  `SystemExit` and will hide instrument faults. Catch `Exception` at minimum
  and log the traceback.
- 106 `print()` calls. Move to `logging` with a level per subsystem, so
  instrument chatter and model chatter can be enabled independently.

## 7. Instrument and metrology

These bear directly on the quoted targeting precision — every ohm of
systematic error here is an ohm of the error budget.

- **Two-wire measurement.** `R = 0.01 / I_measured` uses the *commanded*
  10 mV, not a sensed voltage, and lead plus contact resistance sits in
  series with the junction. The 2450 supports four-wire; probe geometry
  permitting, `:SENS:RES:RSEN ON` removes a systematic that currently lands
  in every derived feature and in the final targeting figure.
- **Software timing.** `msleep(dt)` inside a Python loop with a `:READ?`
  round trip per sample. Actual sample spacing drifts with USB latency and
  GUI load, and `baking_slope`, `mean_rate` and `second_derivative` are all
  computed on that time base. Instrument-side triggering, or at minimum
  timestamps read back from the instrument, would fix this.
- **No compliance check.** The current limit is set but never tested. A
  junction going resistive or shorting is recorded as an ordinary reading.
  Query limit-tripped status after each read and flag the run.
- **No safe shutdown on exception inside the anneal loop.** If `:READ?`
  raises mid-anneal the output can be left on at full bias. Wrap the loop
  body in `try/finally` with `:OUTP OFF`.
- Fixed `:SENS:CURR:RANG 1E-3` — confirm this suits the full resistance
  range you anneal across; with autorange off, clipping is silent.
- **Repeatability budget.** Worth measuring and publishing: re-measure the
  same junction *n* times without annealing, and report the standard
  deviation. That number sets the floor on any targeting claim and is the
  single most useful missing measurement.

## 8. Architecture

`src/abaa_gui.py` is 4,357 lines; `ABAAController` alone is ~3,000 and mixes
Qt widget construction, SCPI strings, feature engineering, scikit-learn
fitting and filesystem layout. Nothing in it can be unit-tested without a Qt
event loop.

A minimal, low-risk split:

```
src/abaa/hardware.py   Keithley wrapper + demo backend behind one interface
src/abaa/features.py   the single feature definition (fixes finding 1)
src/abaa/models.py     fit / select / load / predict, no Qt imports
src/abaa/storage.py    wafer and run directory layout
src/abaa/ui/           the PyQt5 layer, importing the above
```

The first three become testable without hardware or a display.

## 9. Repository hygiene

- Measurement data and fitted models are excluded from version control and
  documented as available on request. `.gitignore` blocks `data/`,
  `runtime_data/`, `artifacts/`, `*.pkl`, `*.joblib` and `*.csv`.
- Pickle executes arbitrary code on load and breaks across scikit-learn
  versions. Where models must be exchanged, ship the fitted coefficients and
  metadata as JSON alongside, so a model remains readable when the pickle
  no longer loads.
- No tests, no CI. Both are cheap once finding 7 is done.

## Priority order

1. Unify feature computation (1) — correctness, and a precondition for 2
2. Baselines and metadata in the per-context selector (2) — credibility
3. Fix selection and CV (3) — credibility
4. `try/finally` + compliance check on the anneal loop (6) — hardware safety
5. Delete collinear clones and the substring selector (4) — correctness
6. Four-wire sensing + repeatability measurement (6) — the error budget
7. Split the module and add tests (7) — maintainability
