# Change Request Cycle Time Prediction

Azure ML learning project that predicts how many days a change request takes to complete. The model uses only information available before the change is resolved.

The training data is the Azure ML data asset `change-request-data`, version 1: 10,000 synthetic change requests with 44 columns. The target is `cycle_time_days`.

## Leakage policy

These columns are identifiers or are only known after approval, implementation, or closure. They are dropped before feature engineering and are not model inputs:

`cr_id`, `requester_email`, `final_approval_at`, `actual_start_at`, `actual_end_at`, `resolved_at`, `closed_at`, `closure_code`, `resolution_summary`, `post_implementation_review`, `actual_effort_hours`, `rollback_performed`, `sla_breached`, `record_last_updated_at`.

The remaining pre-resolution columns are candidate features. `created_at` and `requested_implementation_date` are turned into calendar and lead-time features. Encoders and numeric medians are fit on the training split only.

## Project layout

```
src/data/data_loader.py                  load a CSV path into a DataFrame
src/preprocessing/feature_engineering.py row-level features and missing values
src/preprocessing/data_split.py          70/15/15 split
src/preprocessing/encoding.py            train-fitted imputation and one-hot encoding
src/model/train_config.py                XGBoost hyperparameters
src/pipeline/training_pipeline.py        training orchestration
src/utils/metrics.py                     MAE, RMSE, and R²
configs/config.yaml                      seed, ratios, target, leakage columns
jobs/train.yml                           Azure ML command job
environments/conda.yaml                  job environment
```

`time` and `group` split strategies are reserved in `data_split.py` and are not implemented yet.

## Train locally

```bash
python -m pip install -r requirements.txt
python src/pipeline/training_pipeline.py \
  --data-path /path/to/change_requests.csv \
  --config configs/config.yaml
```

Artifacts are written to `outputs/model/`:

- `model.joblib`
- `preprocessor.joblib`
- `metrics.json`

Score new rows by loading `preprocessor.joblib`, transforming with it, then calling `predict` on `model.joblib`. Do not refit the preprocessor.

## Submit the Azure ML job

Set `compute` in `jobs/train.yml` to the compute instance in your workspace, then:

```bash
az ml job create --file jobs/train.yml
```

The job mounts `change-request-data:1`, runs `src/pipeline/training_pipeline.py`, and writes the model and preprocessor to the `model_artifacts` output. No secrets are stored in the repo.
