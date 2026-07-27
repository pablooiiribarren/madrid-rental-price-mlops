# Madrid Rental Price Prediction — End-to-End MLOps on AWS

Production deployment of a rental-price regression model on **AWS SageMaker**, covering the full machine learning lifecycle: data ingestion and ETL, training, hyperparameter tuning, bias auditing, model registry, real-time inference and continuous monitoring.

Built as the final project of the *Professional Programme in Artificial Intelligence & Data Science* (UNIR, 2026), following the CRISP-DM methodology.

---

## Architecture

```
Raw data (S3 /raw)
        │
        ▼
AWS Glue — PySpark ETL ──────► Glue Data Catalog
        │
        ▼
Processed data (S3 /processed)
        │
        ▼
SageMaker Training ──► Automatic Model Tuning (Bayesian)
        │
        ▼
SageMaker Clarify (bias) ──► Model Registry (versioning)
        │
        ▼
Real-time inference endpoint (ml.m5.large)
        │
        ▼
Model Monitor + CloudWatch ──► SageMaker Pipelines (retraining)
```

| Lifecycle stage | AWS services |
|---|---|
| Ingestion & preprocessing | S3, AWS Glue (PySpark), Glue Data Catalog |
| Training & tuning | SageMaker Training, Experiments, Automatic Model Tuning |
| Evaluation & registry | SageMaker Clarify, Model Registry |
| Deployment | SageMaker Hosting (real-time endpoint) |
| Monitoring | Model Monitor, CloudWatch, SageMaker Pipelines |

---

## Results

| Model | R² | Notes |
|---|---|---|
| GradientBoosting (baseline) | 0.7191 | Default hyperparameters |
| GradientBoosting (tuned) | **0.7874** | Bayesian tuning, 4 parallel jobs |

Automatic Model Tuning improved explanatory power by **9.5%** without changing a single line of model code, exploring the hyperparameter space in ~6 minutes through Bayesian optimisation rather than exhaustive grid search.

Baseline test metrics: MAE `[447.61]` € · RMSE `[844.93]` € · R² 0.7191.

> **Note on scale:** the target variable spans a wide price range, so absolute errors in euros should be read alongside R² rather than in isolation.

---

## Dataset

*Madrid Province Rent Data* — thousands of rental listings with physical, location and amenity features.
Source: [`[enlace a Kaggle]`]([URL])

The dataset is **not** included in this repository. To reproduce, download it and upload to `s3://<your-bucket>/raw/`.

---

## Repository structure

```
├── glue/
│   └── etl_job.py                  # PySpark ETL job (cleaning, typing, feature prep)
├── sagemaker/
│   ├── train.py                    # training script executed by SageMaker Training
│   ├── 01_training_and_tuning.ipynb
│   ├── 02_clarify_and_registry.ipynb
│   ├── 03_deploy_endpoint.ipynb
│   └── 04_monitoring_pipeline.ipynb
├── docs/images/                    # console screenshots
└── requirements.txt
```

---

## Pipeline walkthrough

### 1. Ingestion and ETL — AWS Glue

A PySpark job reads raw listings from `s3://<bucket>/raw/`, handles nulls and type inconsistencies, prepares the feature set and writes the processed dataset to `s3://<bucket>/processed/`. Schema is registered in the Glue Data Catalog.

`[1-2 frases: qué transformaciones concretas hace tu ETL — nulos, encoding, columnas derivadas]`

### 2. Training and tuning — SageMaker

`train.py` trains a GradientBoosting regressor on the processed data. Runs are grouped under a **SageMaker Experiment** (`alquiler-madrid-precio`) so metrics are comparable across configurations.

**Automatic Model Tuning** then launches parallel training jobs with a Bayesian search strategy, maximising R². Best configuration: `[hiperparámetros ganadores]`.

![Tuning jobs](docs/images/tuning-jobs.png)

### 3. Bias auditing — SageMaker Clarify

A pre-training bias analysis was configured with `district` as the facet and a €500/month target threshold, checking whether the training data encodes geographic price bias before the model learns it.

`[1 frase: qué encontró el análisis — ¿hubo desequilibrio entre distritos?]`

This step reflects **EU AI Act (Regulation 2024/1689)** data governance requirements: housing-access systems fall under Annex III, and bias examination is an explicit obligation under Art. 10.

### 4. Registry and deployment

The best model is versioned in the **Model Registry** for traceability, then deployed to a real-time inference endpoint (`alquiler-madrid-endpoint`) on an `ml.m5.large` instance, with an `AllTraffic` production variant that allows future A/B or canary deployments by adding variants.

![Endpoint InService](docs/images/endpoint-inservice.png)

Test inference returned a price estimate consistent with real market values, confirming the endpoint works end to end.

### 5. Monitoring and retraining

**Model Monitor** generates a statistical baseline in S3 (`constraints.json`, `statistics.json`) used as the reference for detecting future data drift. A monitoring schedule runs on `[frecuencia]`, logs are centralised in **CloudWatch**, and **SageMaker Pipelines** automates retraining.

![Model Monitor](docs/images/model-monitor.png)

---

## Engineering constraints

Development ran on a restricted AWS lab environment with limited IAM permissions. This forced explicit architectural decisions — reusing an existing execution role instead of provisioning a scoped one, and configuring public internet access for the networking layer — that mirror common constraints in enterprise environments where permissions are centrally controlled.

All resources were torn down after each session to avoid idle billing.

---

## Reproducing

```bash
pip install -r requirements.txt
```

1. Create an S3 bucket with `raw/`, `processed/` and `models/` prefixes.
2. Upload the dataset to `raw/`.
3. Run the Glue job in `glue/etl_job.py`.
4. Execute the SageMaker notebooks in numerical order.

Requires an AWS account with SageMaker, Glue and S3 permissions.

---

## What this project covers

- PySpark ETL at scale on managed infrastructure
- Experiment tracking and reproducible training jobs
- Bayesian hyperparameter optimisation
- Fairness auditing under EU AI Act requirements
- Model versioning and lineage
- Real-time inference serving
- Data drift detection and automated retraining

---

## Related work

This repository is one phase of a four-part AI system for the Madrid property market. See also:
**[NLP Model Benchmark](`[enlace al otro repo]`)** — comparing TF-IDF + SVR, Word2Vec + CNN and a fine-tuned Spanish transformer (BERTIN-RoBERTa) for price prediction from listing text alone.

---

**Pablo Iribarren** · [LinkedIn](https://www.linkedin.com/in/pablo-iribarren-muru) · p11iribarren@gmail.com
