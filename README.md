# NFL Draft Prediction

A machine learning project that predicts whether a player will be selected in the NFL Draft using athletic test results, physical measurements, school, and position information.

The project combines exploratory data analysis, missing-value handling, categorical features, domain-inspired feature engineering, and **CatBoost** classification evaluated with stratified cross-validation.

## Problem

The target is:

```text
Drafted
```

where the model estimates the probability that a player will be selected in the NFL Draft.

## Dataset Snapshot

The recorded experiment used:

- **Training set:** 2,781 rows × 16 columns
- **Test set:** 696 rows × 15 columns
- **Target distribution:** 64.83% drafted, 35.17% not drafted

Features include:

- Age
- Height
- Weight
- 40-yard sprint time
- Vertical jump
- Bench press repetitions
- Broad jump
- 3-cone agility
- Shuttle
- School
- Player type
- Position type
- Position

The competition data itself is not committed to this repository. Place the required CSV files inside an `input/` folder to run the notebook.

## Key EDA Observations

The original analysis found:

- `Agility_3cone` and `Shuttle` were strongly positively correlated (about **0.88**).
- `Sprint_40yd` and `Broad_Jump` were strongly negatively correlated (about **-0.82**).

The sprint relationship makes sense because **lower sprint time is better**, while a larger broad jump is better.

## Preprocessing

The notebook:

- drops the identifier column
- keeps `School` as a categorical feature
- adds school-frequency information
- adds missing-value indicator columns
- imputes missing physical measurements using position-based means with a global fallback
- passes categorical features directly to CatBoost

## Feature Engineering

The enhanced model adds:

- `BMI`
- `Weight_per_Height`
- `Speed_Score`
- `Explosion_Score`
- `Agility_Score`
- `Strength_to_Weight`

## Model

The main model is a `CatBoostClassifier` evaluated with **5-fold stratified cross-validation** using ROC-AUC.

### Recorded Results

| Experiment | 5-Fold Validation ROC-AUC |
| --- | ---: |
| CatBoost baseline | **0.8359** |
| CatBoost + engineered features | **0.8391** |

The engineered features improved the recorded average validation AUC by approximately **0.0032**.

## Repository Structure

```text
NFL-Prediction/
├── nfl_draft_prediction.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/emanbarakat/NFL-Prediction.git
cd NFL-Prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the data

Create an `input/` directory containing:

```text
input/
├── train.csv
├── test.csv
└── sample_submission.csv
```

### 4. Open the notebook

```text
nfl_draft_prediction.ipynb
```

## Tech Stack

- Python
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- CatBoost
- Jupyter Notebook

## Possible Next Improvements

- Fold-local preprocessing
- Hyperparameter optimization
- LightGBM/XGBoost comparison
- Position-specific features
- Probability calibration
- SHAP explanations
- Repeated cross-validation

## Author

**Eman Barakat**

AI & Software Development | Data Analysis | Machine Learning
