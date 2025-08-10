# Liver Disease Diagnosis Using Tree-Based Machine Learning Algorithms  

## Project Overview

This repository presents an advanced diagnostic framework for **Liver Disease Detection** utilizing tree-based machine learning classifiers. The study evaluates the performance of **Decision Tree (DT)**, **Random Forest (RF)**, and **Extreme Gradient Boosting (XGBoost)** models to accurately classify patients with or without liver disease. A comprehensive, aggregated dataset sourced from multiple repositories underpins the training and evaluation processes.

---

## Repository Structure

### Data
- **`LD_raw_data.csv`** — Raw dataset employed for model training and evaluation.

### Code
- **`preprocess.py`** — Implements data ingestion, preprocessing steps including missing value imputation, class balancing via SMOTE, feature ranking using ANOVA F-test, and data scaling.
- **`models.py`** — Defines the tree-based classifiers alongside hyperparameter grids configured for GridSearchCV optimization.
- **`train.py`** — Manages the end-to-end model training, hyperparameter tuning, and performance evaluation pipeline.

---

## Experimental Results

The proposed framework achieved the following classification accuracies:

| Model             | Accuracy  |
|-------------------|-----------|
| Decision Tree (DT) | 99.94%    |
| Random Forest (RF) | 99.95%    |
| XGBoost           | **99.97%**|

These results highlight the efficacy of tree-based ensemble methods in the context of liver disease diagnosis.

---

## Prerequisites

- Python 3.8 or above  
- Required dependencies listed in `requirements.txt`

### Installation

To install all necessary packages, execute:

```bash
pip install -r requirements.txt
```

---

## Usage
### 1. Data Preprocessing
Use the `preprocess.py` script to clean, impute missing values, balance the dataset, and perform feature ranking:
```bash
python preprocess.py
```

### 2. Train the Models
Train and evaluate models using the `train.py` script:
```bash
python train.py
```

### 3. Visualize Feature Ranking
Feature ranking visualization is part of `preprocess.py`. It generates a bar chart displaying the importance of features based on ANOVA F-test.

---
