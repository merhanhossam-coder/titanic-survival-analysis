# Titanic Survival Prediction (EDA + Random Forest)

This project explores the Titanic dataset and builds a machine learning model to predict passenger survival.  
It includes data cleaning, visual analysis, feature engineering, and classification using a **Random Forest** model.

---

## Project Highlights

- Exploratory Data Analysis (EDA) with visualizations
- Handling missing values (Age, Embarked) and dropping Cabin due to high missingness
- Feature engineering:
  - `Has_Family` (based on SibSp/Parch)
  - `Age_Group` (child/adult/senior)
- Model training and evaluation:
  - Train/test split
  - One-hot encoding for categorical features
  - Random Forest classifier
  - Confusion matrix + accuracy
  - Cross-validation (5-fold)
  - Max depth comparison (overfitting check)
  - Baseline comparison with Decision Tree

---

## Dataset

Source file used:
- `train.csv` (Titanic training dataset)

Target:
- `Survived` (0 = No, 1 = Yes)

---

## Visualizations Included

- Survival count plot
- Survival distribution by sex (pie chart)
- Survival rate by passenger class
- Age distribution (histogram + KDE)
- Survival rate by family presence
- Survival by fare (above vs below average fare)
- Survival rate by age group
- Correlation heatmap
- Survival counts by Embarked location
- Confusion matrix heatmap
- Max depth vs train/test accuracy plot

---

## Features Used for Modeling

Model features:
- `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`

Preprocessing:
- One-hot encoding for `Sex` and `Embarked`

---

## Results

- Random Forest accuracy (test set): **~0.81–0.82**
- 5-fold cross-validation average accuracy: **~0.81**
- Random Forest performed better than a baseline Decision Tree in this workflow.

---

## How to Run

### 1) Install dependencies
```bash
pip install -r requirements.txt
```
Create a requirements.txt with:
```bash
numpy
pandas
matplotlib
seaborn
scikit-learn
```
