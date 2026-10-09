# Cardivascular-Disease-Prediction-Using-Logistic-Regression

## Predicting 10-Year Coronary Heart Disease Risk

## Project Goal
To build a logistic regression model that predicts whether a patient will develop coronary heart disease (CHD) within the next 10 years based on demographic, lifestyle and 
clinical characteristics.

## Main Problem Question
*Can we build a reliable logistic regression model that predicts whether a patient is likely to develop coronary heart disease within 10 years using available health indicators?*

## Dataset Source
Kaggle Cardiovascular Study Dataset  
https://www.kaggle.com/datasets/christofel04/cardiovascular-study-dataset-predict-heart-disea

The dataset contains 3,390 patients with 16 features covering demographics, lifestyle factors and clinical measurements.

## Technologies Used
- Python 3.14
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook 

## Data Cleaning
We cleaned the dataset before modeling by standardizing text categories, checking for inconsistent smoking records, and removing duplicate rows. Missing values were then handled after the train-test split using median imputation fitted on training data only and applied to test data, which helps prevent data leakage. We also capped extreme outliers using IQR-based limits learned from the training set and reused those same limits on the test set for consistency.

## Key Findings
- **Diabetes** is the strongest predictor of 10-year CHD   risk (coefficient 0.912)
- **Blood pressure medication use** is the second strongest   predictor (coefficient 0.786); indicating pre-existing cardiovascular vulnerability
- **Age** and **systolic blood pressure** are equally strong predictors (correlation 0.31 each)
- **Optimal classification threshold** of 0.481 achieves sensitivity of 71.9%; catching nearly 3 in 4 CHD cases
- **ROC AUC of 0.73** confirms meaningful discriminatory ability; significantly better than random guessing

## Methodology
1. Train/test split (70/30) performed BEFORE any processing to prevent data leakage
2. Class imbalance addressed through downsampling on training data only (85/15 → 50/50)
3. Missing values imputed using median strategy fitted on training data only
4. Outliers capped using winsorization (IQR boundaries)
5. 9 features selected based on correlation analysis and multicollinearity assessment
6. Logistic regression fitted with k-fold cross-validation (mean AUC 0.71 across 5 folds)
7. Optimal threshold identified using Youden's J statistic

## Challenge: Maintaining Test Data Integrity
I initially split the dataset before conducting EDA to protect the test set. However, analysing only the training set limited my 
understanding of the full dataset. I revisited my approach and conducted EDA on the entire dataset, learning to distinguish data 
exploration from model training while maintaining test data integrity during model development and evaluation.

## Model Performance
| Metric | Value |
|--------|-------|
| Test Accuracy | 68.9% |
| ROC AUC | 0.73 |
| Recall (CHD) | 71.9% |
| Cross-validation AUC | 0.71 |

## Suggestions For Improvement
1. **Family history data**:
genetic predisposition is one of the strongest known CHD risk factors but was not available in this dataset
2. **External population validation**: testing on patients from different geographic regions and ethnicities would confirm generalisability before clinical deployment

## How To Run
1. Clone this repository
2. Install required libraries:
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
3. Place train.csv in the project folder
4. Open the notebook in Jupyter
5. Run all cells from top to bottom
