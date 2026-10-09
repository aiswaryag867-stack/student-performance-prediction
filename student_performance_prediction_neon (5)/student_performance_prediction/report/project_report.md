# Student Academic Performance Prediction System using Python and Machine Learning

## Abstract
This project predicts a student's final performance score (0-100) from attendance, internal marks, assignment score, continuous assessment, current CGPA and number of backlogs. Three regression models (Linear Regression, Decision Tree, Random Forest) were trained and compared using MAE, MSE, RMSE and R². The best model was saved and integrated into a Streamlit application where users enter values manually and receive a predicted score, a performance level and recommendations.

## 1. Introduction
Academic performance depends on several measurable factors. Machine learning can learn how these factors combine to influence the final result.

## 2. Problem Statement
Performance problems are usually noticed only after final results. There is a need for a simple tool that estimates the likely final score from early academic indicators.

## 3. Objectives
- Build a regression model for final score prediction.
- Compare multiple algorithms and select the best using evaluation metrics.
- Provide a Streamlit interface with manual input and instant prediction.
- Explain the prediction using feature importance.

## 4. Literature / Background
Educational data mining studies show that attendance, previous results and continuous assessment are strong indicators of academic outcomes. Regression models are widely used to predict numeric performance scores, while tree-based ensembles are commonly used for non-linear relationships.

## 5. Methodology
Dataset creation → understanding → cleaning → EDA → feature selection → 80/20 train-test split → training → comparison → evaluation → best-model selection → saving with Joblib → Streamlit application.

## 6. Dataset
1000 synthetic records created for academic demonstration (not real student data). Six input features (attendance, internal marks, assignment score, continuous assessment, current CGPA, backlogs) and one target (final_score). Meaningful relationships and Gaussian noise were included so the dataset behaves realistically.

## 7. Data Preprocessing
Checked shape, data types, missing values and duplicates; clipped values to valid ranges; standardised features for Linear Regression using StandardScaler inside a Pipeline.

## 8. Exploratory Data Analysis
Histograms show feature distributions; box plots check outliers; the correlation heatmap shows that internal marks, current CGPA, continuous assessment and assignment score correlate strongly with the final score, attendance moderately, and backlogs negatively; scatter plots confirm these trends.

## 9. Machine Learning
Linear Regression, Decision Tree (max depth 5) and Random Forest (200 trees, max depth 8) were trained with random_state = 42 and evaluated on a 20% test set plus 5-fold cross-validation.

## 10. Model Comparison
| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 3.03 | 3.88 | 0.81 |
| Decision Tree | 4.40 | 5.69 | 0.60 |
| Random Forest | 3.40 | 4.27 | 0.77 |

The model with the lowest RMSE was selected automatically. Linear Regression performed best because the synthetic data has mostly linear relationships.

## 11. Results
The selected model predicts the final score with an average error of about 3 marks. Feature importance (from the model) ranks current CGPA, internal marks and attendance highest.

## 12. Streamlit Application
Single page: manual inputs with validation, Predict button, predicted score, performance level (Excellent / Good / Average / Needs Improvement as application rules) and recommendations.

## 13. Conclusion
The project delivers a complete Python data science workflow and a working localhost prediction tool.

## 14. Future Scope
Real anonymised data, more features, hyperparameter tuning, SHAP explanations and deployment.
