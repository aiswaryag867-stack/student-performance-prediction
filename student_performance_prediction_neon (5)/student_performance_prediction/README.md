# 🎓 Student Academic Performance Prediction System

Predicts a student's **final performance score (0-100)** from six academic inputs using Python and Machine Learning.
It works like an AQI predictor: **manual input values → ML model → predicted score → performance level.**
There is no login, student database or ID lookup.

## 1. Problem Statement
Students and mentors usually learn about weak performance only after results. A data-driven tool can estimate the likely final score early from simple academic indicators.

## 2. Objective
Build a regression model that predicts the final score and a Streamlit app where a user enters values and gets a prediction, a level and recommendations.

## 3. Existing Problem
- Manual judgement is subjective and late.
- A simple fixed formula cannot capture how features interact in real data.

## 4. Proposed Solution
Train and compare Linear Regression, Decision Tree and Random Forest. The best model (by test RMSE) is saved with Joblib and loaded by the Streamlit app for instant predictions.

## 5. Technologies
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib, Streamlit.

## 6. Dataset
`data/student_performance.csv` - **1000 synthetic records** created by `generate_dataset.py` for academic/project demonstration (not real student data). Relationships are realistic (attendance, marks, continuous assessment and CGPA help; backlogs reduce the score) and random noise is added so the model is not unrealistically perfect.

| Feature | Meaning | Valid range |
|---|---|---|
| attendance | Attendance % | 0-100 |
| internal_marks | Internal assessment marks | 0-100 |
| assignment_score | Assignment score | 0-100 |
| continuous_assessment | Continuous Assessment % (class tests, quizzes, regular evaluations) | 0-100 |
| current_cgpa | Current CGPA expressed as a percentage | 0-100 |
| backlogs | Number of backlogs | 0-10 |
| **final_score** (target) | Final performance score | 0-100 |

Performance levels are **application-level rules, not ML classes**: 85-100 Excellent, 70-84 Good, 50-69 Average, below 50 Needs Improvement.

## 7. Data Science Workflow
Dataset → Understanding → Cleaning → EDA → Visualization → Feature selection → Train/Test split (80/20, random_state=42) → Training → Comparison → Evaluation → Best model → Save (Joblib) → Streamlit app.

## 8. Algorithms
- **Linear Regression** - fits a straight-line (weighted sum) relationship; features are standardised first.
- **Decision Tree** - asks a chain of yes/no questions on feature values; leaf gives the average score.
- **Random Forest** - many decision trees on random samples; predictions are averaged.

## 9. Metrics
MAE (average absolute error), MSE, RMSE (error in score points, penalises big mistakes), R² (share of variation explained), plus 5-fold cross-validated R².

## 10. System Architecture
```
generate_dataset.py → data/student_performance.csv
train_model.py → models/best_model.pkl + outputs/*.png + models/*.json/csv
app.py (Streamlit) → loads model → user inputs → prediction + level + tips + importance
```

## 11. Project Structure
```
student_performance_prediction/
├── app.py                 # Streamlit application
├── train_model.py         # EDA + training + comparison + saving
├── generate_dataset.py    # synthetic dataset creator
├── data/                  # student_performance.csv (generated)
├── models/                # best_model.pkl, model_info.json, model_comparison.csv
├── notebooks/             # student_performance_analysis.ipynb
├── outputs/               # EDA and result plots (generated)
├── report/project_report.md
├── requirements.txt
└── README.md
```

## 12. How to Run (localhost)
```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Mac/Linux

# 2. Install packages
pip install -r requirements.txt

# 3. Create the dataset and train the model (creates models/ and outputs/)
python generate_dataset.py
python train_model.py

# 4. Start the app
streamlit run app.py
```
The app opens at **http://localhost:8501**.

## 13. Results (from the reference run)
| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 3.03 | 3.88 | 0.81 |
| Decision Tree | 4.40 | 5.69 | 0.60 |
| Random Forest | 3.40 | 4.27 | 0.77 |

The best model is chosen automatically by the lowest test RMSE, so re-running may show different numbers if you change the data. **Linear Regression won here because the dataset was generated with mostly linear relationships**; on real, more complex data tree-based models may win. Example: attendance 82, internal 75, assignment 80, continuous assessment 78, current CGPA 76, backlogs 0 → about **80.3 / 100 (GOOD)**.

## 14. Future Enhancements
Real anonymised dataset, more features (participation, project work), hyperparameter tuning, SHAP explanations, deployment.

## 15. Conclusion
The project demonstrates the full Data Science and ML lifecycle and a working prediction interface built only with Python tools.

---

# Viva Preparation

**Why Python?** Simple syntax and the strongest data/ML ecosystem (Pandas, Scikit-learn).

**Where is Data Science used?** Data understanding, cleaning, EDA, visualization, modelling, evaluation.

**Why ML instead of a formula?** We used a formula only to *create* synthetic data. In reality the formula is unknown; ML learns it from examples and handles noise.

**Why regression?** The target is a continuous number (0-100). Classification predicts categories. Our levels are just if/else rules on the predicted number.

**Why train/test split?** To test on unseen data. Evaluating on training data hides overfitting. We use 80% to train, 20% to test.

**Linear Regression** - finds weights w so that score ≈ b + w1·x1 + … + w6·x6 by minimising squared errors.

**Decision Tree** - repeatedly splits data on feature thresholds that reduce error most; a leaf predicts the average of its samples. Can overfit if too deep (we limit depth).

**Random Forest** - averages many trees trained on random subsets of rows and features, which reduces overfitting.

**MAE** - average size of the error in score points (3.0 = off by 3 marks on average).

**RMSE** - square root of the mean squared error; punishes large errors more.

**R²** - fraction of score variation explained by the model; 1 is perfect, 0 is no better than predicting the mean.

**Feature importance** - how much each input contributes to predictions. For trees it comes from `feature_importances_`; for Linear Regression from the size of standardised coefficients. It is calculated from the model, not typed by hand.

**Why was the "best model" Linear Regression?** Results decide, not assumption. Our synthetic data was built from mostly linear relationships, so a linear model fits it best.

**Why Streamlit?** Builds an interactive web UI in pure Python with no HTML/JS.

**Why save the model with Joblib?** So the app loads the trained model instantly instead of retraining on every click.

**Is the data real?** No. It is synthetic, for demonstration, and this is stated openly.


## Redesigned dashboard (neon UI)
`app.py` now has a sidebar (Home / Prediction), a typed-input form, an illustration panel and a
**Prediction Insights** card (score, risk level, insights, suggestions). The model, feature order and
prediction call are unchanged. Run:

```
pip install -r requirements.txt
python generate_dataset.py
python train_model.py
python -m streamlit run app.py
```

Risk levels (Low / Medium / High) use the predicted score plus attendance and backlogs; thresholds are in
`RISK_CONFIG` at the top of `app.py`. They are estimates, not judgments about a student. Data is synthetic.

### Typing and keyboard
All six inputs are plain text boxes in a 2-column x 3-row grid, in this order: Attendance Percentage,
Internal Assessment Marks / Assignment Score, Continuous Assessment Marks / Current CGPA, Number of Backlogs.
- Arrow keys move focus to the neighbouring field (right, left, down, up) and stay put if there is none.
- Enter moves to the next field in order (Shift+Enter goes back); Enter on the last field focuses the Predict button.
- Arrow keys never change a value; Tab and Shift+Tab work as normal.
- Ctrl/Alt/Shift + arrows are left alone so text selection still works.
Percent-type fields accept 0-100, backlogs a whole number 0-10; the model features and their order are unchanged.
