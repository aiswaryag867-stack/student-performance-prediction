"""
Full Data Science + ML pipeline:
 load data -> understand -> clean -> EDA plots -> split -> train 3 models
 -> compare -> pick best (lowest test RMSE) -> save model + metrics + importance.
Run:  python train_model.py
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")                      # draw plots to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

DATA_PATH = Path("data/student_performance.csv")
MODEL_DIR, OUT_DIR = Path("models"), Path("outputs")
MODEL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
TARGET = "final_score"
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------- 1. LOAD
if not DATA_PATH.exists():
    import generate_dataset  # noqa: F401  (creates the CSV)
df = pd.read_csv(DATA_PATH)

# ---------------------------------------------------------------- 2. UNDERSTAND
print("Shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
print("\nDescriptive statistics:\n", df.describe().round(2))

# ---------------------------------------------------------------- 3. CLEAN
before = len(df)
df = df.drop_duplicates()
df = df.fillna(df.median(numeric_only=True))      # no-op here, safe for real data
for col in ["attendance", "internal_marks", "assignment_score",
            "continuous_assessment", "current_cgpa", "final_score"]:
    df[col] = df[col].clip(0, 100)
df["backlogs"] = df["backlogs"].clip(lower=0)
print(f"\nCleaning: {before} -> {len(df)} rows")

features = ["attendance", "internal_marks", "assignment_score",
            "continuous_assessment", "current_cgpa", "backlogs"]
X, y = df[features], df[TARGET]

# ---------------------------------------------------------------- 4. EDA PLOTS
df[features + [TARGET]].hist(bins=25, figsize=(12, 8), color="#4C72B0")
plt.suptitle("Feature Distributions")
plt.tight_layout(); plt.savefig(OUT_DIR / "distributions.png", dpi=120); plt.close()

plt.figure(figsize=(12, 5))
sns.boxplot(data=df[["attendance", "internal_marks", "assignment_score",
                     "continuous_assessment", "current_cgpa", "final_score"]], palette="Set2")
plt.title("Box Plot (outlier check)")
plt.tight_layout(); plt.savefig(OUT_DIR / "boxplot.png", dpi=120); plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout(); plt.savefig(OUT_DIR / "correlation_heatmap.png", dpi=120); plt.close()

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, col in zip(axes.ravel(), features):
    sns.scatterplot(data=df, x=col, y=TARGET, alpha=0.4, ax=ax)
    ax.set_title(f"{col} vs {TARGET}")
plt.tight_layout(); plt.savefig(OUT_DIR / "scatter_plots.png", dpi=120); plt.close()

# ---------------------------------------------------------------- 5. SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE)
print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows")

# ---------------------------------------------------------------- 6. MODELS
models = {
    "Linear Regression": Pipeline([("scaler", StandardScaler()),
                                   ("model", LinearRegression())]),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, min_samples_leaf=10,
                                           random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=8,
                                           min_samples_leaf=5,
                                           random_state=RANDOM_STATE, n_jobs=-1),
}

rows, fitted = [], {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    rows.append({"Model": name,
                 "MAE": mean_absolute_error(y_test, pred),
                 "MSE": mse,
                 "RMSE": float(np.sqrt(mse)),
                 "R2": r2_score(y_test, pred),
                 "CV R2 (5-fold)": cv.mean()})
    fitted[name] = model

results = pd.DataFrame(rows).round(4)
print("\nMODEL COMPARISON\n", results.to_string(index=False))

# ---------------------------------------------------------------- 7. BEST MODEL
best_name = results.sort_values("RMSE").iloc[0]["Model"]    # lowest error wins
best_model = fitted[best_name]
print(f"\nBest model (lowest test RMSE): {best_name}")

# ---------------------------------------------------------------- 8. IMPORTANCE
if hasattr(best_model, "feature_importances_"):
    imp = best_model.feature_importances_
    method = "Tree feature_importances_ (impurity reduction)"
else:  # linear model -> size of standardized coefficients
    imp = np.abs(best_model.named_steps["model"].coef_)
    method = "Absolute standardized regression coefficients"
imp = imp / imp.sum() * 100
importance = dict(sorted(zip(features, imp.round(2).tolist()),
                         key=lambda kv: kv[1], reverse=True))

plt.figure(figsize=(8, 4))
sns.barplot(x=list(importance.values()), y=list(importance.keys()), color="#4C72B0")
plt.xlabel("Importance (%)"); plt.title(f"Feature Importance - {best_name}")
plt.tight_layout(); plt.savefig(OUT_DIR / "feature_importance.png", dpi=120); plt.close()

# model comparison chart
ax = results.set_index("Model")[["MAE", "RMSE"]].plot(kind="bar", figsize=(8, 4), rot=0)
ax.set_title("Model Comparison (lower is better)")
plt.tight_layout(); plt.savefig(OUT_DIR / "model_comparison.png", dpi=120); plt.close()

# actual vs predicted
pred = best_model.predict(X_test)
plt.figure(figsize=(5.5, 5.5))
plt.scatter(y_test, pred, alpha=0.4)
lo, hi = float(y.min()), float(y.max())
plt.plot([lo, hi], [lo, hi], "r--")
plt.xlabel("Actual score"); plt.ylabel("Predicted score")
plt.title(f"Actual vs Predicted ({best_name})")
plt.tight_layout(); plt.savefig(OUT_DIR / "actual_vs_predicted.png", dpi=120); plt.close()

# ---------------------------------------------------------------- 9. SAVE
joblib.dump(best_model, MODEL_DIR / "best_model.pkl")
results.to_csv(MODEL_DIR / "model_comparison.csv", index=False)
with open(MODEL_DIR / "model_info.json", "w") as f:
    json.dump({"best_model": best_name,
               "features": features,
               "importance": importance,
               "importance_method": method}, f, indent=2)
print("\nSaved: models/best_model.pkl, model_comparison.csv, model_info.json and plots in outputs/")
