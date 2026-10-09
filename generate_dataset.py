"""
Creates a SYNTHETIC student performance dataset (for academic/project demonstration).
Relationships are realistic (attendance, marks, continuous assessment, CGPA help; backlogs hurt)
and random noise is added so the ML model cannot reach a perfect score.
Run:  python generate_dataset.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

N_STUDENTS = 1000
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# Independent features
attendance = np.clip(rng.normal(78, 12, N_STUDENTS), 40, 100)
current_cgpa = np.clip(rng.normal(68, 12, N_STUDENTS), 30, 98)      # Current CGPA in %

# Dependent features
continuous_assessment = np.clip(
    15 + 0.50 * current_cgpa + 0.20 * attendance
    + rng.normal(0, 8, N_STUDENTS), 20, 100)

internal_marks = np.clip(
    8 + 0.40 * current_cgpa + 0.25 * attendance + 0.15 * continuous_assessment
    + rng.normal(0, 7, N_STUDENTS), 20, 100)

assignment_score = np.clip(
    20 + 0.30 * attendance + 0.25 * current_cgpa + 0.15 * continuous_assessment
    + rng.normal(0, 8, N_STUDENTS), 20, 100)

backlog_rate = np.clip((75 - current_cgpa) / 25, 0.05, 2.0)
backlogs = np.minimum(rng.poisson(backlog_rate), 6)

# Target
final_score = np.clip(
    8 + 0.18 * attendance + 0.22 * internal_marks + 0.13 * assignment_score
    + 0.15 * continuous_assessment + 0.25 * current_cgpa - 2.5 * backlogs
    + rng.normal(0, 4, N_STUDENTS), 0, 100)

df = pd.DataFrame({
    "attendance": attendance.round(1),
    "internal_marks": internal_marks.round(1),
    "assignment_score": assignment_score.round(1),
    "continuous_assessment": continuous_assessment.round(1),
    "current_cgpa": current_cgpa.round(1),
    "backlogs": backlogs.astype(int),
    "final_score": final_score.round(1),
})

Path("data").mkdir(exist_ok=True)
df.to_csv("data/student_performance.csv", index=False)
print("Dataset saved to data/student_performance.csv | shape:", df.shape)
