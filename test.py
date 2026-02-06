import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ======================================================
# 1. Load Data (original DataFrame remains unchanged)
# ======================================================

df = pd.read_csv("ResearchInformation3.csv")
df['Income'] = df['Income'].str.strip()


# ======================================================
# 2. Declare Column Groups
# ======================================================

categorical_cols = ["Department", "Gender", "Hometown", "Job", "Extra"]
numeric_cols = ["HSC", "SSC"]
ordinal_cols = ["Income", "Preparation", "Gaming", "Attendance", "Semester"]

# Declare ORDER for ordinal columns
income_order = [
    "Low (Below 15,000)",
    "Lower middle (15,000-30,000)",
    "Upper middle (30,000-50,000)",
    "High (Above 50,000)"
]

prep_order = ["0-1 Hour", "2-3 Hours", "More than 3 Hours"]
gaming_order = ["0-1 Hour", "2-3 Hours", "More than 3 Hours"]

attendance_order = ["Below 40%", "40%-59%", "60%-79%", "80%-100%"]

semester_order = ["1st","2nd","3rd","4th","5th","6th","7th","8th",
                  "9th","10th","11th","12th"]


ordinal_categories = [
    income_order,
    prep_order,
    gaming_order,
    attendance_order,
    semester_order
]


# ======================================================
# 3. Build Preprocessing Transformer
# ======================================================

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
        ("ord", OrdinalEncoder(categories=ordinal_categories), ordinal_cols)
    ]
)


# ======================================================
# 4. Full Pipeline
# ======================================================

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("regressor", LinearRegression())
])


# ======================================================
# 5. Train-test split
# ======================================================

X = df[categorical_cols + numeric_cols + ordinal_cols]
Y = df["Overall"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)


# ======================================================
# 6. Fit Model
# ======================================================

model.fit(X_train, Y_train)

Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)


# ======================================================
# 7. Evaluation
# ======================================================

mse_train = mean_squared_error(Y_train, Y_train_pred)
mse_test = mean_squared_error(Y_test, Y_test_pred)
r2_train = r2_score(Y_train, Y_train_pred)
r2_test = r2_score(Y_test, Y_test_pred)

print(f"Train MSE = {mse_train:.4f}, Train R^2 = {r2_train:.4f}")
print(f"Test  MSE = {mse_test:.4f}, Test  R^2 = {r2_test:.4f}")


# ======================================================
# 8. Visualization
# ======================================================

plt.scatter(Y_train, Y_train_pred, color="blue", label="Train")
plt.scatter(Y_test, Y_test_pred, color="orange", label="Test")
plt.xlabel("Actual GPA")
plt.ylabel("Predicted GPA")
plt.legend()
plt.title("Predicted GPA vs Actual GPA")
plt.show()
