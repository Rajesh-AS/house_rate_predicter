import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_excel("realistic_housing_data.xlsx")

X = df.drop("price", axis=1)
y = df["price"]

# ----------------------------
# Feature types
# ----------------------------
num_features = [
    "bedrooms", "bathrooms", "sqft", "lot_size", "age",
    "year_built", "garage", "condition", "school_rating",
    "has_pool", "has_fireplace", "has_basement"
]
cat_features = ["location", "house_type"]

# ----------------------------
# Preprocessing
# ----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
    ]
)

# ----------------------------
# Model (smaller forest)
# ----------------------------
model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=60,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

# ----------------------------
# Train/Test
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

# ----------------------------
# Evaluation
# ----------------------------
y_pred = model.predict(X_test)
print("R²:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# ----------------------------
# Save with compression
# ----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/house_price_model.pkl", compress=3)
print("Model saved with compression")
