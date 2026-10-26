"""Prompt 3: Random Forest feature importance (target: cushioning_score)"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("running_shoes.csv").rename(columns={
    "grip_index": "outsole_grip_index", "stack_height_mm": "stack_height"})
FEATURES = ["price_usd", "weight_g", "midsole_foam_density", "outsole_grip_index", "stack_height"]

X_train, X_test, y_train, y_test = train_test_split(
    df[FEATURES], df["cushioning_score"], test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=300, random_state=42).fit(X_train, y_train)
print(f"Test R²: {r2_score(y_test, rf.predict(X_test)):.3f}")

imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()  # ascending -> top bar = highest
sns.set_theme(style="whitegrid", context="talk")
plt.figure(figsize=(10, 6))
imp.plot(kind="barh", color=sns.color_palette("viridis", len(imp)))
plt.title("Feature Importance: Predicting Cushioning Score", weight="bold")
plt.xlabel("Importance")
plt.tight_layout(); plt.savefig("03_feature_importance.png", dpi=200); plt.show()
