"""
Model Training — Regression on World Happiness Score
Run with: python notebooks/02_model_training.py
Outputs: models/happiness_model.pkl, outputs/model_metrics.txt, outputs/fig7_predictions.png
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from src.preprocess import load_all, clean, get_feature_matrix, FEATURES

OUT    = Path(__file__).parent.parent / "outputs"
MODELS = Path(__file__).parent.parent / "models"
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

# ── Load & clean ──────────────────────────────────────────────────────────────
df = clean(load_all())
X, y, feature_names = get_feature_matrix(df)
print(f"Features used: {feature_names}")
print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features")

# ── Split 70/30 ───────────────────────────────────────────────────────────────
indices = np.arange(len(X))
idx_train, idx_test = train_test_split(indices, test_size=0.30, random_state=42)
X_train, X_test = X[idx_train], X[idx_test]
y_train, y_test = y[idx_train], y[idx_test]
print(f"Train: {len(X_train)}  Test: {len(X_test)}")

# Save test split for Kafka producer
test_export = df.iloc[idx_test].copy()
test_export.to_csv(OUT / "test_data.csv", index=False)
print(f"Test data saved ({len(test_export)} rows)")

# ── Train models ──────────────────────────────────────────────────────────────
models = {
    "LinearRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "GradientBoosting": Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                             max_depth=4, random_state=42))
    ]),
}

results = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(pipe, X, y, cv=5, scoring="r2").mean()
    results[name] = {"mae": mae, "rmse": rmse, "r2": r2, "cv_r2": cv, "pipe": pipe, "y_pred": y_pred}
    print(f"{name}: MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}  CV-R2={cv:.4f}")

# ── Pick best model (highest R2) ──────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["r2"])
best      = results[best_name]
print(f"\nBest model: {best_name} (R2={best['r2']:.4f})")

# ── Serialize ─────────────────────────────────────────────────────────────────
model_path = MODELS / "happiness_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump({"model": best["pipe"], "features": feature_names, "model_name": best_name}, f)
print(f"Model saved: {model_path}")

# ── Save metrics ──────────────────────────────────────────────────────────────
lines = [f"Best model: {best_name}", "=" * 40]
for k, v in {k: v for k, v in best.items() if k not in ("pipe", "y_pred")}.items():
    lines.append(f"{k}: {v:.4f}")
lines.append("\nAll models:")
for name, res in results.items():
    lines.append(f"  {name}: MAE={res['mae']:.4f}  R2={res['r2']:.4f}")
(OUT / "model_metrics.txt").write_text("\n".join(lines))

# ── Plot: actual vs predicted ─────────────────────────────────────────────────
y_pred_best = best["y_pred"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(y_test, y_pred_best, alpha=0.6, edgecolors="white", linewidths=0.3)
lim = [min(y_test.min(), y_pred_best.min()) - 0.2, max(y_test.max(), y_pred_best.max()) + 0.2]
axes[0].plot(lim, lim, "r--", linewidth=1.5, label="Perfect fit")
axes[0].set(title=f"Actual vs Predicted ({best_name})", xlabel="Actual", ylabel="Predicted", xlim=lim, ylim=lim)
axes[0].legend()
axes[0].text(0.05, 0.92, f"R²={best['r2']:.3f}\nMAE={best['mae']:.3f}", transform=axes[0].transAxes,
             fontsize=9, verticalalignment="top", bbox=dict(boxstyle="round", alpha=0.2))

residuals = y_test - y_pred_best
axes[1].hist(residuals, bins=30, color="steelblue", edgecolor="white")
axes[1].axvline(0, color="red", linestyle="--")
axes[1].set(title="Residual Distribution", xlabel="Residual", ylabel="Count")

plt.tight_layout()
fig.savefig(OUT / "fig7_predictions.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig7_predictions.png")

# ── Feature importance (for GradientBoosting) ────────────────────────────────
if best_name == "GradientBoosting":
    importances = best["pipe"].named_steps["model"].feature_importances_
    feat_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(feat_df["feature"], feat_df["importance"], color="steelblue")
    ax.set(title="Feature Importances (Gradient Boosting)", xlabel="Importance")
    fig.savefig(OUT / "fig8_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig8_feature_importance.png")

print("\nTraining complete.")
