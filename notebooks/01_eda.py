"""
EDA — World Happiness Dataset 2015-2019
Run with: python notebooks/01_eda.py
Outputs are saved to outputs/
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from pathlib import Path
from src.preprocess import load_all, clean, FEATURES, TARGET

OUT = Path(__file__).parent.parent / "outputs"
OUT.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")

# ── 1. Load ──────────────────────────────────────────────────────────────────
raw = load_all()
df  = clean(raw)
print(f"Dataset shape: {df.shape}")
print(f"Years: {sorted(df['year'].unique())}")
print(f"\nMissing values before cleaning:\n{raw.isnull().sum()}")
print(f"\nMissing values after cleaning:\n{df.isnull().sum()}")

# ── 2. Descriptive statistics ─────────────────────────────────────────────────
stats = df[[TARGET] + FEATURES].describe().T
stats.to_csv(OUT / "descriptive_stats.csv")
print("\nDescriptive statistics:\n", stats)

# ── 3. Distribution of Happiness Score per year ───────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for year, grp in df.groupby("year"):
    ax.hist(grp[TARGET], bins=20, alpha=0.5, label=str(year))
ax.set(title="Happiness Score Distribution by Year", xlabel="Happiness Score", ylabel="Count")
ax.legend()
fig.savefig(OUT / "fig1_score_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig1")

# ── 4. Mean score trend ────────────────────────────────────────────────────────
trend = df.groupby("year")[TARGET].mean().reset_index()
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(trend["year"], trend[TARGET], marker="o", linewidth=2)
ax.set(title="Mean Happiness Score 2015-2019", xlabel="Year", ylabel="Avg Score")
fig.savefig(OUT / "fig2_trend.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig2")

# ── 5. Correlation heatmap ─────────────────────────────────────────────────────
available = [f for f in FEATURES if f in df.columns]
corr = df[[TARGET] + available].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Feature Correlation Matrix")
fig.savefig(OUT / "fig3_correlation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig3")

# ── 6. Scatter: GDP vs Happiness ──────────────────────────────────────────────
if "gdp_per_capita" in df.columns:
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(df["gdp_per_capita"], df[TARGET], c=df["year"],
                         cmap="plasma", alpha=0.6, edgecolors="white", linewidths=0.3)
    plt.colorbar(scatter, ax=ax, label="Year")
    ax.set(title="GDP per Capita vs Happiness Score", xlabel="GDP per Capita", ylabel="Happiness Score")
    fig.savefig(OUT / "fig4_gdp_vs_score.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig4")

# ── 7. Box plots per feature ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, feat in enumerate(available[:6]):
    sns.boxplot(data=df, x="year", y=feat, ax=axes[i])
    axes[i].set_title(feat)
    axes[i].set_xlabel("")
plt.suptitle("Feature Distributions by Year", fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(OUT / "fig5_boxplots.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig5")

# ── 8. Top/bottom countries (2019) ────────────────────────────────────────────
df19 = df[df["year"] == 2019].nlargest(10, TARGET)[["country", TARGET]]
df19b = df[df["year"] == 2019].nsmallest(10, TARGET)[["country", TARGET]]
combined_rank = pd.concat([df19, df19b])

fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#2ecc71"] * 10 + ["#e74c3c"] * 10
ax.barh(combined_rank["country"], combined_rank[TARGET], color=colors)
ax.axvline(df[TARGET].mean(), linestyle="--", color="gray", label="Global mean")
ax.set(title="Top 10 Happiest & Least Happy Countries (2019)", xlabel="Happiness Score")
ax.legend()
fig.savefig(OUT / "fig6_top_bottom_countries.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig6")

# Save cleaned dataset
df.to_csv(OUT / "combined.csv", index=False)
print(f"\nAll outputs saved to {OUT}")
print("EDA complete.")
