"""Prompt 1: Price vs Energy Return scatter + regression trendline"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("running_shoes.csv").rename(columns={
    "price_usd": "price", "energy_return_pct": "energy_return", "weight_g": "weight"})

sns.set_theme(style="whitegrid", context="talk", palette="deep")
fig, ax = plt.subplots(figsize=(12, 8))

sns.regplot(data=df, x="price", y="energy_return", scatter=False, ci=95,
            line_kws={"color": "#2b2b2b", "lw": 2.5, "ls": "--"}, ax=ax)
sns.scatterplot(data=df, x="price", y="energy_return", hue="brand", size="weight",
                sizes=(40, 400), alpha=0.75, edgecolor="white", linewidth=0.6, ax=ax)

ax.set_title("Running Shoes: Price vs. Energy Return", fontsize=20, weight="bold", pad=15)
ax.set_xlabel("Price (USD)")
ax.set_ylabel("Energy Return (%)")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=11)
sns.despine()
plt.tight_layout()
plt.savefig("01_price_vs_energy_return.png", dpi=200, bbox_inches="tight")
plt.show()
