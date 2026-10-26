"""Prompt A: synthetic running-shoe dataset (200 rows) -> running_shoes.csv"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(404)
N = 200

brands = {
    "Nike": ["Pegasus", "Vaporfly", "Invincible", "Vomero"],
    "Adidas": ["Adizero", "Ultraboost", "Boston", "Supernova"],
    "Asics": ["Novablast", "Gel-Nimbus", "Metaspeed", "Gel-Kayano"],
    "Brooks": ["Ghost", "Glycerin", "Hyperion", "Launch"],
    "Hoka": ["Clifton", "Bondi", "Mach", "Rocket X"],
    "New Balance": ["1080", "FuelCell", "880", "Rebel"],
    "Saucony": ["Endorphin", "Ride", "Kinvara", "Triumph"],
}
brand_bonus = {"Nike": 1.5, "Adidas": 1.0, "Asics": 0.5, "Brooks": 0.0,
               "Hoka": 1.0, "New Balance": 0.5, "Saucony": 1.0}

brand = rng.choice(list(brands), N)
model = [f"{rng.choice(brands[b])} {rng.integers(1, 16)}" for b in brand]
price = np.round(rng.uniform(50, 250, N), 2)
weight = np.round(np.clip(rng.normal(265, 40, N) - 0.1 * (price - 150), 180, 350), 1)

# Energy return: steep gain up to $150, flat-ish (diminishing returns) above
base = 52 + 0.16 * (np.minimum(price, 150) - 50) + 0.03 * np.maximum(price - 150, 0)
energy = base + np.array([brand_bonus[b] for b in brand]) + rng.normal(0, 2.5, N)
energy = np.round(np.clip(energy, 50, 85), 1)

cushion = 30 + 0.22 * price + 0.15 * (weight - 180) + rng.normal(0, 7, N)
cushion = np.round(np.clip(cushion, 1, 100), 1)

df = pd.DataFrame({
    "shoe_id": [f"S{i:03d}" for i in range(1, N + 1)],
    "brand": brand,
    "model_name": model,
    "price_usd": price,
    "weight_g": weight,
    "cushioning_score": cushion,
    "energy_return_pct": energy,
    "grip_index": np.round(np.clip(rng.normal(6, 1.8, N), 1, 10), 1),
    # extra columns needed by the Random Forest prompt
    "midsole_foam_density": np.round(np.clip(0.12 + 0.0004 * (price - 50) + rng.normal(0, 0.02, N), 0.08, 0.30), 3),
    "stack_height_mm": np.round(np.clip(rng.normal(34, 5, N), 22, 48), 1),
})
df.to_csv("running_shoes.csv", index=False)
print(df.head(), f"\nSaved {len(df)} rows -> running_shoes.csv")
