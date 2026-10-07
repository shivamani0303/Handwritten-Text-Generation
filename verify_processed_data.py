from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "processed_data"


# Load processed data
X = np.load(PROCESSED_DIR / "X.npy")
y = np.load(PROCESSED_DIR / "y.npy")

label_mapping = pd.read_csv(
    PROCESSED_DIR / "label_mapping.csv"
)


print("=" * 60)
print("PROCESSED DATA VERIFICATION")
print("=" * 60)

print("\nX shape:", X.shape)
print("y shape:", y.shape)

print("\nPixel range:")
print("Minimum:", X.min())
print("Maximum:", X.max())

print("\nNumber of classes:", len(label_mapping))

print("\nLabel mapping:")
print(label_mapping.head(15))


# ------------------------------------------------------------
# DISPLAY PROCESSED IMAGES
# ------------------------------------------------------------

fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i, ax in enumerate(axes.flat):

    ax.imshow(X[i], cmap="gray")

    label_index = y[i]

    label = label_mapping.loc[
        label_mapping["index"] == label_index,
        "label"
    ].iloc[0]

    ax.set_title(f"Label: {label}")
    ax.axis("off")

plt.tight_layout()
plt.show()