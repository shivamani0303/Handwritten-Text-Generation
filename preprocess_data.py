from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

CSV_PATH = BASE_DIR / "dataset" / "english.csv"
OUTPUT_DIR = BASE_DIR / "processed_data"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (64, 64)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("HANDWRITTEN CHARACTER DATA PREPROCESSING")
print("=" * 60)

df = pd.read_csv(CSV_PATH)

print("\nDataset loaded successfully.")
print("Total images:", len(df))
print("Unique labels:", df["label"].nunique())


# ============================================================
# PREPARE LABELS
# ============================================================

labels = sorted(df["label"].astype(str).unique())

label_to_index = {
    label: index
    for index, label in enumerate(labels)
}

index_to_label = {
    index: label
    for label, index in label_to_index.items()
}


# ============================================================
# PROCESS IMAGES
# ============================================================

images = []
encoded_labels = []

print("\nProcessing images...")

for i, row in df.iterrows():

    image_path = BASE_DIR / "dataset" / row["image"]

    try:
        # Open image
        img = Image.open(image_path)

        # Convert RGB to grayscale
        img = img.convert("L")

        # Resize
        img = img.resize(IMAGE_SIZE)

        # Convert to NumPy array
        img_array = np.array(img, dtype=np.float32)

        # Normalize pixel values from 0-255 to 0-1
        img_array = img_array / 255.0

        images.append(img_array)

        label = str(row["label"])
        encoded_labels.append(label_to_index[label])

    except Exception as e:
        print(f"Error processing {image_path}: {e}")


# ============================================================
# CONVERT TO NUMPY ARRAYS
# ============================================================

X = np.array(images, dtype=np.float32)
y = np.array(encoded_labels, dtype=np.int32)


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

np.save(OUTPUT_DIR / "X.npy", X)
np.save(OUTPUT_DIR / "y.npy", y)

# Save label mapping
label_mapping = pd.DataFrame({
    "index": list(index_to_label.keys()),
    "label": list(index_to_label.values())
})

label_mapping.to_csv(
    OUTPUT_DIR / "label_mapping.csv",
    index=False
)


# ============================================================
# DISPLAY INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)

print("\nProcessed image shape:", X.shape)
print("Label shape:", y.shape)

print("\nImage size:", IMAGE_SIZE)
print("Pixel value range:", X.min(), "to", X.max())

print("\nNumber of classes:", len(labels))

print("\nFirst 10 labels:")
print(labels[:10])

print("\nSaved files:")

print("1.", OUTPUT_DIR / "X.npy")
print("2.", OUTPUT_DIR / "y.npy")
print("3.", OUTPUT_DIR / "label_mapping.csv")

print("\nDone!")