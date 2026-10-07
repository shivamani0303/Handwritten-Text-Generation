from pathlib import Path
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent

csv_path = BASE_DIR / "dataset" / "english.csv"
df = pd.read_csv(csv_path)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i, ax in enumerate(axes.flat):
    image_path = BASE_DIR / "dataset" / df.iloc[i]["image"]

    img = Image.open(image_path)

    ax.imshow(img)
    ax.set_title(f"Label: {df.iloc[i]['label']}")
    ax.axis("off")

plt.tight_layout()
plt.show()