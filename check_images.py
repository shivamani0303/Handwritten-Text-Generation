from pathlib import Path
from PIL import Image
import pandas as pd

BASE_DIR = Path(__file__).parent

csv_path = BASE_DIR / "dataset" / "english.csv"
df = pd.read_csv(csv_path)

print("=" * 50)
print("IMAGE CHECK")
print("=" * 50)

for i in range(5):
    image_path = BASE_DIR / "dataset" / df.iloc[i]["image"]

    print(f"\nImage {i + 1}")
    print("Path:", image_path)
    print("Exists:", image_path.exists())

    if image_path.exists():
        img = Image.open(image_path)

        print("Size:", img.size)
        print("Mode:", img.mode)