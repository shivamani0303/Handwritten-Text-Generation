from pathlib import Path
import random
import pandas as pd
from PIL import Image, ImageOps, ImageDraw

BASE_DIR = Path(__file__).parent

DATASET_DIR = BASE_DIR / "dataset"
CSV_PATH = DATASET_DIR / "english.csv"
OUTPUT_DIR = BASE_DIR / "generated_output"

OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

print("=" * 60)
print("HANDWRITTEN TEXT IMAGE GENERATION")
print("=" * 60)

df = pd.read_csv(CSV_PATH)

df["label"] = df["label"].astype(str)

print("\nDataset loaded successfully!")
print("Total handwritten samples:", len(df))
print("Unique characters:", df["label"].nunique())

# --------------------------------------------------
# CREATE CHARACTER IMAGE DATABASE
# --------------------------------------------------

character_images = {}

for _, row in df.iterrows():

    label = str(row["label"])
    image_path = DATASET_DIR / row["image"]

    if image_path.exists():

        if label not in character_images:
            character_images[label] = []

        character_images[label].append(image_path)

print("Character image database created!")

# --------------------------------------------------
# TEXT TO GENERATE
# --------------------------------------------------

text = input(
    "\nEnter text using characters from the dataset: "
).strip().upper()

if not text:
    text = "HELLO"

print("\nText:", text)

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

CHAR_WIDTH = 100
CHAR_HEIGHT = 100
SPACING = 10

# --------------------------------------------------
# CREATE HANDWRITTEN TEXT
# --------------------------------------------------

character_images_output = []

for character in text:

    # Ignore spaces for now
    if character == " ":
        continue

    if character not in character_images:

        print(
            f"Warning: No handwritten image found for '{character}'"
        )

        continue

    # Pick a random handwritten sample
    image_path = random.choice(
        character_images[character]
    )

    img = Image.open(image_path).convert("L")

    # Resize while maintaining a consistent output size
    img = ImageOps.contain(
        img,
        (CHAR_WIDTH, CHAR_HEIGHT)
    )

    # Create white character canvas
    canvas = Image.new(
        "L",
        (CHAR_WIDTH, CHAR_HEIGHT),
        255
    )

    # Center character
    x = (CHAR_WIDTH - img.width) // 2
    y = (CHAR_HEIGHT - img.height) // 2

    canvas.paste(img, (x, y))

    character_images_output.append(canvas)

# --------------------------------------------------
# CHECK RESULT
# --------------------------------------------------

if not character_images_output:

    print("\nNo valid characters found.")
    exit()

# --------------------------------------------------
# CREATE FINAL CANVAS
# --------------------------------------------------

total_width = (
    len(character_images_output) * CHAR_WIDTH
    + (len(character_images_output) - 1) * SPACING
)

final_image = Image.new(
    "L",
    (total_width, CHAR_HEIGHT),
    255
)

# --------------------------------------------------
# JOIN CHARACTERS
# --------------------------------------------------

current_x = 0

for char_img in character_images_output:

    final_image.paste(
        char_img,
        (current_x, 0)
    )

    current_x += CHAR_WIDTH + SPACING

# --------------------------------------------------
# SAVE IMAGE
# --------------------------------------------------

output_path = OUTPUT_DIR / "generated_handwritten_text.png"

final_image.save(output_path)

print("\n" + "=" * 60)
print("HANDWRITTEN IMAGE GENERATED")
print("=" * 60)

print("\nSaved to:")
print(output_path)

print("\nGeneration completed successfully!")