from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "handwritten_rnn.keras"
MAPPING_PATH = MODEL_DIR / "character_mapping.csv"

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("=" * 60)
print("HANDWRITTEN TEXT GENERATION")
print("=" * 60)

print("\nLoading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

mapping_df = pd.read_csv(MAPPING_PATH)

char_to_int = {
    row["character"]: int(row["index"])
    for _, row in mapping_df.iterrows()
}

int_to_char = {
    int(row["index"]): row["character"]
    for _, row in mapping_df.iterrows()
}

print("Model loaded successfully!")

# --------------------------------------------------
# GENERATE TEXT
# --------------------------------------------------

def generate_text(seed_text, length=30):

    seed_text = str(seed_text)

    # Keep only characters known by the model
    seed_text = "".join(
        char for char in seed_text
        if char in char_to_int
    )

    if not seed_text:
        print("Seed contains no characters from the dataset.")
        return ""

    generated_text = seed_text

    sequence_length = model.input_shape[1]

    # If seed is shorter than required sequence length,
    # repeat the seed until it is long enough.
    while len(seed_text) < sequence_length:
        seed_text = seed_text + seed_text

    seed_text = seed_text[-sequence_length:]

    for _ in range(length):

        encoded = [
            char_to_int[char]
            for char in seed_text
        ]

        input_sequence = np.array(
            [encoded],
            dtype=np.int32
        )

        prediction = model.predict(
            input_sequence,
            verbose=0
        )[0]

        next_index = np.argmax(prediction)

        next_char = int_to_char[next_index]

        generated_text += next_char

        seed_text = seed_text[1:] + next_char

    return generated_text


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

print("\nAvailable characters:")
print("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

seed = input("\nEnter a starting character or word: ").strip().upper()

if not seed:
    seed = "HELLO"

generated = generate_text(
    seed,
    length=30
)

print("\n" + "=" * 60)
print("GENERATED TEXT")
print("=" * 60)

print("\n" + generated)

print("\n" + "=" * 60)
print("GENERATION COMPLETED")
print("=" * 60)