from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

CSV_PATH = DATASET_DIR / "english.csv"


# ============================================================
# LOAD EXACT DATASET
# ============================================================

print("=" * 60)
print("HANDWRITTEN TEXT GENERATION - RNN TRAINING")
print("=" * 60)

df = pd.read_csv(CSV_PATH)

df["label"] = df["label"].astype(str)

print("\nDataset records:", len(df))
print("Unique characters:", df["label"].nunique())


# ============================================================
# GET ALL CHARACTERS
# ============================================================

characters = sorted(df["label"].unique().tolist())

print("\nCharacters:")
print(characters)

vocab_size = len(characters)

print("\nNumber of character classes:", vocab_size)


# ============================================================
# CREATE BALANCED CHARACTER SEQUENCE
# ============================================================

# Each character is represented equally.
# We shuffle the character order many times to avoid
# consecutive repetitions such as OOOOOOOO.

rng = np.random.default_rng(42)

sequences = []

number_of_sequences = 300

for _ in range(number_of_sequences):

    shuffled = characters.copy()

    rng.shuffle(shuffled)

    sequences.extend(shuffled)


text = "".join(sequences)

print("\nBalanced training sequence length:", len(text))


# ============================================================
# CHARACTER MAPPING
# ============================================================

char_to_int = {
    char: index
    for index, char in enumerate(characters)
}

int_to_char = {
    index: char
    for char, index in char_to_int.items()
}


# ============================================================
# ENCODE CHARACTERS
# ============================================================

encoded_text = np.array(
    [char_to_int[char] for char in text],
    dtype=np.int32
)


# ============================================================
# CREATE TRAINING SEQUENCES
# ============================================================

sequence_length = 20

X = []
y = []

for i in range(
    len(encoded_text) - sequence_length
):

    X.append(
        encoded_text[
            i:i + sequence_length
        ]
    )

    y.append(
        encoded_text[
            i + sequence_length
        ]
    )


X = np.array(X, dtype=np.int32)
y = np.array(y, dtype=np.int32)


print("\nTraining input shape:", X.shape)
print("Training output shape:", y.shape)


# ============================================================
# BUILD LSTM MODEL
# ============================================================

model = Sequential([

    Embedding(
        input_dim=vocab_size,
        output_dim=64
    ),

    LSTM(
        128,
        return_sequences=False
    ),

    Dropout(0.2),

    Dense(
        64,
        activation="relu"
    ),

    Dense(
        vocab_size,
        activation="softmax"
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


print("\n")
model.summary()


# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...")

early_stopping = EarlyStopping(
    monitor="loss",
    patience=3,
    restore_best_weights=True
)


history = model.fit(
    X,
    y,
    epochs=30,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = MODEL_DIR / "handwritten_rnn.keras"

model.save(model_path)


# ============================================================
# SAVE CHARACTER MAPPING
# ============================================================

mapping_df = pd.DataFrame({
    "index": list(int_to_char.keys()),
    "character": list(int_to_char.values())
})

mapping_path = MODEL_DIR / "character_mapping.csv"

mapping_df.to_csv(
    mapping_path,
    index=False
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(
    MODEL_DIR / "training_history.csv",
    index=False
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print("\nModel saved to:")
print(model_path)

print("\nCharacter mapping saved to:")
print(mapping_path)

print("\nFinal training accuracy:")
print(
    f"{history.history['accuracy'][-1] * 100:.2f}%"
)

print("\nFinal training loss:")
print(
    f"{history.history['loss'][-1]:.4f}"
)

print("\nRNN model is ready for generation!")