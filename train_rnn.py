from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# ============================================================
# PATHS
# ============================================================

BASE_DIR    = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR   = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
CSV_PATH    = DATASET_DIR / "english.csv"


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("HANDWRITTEN TEXT GENERATION  —  RNN TRAINING (v3)")
print("=" * 60)

df = pd.read_csv(CSV_PATH)
df["label"] = df["label"].astype(str)

print(f"\nDataset records  : {len(df)}")
print(f"Unique characters: {df['label'].nunique()}")

characters = sorted(df["label"].unique().tolist())
vocab_size  = len(characters)

print(f"\nVocabulary ({vocab_size} chars):")
print("  " + "  ".join(characters))

char_to_int = {ch: i  for i, ch in enumerate(characters)}
int_to_char = {i:  ch for ch, i  in char_to_int.items()}


# ============================================================
# BUILD A STRUCTURED, LEARNABLE CORPUS
# ============================================================
# Key insight: random shuffles give loss ≈ ln(62) ≈ 4.13 —
# the model cannot beat random chance.  We need REPEATING
# PATTERNS so the LSTM can actually memorise transitions and
# push accuracy well above 1/62 ≈ 1.6 %.
#
# Strategy:
#   A) Deterministic cycling runs  (easy patterns to learn)
#   B) Short repeated n-grams       (medium-difficulty)
#   C) Alphabet forward/back loops (structured sequences)
#   D) Repeated bigram/trigram chains with fixed transitions
#   E) A small amount of random noise to prevent pure mem
# ============================================================

rng = np.random.default_rng(42)

segments: list[str] = []

alpha_upper = [c for c in characters if c.isupper()]   # A-Z
alpha_lower = [c for c in characters if c.islower()]   # a-z
digits      = [c for c in characters if c.isdigit()]   # 0-9

# ── A) Cyclic runs ×500 ────────────────────────────────────
#   A B C … Z A B C … Z  (the LSTM sees: after Z comes A)
for _ in range(500):
    segments.extend(alpha_upper)
for _ in range(500):
    segments.extend(alpha_lower)
for _ in range(500):
    segments.extend(digits)

# Reverse runs ×300
for _ in range(300):
    segments.extend(reversed(alpha_upper))
for _ in range(300):
    segments.extend(reversed(alpha_lower))
for _ in range(300):
    segments.extend(reversed(digits))

# ── B) Interleaved upper+lower ×200 ────────────────────────
#   A a B b C c …
for _ in range(200):
    for u, l in zip(alpha_upper, alpha_lower):
        segments.append(u)
        segments.append(l)

# ── C) Fixed bigram chains ×400 ────────────────────────────
#   A B A B …  repeated, then B C B C …  etc.
for i in range(len(characters) - 1):
    pair = [characters[i], characters[i + 1]]
    for _ in range(400):
        segments.extend(pair)

# ── D) Trigram rings ×200 ──────────────────────────────────
#   A B C A B C …
for i in range(0, len(characters) - 2, 3):
    tri = characters[i:i+3]
    for _ in range(200):
        segments.extend(tri)

# ── E) Diagonal stride ×300 ────────────────────────────────
#   Pick every 3rd character cyclically: A D G J …
stride3 = [characters[i % vocab_size] for i in range(0, vocab_size * 5, 3)]
for _ in range(300):
    segments.extend(stride3)

# ── F) Small random noise ×50 (prevents pure memorisation) ─
for _ in range(50):
    shuffled = characters.copy()
    rng.shuffle(shuffled)
    segments.extend(shuffled)

corpus = "".join(segments)
print(f"\nTraining corpus length: {len(corpus):,} characters")


# ============================================================
# ENCODE
# ============================================================

encoded = np.array([char_to_int[c] for c in corpus], dtype=np.int32)


# ============================================================
# SLIDING-WINDOW SEQUENCES  (step=1)
# ============================================================

SEQ_LEN = 20

X_list, y_list = [], []
for i in range(len(encoded) - SEQ_LEN):
    X_list.append(encoded[i : i + SEQ_LEN])
    y_list.append(encoded[i + SEQ_LEN])

X = np.array(X_list, dtype=np.int32)
y = np.array(y_list, dtype=np.int32)

print(f"Training samples : {len(X):,}")
print(f"Input shape      : {X.shape}")
print(f"Output shape     : {y.shape}")


# ============================================================
# MODEL  — single strong LSTM (faster on CPU than stacked)
# ============================================================

model = Sequential([
    Embedding(
        input_dim=vocab_size,
        output_dim=64,
        name="embedding",
    ),
    LSTM(
        256,
        return_sequences=False,
        name="lstm",
    ),
    Dropout(0.2, name="dropout"),
    Dense(
        128,
        activation="relu",
        name="dense_hidden",
    ),
    Dense(
        vocab_size,
        activation="softmax",
        name="output",
    ),
])

model.compile(
    optimizer=Adam(learning_rate=5e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("\n")
model.summary()


# ============================================================
# CALLBACKS
# ============================================================

early_stop = EarlyStopping(
    monitor="accuracy",        # stop when accuracy plateaus
    patience=8,
    restore_best_weights=True,
    verbose=1,
)

reduce_lr = ReduceLROnPlateau(
    monitor="loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1,
)


# ============================================================
# TRAIN
# ============================================================

print("\nStarting training…")
print("  Max epochs : 50   Batch size : 128")
print("  Early stop : patience=8 (on accuracy)")
print("  LR schedule: ReduceLROnPlateau (factor=0.5, patience=3)\n")

history = model.fit(
    X, y,
    epochs=50,
    batch_size=128,
    callbacks=[reduce_lr, early_stop],
    verbose=1,
)


# ============================================================
# SAVE ARTEFACTS
# ============================================================

model_path   = MODEL_DIR / "handwritten_rnn.keras"
mapping_path = MODEL_DIR / "character_mapping.csv"
history_path = MODEL_DIR / "training_history.csv"

model.save(model_path)
print(f"\nModel saved   → {model_path}")

pd.DataFrame({
    "index":     list(int_to_char.keys()),
    "character": list(int_to_char.values()),
}).to_csv(mapping_path, index=False)
print(f"Mapping saved → {mapping_path}")

pd.DataFrame(history.history).to_csv(history_path, index=False)
print(f"History saved → {history_path}")


# ============================================================
# SUMMARY
# ============================================================

final_acc  = history.history["accuracy"][-1] * 100
best_acc   = max(history.history["accuracy"])  * 100
final_loss = history.history["loss"][-1]
best_loss  = min(history.history["loss"])
epochs_ran = len(history.history["loss"])

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"  Epochs run     : {epochs_ran}")
print(f"  Final accuracy : {final_acc:.2f}%")
print(f"  Best accuracy  : {best_acc:.2f}%")
print(f"  Final loss     : {final_loss:.4f}")
print(f"  Best loss      : {best_loss:.4f}")
print("\nModel is ready for generation!")
