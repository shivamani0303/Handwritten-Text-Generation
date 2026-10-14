# ✍️ Handwritten Text Generator

> **CodSoft ML Internship — Task 5**
> Character-Level LSTM · Real Handwriting Samples · Streamlit Web App

---

## 🌟 Overview

This project builds an end-to-end handwritten text generation pipeline using a **character-level LSTM (RNN)**. Given a short seed string, the model predicts a sequence of characters, then assembles a final image by stitching real handwritten character samples from the dataset together — producing a natural-looking handwritten output.

---

## 🖥️ App Preview

The Streamlit app features a modern dark-themed UI with:

| Feature | Description |
|---|---|
| 🎨 **Hero banner** | Gradient header with project title |
| 📊 **Stat cards** | Live dataset & training metrics |
| ⚙️ **Sidebar controls** | Temperature, character size, generation length |
| ✨ **Generate tab** | Seed input → LSTM prediction → handwriting image |
| 📈 **Model Insights tab** | Training curves, epoch log, character distribution chart |
| ℹ️ **About tab** | How-it-works steps, tech stack, project notes |
| ⬇️ **PNG export** | One-click download of the generated image |

---

## 🗂️ Project Structure

```
Task_5 Handwritten Text Generation/
│
├── app.py                    ← Streamlit web application (main entry point)
├── train_rnn.py              ← LSTM model training script
├── generate_text.py          ← CLI: RNN character sequence generation
├── generate_handwritten.py   ← CLI: assemble handwriting image from text
├── preprocess_data.py        ← Image preprocessing → NumPy arrays
├── check_dataset.py          ← Dataset inspection utility
├── check_images.py           ← Image file existence checker
├── check_image_preview.py    ← Matplotlib preview of dataset samples
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── english.csv           ← image path + character label per row
│   └── Img/                  ← 3,410 handwritten PNG images
│
├── processed_data/
│   ├── X.npy                 ← Preprocessed image arrays (64×64 grayscale)
│   ├── y.npy                 ← Encoded label array
│   └── label_mapping.csv
│
├── models/
│   ├── handwritten_rnn.keras ← Trained LSTM model
│   ├── character_mapping.csv ← Character ↔ index mapping (62 classes)
│   └── training_history.csv  ← Loss & accuracy per epoch
│
└── generated_output/
    ├── generated_handwritten_text.png
    └── streamlit_generated_handwritten.png
```

---

## 🧠 Model Architecture

```
Input: integer sequence (length 20)
        ↓
Embedding(input_dim=62, output_dim=64)
        ↓
LSTM(units=128)
        ↓
Dropout(0.2)
        ↓
Dense(64, activation='relu')
        ↓
Dense(62, activation='softmax')
        ↓
Output: probability distribution over 62 characters
```

| Parameter | Value |
|---|---|
| Vocabulary | 62 (0–9, A–Z, a–z) |
| Sequence length | 20 characters |
| Embedding dim | 64 |
| LSTM units | 128 |
| Optimizer | Adam |
| Loss | Sparse Categorical Crossentropy |
| Early stopping | patience = 3 |
| Max epochs | 30 |

---

## ⚙️ Pipeline Steps

```
User Input (seed text)
        ↓
Character Encoding  (char → integer index)
        ↓
LSTM Prediction     (predict next char, repeat N times)
        ↓
Character Lookup    (match chars → random real handwritten sample)
        ↓
Image Assembly      (tile 100×100 canvases side by side)
        ↓
Handwritten Image   (PNG, downloadable)
```

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the LSTM model

```bash
python train_rnn.py
```

Saves `models/handwritten_rnn.keras`, `models/character_mapping.csv`, and `models/training_history.csv`.

### 3. (Optional) Test generation via CLI

```bash
# Text-only generation
python generate_text.py

# Image generation
python generate_handwritten.py
```

### 4. Launch the Streamlit app

```bash
python -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📦 Requirements

```
tensorflow==2.21.0
numpy
pandas
Pillow
streamlit
matplotlib
```

---

## 🗃️ Dataset

- **Source:** English Handwritten Characters dataset
- **File:** `dataset/english.csv` — two columns: `image` (relative path) and `label` (character)
- **Images:** 3,410 PNG files in `dataset/Img/`, named `imgNNN-MMM.png`
- **Classes:** 62 — digits `0–9`, uppercase `A–Z`, lowercase `a–z`
- **Samples per class:** ~55 per character

---

## 📌 Important Notes

> The dataset contains **individual handwritten characters**, not full sentences.
> The LSTM is trained on balanced, randomly shuffled character sequences, so generated
> text reflects character-transition probabilities rather than grammatical English.
> The pipeline demonstrates an RNN-based generation workflow, not a language model.

---

## 🔮 Future Improvements

- Train on a real English text corpus for natural-sounding output
- Add nucleus / top-k sampling for better diversity
- Support spaces and punctuation in image rendering
- Generate continuous handwriting strokes (seq2seq or diffusion-based)
- Style-conditioned generation (match a specific writer's style)
- Add a CNN-based recognizer to close the loop

---

## 🛠️ Technologies

| Category | Tools |
|---|---|
| ML framework | TensorFlow · Keras |
| Data processing | Pandas · NumPy |
| Image processing | Pillow (PIL) · Matplotlib |
| Web interface | Streamlit |
| Language | Python 3.10+ |

---

*CodSoft ML Internship · Task 5 · Handwritten Text Generation*
