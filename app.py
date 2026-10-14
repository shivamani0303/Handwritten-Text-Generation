from pathlib import Path
import random
import io

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

DATASET_DIR   = BASE_DIR / "dataset"
CSV_PATH      = DATASET_DIR / "english.csv"

MODEL_DIR     = BASE_DIR / "models"
MODEL_PATH    = MODEL_DIR / "handwritten_rnn.keras"
MAPPING_PATH  = MODEL_DIR / "character_mapping.csv"
HISTORY_PATH  = MODEL_DIR / "training_history.csv"

OUTPUT_DIR    = BASE_DIR / "generated_output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# PAGE CONFIGURATION  (must be first Streamlit call)
# ============================================================

st.set_page_config(
    page_title="Handwritten Text Generator",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL STYLES
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- Google Font ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---------- Hide Streamlit chrome ---------- */
    #MainMenu  { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }

    /* ---------- App background ---------- */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
        min-height: 100vh;
    }

    /* ---------- Hero banner ---------- */
    .hero-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 24px;
        padding: 52px 40px 44px;
        text-align: center;
        margin-bottom: 32px;
        box-shadow: 0 20px 60px rgba(102,126,234,0.4);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: "";
        position: absolute;
        top: -60%; left: -20%;
        width: 140%; height: 200%;
        background: radial-gradient(ellipse at 60% 40%, rgba(255,255,255,0.12) 0%, transparent 65%);
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -1px;
        margin: 0 0 10px;
        text-shadow: 0 4px 24px rgba(0,0,0,0.25);
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: rgba(255,255,255,0.85);
        font-weight: 300;
        margin: 0;
        letter-spacing: 0.5px;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.22);
        border: 1px solid rgba(255,255,255,0.35);
        color: #fff;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 4px 14px;
        border-radius: 20px;
        margin-bottom: 18px;
    }

    /* ---------- Stat cards ---------- */
    .stat-row {
        display: flex;
        gap: 18px;
        margin-bottom: 32px;
    }
    .stat-card {
        flex: 1;
        border-radius: 18px;
        padding: 28px 20px 24px;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 40px rgba(0,0,0,0.4);
    }
    .stat-card.purple {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        box-shadow: 0 8px 30px rgba(102,126,234,0.35);
    }
    .stat-card.teal {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        box-shadow: 0 8px 30px rgba(17,153,142,0.35);
    }
    .stat-card.orange {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        box-shadow: 0 8px 30px rgba(245,87,108,0.35);
    }
    .stat-card.blue {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        box-shadow: 0 8px 30px rgba(79,172,254,0.35);
    }
    .stat-icon {
        font-size: 2.2rem;
        margin-bottom: 10px;
        display: block;
    }
    .stat-value {
        font-size: 2.4rem;
        font-weight: 800;
        color: #fff;
        line-height: 1;
        margin-bottom: 6px;
    }
    .stat-label {
        font-size: 0.82rem;
        color: rgba(255,255,255,0.80);
        font-weight: 500;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* ---------- Panel cards (sections) ---------- */
    .panel {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 20px;
        padding: 32px 28px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
    }
    .panel-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 0 0 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* ---------- Section divider ---------- */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102,126,234,0.6), transparent);
        margin: 28px 0;
        border: none;
    }

    /* ---------- Step cards (How it Works) ---------- */
    .step-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-top: 8px;
    }
    .step-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 24px 16px 20px;
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
    }
    .step-card:hover {
        transform: translateY(-3px);
        border-color: rgba(102,126,234,0.5);
    }
    .step-num {
        display: inline-block;
        width: 36px; height: 36px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 50%;
        line-height: 36px;
        font-size: 0.95rem;
        font-weight: 700;
        color: #fff;
        margin-bottom: 12px;
    }
    .step-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 8px;
    }
    .step-desc {
        font-size: 0.80rem;
        color: rgba(255,255,255,0.55);
        line-height: 1.5;
    }

    /* ---------- Generated text display ---------- */
    .gen-text-box {
        background: rgba(0,0,0,0.35);
        border: 1px solid rgba(102,126,234,0.4);
        border-radius: 14px;
        padding: 20px 24px;
        font-family: 'Courier New', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: 6px;
        color: #a78bfa;
        text-align: center;
        word-break: break-all;
        text-shadow: 0 0 20px rgba(167,139,250,0.5);
        margin: 12px 0;
    }

    /* ---------- Output image frame ---------- */
    .image-frame {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px;
        padding: 24px;
        text-align: center;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: rgba(15,12,41,0.95) !important;
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
    }
    .sidebar-logo {
        font-size: 2.8rem;
        text-align: center;
        margin-bottom: 4px;
    }
    .sidebar-brand {
        font-size: 1.05rem;
        font-weight: 700;
        text-align: center;
        color: #a78bfa !important;
        margin-bottom: 4px;
    }
    .sidebar-tagline {
        font-size: 0.75rem;
        text-align: center;
        color: rgba(255,255,255,0.45) !important;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 24px;
    }
    .sidebar-info {
        background: rgba(102,126,234,0.12);
        border: 1px solid rgba(102,126,234,0.25);
        border-radius: 12px;
        padding: 14px 16px;
        font-size: 0.80rem;
        color: rgba(255,255,255,0.65) !important;
        line-height: 1.65;
        margin-top: 10px;
    }

    /* ---------- Streamlit widget overrides ---------- */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        font-size: 1.05rem !important;
        padding: 10px 14px !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102,126,234,0.20) !important;
    }
    .stSlider > div { padding-top: 4px; }
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding: 14px 28px !important;
        letter-spacing: 0.5px;
        box-shadow: 0 6px 24px rgba(102,126,234,0.45) !important;
        transition: opacity 0.2s, transform 0.15s !important;
    }
    div[data-testid="stButton"] > button:hover {
        opacity: 0.88 !important;
        transform: translateY(-2px) !important;
    }
    div[data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
        color: #0f2027 !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 6px 24px rgba(17,153,142,0.35) !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: rgba(255,255,255,0.55) !important;
        border-radius: 9px;
        font-weight: 600;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: #fff !important;
    }
    div[data-testid="stSelectbox"] > div > div {
        background: rgba(255,255,255,0.06) !important;
        border-color: rgba(255,255,255,0.15) !important;
        color: #e2e8f0 !important;
        border-radius: 10px !important;
    }

    /* ---------- Alert overrides ---------- */
    .stAlert {
        border-radius: 12px !important;
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center;
        padding: 28px 0 8px;
        color: rgba(255,255,255,0.30);
        font-size: 0.78rem;
        letter-spacing: 0.5px;
    }
    .footer span {
        color: rgba(167,139,250,0.7);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING  (cached)
# ============================================================

@st.cache_data(show_spinner=False)
def load_dataset():
    df = pd.read_csv(CSV_PATH)
    df["label"] = df["label"].astype(str)
    character_images: dict[str, list] = {}
    for _, row in df.iterrows():
        label = str(row["label"])
        image_path = DATASET_DIR / row["image"]
        if image_path.exists():
            character_images.setdefault(label, []).append(image_path)
    return df, character_images


@st.cache_resource(show_spinner=False)
def load_model():
    m = tf.keras.models.load_model(MODEL_PATH)
    mapping_df = pd.read_csv(MAPPING_PATH)
    c2i = {r["character"]: int(r["index"]) for _, r in mapping_df.iterrows()}
    i2c = {int(r["index"]): r["character"] for _, r in mapping_df.iterrows()}
    return m, c2i, i2c


@st.cache_data(show_spinner=False)
def load_history():
    if HISTORY_PATH.exists():
        return pd.read_csv(HISTORY_PATH)
    return None


# ============================================================
# GENERATION HELPERS
# ============================================================

def generate_text(seed: str, model, c2i: dict, i2c: dict,
                  length: int = 20, temperature: float = 1.0) -> str:
    seed = seed.upper()
    seed = "".join(ch for ch in seed if ch in c2i)
    if not seed:
        return ""

    generated = seed
    seq_len   = model.input_shape[1]

    while len(seed) < seq_len:
        seed += seed
    seed = seed[-seq_len:]

    for _ in range(length):
        encoded = [c2i[ch] for ch in seed]
        inp     = tf.convert_to_tensor([encoded], dtype=tf.int32)
        probs   = model.predict(inp, verbose=0)[0].astype(float)

        if temperature != 1.0:
            probs = np.log(probs + 1e-8) / temperature
            probs = np.exp(probs)
        probs /= probs.sum()

        next_idx  = int(np.random.choice(len(probs), p=probs))
        next_char = i2c[next_idx]
        generated += next_char
        seed       = seed[1:] + next_char

    return generated


def create_handwritten_image(text: str, character_images: dict,
                              char_size: int = 100, spacing: int = 12) -> Image.Image | None:
    tiles = []
    for ch in text:
        if ch == " " or ch not in character_images:
            continue
        path = random.choice(character_images[ch])
        img  = Image.open(path).convert("L")
        img  = ImageOps.contain(img, (char_size, char_size))
        canvas = Image.new("L", (char_size, char_size), 255)
        x = (char_size - img.width)  // 2
        y = (char_size - img.height) // 2
        canvas.paste(img, (x, y))
        tiles.append(canvas)

    if not tiles:
        return None

    total_w = len(tiles) * char_size + (len(tiles) - 1) * spacing
    output  = Image.new("L", (total_w, char_size), 255)
    x_pos   = 0
    for tile in tiles:
        output.paste(tile, (x_pos, 0))
        x_pos += char_size + spacing
    return output


def pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ============================================================
# LOAD RESOURCES  (with graceful error handling)
# ============================================================

with st.spinner("🔄 Loading model and dataset…"):
    try:
        df, character_images = load_dataset()
        model, c2i, i2c     = load_model()
        history_df           = load_history()
        load_ok              = True
    except Exception as exc:
        load_ok = False
        load_err = str(exc)

if not load_ok:
    st.error("⚠️ Could not load the model or dataset.")
    st.code(load_err)
    st.stop()

TOTAL_SAMPLES    = len(df)
TOTAL_CHARS      = df["label"].nunique()
AVAILABLE_CHARS  = sorted(c2i.keys())


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown('<div class="sidebar-logo">✍️</div>',      unsafe_allow_html=True)
    st.markdown('<div class="sidebar-brand">HandwriteAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">RNN • LSTM • Character Generation</div>',
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Generation Settings")

    gen_length = st.slider(
        "Generated characters",
        min_value=5, max_value=60, value=25, step=5,
        help="How many characters the LSTM will predict after your seed text."
    )

    temperature = st.select_slider(
        "Creativity (temperature)",
        options=[0.5, 0.7, 1.0, 1.3, 1.6],
        value=1.0,
        help="Lower = more predictable output. Higher = more creative / random."
    )

    char_size = st.slider(
        "Character tile size (px)",
        min_value=60, max_value=140, value=100, step=10,
        help="Pixel size of each character tile in the output image."
    )

    st.markdown("---")
    st.markdown("### 📋 Dataset Info")
    st.markdown(
        f"""
        <div class="sidebar-info">
        📁 <b>Dataset:</b> English Handwritten<br>
        🖼️ <b>Images:</b> {TOTAL_SAMPLES:,}<br>
        🔡 <b>Classes:</b> {TOTAL_CHARS} characters<br>
        🏗️ <b>Architecture:</b> Embedding → LSTM → Dense<br>
        📐 <b>Sequence length:</b> 20 characters<br>
        🎯 <b>Optimizer:</b> Adam<br>
        📉 <b>Loss:</b> Sparse Categorical CE
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:rgba(255,255,255,0.3);font-size:0.72rem'>"
        "CodSoft ML Internship • Task 5</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# HERO BANNER
# ============================================================

st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">✍️ Handwritten Text Generator</div>
        <div class="hero-subtitle">
            Character-Level LSTM · Real Handwriting Samples · Instant Image Export
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# STAT CARDS
# ============================================================

if history_df is not None:
    best_acc  = f"{history_df['accuracy'].max()*100:.1f}%"
    min_loss  = f"{history_df['loss'].min():.3f}"
    epochs_run = str(len(history_df))
else:
    best_acc   = "—"
    min_loss   = "—"
    epochs_run = "—"

st.markdown(
    f"""
    <div class="stat-row">
        <div class="stat-card purple">
            <span class="stat-icon">🖼️</span>
            <div class="stat-value">{TOTAL_SAMPLES:,}</div>
            <div class="stat-label">Handwriting Samples</div>
        </div>
        <div class="stat-card teal">
            <span class="stat-icon">🔡</span>
            <div class="stat-value">{TOTAL_CHARS}</div>
            <div class="stat-label">Character Classes</div>
        </div>
        <div class="stat-card orange">
            <span class="stat-icon">🎯</span>
            <div class="stat-value">{best_acc}</div>
            <div class="stat-label">Best Accuracy</div>
        </div>
        <div class="stat-card blue">
            <span class="stat-icon">🔁</span>
            <div class="stat-value">{epochs_run}</div>
            <div class="stat-label">Training Epochs</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MAIN TABS
# ============================================================

tab_gen, tab_explore, tab_about = st.tabs(
    ["✨  Generate", "📊  Model Insights", "ℹ️  About"]
)


# ────────────────────────────────────────────────────────────
# TAB 1 — GENERATE
# ────────────────────────────────────────────────────────────

with tab_gen:

    # ----- Input panel -----
    st.markdown(
        '<div class="panel-title" style="margin-top:8px;">📝 Seed Text</div>',
        unsafe_allow_html=True,
    )

    col_input, col_hint = st.columns([3, 1])

    with col_input:
        seed_input = st.text_input(
            "Enter characters to seed the LSTM (A–Z, 0–9):",
            value="HELLO",
            max_chars=20,
            label_visibility="collapsed",
            placeholder="e.g. HELLO or ABC123",
        )

    with col_hint:
        st.markdown(
            "<div style='padding-top:8px; color:rgba(255,255,255,0.45);font-size:0.78rem'>"
            "Uppercase letters &amp; digits only.</div>",
            unsafe_allow_html=True,
        )

    # ----- Generate button -----
    generate_clicked = st.button(
        "✨  Generate Handwritten Text",
        use_container_width=True,
        key="generate_btn",
    )

    # ----- Output -----
    if generate_clicked:
        seed_clean = seed_input.strip().upper()
        known_seed = "".join(ch for ch in seed_clean if ch in c2i)

        if not known_seed:
            st.warning(
                "⚠️ None of the entered characters exist in the dataset. "
                f"Use characters from: `{'  '.join(AVAILABLE_CHARS)}`"
            )
        else:
            with st.spinner("🤖 LSTM is thinking…"):
                generated = generate_text(
                    known_seed, model, c2i, i2c,
                    length=gen_length, temperature=temperature
                )

            with st.spinner("🖌️ Assembling handwriting…"):
                hw_image = create_handwritten_image(
                    generated, character_images, char_size=char_size
                )

            if not generated:
                st.error("Generation failed — no valid characters produced.")
            elif hw_image is None:
                st.error("Could not assemble the handwriting image.")
            else:
                st.success("✅ Handwritten image generated successfully!")

                # ----- Generated text -----
                st.markdown(
                    f'<div class="gen-text-box">{generated}</div>',
                    unsafe_allow_html=True,
                )

                # ----- Image output -----
                st.markdown('<div class="image-frame">', unsafe_allow_html=True)
                st.image(
                    hw_image,
                    caption="Generated Handwritten Text",
                    use_container_width=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

                # ----- Download -----
                img_bytes  = pil_to_bytes(hw_image)
                save_path  = OUTPUT_DIR / "streamlit_generated_handwritten.png"
                hw_image.save(save_path)

                st.download_button(
                    label="⬇️  Download as PNG",
                    data=img_bytes,
                    file_name="generated_handwriting.png",
                    mime="image/png",
                    use_container_width=True,
                )

                # ----- Quick stats -----
                st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("Seed length",     f"{len(known_seed)} chars")
                m2.metric("Generated",       f"{gen_length} chars")
                m3.metric("Image width",     f"{hw_image.width} px")

    # ----- Sample preview panel -----
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-title">🔍 Dataset Sample Preview</div>',
        unsafe_allow_html=True,
    )

    sample_char = st.selectbox(
        "Pick a character to preview its handwritten samples:",
        options=AVAILABLE_CHARS,
        index=AVAILABLE_CHARS.index("A") if "A" in AVAILABLE_CHARS else 0,
        label_visibility="collapsed",
    )

    if sample_char and sample_char in character_images:
        sample_paths = character_images[sample_char]
        show_n       = min(8, len(sample_paths))
        chosen       = random.sample(sample_paths, show_n)

        preview_cols = st.columns(show_n)
        for col, path in zip(preview_cols, chosen):
            img = Image.open(path).convert("L")
            col.image(img, use_container_width=True)

        st.caption(
            f"Showing {show_n} of {len(sample_paths)} samples for '{sample_char}'. "
            "Refresh to see different samples."
        )


# ────────────────────────────────────────────────────────────
# TAB 2 — MODEL INSIGHTS
# ────────────────────────────────────────────────────────────

with tab_explore:

    if history_df is None:
        st.info("Training history file not found at `models/training_history.csv`.")
    else:
        # ----- Training curves -----
        st.markdown(
            '<div class="panel-title" style="margin-top:8px;">📈 Training History</div>',
            unsafe_allow_html=True,
        )

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        fig.patch.set_facecolor("#0f0c29")

        for ax in axes:
            ax.set_facecolor("#1a1a2e")
            ax.tick_params(colors="white", labelsize=9)
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_edgecolor((1, 1, 1, 0.15))

        epochs = range(1, len(history_df) + 1)

        # Loss
        axes[0].plot(epochs, history_df["loss"], color="#a78bfa", linewidth=2.5,
                     marker="o", markersize=4, label="Training Loss")
        axes[0].fill_between(epochs, history_df["loss"], alpha=0.15, color="#a78bfa")
        axes[0].set_title("Loss over Epochs", fontsize=13, fontweight="bold", pad=12)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend(facecolor="#1a1a2e", edgecolor=(1, 1, 1, 0.2),
                        labelcolor="white", fontsize=9)
        axes[0].grid(alpha=0.15)

        # Accuracy
        axes[1].plot(epochs, history_df["accuracy"] * 100, color="#38ef7d",
                     linewidth=2.5, marker="o", markersize=4, label="Training Accuracy")
        axes[1].fill_between(epochs, history_df["accuracy"] * 100, alpha=0.15, color="#38ef7d")
        axes[1].set_title("Accuracy over Epochs", fontsize=13, fontweight="bold", pad=12)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].legend(facecolor="#1a1a2e", edgecolor=(1, 1, 1, 0.2),
                        labelcolor="white", fontsize=9)
        axes[1].grid(alpha=0.15)

        plt.tight_layout(pad=2.0)
        st.pyplot(fig)
        plt.close(fig)

        # ----- History table -----
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-title">📋 Epoch-by-Epoch Log</div>',
            unsafe_allow_html=True,
        )

        display_df = history_df.copy()
        display_df.index = range(1, len(display_df) + 1)
        display_df.index.name = "Epoch"
        display_df["accuracy"] = (display_df["accuracy"] * 100).round(2).astype(str) + "%"
        display_df["loss"]     = display_df["loss"].round(4)
        display_df.columns     = ["Loss", "Accuracy"]

        st.dataframe(
            display_df,
            use_container_width=True,
            height=280,
        )

    # ----- Model architecture card -----
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-title">🏗️ Model Architecture</div>',
        unsafe_allow_html=True,
    )

    arch_col1, arch_col2 = st.columns(2)

    with arch_col1:
        st.markdown(
            """
            | Layer | Config |
            |---|---|
            | **Embedding** | vocab=62, dim=64 |
            | **LSTM** | units=128 |
            | **Dropout** | rate=0.2 |
            | **Dense** | units=64, ReLU |
            | **Output Dense** | units=62, Softmax |
            """,
        )

    with arch_col2:
        st.markdown(
            """
            | Parameter | Value |
            |---|---|
            | **Sequence length** | 20 chars |
            | **Vocabulary size** | 62 |
            | **Optimizer** | Adam |
            | **Loss function** | Sparse Cat. CE |
            | **Early stopping** | patience=3 |
            """,
        )

    # ----- Character distribution -----
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-title">🔡 Character Distribution in Dataset</div>',
        unsafe_allow_html=True,
    )

    counts = df["label"].value_counts().sort_index()

    fig2, ax2 = plt.subplots(figsize=(14, 3.5))
    fig2.patch.set_facecolor("#0f0c29")
    ax2.set_facecolor("#1a1a2e")

    bar_colors = [
        "#667eea" if c.isdigit() else
        "#38ef7d" if c.isupper() else
        "#f093fb"
        for c in counts.index
    ]
    ax2.bar(counts.index, counts.values, color=bar_colors, width=0.7, alpha=0.9)
    ax2.set_xlabel("Character", color="white", fontsize=10)
    ax2.set_ylabel("Sample count", color="white", fontsize=10)
    ax2.set_title("Samples per Character  (🟦 digits  🟩 uppercase  🟪 lowercase)",
                  color="white", fontsize=11, pad=10)
    ax2.tick_params(colors="white", labelsize=7.5)
    for spine in ax2.spines.values():
        spine.set_edgecolor((1, 1, 1, 0.15))
    ax2.grid(axis="y", alpha=0.12)

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)


# ────────────────────────────────────────────────────────────
# TAB 3 — ABOUT
# ────────────────────────────────────────────────────────────

with tab_about:

    st.markdown(
        '<div class="panel-title" style="margin-top:8px;">⚙️ How It Works</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="step-grid">
            <div class="step-card">
                <div class="step-num">1</div>
                <div class="step-title">You Type a Seed</div>
                <div class="step-desc">Enter any letters or digits. The app filters to known characters and uses them as the LSTM's starting context.</div>
            </div>
            <div class="step-card">
                <div class="step-num">2</div>
                <div class="step-title">LSTM Predicts</div>
                <div class="step-desc">A character-level LSTM model predicts the most likely next character, one step at a time, extending your seed.</div>
            </div>
            <div class="step-card">
                <div class="step-num">3</div>
                <div class="step-title">Images Are Looked Up</div>
                <div class="step-desc">Each predicted character is matched against 3,410 real handwritten samples in the dataset. One is picked at random.</div>
            </div>
            <div class="step-card">
                <div class="step-num">4</div>
                <div class="step-title">Image Is Assembled</div>
                <div class="step-desc">Character tiles are centered on a white canvas and stitched together into a single handwritten text image for download.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----- Tech stack -----
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-title">🛠️ Tech Stack</div>',
        unsafe_allow_html=True,
    )

    t1, t2, t3 = st.columns(3)

    with t1:
        st.markdown(
            """
            **Machine Learning**
            - TensorFlow / Keras
            - LSTM (character-level RNN)
            - Adam optimiser
            - Early stopping
            """
        )
    with t2:
        st.markdown(
            """
            **Data & Image Processing**
            - Pandas (CSV loading)
            - NumPy (sequence encoding)
            - Pillow (image assembly)
            - Matplotlib (charts)
            """
        )
    with t3:
        st.markdown(
            """
            **Interface**
            - Streamlit (web app)
            - Custom CSS theming
            - Responsive layout
            - PNG export
            """
        )

    # ----- Project notes -----
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-title">📌 Project Notes</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "**Dataset:** The model trains on individual handwritten characters, not full sentences. "
        "The LSTM learns character-transition probabilities from shuffled sequences, so the "
        "generated text is not natural English — it demonstrates the RNN pipeline rather than "
        "language modelling."
    )




# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Built for <span>CodSoft ML Internship — Task 5</span> &nbsp;·&nbsp;
        Character-Level RNN &nbsp;·&nbsp; LSTM &nbsp;·&nbsp; Handwritten Character Dataset
    </div>
    """,
    unsafe_allow_html=True,
)
