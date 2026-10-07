from pathlib import Path
import random

import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

DATASET_DIR = BASE_DIR / "dataset"
CSV_PATH = DATASET_DIR / "english.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "handwritten_rnn.keras"
MAPPING_PATH = MODEL_DIR / "character_mapping.csv"

OUTPUT_DIR = BASE_DIR / "generated_output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Handwritten Text Generator",
    page_icon="✍️",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #888888;
        font-size: 18px;
        margin-bottom: 35px;
    }

    .info-card {
        padding: 20px;
        border-radius: 15px;
        background-color: rgba(128,128,128,0.08);
        text-align: center;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    df = pd.read_csv(CSV_PATH)

    df["label"] = df["label"].astype(str)

    character_images = {}

    for _, row in df.iterrows():

        label = str(row["label"])
        image_path = DATASET_DIR / row["image"]

        if image_path.exists():

            if label not in character_images:
                character_images[label] = []

            character_images[label].append(image_path)

    return df, character_images


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

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

    return model, char_to_int, int_to_char


# ============================================================
# GENERATE TEXT
# ============================================================

def generate_text(
    seed_text,
    model,
    char_to_int,
    int_to_char,
    length=20
):

    seed_text = seed_text.upper()

    seed_text = "".join(
        char
        for char in seed_text
        if char in char_to_int
    )

    if not seed_text:
        return ""

    generated_text = seed_text

    sequence_length = model.input_shape[1]

    while len(seed_text) < sequence_length:
        seed_text += seed_text

    seed_text = seed_text[-sequence_length:]

    for _ in range(length):

        encoded = [
            char_to_int[char]
            for char in seed_text
        ]

        input_sequence = tf.convert_to_tensor(
            [encoded],
            dtype=tf.int32
        )

        prediction = model.predict(
            input_sequence,
            verbose=0
        )[0]

        next_index = int(tf.argmax(prediction))

        next_char = int_to_char[next_index]

        generated_text += next_char

        seed_text = seed_text[1:] + next_char

    return generated_text


# ============================================================
# CREATE HANDWRITTEN IMAGE
# ============================================================

def create_handwritten_image(
    text,
    character_images
):

    char_width = 100
    char_height = 100
    spacing = 10

    output_characters = []

    for character in text:

        if character == " ":
            continue

        if character not in character_images:
            continue

        image_path = random.choice(
            character_images[character]
        )

        img = Image.open(image_path).convert("L")

        img = ImageOps.contain(
            img,
            (char_width, char_height)
        )

        canvas = Image.new(
            "L",
            (char_width, char_height),
            255
        )

        x = (char_width - img.width) // 2
        y = (char_height - img.height) // 2

        canvas.paste(
            img,
            (x, y)
        )

        output_characters.append(canvas)

    if not output_characters:
        return None

    total_width = (
        len(output_characters) * char_width
        + (len(output_characters) - 1) * spacing
    )

    final_image = Image.new(
        "L",
        (total_width, char_height),
        255
    )

    current_x = 0

    for character_image in output_characters:

        final_image.paste(
            character_image,
            (current_x, 0)
        )

        current_x += char_width + spacing

    return final_image


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    df, character_images = load_dataset()

    model, char_to_int, int_to_char = load_model()

except Exception as e:

    st.error(
        "Unable to load the trained model or dataset."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">✍️ Handwritten Text Generator</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Character-Level RNN Based Handwritten Text Generation'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# INFORMATION CARDS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        f"""
        <div class="info-card">
        <h3>{len(df)}</h3>
        <p>Handwritten Samples</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        f"""
        <div class="info-card">
        <h3>{df["label"].nunique()}</h3>
        <p>Character Classes</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        """
        <div class="info-card">
        <h3>LSTM</h3>
        <p>RNN Architecture</p>
        </div>
        """,
        unsafe_allow_html=True
    )


st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("📝 Enter Text")

text_input = st.text_input(
    "Enter characters:",
    value="HELLO",
    max_chars=20
)

generation_length = st.slider(
    "Number of characters to generate",
    min_value=5,
    max_value=50,
    value=20
)


# ============================================================
# GENERATE BUTTON
# ============================================================

if st.button(
    "✨ Generate Handwritten Text",
    use_container_width=True
):

    if not text_input.strip():

        st.warning(
            "Please enter some characters."
        )

    else:

        with st.spinner(
            "Generating handwritten text..."
        ):

            generated_text = generate_text(
                text_input,
                model,
                char_to_int,
                int_to_char,
                generation_length
            )

            handwritten_image = create_handwritten_image(
                generated_text,
                character_images
            )

        if not generated_text:

            st.error(
                "The entered characters are not available "
                "in the dataset."
            )

        elif handwritten_image is None:

            st.error(
                "Unable to create handwritten image."
            )

        else:

            st.success(
                "Handwritten text generated successfully!"
            )

            # ------------------------------------------------
            # GENERATED TEXT
            # ------------------------------------------------

            st.subheader("Generated Text")

            st.code(
                generated_text,
                language="text"
            )

            # ------------------------------------------------
            # HANDWRITTEN IMAGE
            # ------------------------------------------------

            st.subheader("Handwritten Output")

            st.image(
                handwritten_image,
                caption="Generated Handwritten Text",
                use_container_width=True
            )

            # ------------------------------------------------
            # SAVE IMAGE
            # ------------------------------------------------

            output_path = (
                OUTPUT_DIR /
                "streamlit_generated_handwritten.png"
            )

            handwritten_image.save(
                output_path
            )

            st.download_button(
                label="⬇️ Download Handwritten Image",
                data=output_path.read_bytes(),
                file_name="generated_handwritten_text.png",
                mime="image/png",
                use_container_width=True
            )


# ============================================================
# HOW IT WORKS
# ============================================================

st.divider()

st.subheader("⚙️ How It Works")

step1, step2, step3, step4 = st.columns(4)

with step1:

    st.markdown(
        """
        **1️⃣ Input**

        User enters starting characters.
        """
    )

with step2:

    st.markdown(
        """
        **2️⃣ RNN**

        LSTM predicts the next characters.
        """
    )

with step3:

    st.markdown(
        """
        **3️⃣ Dataset**

        Handwritten character samples are selected
        from the dataset.
        """
    )

with step4:

    st.markdown(
        """
        **4️⃣ Output**

        Characters are combined into a handwritten
        text image.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Character-Level RNN • LSTM • Handwritten Character Dataset"
)