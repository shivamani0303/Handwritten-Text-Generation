### 1. Dataset Loading

The `english.csv` file is loaded using Pandas.

Each record contains:

- Image path
- Character label

### 2. Character Encoding

Each unique character is assigned a numerical index.

Example:

```text
A → 0
B → 1
C → 2
...

3. Sequence Creation
Balanced character sequences are created and divided into:
- Input sequence
- Target character
The sequence length used for training is 20 characters.
4. RNN Model
The model uses:
- Embedding layer
- LSTM layer
- Dropout layer
- Dense layer
- Softmax output layer
5. Character Generation
After training, the model predicts the next character based on the previous sequence.
6. Handwritten Output
The predicted characters are matched with handwritten samples from the dataset.
These character images are combined to create the final handwritten text image.
Project Structure
Task_5 Handwritten Text Generation/
│
├── app.py
├── train_rnn.py
├── generate_text.py
├── generate_handwritten.py
├── check_dataset.py
├── check_images.py
├── check_image_preview.py
├── preprocess_data.py
├── verify_processed_data.py
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── english.csv
│   └── Img/
│
├── processed_data/
│   ├── X.npy
│   ├── y.npy
│   └── label_mapping.csv
│
├── models/
│   ├── handwritten_rnn.keras
│   ├── character_mapping.csv
│   └── training_history.csv
│
└── generated_output/
    └── generated_handwritten_text.png

Technologies Used
- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Pillow
- Matplotlib
- Streamlit
How to Run
1. Install dependencies
pip install -r requirements.txt

2. Train the model
python train_rnn.py

3. Test character generation
python generate_text.py

4. Generate handwritten text image
python generate_handwritten.py

5. Run the Streamlit application
python -m streamlit run app.py

Application Workflow
User Input
     ↓
Character Encoding
     ↓
LSTM-Based RNN
     ↓
Character Prediction
     ↓
Handwritten Character Selection
     ↓
Image Combination
     ↓
Handwritten Text Image

Important Note
The dataset consists of individual handwritten characters rather than complete English sentences.
Therefore, the trained RNN demonstrates character-level sequence generation rather than grammatical English sentence generation.
The final handwritten output is created using handwritten character samples from the same dataset.
Future Improvements
- Train on a larger text corpus containing complete sentences
- Generate more natural English text
- Improve handwriting style consistency
- Generate continuous handwriting instead of separate character images
- Add support for spaces and punctuation
- Use a more advanced sequence-generation architecture
Project Objective
The objective is to demonstrate how a character-level recurrent neural network can learn patterns from handwritten character data and generate character sequences that can be represented as handwritten-style text.