from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import re
import unicodedata
import emoji
import saka
import gradio as gr

app = FastAPI(title="BiGRU Judol API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Model Architecture (from Notebook/GRU_model.ipynb) ---
class BiGRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super(BiGRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers,
                          bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, hidden = self.gru(embedded)
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        hidden_concat = torch.cat((hidden_forward, hidden_backward), dim=1)
        return self.fc(self.dropout(hidden_concat))


# --- Preprocessing (2-layer pipeline: AdvancedPreprocessing + GRU_model clean_text) ---

def advanced_preprocessing(text):
    """Layer 1: AdvancedPreprocessing.ipynb — applied BEFORE GRU_model training"""
    if not isinstance(text, str):
        return ""
    # 1. Emoji to Text (Demojize) — harus paling awal
    text = emoji.demojize(text, delimiters=(" ", " "))
    # 2. Unicode Normalization (Bold/Italic → Normal) + drop non-ASCII
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    # 3. URL Masking
    text = re.sub(r'http\S+|www\S+|https\S+', '[URL]', text)
    text = re.sub(r'bit\s?[\.\,]\s?ly/\w+', '[URL]', text)
    text = re.sub(r'\w+\s?\[dot\]\s?\w+', '[URL]', text)
    # 4. Lowercase
    text = text.lower()
    # 5. Keep alphanumeric, [URL], and _: (demojize uses underscore)
    text = re.sub(r'[^a-zA-Z0-9\s\[\]\_:]', ' ', text)
    # 6. Indonesian slang normalization
    text = saka.normalize(text)
    # 7. Reduce repeated chars
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    # 8. Final whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_text(text, slang_map):
    """Layer 2: GRU_model.ipynb clean_text — applied right before model input"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', 'link_website', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = text.split()
    normalized_words = [slang_map.get(word, word) for word in words]
    return " ".join(normalized_words)


def full_preprocessing(text, slang_map):
    """Full 2-layer preprocessing matching training pipeline"""
    return clean_text(advanced_preprocessing(text), slang_map)


def text_to_sequence(text, vocab, max_len):
    tokens = text.split()
    sequence = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    if len(sequence) < max_len:
        sequence = sequence + [vocab['<PAD>']] * (max_len - len(sequence))
    else:
        sequence = sequence[:max_len]
    return sequence


# --- Load Model & Config on Startup ---
print("Sedang memuat model BiGRU...")
config = torch.load("bigru_config.pt", weights_only=False)
vocab = config['vocab']
MAX_LEN = config['MAX_LEN']
VOCAB_SIZE = config['VOCAB_SIZE']
EMBEDDING_DIM = config['EMBEDDING_DIM']
HIDDEN_DIM = config['HIDDEN_DIM']
SLANG_JUDOL_MAP = config['SLANG_JUDOL_MAP']

model = BiGRUClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, output_dim=1)
model.load_state_dict(torch.load("bigru_model.pt", map_location="cpu", weights_only=True))
model.eval()
print("Model BiGRU berhasil dimuat!")


# --- API Endpoints ---
class CommentRequest(BaseModel):
    text: str


@app.post("/predict")
def predict_comment(req: CommentRequest):
    cleaned = full_preprocessing(req.text, SLANG_JUDOL_MAP)
    seq = text_to_sequence(cleaned, vocab, MAX_LEN)
    tensor_input = torch.tensor([seq], dtype=torch.long)
    with torch.no_grad():
        score = torch.sigmoid(model(tensor_input)).item()
    label = "judol" if score >= 0.5 else "bukan_judol"
    return {"label": label, "score": score}


@app.get("/api_health")
def api_health():
    return {"status": "API BiGRU Judol is running!"}


# --- Gradio UI ---
def predict_comment_ui(text):
    """For Gradio UI"""
    cleaned = full_preprocessing(text, SLANG_JUDOL_MAP)
    seq = text_to_sequence(cleaned, vocab, MAX_LEN)
    tensor_input = torch.tensor([seq], dtype=torch.long)
    with torch.no_grad():
        score = torch.sigmoid(model(tensor_input)).item()
    label = "judol" if score >= 0.5 else "bukan_judol"
    confidence = score if label == "judol" else 1 - score

    if label == "judol":
        emoji = "🚨"
        color = "red"
    else:
        emoji = "✅"
        color = "green"

    return (
        f"### {emoji} Hasil Deteksi\n\n"
        f"**Label:** <span style='color:{color};font-weight:bold'>{label.upper()}</span>\n\n"
        f"**Confidence:** {confidence * 100:.1f}%\n\n"
        f"---\n"
        f"**Teks asli:** _{text}_\n"
        f"**Teks bersih:** _{cleaned}_"
    )


with gr.Blocks(
    title="Deteksi Judol Comment - BiGRU",
    theme=gr.themes.Soft(),
    css="""
    .judol-header { text-align: center; margin-bottom: 1em; }
    .footer { text-align: center; font-size: 0.8em; color: #888; margin-top: 2em; }
    """
) as demo:
    gr.Markdown(
        "# 🎯 Deteksi Komentar Judol YouTube\n"
        "Masukkan komentar YouTube untuk mendeteksi apakah mengandung **judol (judi online)** atau tidak.",
        elem_classes="judol-header"
    )

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Komentar",
                placeholder="Contoh: gacor maxwin hari ini langsung cair...",
                lines=5
            )
            predict_btn = gr.Button("🔍 Deteksi", variant="primary", size="lg")
            gr.Examples(
                examples=[
                    "gacor maxwin hari ini langsung cair",
                    "video nya bagus banget kak, lanjutkan",
                    "daftar sekarang juga dapat bonus new member 100%",
                    "mantap kontennya sangat bermanfaat",
                    "link slot gacor ada di bio",
                ],
                inputs=text_input,
                label="Contoh komentar (klik untuk mencoba)",
            )

        with gr.Column(scale=1):
            output = gr.Markdown(
                value="### 👈 Masukkan komentar lalu klik **Deteksi**",
            )

    predict_btn.click(
        fn=predict_comment_ui,
        inputs=text_input,
        outputs=output,
    )

    gr.Markdown(
        "---\n"
        "🔌 **API**: Gunakan endpoint `POST /predict` dengan JSON `{\"text\": \"...\"}`\n\n"
        "*Model: BiGRU • NLP Final Project*",
        elem_classes="footer"
    )

# Mount Gradio on root, FastAPI endpoints remain accessible
app = gr.mount_gradio_app(app, demo, path="/")