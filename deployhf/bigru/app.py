from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import re

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


# --- Preprocessing (from Notebook/GRU_model.ipynb) ---
def clean_text(text, slang_map):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', 'link_website', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = text.split()
    normalized_words = [slang_map.get(word, word) for word in words]
    return " ".join(normalized_words)


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
    cleaned = clean_text(req.text, SLANG_JUDOL_MAP)
    seq = text_to_sequence(cleaned, vocab, MAX_LEN)
    tensor_input = torch.tensor([seq], dtype=torch.long)
    with torch.no_grad():
        score = torch.sigmoid(model(tensor_input)).item()
    label = "judol" if score >= 0.5 else "bukan_judol"
    return {"label": label, "score": score}


@app.get("/")
def health_check():
    return {"status": "API BiGRU Judol is running!"}
