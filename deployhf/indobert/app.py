from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import os

app = FastAPI(title="IndoBERT Judol API")

# PENTING: Izinkan ekstensi browser Anda untuk memanggil API ini (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Di produksi, Anda bisa membatasi ini
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🔄 Sedang memuat model IndoBERT... (Ini mungkin memakan waktu beberapa menit)")
# Load model langsung dari folder model atau dari repo HF Anda
# Pastikan file model (config.json, model.safetensors, dll) ada di repo Space ini
model_path = "./" 
classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)
print("✅ Model berhasil dimuat!")

class CommentRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_comment(req: CommentRequest):
    # Jalankan inferensi
    result = classifier(req.text)[0]
    
    # Sesuaikan output agar langsung bisa dibaca oleh service-worker.js Anda
    # Ingat: Jika config.json sudah diperbaiki, labelnya adalah 'JUDOL' atau 'BUKAN JUDOL'
    raw_label = result['label'].upper()
    label = "judol" if raw_label == "JUDOL" else "bukan_judol"
    
    return {
        "label": label,
        "score": float(result['score'])
    }

@app.get("/")
def health_check():
    return {"status": "API IndoBERT Judol is running!"}