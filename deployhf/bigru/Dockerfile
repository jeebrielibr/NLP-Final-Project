FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy file aplikasi dan model
COPY . .

# Jalankan server menggunakan Uvicorn
# Port 7860 adalah port standar yang digunakan Hugging Face Spaces
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
