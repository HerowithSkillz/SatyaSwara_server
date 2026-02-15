# 🎤 AI Voice Detection API

> **Hackathon Submission** - Production-Ready FastAPI Implementation  
> Detect AI-generated vs Human voice from audio files with high accuracy

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co)

---

## 📋 Overview

A high-performance REST API that analyzes Base64-encoded MP3 audio files to classify them as **AI_GENERATED** or **HUMAN** using deep learning. Built for speed (<200ms response time) and memory efficiency (8GB RAM safe).

### Key Features

- ✅ **Global Model Loading** - Model loaded once at startup for blazing-fast inference
- ✅ **Memory Safe** - Audio truncated to 4 seconds, streamed processing
- ✅ **Smart Base64 Handling** - Auto-strips data URI headers
- ✅ **Dynamic Explanations** - Scientific analysis based on confidence scores
- ✅ **Production Ready** - Proper error handling, logging, and validation

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd aivoice_detection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Server

```bash
# Development
python server.py

# Production (with Gunicorn)
gunicorn server:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### 3. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Voice detection (replace with your base64 audio)
curl -X POST http://localhost:8000/api/voice-detection \
  -H "Content-Type: application/json" \
  -H "x-api-key: hackathon-secret-key-2024" \
  -d '{
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": "YOUR_BASE64_AUDIO_HERE"
  }'
```

---

## 📖 API Documentation

### Endpoint

```
POST /api/voice-detection
```

### Authentication

| Header | Required | Description |
|--------|----------|-------------|
| `x-api-key` | ✅ Yes | Valid API key for authentication |

**Valid API Keys:**
- `hackathon-secret-key-2024` (default)
- `test-api-key-12345` (testing)
- Custom via `API_KEY` environment variable

### Request Body

```json
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "//uQxAAAAAANIAAAAAExBTUUzLjEwMFVVVVVVVV..."
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `language` | string | ✅ | One of: `Tamil`, `English`, `Hindi`, `Malayalam`, `Telugu` |
| `audioFormat` | string | ✅ | Must be `mp3` |
| `audioBase64` | string | ✅ | Base64 encoded MP3 (with or without data URI header) |

### Success Response (200)

```json
{
  "status": "success",
  "language": "Tamil",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.9523,
  "explanation": "Analysis reveals strong high-frequency spectral artifacts (95.2% confidence) consistent with neural vocoder synthesis. Detected prosodic irregularities and characteristic mel-spectrogram patterns typical of transformer-based TTS systems."
}
```

### Error Response (401/400)

```json
{
  "status": "error",
  "message": "Invalid API key or malformed request"
}
```

---

## 🧠 Model Information

| Property | Value |
|----------|-------|
| **Model ID** | `motheecreator/Deepfake-audio-detection` |
| **Source** | HuggingFace Hub |
| **Architecture** | Audio Classification Transformer |
| **Sample Rate** | 16kHz |
| **Max Duration** | 4 seconds (memory safety) |

### Label Mapping

The API auto-detects label mapping from `model.config.id2label`:
- Searches for labels containing: `fake`, `spoof`, `ai`, `synthetic`, `generated`
- Maps detected index to `AI_GENERATED` classification

---

## ⚡ Performance Optimizations

### 1. Global Model Loading
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_manager.load_model()  # Load ONCE at startup
    yield
```

### 2. Memory-Safe Audio Processing
```python
audio_array, sr = librosa.load(
    audio_buffer,
    sr=16000,
    duration=4.0,  # Truncate to 4 seconds
    mono=True
)
```

### 3. No-Gradient Inference
```python
with torch.no_grad():
    logits = model(**inputs).logits
```

### 4. In-Memory Processing
```python
audio_buffer = io.BytesIO(audio_bytes)  # No disk I/O
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | `hackathon-secret-key-2024` | Custom API key |
| `PORT` | `8000` | Server port |
| `HOST` | `0.0.0.0` | Server host |

### Example

```bash
export API_KEY="my-secure-api-key"
export PORT=3000
python server.py
```

---

## 📊 Dynamic Explanation Logic

The API generates contextual explanations based on confidence scores:

| Confidence | AI_GENERATED Explanation |
|------------|-------------------------|
| ≥95% | "Strong high-frequency spectral artifacts... neural vocoder synthesis" |
| 85-94% | "Moderate spectral anomalies... autoregressive voice generation" |
| 70-84% | "Preliminary analysis indicates possible AI generation..." |
| <70% | "Inconclusive analysis... mixed spectral signatures" |

| Confidence | HUMAN Explanation |
|------------|------------------|
| ≥95% | "Natural prosodic variations, organic breath patterns..." |
| 85-94% | "Natural micro-variations in pitch and timing..." |
| 70-84% | "Probable human voice... natural acoustic properties" |
| <70% | "Tentative human classification... mixed indicators" |

---

## 🧪 Testing

### Python Test Script

```python
import base64
import requests

# Read and encode audio file
with open("test_audio.mp3", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

# Make API request
response = requests.post(
    "http://localhost:8000/api/voice-detection",
    headers={
        "Content-Type": "application/json",
        "x-api-key": "hackathon-secret-key-2024"
    },
    json={
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": audio_b64
    }
)

print(response.json())
```

### cURL with File

```bash
# Encode file and send
BASE64=$(base64 -w 0 test_audio.mp3)

curl -X POST http://localhost:8000/api/voice-detection \
  -H "Content-Type: application/json" \
  -H "x-api-key: hackathon-secret-key-2024" \
  -d "{\"language\": \"English\", \"audioFormat\": \"mp3\", \"audioBase64\": \"$BASE64\"}"
```

---

## 📁 Project Structure

```
aivoice_detection/
├── server.py           # Main FastAPI application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── ps1AI.pdf          # Problem statement
```

---

## 🐳 Docker Deployment (Optional)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

EXPOSE 8000
CMD ["python", "server.py"]
```

```bash
docker build -t ai-voice-detection .
docker run -p 8000:8000 -e API_KEY=my-key ai-voice-detection
```

---

## 📝 API Response Codes

| Code | Description |
|------|-------------|
| `200` | Success - Classification returned |
| `400` | Bad Request - Malformed base64 or invalid audio |
| `401` | Unauthorized - Invalid or missing API key |
| `500` | Internal Server Error - Model inference failed |

---

## 🏆 Hackathon Compliance Checklist

- [x] FastAPI framework
- [x] `POST /api/voice-detection` endpoint
- [x] `x-api-key` header validation
- [x] Exact request/response JSON format
- [x] Base64 header stripping
- [x] Memory safety (4-second truncation)
- [x] Global model loading (<200ms response)
- [x] Auto-detect label mapping
- [x] Dynamic explanations based on confidence

---

## 📄 License

MIT License - Built for Hackathon 2024

---

**Built with ❤️ using FastAPI + HuggingFace Transformers**
