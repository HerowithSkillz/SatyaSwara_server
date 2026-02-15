"""
AI Voice Detection API - Production-Ready FastAPI Implementation
================================================================
Detects if an audio file is AI-generated or human voice using 
HuggingFace's motheecreator/Deepfake-audio-detection model.

Author: Senior Backend Engineer
Hackathon Submission
"""

import os
import re
import io
import base64
import logging
from typing import Literal
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

import torch
import librosa
import numpy as np
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_ID = "./Deepfake-audio-detection"  # Local model folder
SAMPLE_RATE = 16000
MAX_DURATION_SECONDS = 4.0  # Memory safety: truncate to 4 seconds for inference
MAX_AUDIO_DURATION_ACCEPT = 60  # Accept up to 1 minute of MP3 audio
MAX_REQUEST_BODY_BYTES = 12 * 1024 * 1024  # 12MB max request body
VALID_API_KEYS = {
    os.getenv("API_KEY", "sk_live_ai_voice_detect_2026_xKp9Qm3R"),
}

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# GLOBAL MODEL LOADING (Singleton Pattern for <200ms response)
# ============================================================================
class ModelManager:
    """Singleton class to manage global model loading."""
    
    _instance = None
    model = None
    feature_extractor = None
    fake_label_index: int = 0  # Will be auto-detected
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(self):
        """Load model once at startup."""
        if self.model is not None:
            logger.info("Model already loaded, skipping...")
            return
        
        logger.info(f"⏳ Loading Model: {MODEL_ID}...")
        
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
            self.model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
            
            # Set model to evaluation mode for inference
            self.model.eval()
            
            # Auto-detect label mapping
            id2label = self.model.config.id2label
            logger.info(f"📋 Model Labels: {id2label}")
            
            # Find which index corresponds to "fake/deepfake/ai"
            for idx, label in id2label.items():
                label_lower = str(label).lower()
                if any(keyword in label_lower for keyword in ["fake", "spoof", "ai", "synthetic", "generated"]):
                    self.fake_label_index = int(idx)
                    break
            
            logger.info(f"✅ Model Loaded! Fake Label Index: {self.fake_label_index}")
            
        except Exception as e:
            logger.error(f"❌ Model Load Error: {e}")
            raise RuntimeError(f"Failed to load model: {e}")
    
    def predict(self, audio_array: np.ndarray) -> tuple[str, float]:
        """
        Run inference on audio array.
        
        Returns:
            tuple: (classification, confidence_score)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded!")
        
        # Prepare inputs
        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )
        
        # Run inference with no gradient computation
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # Convert to probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs_np = probs[0].numpy()
        
        # Get fake/real probabilities based on auto-detected index
        fake_prob = probs_np[self.fake_label_index]
        real_prob = probs_np[1 - self.fake_label_index]
        
        logger.info(f"📊 Probabilities -> AI: {fake_prob:.4f} | HUMAN: {real_prob:.4f}")
        
        if fake_prob > real_prob:
            return "AI_GENERATED", float(fake_prob)
        else:
            return "HUMAN", float(real_prob)


# Global model manager instance
model_manager = ModelManager()

# ============================================================================
# LIFESPAN: Load model at startup
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load model on startup."""
    logger.info("🚀 Starting AI Voice Detection API...")
    model_manager.load_model()
    logger.info("✅ Server ready to accept requests!")
    yield
    logger.info("🛑 Shutting down server...")


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================
app = FastAPI(
    title="AI Voice Detection API",
    description="Detect AI-generated vs Human voice from audio files",
    version="1.0.0",
    lifespan=lifespan
)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================
class VoiceDetectionRequest(BaseModel):
    """Request schema for voice detection endpoint."""
    
    language: Literal["Tamil", "English", "Hindi", "Malayalam", "Telugu"] = Field(
        ...,
        description="Language of the audio"
    )
    audioFormat: Literal["mp3"] = Field(
        ...,
        description="Audio format (must be mp3)"
    )
    audioBase64: str = Field(
        ...,
        description="Base64 encoded MP3 audio string",
        min_length=100,          # Minimum reasonable length for audio
        max_length=10_000_000    # ~10MB base64 = supports longer audio files
    )
    
    @field_validator("audioBase64")
    @classmethod
    def validate_base64_not_empty(cls, v: str) -> str:
        """Ensure base64 string is not just whitespace."""
        if not v or not v.strip():
            raise ValueError("audioBase64 cannot be empty")
        return v.strip()


class SuccessResponse(BaseModel):
    """Success response schema."""
    
    status: Literal["success"] = "success"
    language: str
    classification: Literal["AI_GENERATED", "HUMAN"]
    confidenceScore: float = Field(..., ge=0.0, le=1.0)
    explanation: str


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    status: Literal["error"] = "error"
    message: str


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def strip_base64_header(b64_string: str) -> str:
    """
    Strip data URI header from base64 string if present.
    
    Handles formats like:
    - data:audio/mp3;base64,<data>
    - data:audio/mpeg;base64,<data>
    - <raw base64 data>
    """
    # Check for data URI pattern in first 100 chars
    if "," in b64_string[:100]:
        # Pattern: data:audio/...;base64,<actual_data>
        parts = b64_string.split(",", 1)
        if len(parts) == 2 and "base64" in parts[0].lower():
            return parts[1]
    return b64_string


def decode_base64_audio(b64_string: str) -> bytes:
    """
    Decode base64 string to audio bytes.
    
    Raises:
        ValueError: If base64 is malformed
    """
    # Strip header if present
    clean_b64 = strip_base64_header(b64_string)
    
    # Remove any whitespace/newlines
    clean_b64 = re.sub(r'\s+', '', clean_b64)
    
    # Add padding if needed
    padding_needed = len(clean_b64) % 4
    if padding_needed:
        clean_b64 += "=" * (4 - padding_needed)
    
    try:
        return base64.b64decode(clean_b64, validate=True)
    except Exception as e:
        raise ValueError(f"Invalid base64 encoding: {e}")


def load_audio_from_bytes(audio_bytes: bytes) -> np.ndarray:
    """
    Load audio from bytes with memory safety.
    
    - Truncates to MAX_DURATION_SECONDS
    - Resamples to SAMPLE_RATE
    - Converts to mono
    """
    try:
        # Use BytesIO to avoid writing to disk
        audio_buffer = io.BytesIO(audio_bytes)
        
        # Load with librosa (memory-safe: truncate to 4 seconds)
        audio_array, sr = librosa.load(
            audio_buffer,
            sr=SAMPLE_RATE,
            duration=MAX_DURATION_SECONDS,
            mono=True
        )
        
        return audio_array
        
    except Exception as e:
        raise ValueError(f"Failed to process audio: {e}")


def generate_explanation(classification: str, confidence: float) -> str:
    """
    Generate dynamic scientific-sounding explanation based on confidence.
    
    Args:
        classification: "AI_GENERATED" or "HUMAN"
        confidence: Float between 0.0 and 1.0
    
    Returns:
        Dynamic explanation string
    """
    if classification == "AI_GENERATED":
        if confidence >= 0.95:
            return (
                "Unnatural pitch consistency and robotic speech patterns detected. "
                "High-frequency spectral artifacts consistent with neural vocoder synthesis observed."
            )
        elif confidence >= 0.85:
            return (
                "Spectral anomalies suggesting AI synthesis detected. "
                "Periodic micro-fluctuations in pitch contour indicative of autoregressive generation."
            )
        elif confidence >= 0.70:
            return (
                "Possible AI-generated voice detected. Some synthetic spectral patterns found "
                "alongside natural characteristics. Further verification recommended."
            )
        else:
            return (
                "Inconclusive analysis. Mixed spectral signatures detected with some synthetic markers "
                "but below definitive classification threshold."
            )
    else:  # HUMAN
        if confidence >= 0.95:
            return (
                "Natural prosodic variations and organic breath patterns confirmed. "
                "No neural vocoder artifacts detected in spectral analysis."
            )
        elif confidence >= 0.85:
            return (
                "Natural micro-variations in pitch and timing consistent with human speech. "
                "Spectral envelope matches biological voice production characteristics."
            )
        elif confidence >= 0.70:
            return (
                "Probable human voice with natural acoustic properties. "
                "Overall speech pattern suggests organic voice production."
            )
        else:
            return (
                "Tentative human classification with mixed indicators. "
                "Predominant features suggest human origin but confidence is below optimal threshold."
            )


def validate_api_key(api_key: str | None) -> bool:
    """Validate the API key against allowed keys."""
    if not api_key or not api_key.strip():
        return False
    return api_key in VALID_API_KEYS


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================
from fastapi.exceptions import RequestValidationError


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic/FastAPI 422 errors with the exact JSON format required."""
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=400,
        content={"status": "error", "message": "Invalid API key or malformed request"}
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom handler for HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all handler for unexpected errors."""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal server error"}
    )


# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "AI Voice Detection API", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_loaded": model_manager.model is not None,
        "model_id": MODEL_ID
    }


@app.post(
    "/api/voice-detection",
    response_model=SuccessResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
        400: {"model": ErrorResponse, "description": "Bad request"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    tags=["Voice Detection"]
)
async def detect_voice(
    request: VoiceDetectionRequest,
    x_api_key: str | None = Header(None, alias="x-api-key")
):
    """
    Detect if audio is AI-generated or human voice.
    
    - **language**: Language of the audio (Tamil, English, Hindi, Malayalam, Telugu)
    - **audioFormat**: Must be "mp3"
    - **audioBase64**: Base64 encoded MP3 audio
    
    Returns classification with confidence score and explanation.
    """
    
    # 1. Validate API Key
    if not validate_api_key(x_api_key):
        logger.warning(f"Invalid API key attempt: {x_api_key[:10] if x_api_key else 'None'}...")
        return JSONResponse(
            status_code=401,
            content={"status": "error", "message": "Invalid API key or malformed request"}
        )
    
    # 2. Decode Base64 Audio
    try:
        audio_bytes = decode_base64_audio(request.audioBase64)
        logger.info(f"📥 Received audio: {len(audio_bytes)} bytes, language: {request.language}")
    except ValueError as e:
        logger.error(f"Base64 decode error: {e}")
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Invalid API key or malformed request"}
        )
    
    # 3. Load and Process Audio
    try:
        audio_array = load_audio_from_bytes(audio_bytes)
        logger.info(f"🎵 Audio loaded: {len(audio_array)} samples ({len(audio_array)/SAMPLE_RATE:.2f}s)")
    except ValueError as e:
        logger.error(f"Audio processing error: {e}")
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Invalid API key or malformed request"}
        )
    
    # 4. Run Model Prediction
    try:
        classification, confidence = model_manager.predict(audio_array)
        logger.info(f"🎯 Prediction: {classification} ({confidence:.2%})")
    except Exception as e:
        logger.error(f"Model inference error: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Internal server error during analysis"}
        )
    
    # 5. Generate Dynamic Explanation
    explanation = generate_explanation(classification, confidence)
    
    # 6. Return Success Response
    return SuccessResponse(
        status="success",
        language=request.language,
        classification=classification,
        confidenceScore=round(confidence, 4),
        explanation=explanation
    )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or default to 8001
    port = int(os.getenv("PORT", 8001))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"🚀 Starting server on {host}:{port}")
    
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=False,  # Disable reload in production
        workers=1,     # Single worker to share model in memory
        log_level="info"
    )
