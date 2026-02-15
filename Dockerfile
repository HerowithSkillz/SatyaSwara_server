FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server.py .
COPY .env .

# Copy the pre-downloaded model
COPY Deepfake-audio-detection ./Deepfake-audio-detection

# Expose port
EXPOSE 8000

# Run the server
CMD ["python", "server.py"]
