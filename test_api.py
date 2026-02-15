#!/usr/bin/env python3
"""Test script for AI Voice Detection API"""

import requests
import json

API_URL = "http://localhost:8000/api/voice-detection"
API_KEY = "sk_live_ai_voice_detect_2026_xKp9Qm3R"

def test_audio_file(filename, language="English"):
    """Test the API with a base64 file"""
    
    print(f"\n{'='*60}")
    print(f"Testing: {filename}")
    print('='*60)
    
    # Read base64 audio
    try:
        with open(filename, "r") as f:
            audio_b64 = f.read().strip()
        print(f"✓ Loaded {len(audio_b64)} chars of base64")
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return
    
    # Make API request
    try:
        response = requests.post(
            API_URL,
            headers={
                "Content-Type": "application/json",
                "x-api-key": API_KEY
            },
            json={
                "language": language,
                "audioFormat": "mp3",
                "audioBase64": audio_b64
            },
            timeout=30
        )
        
        print(f"\n📡 Response Status: {response.status_code}")
        print(f"\n📄 Response Body:")
        print(json.dumps(response.json(), indent=2))
        
    except requests.exceptions.Timeout:
        print("✗ Request timed out")
    except Exception as e:
        print(f"✗ Error: {e}")

# Test both files
if __name__ == "__main__":
    test_audio_file("base64.txt")
    test_audio_file("base64 (1).txt")
