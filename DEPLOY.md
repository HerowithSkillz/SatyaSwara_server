# 🚀 Deploy to DigitalOcean Droplet

## Quick Deployment Steps

### 1. Upload to Droplet
```bash
# From your local machine
rsync -avz --exclude 'venv' --exclude '.git' --exclude '__pycache__' \
  /home/dashrath/Desktop/dumb_workspace/aivoice_detection/ \
  root@YOUR_DROPLET_IP:/root/aivoice_detection/
```

### 2. SSH into Droplet
```bash
ssh root@YOUR_DROPLET_IP
```

### 3. Install Docker (if not installed)
```bash
# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
apt install docker-compose -y

# Verify installation
docker --version
docker-compose --version
```

### 4. Build and Run
```bash
cd /root/aivoice_detection

# Build the Docker image
docker-compose build

# Start the service
docker-compose up -d

# Check if it's running
docker-compose ps
docker-compose logs -f
```

### 5. Install and Setup ngrok
```bash
# Download ngrok
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar xvzf ngrok-v3-stable-linux-amd64.tgz
sudo mv ngrok /usr/local/bin/

# Authenticate with your token (removes banner, enables more features)
ngrok authtoken 38vlNGmeEYMB8Me64qbih2dkDy9_6qFouJKJvigaXSuuiodqE

# Start ngrok in background
nohup ngrok http 8000 > ngrok.log 2>&1 &

# Wait a few seconds for ngrok to start
sleep 5

# Get your public URL
curl -s http://localhost:4040/api/tunnels | grep -o 'https://[^"]*ngrok[^"]*'
```

### 6. Test It
```bash
# From droplet
curl http://localhost:8000/health

# Get ngrok URL and test
NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o 'https://[^"]*ngrok[^"]*' | head -1)
echo "Your public URL: $NGROK_URL"
curl $NGROK_URL/health
```

### 7. Your Public Endpoint
```
https://YOUR-RANDOM-ID.ngrok-free.app/api/voice-detection
```

Get the URL from step 6 above.

## 🔧 Management Commands

```bash
# View API logs
docker-compose logs -f

# View ngrok logs
tail -f ngrok.log

# Get ngrok URL anytime
curl -s http://localhost:4040/api/tunnels | grep -o 'https://[^"]*ngrok[^"]*'

# Restart API service
docker-compose restart

# Stop everything
docker-compose down
pkill ngrok

# Rebuild after changes
docker-compose down
docker-compose build --no-cache
docker-compose up -d
nohup ngrok http 8000 > ngrok.log 2>&1 &
```

## 🔄 Keep ngrok Running (Auto-restart)

Create a systemd service for ngrok:

```bash
# Create service file
cat > /etc/systemd/system/ngrok.service << 'EOF'
[Unit]
Description=ngrok tunnel
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/aivoice_detection
ExecStart=/usr/local/bin/ngrok http 8000 --log=stdout
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
systemctl daemon-reload
systemctl enable ngrok
systemctl start ngrok

# Check status
systemctl status ngrok

# Get URL
sleep 3
curl -s http://localhost:4040/api/tunnels | grep -o 'https://[^"]*ngrok[^"]*'
```

## 📝 API Details for Evaluators

**Endpoint:** `https://YOUR-RANDOM-ID.ngrok-free.app/api/voice-detection`

(Get exact URL by running: `curl -s http://localhost:4040/api/tunnels | grep -o 'https://[^"]*ngrok[^"]*'` on the droplet)

**Method:** `POST`

**Headers:**
```
x-api-key: sk_live_ai_voice_detect_2026_xKp9Qm3R
Content-Type: application/json
```

**Request Body:**
```json
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "BASE64_ENCODED_MP3_STRING"
}
```

**Response:**
```json
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.9523,
  "explanation": "Unnatural pitch consistency and robotic speech patterns detected..."
}
```

## 🛡️ Security Notes (localhost only)
- ngrok provides the public HTTPS URL
- API key authentication is required
- Container auto-restarts if it crashes
- Health checks run every 30 seconds
- Droplet IP is hidden behind ngrok tunnel

## 💡 Important Notes

**ngrok free tier limitations:**
- URL changes if ngrok restarts (use systemd service to keep it stable)
- Session timeout after 8 hours (systemd auto-restarts it)
- For production: get ngrok paid plan for static domain

**Recommended: ngrok paid plan ($8/month)**
```bash
# With paid plan, get a static domain
ngrok http 8000 --domain=your-static-name.ngrok-free.app
```

This gives you a permanent URL that never changes.
- Container auto-restarts if it crashes
- Health checks run every 30 seconds
