#!/bin/bash
# Restart the ZTI API on the GPU server
# This script should be run ON the GPU server (192.168.87.134)

set -e

echo "🔄 Restarting ZTI API on GPU server..."

# Kill existing API process
echo "Stopping existing API process..."
sudo pkill -f 'uvicorn.*services.api.main:app' || echo "No existing process found"

# Wait for process to stop
sleep 3

# Navigate to project directory
cd ~/zendesk-ticket-intelligence

# Pull latest changes
echo "Pulling latest changes..."
git pull origin parallel

# Start API in background
echo "Starting API..."
sudo /usr/local/bin/uvicorn services.api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --log-level info \
  > /tmp/zti-api.log 2>&1 &

# Wait for API to start
sleep 5

# Check if API is running
if curl -s http://localhost:8000/health > /dev/null; then
  echo "✅ API restarted successfully!"
  echo "API is running at http://192.168.87.134:8000"
  echo "Logs: tail -f /tmp/zti-api.log"
else
  echo "❌ API failed to start. Check logs:"
  echo "tail -f /tmp/zti-api.log"
  exit 1
fi

