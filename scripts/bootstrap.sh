#!/bin/bash
set -e

echo "🚀 Bootstrapping Zendesk Ticket Intelligence (ZTI)..."

# Check if running on GPU machine
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name --format=csv,noheader
else
    echo "⚠️  No NVIDIA GPU detected - Ollama will run on CPU"
fi

# Initialize git submodules
echo "📦 Initializing txt2kg submodule..."
git submodule update --init --recursive

# Verify txt2kg path (fix common path typo issue)
if [ ! -d "dgx-spark-playbooks/nvidia/txt2kg" ]; then
    echo "❌ txt2kg path not found at dgx-spark-playbooks/nvidia/txt2kg"
    echo "   This is a known issue - checking alternative paths..."
    find dgx-spark-playbooks -name "txt2kg" -type d 2>/dev/null || true
    exit 1
fi

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/{raw,qdrant,arangodb,ollama}
mkdir -p logs

# Create shared directory structure
echo "📁 Creating shared directories..."
mkdir -p shared/{schemas,prompts,config,utils}

# Set permissions
chmod +x scripts/*.sh

# Create environment file template
if [ ! -f .env ]; then
    echo "📝 Creating .env template..."
    cat > .env << EOF
# Zendesk Configuration
ZENDESK_AGENT_URL=http://192.168.87.79:10004
ZENDESK_API_TOKEN=your_token_here

# Database Configuration
ARANGODB_ROOT_PASSWORD=zti_password
QDRANT_API_KEY=

# Ollama Configuration
OLLAMA_LLM_LIBRARY=cuda_v13
OLLAMA_MODEL=llama2:7b

# Development
DEBUG=0
LOG_LEVEL=INFO
EOF
    echo "⚠️  Please configure .env file before starting services"
fi

# Pull required Docker images
echo "🐳 Pulling Docker images..."
docker compose -f docker/docker-compose.yml pull

# Download Ollama model
echo "🤖 Setting up Ollama model..."
docker compose -f docker/docker-compose.yml up -d ollama
sleep 10
docker compose -f docker/docker-compose.yml exec ollama ollama pull llama2:7b

echo "✅ Bootstrap complete!"
echo ""
echo "Next steps:"
echo "1. Configure .env file"
echo "2. Run: docker compose -f docker/docker-compose.yml up -d"
echo "3. Run: ./scripts/healthcheck.sh"