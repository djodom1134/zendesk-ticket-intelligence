#!/bin/bash
set -e

echo "🔍 Running ZTI Health Checks..."

# Function to check service health
check_service() {
    local service=$1
    local url=$2
    local max_attempts=30
    local attempt=1

    echo -n "Checking $service... "
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            echo "✅ OK"
            return 0
        fi
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "❌ FAILED"
    return 1
}

# Check if services are running
echo "📋 Checking Docker services..."
docker-compose ps

echo ""
echo "🌐 Checking service endpoints..."

# Check Qdrant
check_service "Qdrant" "http://localhost:6333/health"

# Check ArangoDB
check_service "ArangoDB" "http://localhost:8529/_api/version"

# Check Ollama
check_service "Ollama" "http://localhost:11434/api/tags"

# Verify GPU usage for Ollama
echo ""
echo "🎮 Checking GPU acceleration..."
if docker-compose exec ollama nvidia-smi > /dev/null 2>&1; then
    echo "✅ GPU accessible in Ollama container"
    
    # Check if Ollama is actually using GPU
    if docker-compose exec ollama ollama ps | grep -q "GPU"; then
        echo "✅ Ollama using GPU acceleration"
    else
        echo "⚠️  Ollama may be running on CPU - check OLLAMA_LLM_LIBRARY setting"
    fi
else
    echo "⚠️  GPU not accessible in Ollama container"
fi

# Check UI accessibility
echo ""
echo "🖥️  Checking UI accessibility..."
if check_service "ZTI UI" "http://localhost:3000"; then
    echo "🌐 UI available at: http://localhost:3000"
fi

# Check Chat API
if check_service "Chat API" "http://localhost:8001/health"; then
    echo "💬 Chat API available at: http://localhost:8001"
fi

echo ""
echo "✅ Health check complete!"
echo ""
echo "Services:"
echo "- UI: http://localhost:3000"
echo "- Chat API: http://localhost:8001"
echo "- ArangoDB: http://localhost:8529"
echo "- Qdrant: http://localhost:6333"
echo "- Ollama: http://localhost:11434"