#!/bin/bash
# ZTI Health Check Script
# Verifies all services are running and GPU is properly configured
#
# Addresses PRD Section 5 requirements:
# - GPU self-test that FAILS LOUDLY if Ollama not using GPU
# - Validates all infrastructure services
# - Checks database initialization

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
STRICT_GPU=true
COMPOSE_FILE="${COMPOSE_FILE:-docker/docker-compose.yml}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-strict-gpu)
            STRICT_GPU=false
            shift
            ;;
        -f|--file)
            COMPOSE_FILE="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "🔍 Running ZTI Health Checks..."
echo "   Compose file: $COMPOSE_FILE"
echo ""

FAILED=0

# Function to check service health
check_service() {
    local service=$1
    local url=$2
    local max_attempts=${3:-10}
    local attempt=1

    printf "  %-20s" "$service..."

    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ OK${NC}"
            return 0
        fi
        sleep 2
        attempt=$((attempt + 1))
    done

    echo -e "${RED}❌ FAILED${NC}"
    return 1
}

# Check if Docker is running
echo "📋 Checking Docker..."
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running or not accessible${NC}"
    exit 1
fi
echo -e "  Docker:             ${GREEN}✅ OK${NC}"

# Check if services are running
echo ""
echo "📋 Docker Service Status:"
docker compose -f "$COMPOSE_FILE" ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || docker compose -f "$COMPOSE_FILE" ps

echo ""
echo "🌐 Service Health Endpoints:"

# Check Qdrant
if ! check_service "Qdrant" "http://localhost:6333/health"; then
    FAILED=$((FAILED + 1))
fi

# Check ArangoDB
if ! check_service "ArangoDB" "http://localhost:8529/_api/version"; then
    FAILED=$((FAILED + 1))
fi

# Check Ollama
if ! check_service "Ollama" "http://localhost:11434/api/tags"; then
    FAILED=$((FAILED + 1))
fi

# ================== GPU VERIFICATION (CRITICAL) ==================
echo ""
echo "🎮 GPU Verification (CRITICAL):"

# Check host GPU
printf "  %-20s" "Host GPU..."
if nvidia-smi > /dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo -e "${GREEN}✅ $GPU_NAME${NC}"
else
    echo -e "${YELLOW}⚠️  Not detected on host${NC}"
fi

# Check GPU in Ollama container
printf "  %-20s" "Container GPU..."
if docker compose -f "$COMPOSE_FILE" exec -T zti-ollama nvidia-smi > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Accessible${NC}"

    # CRITICAL: Check if Ollama is actually using GPU
    printf "  %-20s" "Ollama GPU Mode..."
    OLLAMA_PS=$(docker compose -f "$COMPOSE_FILE" exec -T zti-ollama ollama ps 2>/dev/null || echo "")

    if echo "$OLLAMA_PS" | grep -qi "gpu\|cuda"; then
        echo -e "${GREEN}✅ Using GPU${NC}"
    elif [ -z "$OLLAMA_PS" ] || echo "$OLLAMA_PS" | grep -q "no models"; then
        echo -e "${YELLOW}⚠️  No models loaded (pull a model first)${NC}"
    else
        echo -e "${RED}❌ RUNNING ON CPU!${NC}"
        echo ""
        echo -e "${RED}╔══════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║  CRITICAL: Ollama is NOT using GPU acceleration!            ║${NC}"
        echo -e "${RED}║                                                              ║${NC}"
        echo -e "${RED}║  This will result in extremely slow inference.              ║${NC}"
        echo -e "${RED}║  Check NVIDIA_VISIBLE_DEVICES and container GPU access.     ║${NC}"
        echo -e "${RED}╚══════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        if [ "$STRICT_GPU" = true ]; then
            echo "Failing health check due to GPU requirement."
            echo "Use --no-strict-gpu to bypass this check."
            exit 1
        fi
        FAILED=$((FAILED + 1))
    fi
else
    echo -e "${RED}❌ NOT accessible in container${NC}"
    if [ "$STRICT_GPU" = true ]; then
        echo ""
        echo -e "${RED}CRITICAL: GPU not accessible in Ollama container!${NC}"
        echo "Check NVIDIA Container Toolkit installation."
        exit 1
    fi
    FAILED=$((FAILED + 1))
fi

# ================== DATABASE VERIFICATION ==================
echo ""
echo "🗄️  Database Verification:"

# Check ArangoDB ZTI database
printf "  %-20s" "ArangoDB 'zti' DB..."
if curl -s "http://localhost:8529/_db/zti/_api/version" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Ready${NC}"
else
    echo -e "${YELLOW}⚠️  Not initialized (run init container)${NC}"
fi

# Check Qdrant collections
printf "  %-20s" "Qdrant collections..."
TICKET_COL=$(curl -s "http://localhost:6333/collections/ticket-embeddings" 2>/dev/null || echo "")
if echo "$TICKET_COL" | grep -q '"status":"ok"'; then
    echo -e "${GREEN}✅ Ready${NC}"
else
    echo -e "${YELLOW}⚠️  Not initialized (run init container)${NC}"
fi

# ================== SUMMARY ==================
echo ""
echo "════════════════════════════════════════════════════════════════"
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All health checks passed!${NC}"
else
    echo -e "${YELLOW}⚠️  $FAILED check(s) failed${NC}"
fi
echo ""
echo "📍 Service URLs:"
echo "   • ArangoDB:  http://localhost:8529"
echo "   • Qdrant:    http://localhost:6333"
echo "   • Ollama:    http://localhost:11434"
echo ""
echo "📝 Next Steps:"
echo "   1. Pull an Ollama model:"
echo "      docker compose -f $COMPOSE_FILE exec zti-ollama ollama pull llama3.1:8b"
echo "   2. Verify GPU usage:"
echo "      docker compose -f $COMPOSE_FILE exec zti-ollama ollama ps"
echo ""

exit $FAILED