#!/bin/bash
# Setup script for GPU machine - run this once the machine is accessible
# Usage: ./scripts/setup-gpu-machine.sh

set -e

GPU_HOST="${GPU_HOST:-zti-gpu}"
REPO_URL="https://github.com/djodom1134/zendesk-ticket-intelligence.git"
REMOTE_DIR="~/zendesk-ticket-intelligence"

echo "🔧 ZTI GPU Machine Setup Script"
echo "================================"
echo "Target: $GPU_HOST"
echo ""

# Test SSH connection
echo "1. Testing SSH connection..."
if ! ssh -o ConnectTimeout=10 "$GPU_HOST" "echo 'Connected successfully'" 2>/dev/null; then
    echo "❌ Cannot connect to $GPU_HOST"
    echo ""
    echo "Troubleshooting steps:"
    echo "  1. Ensure the GPU machine is powered on"
    echo "  2. Verify SSH is running: ssh d@192.168.87.79"
    echo "  3. Copy SSH key if needed: ssh-copy-id -i ~/.ssh/id_rsa.pub $GPU_HOST"
    exit 1
fi
echo "✅ SSH connection successful"
echo ""

# Check GPU availability
echo "2. Checking GPU availability..."
ssh "$GPU_HOST" "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv" || {
    echo "⚠️  nvidia-smi not available - GPU may not be configured"
}
echo ""

# Check Docker
echo "3. Checking Docker installation..."
ssh "$GPU_HOST" "docker --version && docker-compose --version" || {
    echo "❌ Docker not installed or not accessible"
    echo "   Please install Docker and Docker Compose on the GPU machine"
    exit 1
}
echo "✅ Docker available"
echo ""

# Check NVIDIA Container Toolkit
echo "4. Checking NVIDIA Container Toolkit..."
if ssh "$GPU_HOST" "docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi" 2>/dev/null; then
    echo "✅ NVIDIA Container Toolkit working"
else
    echo "⚠️  NVIDIA Container Toolkit may not be installed"
    echo "   Install with: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi
echo ""

# Clone or update repository
echo "5. Setting up repository..."
ssh "$GPU_HOST" "
    if [ -d $REMOTE_DIR ]; then
        echo 'Repository exists, pulling latest...'
        cd $REMOTE_DIR && git pull
    else
        echo 'Cloning repository...'
        git clone --recursive $REPO_URL $REMOTE_DIR
    fi
"
echo "✅ Repository ready"
echo ""

# Create .env file if not exists
echo "6. Checking environment configuration..."
ssh "$GPU_HOST" "
    cd $REMOTE_DIR
    if [ ! -f .env ]; then
        cp .env.example .env
        echo '⚠️  Created .env from template - please configure it'
    else
        echo '✅ .env file exists'
    fi
"
echo ""

echo "================================"
echo "✅ GPU machine setup complete!"
echo ""
echo "Next steps:"
echo "  1. SSH to GPU machine: ssh $GPU_HOST"
echo "  2. Configure .env: nano $REMOTE_DIR/.env"
echo "  3. Run bootstrap: cd $REMOTE_DIR && ./scripts/bootstrap.sh"
echo "  4. Start services: docker-compose -f docker/docker-compose.yml up -d"
echo "  5. Check health: ./scripts/healthcheck.sh"
echo ""
echo "For port forwarding, use:"
echo "  ssh -L 3000:localhost:3000 -L 8001:localhost:8001 -L 8529:localhost:8529 $GPU_HOST"

