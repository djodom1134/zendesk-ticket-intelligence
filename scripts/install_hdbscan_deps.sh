#!/bin/bash
# Install Python development headers needed for HDBSCAN compilation
# Run this with: sudo bash scripts/install_hdbscan_deps.sh

set -e

echo "Installing Python development headers for HDBSCAN..."
apt-get update
apt-get install -y python3-dev python3.12-dev build-essential

echo "✅ Headers installed. Now run:"
echo "   cd ~/zendesk-ticket-intelligence"
echo "   source venv/bin/activate"
echo "   pip install -U hdbscan"

