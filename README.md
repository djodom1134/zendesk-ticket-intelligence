# Zendesk Ticket Intelligence (ZTI)

A local-first system built on NVIDIA's txt2kg playbook that ingests Zendesk tickets, converts them into knowledge graphs with embeddings, clusters them into "problem families," and provides interactive visualization and Tier-0 chatbot capabilities.

## Overview

- **Batch + Real-time Ingestion**: Pull tickets from local Zendesk agent (http://192.168.87.79:10004)
- **Knowledge Graph + Clustering**: Convert tickets to graph entities and semantic clusters
- **Interactive UI**: Explore clusters, trends, duplicates, and root causes
- **Tier-0 Chatbot**: Local Ollama LLM for ticket routing and response suggestions

## Architecture

Built on NVIDIA DGX Spark playbook with:
- **Graph DB**: ArangoDB
- **Vector DB**: Qdrant (ARM-friendly)
- **LLM**: Local Ollama with GPU acceleration
- **UI**: Adapted from txt2kg visualization

## Quick Start

```bash
# Clone with txt2kg dependency
git clone --recursive https://github.com/[username]/zendesk-ticket-intelligence.git
cd zendesk-ticket-intelligence

# Bootstrap environment
./scripts/bootstrap.sh

# Start services
docker-compose up -d

# Verify GPU acceleration
./scripts/healthcheck.sh
```

## Project Status

🚧 **In Development** - See [Project Board](../../projects/1) for current progress

## Documentation

- [PRD](prd/zen.md) - Complete project requirements
- [Architecture](docs/architecture.md) - System design details
- [Deployment](docs/deployment.md) - Setup and configuration