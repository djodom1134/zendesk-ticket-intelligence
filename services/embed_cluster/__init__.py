"""
ZTI Embed/Cluster Service
Generates embeddings and clusters tickets into problem families

Components:
- embedder.py: TicketEmbedder for generating embeddings via Ollama or sentence-transformers
- vector_store.py: VectorStore for Qdrant integration
- cli.py: Command-line interface for embedding and search
"""

from .embedder import TicketEmbedder, get_embedder
from .vector_store import VectorStore

__all__ = ["TicketEmbedder", "get_embedder", "VectorStore"]

