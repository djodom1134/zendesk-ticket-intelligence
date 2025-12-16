"""
ZTI Embed/Cluster Service
Generates embeddings and clusters tickets into problem families

Components:
- embedder.py: TicketEmbedder for generating embeddings via Ollama or sentence-transformers
- vector_store.py: VectorStore for Qdrant integration
- clusterer.py: TicketClusterer for UMAP + HDBSCAN clustering
- labeler.py: ClusterLabeler for LLM-generated cluster summaries
- cli.py: Command-line interface for embedding and search
"""

from .embedder import TicketEmbedder, get_embedder
from .vector_store import VectorStore
from .clusterer import TicketClusterer, ClusterResult
from .labeler import ClusterLabeler, ClusterSummary, summary_to_dict

__all__ = [
    "TicketEmbedder",
    "get_embedder",
    "VectorStore",
    "TicketClusterer",
    "ClusterResult",
    "ClusterLabeler",
    "ClusterSummary",
    "summary_to_dict",
]

