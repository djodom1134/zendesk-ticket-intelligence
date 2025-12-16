"""
Ticket Embedding Service
Generates embeddings for ticket fulltext using sentence-transformers or Ollama
"""

import hashlib
from typing import Any, Optional

import httpx
import numpy as np
import structlog

logger = structlog.get_logger()


class TicketEmbedder:
    """
    Generates embeddings for ticket text.

    Supports two backends:
    1. Ollama (default, GPU-accelerated on server)
    2. Sentence-transformers (local fallback)
    """

    # Max characters to send to embedding model
    # qwen3-embedding has 40K token context window
    # Using 60000 chars to be safe with average of 1.5 chars/token
    MAX_TEXT_LENGTH = 60000

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen3-embedding:8b",
        use_local: bool = False,
        max_length: int = None,
    ):
        """
        Args:
            ollama_url: URL of Ollama server
            model: Embedding model to use
            use_local: If True, use sentence-transformers locally
            max_length: Max text length (default: MAX_TEXT_LENGTH)
        """
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.use_local = use_local
        self.max_length = max_length or self.MAX_TEXT_LENGTH
        self._local_model = None
        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> int:
        """Get embedding dimension (lazy init)"""
        if self._dimension is None:
            # Get dimension from a test embedding
            test_emb = self.embed("test")
            self._dimension = len(test_emb)
        return self._dimension

    def embed(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats (embedding vector)
        """
        if not text or not text.strip():
            # Return zero vector for empty text (4096 dims for qwen3-embedding:8b)
            return [0.0] * (self._dimension or 4096)

        # Truncate text if too long
        original_len = len(text)
        if len(text) > self.max_length:
            text = text[:self.max_length]
            logger.debug("Truncated text", original=original_len, truncated=len(text), max=self.max_length)

        if self.use_local:
            return self._embed_local(text)
        else:
            return self._embed_ollama(text)

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for local model

        Returns:
            List of embedding vectors
        """
        if self.use_local:
            return self._embed_batch_local(texts, batch_size)
        else:
            # Ollama doesn't support batch API, process sequentially
            return [self.embed(text) for text in texts]

    def _embed_ollama(self, text: str) -> list[float]:
        """Generate embedding using Ollama API"""
        # Ensure text is truncated (should already be done by embed())
        if len(text) > self.max_length:
            text = text[:self.max_length]
            logger.debug("Truncated text in _embed_ollama", length=len(text))

        try:
            response = httpx.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                headers={"Content-Type": "application/json"},
                timeout=120.0,
            )
            response.raise_for_status()
            result = response.json()
            embedding = result.get("embedding", [])
            if not embedding:
                raise ValueError("Empty embedding returned")
            return embedding
        except httpx.HTTPStatusError as e:
            # Log the actual error message
            error_text = e.response.text if e.response else str(e)
            logger.error("Ollama HTTP error", status=e.response.status_code, error=error_text[:200])
            raise
        except Exception as e:
            logger.error("Ollama embedding failed", error=str(e), model=self.model)
            # Fallback to local if available
            if not self.use_local:
                logger.info("Falling back to local embedding")
                return self._embed_local(text)
            raise

    def _embed_local(self, text: str) -> list[float]:
        """Generate embedding using sentence-transformers"""
        if self._local_model is None:
            self._init_local_model()

        embedding = self._local_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def _embed_batch_local(self, texts: list[str], batch_size: int) -> list[list[float]]:
        """Generate batch embeddings using sentence-transformers"""
        if self._local_model is None:
            self._init_local_model()

        embeddings = self._local_model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return embeddings.tolist()

    def _init_local_model(self):
        """Initialize local sentence-transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            # Use a good model for semantic search
            self._local_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._dimension = self._local_model.get_sentence_embedding_dimension()
            logger.info("Initialized local embedding model", dim=self._dimension)
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )


def text_hash(text: str) -> str:
    """Generate a hash for text (for deduplication)"""
    return hashlib.md5(text.encode()).hexdigest()[:16]


# Default embedder instance (will use Ollama)
default_embedder: Optional[TicketEmbedder] = None


def get_embedder(ollama_url: str = "http://localhost:11434") -> TicketEmbedder:
    """Get or create default embedder"""
    global default_embedder
    if default_embedder is None:
        default_embedder = TicketEmbedder(ollama_url=ollama_url)
    return default_embedder

