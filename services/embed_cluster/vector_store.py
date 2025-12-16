"""
Qdrant Vector Store
Manages embedding storage and similarity search
"""

from typing import Any, Optional

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, PointStruct, VectorParams

logger = structlog.get_logger()

# Collection names
TICKET_COLLECTION = "ticket_embeddings"
CLUSTER_COLLECTION = "cluster_embeddings"


class VectorStore:
    """
    Qdrant vector store for ticket embeddings.

    Supports:
    - Storing ticket embeddings with metadata
    - Similarity search
    - Batch operations
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection: str = TICKET_COLLECTION,
        dimension: int = 768,
    ):
        """
        Args:
            host: Qdrant host
            port: Qdrant port
            collection: Collection name
            dimension: Vector dimension
        """
        self.host = host
        self.port = port
        self.collection = collection
        self.dimension = dimension
        self._client: Optional[QdrantClient] = None
        self._connect()

    def _connect(self):
        """Connect to Qdrant"""
        try:
            self._client = QdrantClient(host=self.host, port=self.port)
            self._ensure_collection()
            logger.info("Connected to Qdrant", host=self.host, collection=self.collection)
        except Exception as e:
            logger.error("Failed to connect to Qdrant", error=str(e))
            raise

    def _ensure_collection(self):
        """Ensure collection exists with proper config"""
        collections = self._client.get_collections().collections
        exists = any(c.name == self.collection for c in collections)

        if not exists:
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created collection", name=self.collection)

    def check_health(self) -> bool:
        """Check Qdrant connectivity"""
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False

    def upsert(
        self,
        ticket_id: str,
        vector: list[float],
        payload: dict[str, Any] = None,
    ):
        """
        Upsert a single ticket embedding.

        Args:
            ticket_id: Ticket ID (used as point ID)
            vector: Embedding vector
            payload: Additional metadata
        """
        point = PointStruct(
            id=int(ticket_id),
            vector=vector,
            payload=payload or {},
        )
        self._client.upsert(
            collection_name=self.collection,
            points=[point],
        )

    def upsert_batch(
        self,
        ticket_ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict] = None,
    ):
        """
        Upsert batch of ticket embeddings.

        Args:
            ticket_ids: List of ticket IDs
            vectors: List of embedding vectors
            payloads: List of metadata dicts (optional)
        """
        if payloads is None:
            payloads = [{}] * len(ticket_ids)

        points = [
            PointStruct(
                id=int(tid),
                vector=vec,
                payload=payload,
            )
            for tid, vec, payload in zip(ticket_ids, vectors, payloads)
        ]

        # Batch upsert in chunks
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self._client.upsert(
                collection_name=self.collection,
                points=batch,
            )

        logger.info("Upserted embeddings", count=len(points))

    def search(
        self,
        vector: list[float],
        limit: int = 10,
        score_threshold: float = 0.0,
    ) -> list[dict]:
        """
        Search for similar tickets.

        Args:
            vector: Query embedding
            limit: Max results
            score_threshold: Minimum similarity score

        Returns:
            List of {id, score, payload}
        """
        # qdrant-client 1.7+ uses query() instead of search()
        results = self._client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            score_threshold=score_threshold if score_threshold > 0 else None,
        ).points
        return [
            {"id": str(r.id), "score": r.score, "payload": r.payload}
            for r in results
        ]

