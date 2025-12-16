"""
Incremental Ingest Pipeline
Processes new tickets in near-real-time with delta detection.
Can be run on a schedule (cron) or triggered via webhook.
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Optional

import httpx
import structlog
from arango import ArangoClient

from services.embed_cluster.embedder import TicketEmbedder
from services.embed_cluster.normalizer import TicketNormalizer
from services.embed_cluster.vector_store import VectorStore

logger = structlog.get_logger()

# Configuration
ARANGODB_HOST = os.getenv("ARANGODB_HOST", "localhost")
ARANGODB_PORT = int(os.getenv("ARANGODB_PORT", "8529"))
ARANGODB_DB = os.getenv("ARANGODB_DB", "zti")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:8b")
MCP_URL = os.getenv("MCP_URL", "http://localhost:10005/sse")


class IncrementalPipeline:
    """Process new tickets incrementally with cluster assignment."""

    def __init__(
        self,
        arango_host: str = ARANGODB_HOST,
        arango_port: int = ARANGODB_PORT,
        arango_db: str = ARANGODB_DB,
        qdrant_host: str = QDRANT_HOST,
        qdrant_port: int = QDRANT_PORT,
        ollama_url: str = OLLAMA_URL,
        embed_model: str = EMBED_MODEL,
    ):
        self.arango_client = ArangoClient(hosts=f"http://{arango_host}:{arango_port}")
        self.db = self.arango_client.db(arango_db, verify=True)
        self.normalizer = TicketNormalizer()
        self.embedder = TicketEmbedder(ollama_url=ollama_url, model=embed_model)
        self.vector_store = VectorStore(host=qdrant_host, port=qdrant_port)
        self.vector_store.ensure_collection()

    def get_last_sync_time(self) -> Optional[datetime]:
        """Get timestamp of last successful sync."""
        try:
            meta = self.db.collection("_sync_meta").get("last_sync")
            if meta and meta.get("timestamp"):
                return datetime.fromisoformat(meta["timestamp"])
        except Exception:
            pass
        return None

    def set_last_sync_time(self, ts: datetime):
        """Update last sync timestamp."""
        try:
            if not self.db.has_collection("_sync_meta"):
                self.db.create_collection("_sync_meta")
            self.db.collection("_sync_meta").insert(
                {"_key": "last_sync", "timestamp": ts.isoformat()},
                overwrite=True
            )
        except Exception as e:
            logger.warning("Failed to update sync meta", error=str(e))

    def get_known_ticket_ids(self) -> set[str]:
        """Get set of ticket IDs already in the database."""
        try:
            cursor = self.db.aql.execute("FOR t IN tickets RETURN t._key")
            return set(cursor)
        except Exception:
            return set()

    def fetch_new_tickets(self, since: Optional[datetime] = None, days: int = 1) -> list[dict]:
        """Fetch tickets from Zendesk MCP that are newer than since."""
        # This would call the Zendesk MCP - for now stub with file-based approach
        logger.info("Fetching new tickets", since=since, days=days)
        # In production, this would use MCP client to fetch recent tickets
        return []

    def assign_cluster(self, embedding: list[float]) -> tuple[Optional[str], float]:
        """Assign ticket to nearest cluster using k-NN voting."""
        try:
            results = self.vector_store.search(embedding, top_k=10)
            if not results:
                return None, 0.0

            # Vote on cluster
            cluster_votes: dict[str, float] = {}
            for score, payload in results:
                if payload and payload.get("cluster_id"):
                    cid = payload["cluster_id"]
                    cluster_votes[cid] = cluster_votes.get(cid, 0) + score

            if not cluster_votes:
                return None, 0.0

            best_cluster = max(cluster_votes, key=cluster_votes.get)
            confidence = cluster_votes[best_cluster] / sum(cluster_votes.values())
            return best_cluster, confidence
        except Exception as e:
            logger.warning("Cluster assignment failed", error=str(e))
            return None, 0.0

    def process_ticket(self, raw_ticket: dict) -> dict:
        """Process a single ticket: normalize, embed, assign cluster, store."""
        ticket_id = str(raw_ticket.get("id") or raw_ticket.get("ticket_id"))
        logger.info("Processing ticket", ticket_id=ticket_id)

        # Normalize
        normalized = self.normalizer.normalize(raw_ticket)

        # Embed
        text = normalized.get("ticket_fulltext", "")
        embedding = self.embedder.embed(text)

        # Assign cluster
        cluster_id, confidence = self.assign_cluster(embedding)
        normalized["cluster_id"] = cluster_id
        normalized["cluster_confidence"] = confidence

        # Store in Qdrant
        self.vector_store.upsert(
            ticket_id=ticket_id,
            embedding=embedding,
            payload={
                "subject": normalized.get("subject", ""),
                "status": normalized.get("status", ""),
                "cluster_id": cluster_id,
                "created_at": normalized.get("created_at"),
            }
        )

        # Store in ArangoDB
        normalized["_key"] = ticket_id
        self.db.collection("tickets").insert(normalized, overwrite=True)

        logger.info("Ticket processed", ticket_id=ticket_id, cluster=cluster_id, confidence=f"{confidence:.2f}")
        return normalized

    def run(self, tickets: Optional[list[dict]] = None, days: int = 1) -> dict:
        """Run incremental pipeline on new tickets."""
        start = time.time()
        
        if tickets is None:
            since = self.get_last_sync_time()
            tickets = self.fetch_new_tickets(since=since, days=days)

        known_ids = self.get_known_ticket_ids()
        new_tickets = [t for t in tickets if str(t.get("id") or t.get("ticket_id")) not in known_ids]

        logger.info("Incremental run", total=len(tickets), new=len(new_tickets), known=len(known_ids))

        processed = 0
        for ticket in new_tickets:
            try:
                self.process_ticket(ticket)
                processed += 1
            except Exception as e:
                logger.error("Failed to process ticket", error=str(e))

        self.set_last_sync_time(datetime.utcnow())
        elapsed = time.time() - start

        return {"processed": processed, "skipped": len(tickets) - len(new_tickets), "elapsed_seconds": elapsed}

