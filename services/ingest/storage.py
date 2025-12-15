"""
ArangoDB Storage for ZTI Ingest Service
Handles raw ticket persistence and job status tracking
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import structlog
from arango import ArangoClient
from arango.database import StandardDatabase

logger = structlog.get_logger()

# Data directory for raw JSON backup
DATA_DIR = Path(os.getenv("ZTI_DATA_DIR", "/data/zti"))


class ArangoStorage:
    """ArangoDB storage for tickets and job status"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8529,
        database: str = "zti",
        username: str = "root",
        password: str = "",
    ):
        self.host = host
        self.port = port
        self.database_name = database
        self.username = username
        self.password = password
        self._client: Optional[ArangoClient] = None
        self._db: Optional[StandardDatabase] = None
        self._connect()

    def _connect(self):
        """Establish connection to ArangoDB"""
        try:
            self._client = ArangoClient(hosts=f"http://{self.host}:{self.port}")
            # Connect without auth (dev mode) or with credentials
            if self.password:
                self._db = self._client.db(
                    self.database_name, username=self.username, password=self.password
                )
            else:
                self._db = self._client.db(self.database_name)
            logger.info("Connected to ArangoDB", host=self.host, database=self.database_name)
        except Exception as e:
            logger.error("Failed to connect to ArangoDB", error=str(e))
            raise

    def check_health(self) -> bool:
        """Check database connectivity"""
        try:
            if self._db:
                self._db.version()
                return True
        except Exception as e:
            logger.warning("ArangoDB health check failed", error=str(e))
        return False

    def store_raw_tickets(self, tickets: list[dict], job_id: str) -> int:
        """
        Store raw tickets in ArangoDB and backup to disk

        Args:
            tickets: List of raw ticket dictionaries
            job_id: Ingest job ID for tracking

        Returns:
            Number of tickets stored
        """
        if not tickets:
            return 0

        collection = self._db.collection("raw_tickets")
        stored = 0

        # Backup to disk first
        self._backup_to_disk(tickets, job_id)

        for ticket in tickets:
            try:
                # Add metadata
                doc = {
                    "_key": str(ticket.get("id", ticket.get("ticket_id", stored))),
                    "raw_payload": ticket,
                    "ingested_at": datetime.utcnow().isoformat(),
                    "job_id": job_id,
                    "source": "zendesk",
                }

                # Upsert (insert or update)
                if collection.has(doc["_key"]):
                    collection.update(doc)
                else:
                    collection.insert(doc)

                stored += 1

            except Exception as e:
                logger.warning("Failed to store ticket", ticket_id=doc.get("_key"), error=str(e))

        logger.info("Stored raw tickets", count=stored, job_id=job_id)
        return stored

    def _backup_to_disk(self, tickets: list[dict], job_id: str):
        """Backup raw tickets to disk as JSON"""
        try:
            backup_dir = DATA_DIR / "raw" / job_id
            backup_dir.mkdir(parents=True, exist_ok=True)

            backup_file = backup_dir / f"tickets_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            with open(backup_file, "w") as f:
                json.dump(tickets, f, indent=2, default=str)

            logger.info("Backed up tickets to disk", file=str(backup_file), count=len(tickets))
        except Exception as e:
            logger.warning("Failed to backup tickets to disk", error=str(e))

    def get_job_status(self, job_id: str) -> Optional[dict]:
        """Get status of an ingest job"""
        try:
            collection = self._db.collection("ingest_jobs")
            if collection.has(job_id):
                return collection.get(job_id)
        except Exception:
            pass
        return None

    def update_job_status(self, job_id: str, status: str, metadata: dict = None):
        """Update ingest job status"""
        try:
            # Ensure collection exists
            if not self._db.has_collection("ingest_jobs"):
                self._db.create_collection("ingest_jobs")

            collection = self._db.collection("ingest_jobs")

            doc = {
                "_key": job_id,
                "status": status,
                "updated_at": datetime.utcnow().isoformat(),
                **(metadata or {}),
            }

            if collection.has(job_id):
                collection.update(doc)
            else:
                doc["created_at"] = datetime.utcnow().isoformat()
                collection.insert(doc)

        except Exception as e:
            logger.error("Failed to update job status", job_id=job_id, error=str(e))

