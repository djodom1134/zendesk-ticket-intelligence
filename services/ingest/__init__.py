"""
ZTI Ingest Service
Pulls tickets from Zendesk A2A agent and persists to databases

Components:
- main.py: FastAPI application with ingest endpoints
- client.py: Zendesk A2A protocol client
- storage.py: ArangoDB storage layer
"""

from .client import ZendeskA2AClient
from .storage import ArangoStorage

__all__ = ["ZendeskA2AClient", "ArangoStorage"]

