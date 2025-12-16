"""
ZTI Ingest Service
Pulls tickets from Zendesk MCP server and persists to databases

Components:
- main.py: FastAPI application with ingest endpoints
- client.py: Zendesk MCP protocol client (SSE transport)
- storage.py: ArangoDB storage layer
"""

from .client import ZendeskMCPClient
from .storage import ArangoStorage

__all__ = ["ZendeskMCPClient", "ArangoStorage"]

