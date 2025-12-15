"""
Zendesk Ticket Intelligence - Ticket Schemas

Defines the canonical TicketDocument and related models per PRD Section 8.2
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Comment(BaseModel):
    """Individual ticket comment"""

    author_role: str = Field(..., description="Role: agent, end-user, system")
    created_at: datetime
    body_text: str


class TicketDocument(BaseModel):
    """
    Canonical normalized ticket document.
    This is the unit for embeddings and LLM extraction.
    """

    ticket_id: str
    created_at: datetime
    updated_at: datetime
    solved_at: Optional[datetime] = None

    status: str = Field(..., description="open, pending, solved, closed")
    priority: Optional[str] = None
    type: Optional[str] = None

    brand: Optional[str] = None
    product_line: Optional[str] = None
    platform: Optional[str] = Field(None, description="Win/Linux/Mac/ARM, etc")

    subject: str
    description: str
    comments: list[Comment] = Field(default_factory=list)

    tags: list[str] = Field(default_factory=list)
    custom_fields: dict = Field(default_factory=dict)
    attachments: list[dict] = Field(
        default_factory=list, description="Metadata only; no binaries"
    )

    pii_redacted_text: Optional[str] = Field(
        None, description="The text that downstream models see"
    )
    source_url: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "ticket_id": "12345",
                "created_at": "2025-01-15T10:30:00Z",
                "updated_at": "2025-01-16T14:00:00Z",
                "status": "open",
                "priority": "high",
                "subject": "Cannot login to application",
                "description": "When I try to login, I get error code 500...",
                "comments": [],
                "tags": ["login", "error-500"],
            }
        }


class RawTicketRecord(BaseModel):
    """
    Raw ticket record from Zendesk API.
    Persisted exactly as received (immutable).
    """

    raw_payload: dict
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    source: str = "zendesk"

