"""
Zendesk Ticket Intelligence - Ticket Schemas

Defines the canonical TicketDocument and related models per PRD Section 8.2
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TicketStatus(str, Enum):
    """Zendesk ticket status"""
    NEW = "new"
    OPEN = "open"
    PENDING = "pending"
    HOLD = "hold"
    SOLVED = "solved"
    CLOSED = "closed"


class TicketPriority(str, Enum):
    """Zendesk ticket priority"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class AuthorRole(str, Enum):
    """Role of comment author"""
    AGENT = "agent"
    END_USER = "end-user"
    SYSTEM = "system"


class Comment(BaseModel):
    """Individual ticket comment"""
    id: Optional[int] = None
    author_id: Optional[int] = None
    author_role: AuthorRole = AuthorRole.END_USER
    created_at: datetime
    body_text: str
    body_redacted: str = ""  # PII-redacted version
    is_public: bool = True


class Attachment(BaseModel):
    """Ticket attachment metadata (no binary data)"""
    id: Optional[int] = None
    filename: str = ""
    content_type: str = ""
    size: int = 0
    url: Optional[str] = None


class TicketDocument(BaseModel):
    """
    Canonical normalized ticket document.
    This is the unit for embeddings and LLM extraction.
    """
    # Core identifiers
    ticket_id: str
    source_url: Optional[str] = None

    # Timestamps
    created_at: datetime
    updated_at: datetime
    solved_at: Optional[datetime] = None

    # Status & classification
    status: TicketStatus = TicketStatus.OPEN
    priority: Optional[TicketPriority] = None
    ticket_type: Optional[str] = None

    # Product/platform info (derived from tags/custom fields)
    brand: Optional[str] = None
    product_line: Optional[str] = None
    platform: Optional[str] = Field(None, description="Win/Linux/Mac/ARM, etc")
    version: Optional[str] = None

    # Content
    subject: str
    description: str
    description_redacted: str = ""  # PII-redacted version
    comments: list[Comment] = Field(default_factory=list)

    # Metadata
    tags: list[str] = Field(default_factory=list)
    custom_fields: dict[str, Any] = Field(default_factory=dict)
    attachments: list[Attachment] = Field(default_factory=list)

    # Assignee/requester info
    assignee_id: Optional[int] = None
    requester_id: Optional[int] = None
    group_id: Optional[int] = None

    # Computed fields for downstream processing
    ticket_fulltext: str = ""  # subject + description + all comments
    pii_redacted_text: str = ""  # Full text with PII redacted

    # Normalization metadata
    normalized_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True
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
    _key: Optional[str] = None
    raw_payload: dict
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    job_id: Optional[str] = None
    source: str = "zendesk"

