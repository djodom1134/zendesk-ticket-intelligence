"""
Zendesk Ticket Intelligence - Cluster Schemas

Defines IssueCluster and related models per PRD Section 9
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ClusterSummary(BaseModel):
    """Generated summary for a cluster"""

    what_is_this_issue: str
    how_to_reproduce: Optional[str] = None
    common_environment: Optional[str] = None
    recommended_response: Optional[str] = None
    deflection_path: Optional[str] = Field(
        None, description="Docs/Product/Support/Automation"
    )


class IssueCluster(BaseModel):
    """
    Represents a cluster of similar tickets (problem family).
    """

    cluster_id: str
    cluster_label: str
    cluster_keywords: list[str] = Field(default_factory=list)
    representative_ticket_ids: list[str] = Field(default_factory=list)

    ticket_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Cluster metadata
    severity_distribution: dict = Field(
        default_factory=dict, description="Count by priority level"
    )
    trend_data: list[dict] = Field(
        default_factory=list, description="Weekly ticket counts"
    )

    # Generated summary
    summary: Optional[ClusterSummary] = None

    # Embedding for retrieval
    embedding: Optional[list[float]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "cluster_id": "cluster_001",
                "cluster_label": "Login Authentication Failures",
                "cluster_keywords": ["login", "auth", "500", "timeout"],
                "ticket_count": 45,
                "representative_ticket_ids": ["12345", "12346", "12347"],
            }
        }


class ClusterAssignment(BaseModel):
    """Assignment of a ticket to a cluster with confidence"""

    ticket_id: str
    cluster_id: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    assigned_at: datetime = Field(default_factory=datetime.utcnow)
    assigned_by: str = Field(default="auto", description="auto or human reviewer")

