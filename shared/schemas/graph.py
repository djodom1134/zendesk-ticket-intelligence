"""
Zendesk Ticket Intelligence - Knowledge Graph Schemas

Defines graph nodes and edges per PRD Section 8.3
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ================== Node Types ==================


class GraphNode(BaseModel):
    """Base class for all graph nodes"""

    node_id: str
    node_type: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    properties: dict = Field(default_factory=dict)


class TicketNode(GraphNode):
    """Ticket node in the knowledge graph"""

    node_type: Literal["Ticket"] = "Ticket"
    ticket_id: str
    subject: str
    status: str


class ComponentNode(GraphNode):
    """Component node (e.g., Media Server, Client, Cloud, Mobile)"""

    node_type: Literal["Component"] = "Component"
    name: str
    category: Optional[str] = None


class ErrorSignatureNode(GraphNode):
    """Error signature node (error codes, log phrases)"""

    node_type: Literal["ErrorSignature"] = "ErrorSignature"
    signature: str
    error_code: Optional[str] = None


class EnvironmentNode(GraphNode):
    """Environment node (OS/GPU/driver/version)"""

    node_type: Literal["Environment"] = "Environment"
    os: Optional[str] = None
    gpu: Optional[str] = None
    driver_version: Optional[str] = None
    app_version: Optional[str] = None


class FeatureNode(GraphNode):
    """Feature/Workflow node"""

    node_type: Literal["Feature"] = "Feature"
    name: str
    category: Optional[str] = None


class FixNode(GraphNode):
    """Fix node (release, workaround, KB article)"""

    node_type: Literal["Fix"] = "Fix"
    fix_type: str = Field(..., description="release, workaround, kb_article")
    title: str
    url: Optional[str] = None


# ================== Edge Types ==================


class GraphEdge(BaseModel):
    """Base class for all graph edges"""

    edge_id: str
    edge_type: str
    from_node_id: str
    to_node_id: str
    weight: float = 1.0
    properties: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MentionsEdge(GraphEdge):
    """Ticket -> mentions -> Component|Feature|ErrorSignature|Environment"""

    edge_type: Literal["mentions"] = "mentions"


class BelongsToEdge(GraphEdge):
    """Ticket -> belongs_to -> IssueCluster"""

    edge_type: Literal["belongs_to"] = "belongs_to"
    confidence: float = 1.0


class CausedByEdge(GraphEdge):
    """IssueCluster -> caused_by -> Component|Feature"""

    edge_type: Literal["caused_by"] = "caused_by"


class MitigatedByEdge(GraphEdge):
    """IssueCluster -> mitigated_by -> Fix"""

    edge_type: Literal["mitigated_by"] = "mitigated_by"


class DuplicateOfEdge(GraphEdge):
    """Ticket -> duplicate_of -> Ticket (optional, inferred)"""

    edge_type: Literal["duplicate_of"] = "duplicate_of"
    similarity_score: float = 0.0

