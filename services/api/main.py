"""
ZTI API Service - FastAPI backend for cluster data
Following NVIDIA txt2kg patterns
"""

import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from arango import ArangoClient

app = FastAPI(
    title="ZTI Cluster API",
    description="API for Zendesk Ticket Intelligence cluster data",
    version="0.1.0"
)

# CORS for UI access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
ARANGODB_HOST = os.getenv("ARANGODB_HOST", "localhost")
ARANGODB_PORT = int(os.getenv("ARANGODB_PORT", "8529"))
ARANGODB_DB = os.getenv("ARANGODB_DB", "zti")


def get_db():
    """Get ArangoDB connection."""
    client = ArangoClient(hosts=f"http://{ARANGODB_HOST}:{ARANGODB_PORT}")
    return client.db(ARANGODB_DB, verify=True)


# Pydantic models
class ClusterSummary(BaseModel):
    id: str
    label: str
    size: int
    priority: str
    confidence: float
    keywords: List[str]
    issue_description: Optional[str] = None
    created_at: Optional[str] = None


class ClusterDetail(BaseModel):
    id: str
    label: str
    size: int
    priority: str
    confidence: float
    keywords: List[str]
    issue_description: Optional[str] = None
    environment: Optional[str] = None
    symptoms: Optional[List[str]] = None
    recommended_response: Optional[str] = None
    deflection_path: Optional[str] = None
    representative_tickets: List[str] = []
    trend: List[int] = []
    created_at: Optional[str] = None


class StatsResponse(BaseModel):
    total_clusters: int
    total_tickets: int
    avg_confidence: float
    trending_up: int


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        db = get_db()
        db.collection("clusters").count()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get dashboard statistics."""
    try:
        db = get_db()
        clusters = list(db.collection("clusters").all())
        tickets = db.collection("tickets").count()
        
        if clusters:
            avg_conf = sum(c.get("confidence", 0) for c in clusters) / len(clusters)
            trending = sum(1 for c in clusters if c.get("trend_direction") == "up")
        else:
            avg_conf = 0
            trending = 0
        
        return StatsResponse(
            total_clusters=len(clusters),
            total_tickets=tickets,
            avg_confidence=round(avg_conf, 2),
            trending_up=trending
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/clusters", response_model=List[ClusterSummary])
async def list_clusters(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("size", regex="^(size|confidence|created_at)$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$")
):
    """List all clusters with pagination and sorting."""
    try:
        db = get_db()
        cursor = db.aql.execute(
            """
            FOR c IN clusters
                SORT c.@sort_by @sort_order
                LIMIT @offset, @limit
                RETURN c
            """,
            bind_vars={
                "sort_by": sort_by,
                "sort_order": sort_order.upper(),
                "offset": offset,
                "limit": limit
            }
        )
        
        return [
            ClusterSummary(
                id=c["_key"],
                label=c.get("label", f"Cluster {c['_key']}"),
                size=c.get("size", 0),
                priority=c.get("priority", "medium"),
                confidence=c.get("confidence", 0),
                keywords=c.get("keywords", []),
                issue_description=c.get("issue_description"),
                created_at=c.get("created_at")
            )
            for c in cursor
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/clusters/{cluster_id}", response_model=ClusterDetail)
async def get_cluster(cluster_id: str):
    """Get detailed cluster information."""
    try:
        db = get_db()
        cluster = db.collection("clusters").get(cluster_id)
        
        if not cluster:
            raise HTTPException(status_code=404, detail="Cluster not found")
        
        return ClusterDetail(
            id=cluster["_key"],
            label=cluster.get("label", f"Cluster {cluster['_key']}"),
            size=cluster.get("size", 0),
            priority=cluster.get("priority", "medium"),
            confidence=cluster.get("confidence", 0),
            keywords=cluster.get("keywords", []),
            issue_description=cluster.get("issue_description"),
            environment=cluster.get("environment"),
            symptoms=cluster.get("symptoms"),
            recommended_response=cluster.get("recommended_response"),
            deflection_path=cluster.get("deflection_path"),
            representative_tickets=cluster.get("representative_tickets", []),
            trend=cluster.get("trend", []),
            created_at=cluster.get("created_at")
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

