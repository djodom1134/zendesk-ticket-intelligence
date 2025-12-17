"""
ZTI API Service - FastAPI backend for cluster data, search, and Tier-0 chat
Following NVIDIA txt2kg patterns
"""

import os
from datetime import datetime
from typing import Any, List, Optional

import httpx
import structlog
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from arango import ArangoClient
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

logger = structlog.get_logger()

app = FastAPI(
    title="ZTI API",
    description="Zendesk Ticket Intelligence - Clusters, Search, and Tier-0 Chat",
    version="0.2.0"
)

# CORS for UI access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
ARANGODB_HOST = os.getenv("ARANGODB_HOST", "localhost")
ARANGODB_PORT = int(os.getenv("ARANGODB_PORT", "8529"))
ARANGODB_DB = os.getenv("ARANGODB_DB", "zti")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "ticket_embeddings")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:8b")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-oss:120b")


# Database connections (lazy init)
_arango_client: Optional[ArangoClient] = None
_qdrant_client: Optional[QdrantClient] = None


def get_db():
    """Get ArangoDB connection."""
    global _arango_client
    if _arango_client is None:
        _arango_client = ArangoClient(hosts=f"http://{ARANGODB_HOST}:{ARANGODB_PORT}")
    return _arango_client.db(ARANGODB_DB, verify=True)


def get_qdrant():
    """Get Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _qdrant_client


async def embed_text(text: str) -> List[float]:
    """Generate embedding for text using Ollama."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": text[:60000]},
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]


async def compute_cluster_positions(
    cluster_ids: List[str],
    x_dim: int = 0,
    y_dim: int = 1,
    n_components: int = 10
) -> List[List[float]]:
    """
    Compute cluster positions for visualization using UMAP dimensionality reduction.

    Args:
        cluster_ids: List of cluster IDs to compute positions for
        x_dim: Which UMAP component to use for x-axis (0-9, default 0)
        y_dim: Which UMAP component to use for y-axis (0-9, default 1)
        n_components: Number of UMAP components to compute (default 10)

    Returns:
        List of [x, y] coordinates normalized to 0-100 range

    How it works:
        1. Fetch representative ticket embeddings (4096-dim) from Qdrant
        2. Compute cluster centroids (mean of representative tickets)
        3. Apply UMAP to reduce from 4096-dim to n_components-dim (default 10)
        4. Select x_dim and y_dim components for visualization
        5. Normalize to 0-100 range for consistent graph scaling

    The x and y axes represent:
        - UMAP Component N: A learned combination of all 4096 embedding dimensions
        - Preserves semantic similarity (similar clusters appear close together)
        - Each component captures different aspects of cluster relationships
    """
    try:
        import numpy as np
        import umap

        db = get_db()
        qdrant = get_qdrant()

        # Get cluster centroids from Qdrant
        centroids = []
        cluster_to_index = {}

        for idx, cluster_id in enumerate(cluster_ids):
            # Get all tickets in this cluster
            cluster = db.collection("clusters").get(cluster_id)
            if not cluster:
                continue

            # Get representative tickets to compute centroid
            rep_tickets = cluster.get("representative_tickets", [])
            if not rep_tickets:
                continue

            # Fetch embeddings from Qdrant (4096-dimensional vectors)
            vectors = []
            for ticket_id in rep_tickets[:5]:  # Use up to 5 representative tickets
                try:
                    points = qdrant.retrieve(
                        collection_name=QDRANT_COLLECTION,
                        ids=[int(ticket_id)],
                        with_vectors=True
                    )
                    if points and len(points) > 0:
                        # Access vector from the Record object
                        vector = points[0].vector
                        if vector is not None:
                            vectors.append(vector)
                except Exception as e:
                    logger.warning(f"Failed to get vector for ticket {ticket_id}: {e}")
                    continue

            if vectors:
                # Compute centroid (mean of 4096-dim vectors)
                centroid = np.mean(vectors, axis=0)
                centroids.append(centroid)
                cluster_to_index[idx] = len(centroids) - 1

        if len(centroids) < 2:
            # Not enough data for UMAP - return grid layout
            logger.warning(f"Only {len(centroids)} valid centroids, using grid layout")
            import math
            n = len(cluster_ids)
            cols = math.ceil(math.sqrt(n))
            return [[float(i % cols * 10), float(i // cols * 10)] for i in range(n)]

        # Apply UMAP to reduce from 4096-dim to n_components-dim
        centroids_array = np.array(centroids)
        umap_model = umap.UMAP(
            n_components=min(n_components, len(centroids) - 1),  # Can't have more components than samples
            n_neighbors=min(15, len(centroids) - 1),
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        positions_nd = umap_model.fit_transform(centroids_array)

        # Validate dimension selection
        actual_components = positions_nd.shape[1]
        x_dim = min(x_dim, actual_components - 1)
        y_dim = min(y_dim, actual_components - 1)

        # Extract selected dimensions
        positions_2d = positions_nd[:, [x_dim, y_dim]]

        # Normalize to 0-100 range for visualization
        pos_min = positions_2d.min(axis=0)
        pos_max = positions_2d.max(axis=0)
        pos_range = pos_max - pos_min
        # Avoid division by zero
        pos_range = np.where(pos_range == 0, 1.0, pos_range)
        positions_2d = (positions_2d - pos_min) / pos_range
        positions_2d = positions_2d * 100  # Scale to 0-100 range

        # Map back to original cluster order
        result = []
        for idx in range(len(cluster_ids)):
            if idx in cluster_to_index:
                pos_idx = cluster_to_index[idx]
                result.append([float(positions_2d[pos_idx][0]), float(positions_2d[pos_idx][1])])
            else:
                # Cluster had no valid vectors - place at origin
                result.append([0.0, 0.0])

        logger.info(f"Computed positions using UMAP components {x_dim} and {y_dim} (of {actual_components} total)")
        return result

    except Exception as e:
        logger.error("Failed to compute cluster positions", error=str(e))
        # Return default positions in a grid
        import math
        n = len(cluster_ids)
        cols = math.ceil(math.sqrt(n))
        return [[float(i % cols * 10), float(i // cols * 10)] for i in range(n)]


# ============================================================================
# Pydantic Models
# ============================================================================

class ClusterSummary(BaseModel):
    id: str
    label: str
    size: int
    priority: str
    confidence: float
    keywords: List[str]
    issue_description: Optional[str] = None
    created_at: Optional[str] = None
    x: Optional[float] = None  # 2D UMAP x-coordinate
    y: Optional[float] = None  # 2D UMAP y-coordinate


class TicketPosition(BaseModel):
    """Ticket with 2D position for visualization"""
    ticket_id: str
    cluster_id: Optional[str] = None
    cluster_label: Optional[str] = None
    subject: Optional[str] = None
    x: float
    y: float
    z: Optional[float] = None  # For 3D visualization


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


class TicketSummary(BaseModel):
    id: str
    subject: str
    status: str
    cluster_id: Optional[str] = None
    created_at: Optional[str] = None
    similarity: Optional[float] = None


class TicketDetail(BaseModel):
    id: str
    subject: str
    description: str
    status: str
    priority: Optional[str] = None
    tags: List[str] = []
    cluster_id: Optional[str] = None
    summary: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    limit: int = Field(10, ge=1, le=100)
    cluster_id: Optional[str] = None


class SearchResult(BaseModel):
    tickets: List[TicketSummary]
    query: str
    total: int


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    ticket_id: Optional[str] = None  # Optional: context from existing ticket
    include_citations: bool = True


class Citation(BaseModel):
    type: str  # "cluster" or "ticket"
    id: str
    label: str
    relevance: float


class ChatResponse(BaseModel):
    response: str
    predicted_cluster: Optional[str] = None
    cluster_label: Optional[str] = None
    confidence: float = 0.0
    recommended_actions: List[str] = []
    draft_reply: Optional[str] = None
    citations: List[Citation] = []
    ask_for_info: List[str] = []  # Questions to ask customer


class WeeklyTrend(BaseModel):
    """Weekly ticket volume with trend calculation."""
    week_start: str  # ISO date
    count: int
    change_pct: Optional[float] = None  # vs previous week


class ClusterGrowth(BaseModel):
    """Cluster growth metrics."""
    cluster_id: str
    label: str
    current_size: int
    previous_size: int
    growth_rate: float  # percentage change
    is_new: bool  # appeared this week


class ResolutionMetrics(BaseModel):
    """Resolution time statistics."""
    avg_hours: float
    median_hours: float
    p90_hours: float  # 90th percentile
    trend_pct: float  # change vs previous period


class DeflectionMetrics(BaseModel):
    """Deflection potential based on cluster characteristics."""
    total_deflectable: int  # tickets that could be auto-responded
    deflection_rate: float  # percentage of total
    top_deflectable_clusters: List[dict]  # clusters with highest deflection potential
    estimated_hours_saved: float  # assuming avg 5 min per ticket


class StatsResponse(BaseModel):
    # Basic counts
    total_clusters: int
    total_tickets: int
    avg_confidence: float
    trending_up: int

    # Weekly trends
    tickets_this_week: int = 0
    tickets_last_week: int = 0
    week_over_week_change: float = 0.0  # percentage
    weekly_trend: List[WeeklyTrend] = []  # last 8 weeks

    # Cluster growth
    new_clusters_this_week: int = 0
    growing_clusters: List[ClusterGrowth] = []

    # Resolution times
    resolution: Optional[ResolutionMetrics] = None

    # Deflection potential
    deflection: Optional[DeflectionMetrics] = None

    # Top clusters (for backward compat)
    top_clusters: List[dict] = []


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
    """Get comprehensive dashboard statistics with real KPI calculations."""
    try:
        db = get_db()

        # Basic cluster stats
        clusters = list(db.collection("clusters").all())
        total_tickets = db.collection("tickets").count()

        if clusters:
            avg_conf = sum(c.get("confidence", 0) for c in clusters) / len(clusters)
            trending = sum(1 for c in clusters if c.get("trend_direction") == "up")
        else:
            avg_conf = 0
            trending = 0

        # ============================================================
        # 1. Weekly Ticket Trend (last 8 weeks)
        # ============================================================
        weekly_trend = []
        try:
            weekly_data = list(db.aql.execute("""
                LET now = DATE_NOW()
                FOR week_offset IN 0..7
                    LET week_start = DATE_SUBTRACT(now, week_offset, "week")
                    LET week_end = DATE_SUBTRACT(now, week_offset - 1, "week")
                    LET count = LENGTH(
                        FOR t IN tickets
                            FILTER t.created_at >= DATE_ISO8601(week_start)
                            FILTER t.created_at < DATE_ISO8601(week_end)
                            RETURN 1
                    )
                    RETURN {
                        week_start: DATE_ISO8601(week_start),
                        count: count,
                        week_offset: week_offset
                    }
            """))

            # Calculate week-over-week changes
            for i, week in enumerate(weekly_data):
                change_pct = None
                if i < len(weekly_data) - 1 and weekly_data[i + 1]["count"] > 0:
                    prev_count = weekly_data[i + 1]["count"]
                    change_pct = ((week["count"] - prev_count) / prev_count) * 100
                weekly_trend.append(WeeklyTrend(
                    week_start=week["week_start"][:10],
                    count=week["count"],
                    change_pct=round(change_pct, 1) if change_pct is not None else None
                ))

            tickets_this_week = weekly_data[0]["count"] if weekly_data else 0
            tickets_last_week = weekly_data[1]["count"] if len(weekly_data) > 1 else 0
            wow_change = 0.0
            if tickets_last_week > 0:
                wow_change = ((tickets_this_week - tickets_last_week) / tickets_last_week) * 100
        except Exception:
            tickets_this_week = 0
            tickets_last_week = 0
            wow_change = 0.0

        # ============================================================
        # 2. Cluster Growth Rate
        # ============================================================
        growing_clusters = []
        new_clusters_this_week = 0
        try:
            # Compare current cluster sizes to stored historical sizes
            for c in clusters:
                current_size = c.get("size", 0)
                previous_size = c.get("previous_size", current_size)  # Stored from last week
                created_at = c.get("created_at", "")

                # Check if cluster is new (created within last 7 days)
                is_new = False
                if created_at:
                    from datetime import datetime, timedelta
                    try:
                        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                        is_new = (datetime.now(created.tzinfo) - created).days <= 7
                        if is_new:
                            new_clusters_this_week += 1
                    except Exception:
                        pass

                growth_rate = 0.0
                if previous_size > 0:
                    growth_rate = ((current_size - previous_size) / previous_size) * 100

                if abs(growth_rate) > 5 or is_new:  # Only include significant changes
                    growing_clusters.append(ClusterGrowth(
                        cluster_id=c["_key"],
                        label=c.get("label", "Unknown"),
                        current_size=current_size,
                        previous_size=previous_size,
                        growth_rate=round(growth_rate, 1),
                        is_new=is_new
                    ))

            # Sort by growth rate descending
            growing_clusters.sort(key=lambda x: x.growth_rate, reverse=True)
            growing_clusters = growing_clusters[:10]  # Top 10
        except Exception:
            pass

        # ============================================================
        # 3. Resolution Time Metrics
        # ============================================================
        resolution = None
        try:
            resolution_data = list(db.aql.execute("""
                LET resolved_tickets = (
                    FOR t IN tickets
                        FILTER t.status == "solved" OR t.status == "closed"
                        FILTER t.created_at != null AND t.solved_at != null
                        LET hours = DATE_DIFF(t.created_at, t.solved_at, "hour")
                        FILTER hours > 0 AND hours < 720  // Exclude outliers (>30 days)
                        RETURN hours
                )
                LET sorted = SORTED(resolved_tickets)
                LET count = LENGTH(sorted)
                RETURN {
                    avg: count > 0 ? AVERAGE(resolved_tickets) : 0,
                    median: count > 0 ? sorted[FLOOR(count / 2)] : 0,
                    p90: count > 0 ? sorted[FLOOR(count * 0.9)] : 0,
                    count: count
                }
            """))

            if resolution_data and resolution_data[0]["count"] > 0:
                rd = resolution_data[0]
                resolution = ResolutionMetrics(
                    avg_hours=round(rd["avg"], 1),
                    median_hours=round(rd["median"], 1),
                    p90_hours=round(rd["p90"], 1),
                    trend_pct=0.0  # TODO: Compare to previous period
                )
        except Exception:
            pass

        # ============================================================
        # 4. Deflection Potential
        # ============================================================
        deflection = None
        try:
            # Deflectable = clusters with high confidence + recommended_response
            deflectable_clusters = [
                c for c in clusters
                if c.get("confidence", 0) >= 0.7 and c.get("recommended_response")
            ]

            total_deflectable = sum(c.get("size", 0) for c in deflectable_clusters)
            deflection_rate = (total_deflectable / total_tickets * 100) if total_tickets > 0 else 0

            # Estimate hours saved: 5 min per deflected ticket
            hours_saved = (total_deflectable * 5) / 60

            top_deflectable = [
                {
                    "cluster_id": c["_key"],
                    "label": c.get("label", "Unknown"),
                    "size": c.get("size", 0),
                    "confidence": c.get("confidence", 0),
                }
                for c in sorted(deflectable_clusters, key=lambda x: x.get("size", 0), reverse=True)[:5]
            ]

            deflection = DeflectionMetrics(
                total_deflectable=total_deflectable,
                deflection_rate=round(deflection_rate, 1),
                top_deflectable_clusters=top_deflectable,
                estimated_hours_saved=round(hours_saved, 1)
            )
        except Exception:
            pass

        # ============================================================
        # Top clusters for backward compatibility
        # ============================================================
        top_clusters = [
            {"id": c["_key"], "label": c.get("label", ""), "size": c.get("size", 0)}
            for c in sorted(clusters, key=lambda x: x.get("size", 0), reverse=True)[:5]
        ]

        return StatsResponse(
            total_clusters=len(clusters),
            total_tickets=total_tickets,
            avg_confidence=round(avg_conf, 2),
            trending_up=trending,
            tickets_this_week=tickets_this_week,
            tickets_last_week=tickets_last_week,
            week_over_week_change=round(wow_change, 1),
            weekly_trend=weekly_trend,
            new_clusters_this_week=new_clusters_this_week,
            growing_clusters=growing_clusters,
            resolution=resolution,
            deflection=deflection,
            top_clusters=top_clusters,
        )
    except Exception as e:
        logger.error("Stats calculation failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/clusters", response_model=List[ClusterSummary])
async def list_clusters(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("size", regex="^(size|confidence|created_at)$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    include_positions: bool = Query(False, description="Include 2D UMAP positions"),
    x_dim: int = Query(0, ge=0, le=9, description="UMAP component for x-axis (0-9)"),
    y_dim: int = Query(1, ge=0, le=9, description="UMAP component for y-axis (0-9)")
):
    """
    List all clusters with pagination and sorting.

    Dimension Selection:
        - x_dim, y_dim: Select which UMAP components to plot (0-9)
        - Component 0: Primary semantic dimension (most variance)
        - Component 1: Secondary semantic dimension
        - Components 2-9: Additional semantic dimensions
        - Each component captures different aspects of cluster relationships

    Examples:
        - x_dim=0, y_dim=1: Default view (most common)
        - x_dim=0, y_dim=2: Alternative perspective
        - x_dim=2, y_dim=3: Explore tertiary relationships
    """
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

        clusters = [
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

        # Add 2D positions if requested
        if include_positions and clusters:
            positions = await compute_cluster_positions(
                [c.id for c in clusters],
                x_dim=x_dim,
                y_dim=y_dim
            )
            for cluster, pos in zip(clusters, positions):
                cluster.x = pos[0]
                cluster.y = pos[1]

        return clusters
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tickets/positions", response_model=List[TicketPosition])
async def get_ticket_positions(
    x_dim: int = Query(0, ge=0, le=9, description="UMAP component for x-axis"),
    y_dim: int = Query(1, ge=0, le=9, description="UMAP component for y-axis"),
    z_dim: int = Query(2, ge=0, le=9, description="UMAP component for z-axis (3D only)"),
    use_3d: bool = Query(False, description="Include z-coordinate for 3D visualization"),
    limit: int = Query(2000, ge=1, le=5000, description="Max tickets to return")
):
    """
    Get all tickets with their 2D/3D UMAP positions for scatter plot visualization.

    This endpoint returns individual tickets as points, colored by their cluster.
    Perfect for visualizing cluster density, separation, and overlap.

    Returns:
        List of tickets with x,y (and optionally z) coordinates
        Each ticket includes its cluster assignment for coloring
    """
    try:
        import numpy as np
        import umap

        qdrant = get_qdrant()
        db = get_db()

        # Get all tickets from Qdrant with their embeddings
        logger.info("Fetching all ticket embeddings from Qdrant")

        # Use scroll to get all points
        scroll_result = qdrant.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=limit,
            with_vectors=True,
            with_payload=True
        )

        points = scroll_result[0]  # First element is the list of points

        if not points or len(points) < 2:
            return []

        logger.info(f"Retrieved {len(points)} tickets from Qdrant")

        # Extract vectors and metadata
        vectors = []
        ticket_metadata = []

        for point in points:
            if point.vector is not None:
                vectors.append(point.vector)
                ticket_metadata.append({
                    'ticket_id': str(point.id),
                    'subject': point.payload.get('subject', ''),
                    'cluster_id': point.payload.get('cluster_id')
                })

        if len(vectors) < 2:
            return []

        # Apply UMAP to reduce to 3D or 2D
        n_components = 3 if use_3d else 10  # Use 10 for 2D to allow dimension selection
        vectors_array = np.array(vectors)

        logger.info(f"Applying UMAP to {len(vectors)} tickets, reducing to {n_components} dimensions")

        umap_model = umap.UMAP(
            n_components=min(n_components, len(vectors) - 1),
            n_neighbors=min(15, len(vectors) - 1),
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        positions_nd = umap_model.fit_transform(vectors_array)

        # Get cluster labels for coloring
        cluster_map = {}
        try:
            clusters = list(db.collection("clusters").all())
            for c in clusters:
                cluster_map[c["_key"]] = c.get("label", f"Cluster {c['_key']}")
        except:
            pass

        # Build result
        result = []

        if use_3d:
            # For 3D, use dimensions directly
            positions_3d = positions_nd[:, [x_dim, y_dim, z_dim]]

            # Normalize to 0-100
            for i in range(3):
                col = positions_3d[:, i]
                col_min, col_max = col.min(), col.max()
                col_range = col_max - col_min if col_max != col_min else 1.0
                positions_3d[:, i] = (col - col_min) / col_range * 100

            for i, meta in enumerate(ticket_metadata):
                cluster_id = meta.get('cluster_id')
                result.append(TicketPosition(
                    ticket_id=meta['ticket_id'],
                    cluster_id=cluster_id,
                    cluster_label=cluster_map.get(cluster_id, 'Unclustered'),
                    subject=meta.get('subject', ''),
                    x=float(positions_3d[i][0]),
                    y=float(positions_3d[i][1]),
                    z=float(positions_3d[i][2])
                ))
        else:
            # For 2D, select dimensions
            positions_2d = positions_nd[:, [x_dim, y_dim]]

            # Normalize to 0-100
            for i in range(2):
                col = positions_2d[:, i]
                col_min, col_max = col.min(), col.max()
                col_range = col_max - col_min if col_max != col_min else 1.0
                positions_2d[:, i] = (col - col_min) / col_range * 100

            for i, meta in enumerate(ticket_metadata):
                cluster_id = meta.get('cluster_id')
                result.append(TicketPosition(
                    ticket_id=meta['ticket_id'],
                    cluster_id=cluster_id,
                    cluster_label=cluster_map.get(cluster_id, 'Unclustered'),
                    subject=meta.get('subject', ''),
                    x=float(positions_2d[i][0]),
                    y=float(positions_2d[i][1])
                ))

        logger.info(f"Returning {len(result)} ticket positions")
        return result

    except Exception as e:
        logger.error(f"Failed to get ticket positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/clusters/{cluster_id}/enrich")
async def enrich_cluster(cluster_id: str):
    """
    Enrich cluster data with LLM-generated content for missing fields.
    Generates symptoms, environment details, and recommended responses.
    """
    try:
        db = get_db()
        cluster = db.collection("clusters").get(cluster_id)

        if not cluster:
            raise HTTPException(status_code=404, detail="Cluster not found")

        # Build context from existing cluster data
        context = f"""
Cluster: {cluster.get('label', 'Unknown')}
Issue Description: {cluster.get('issue_description', 'N/A')}
Environment: {cluster.get('environment', 'N/A')}
Keywords: {', '.join(cluster.get('keywords', []))}
Recommended Response: {cluster.get('recommended_response', 'N/A')}
Number of Tickets: {cluster.get('size', 0)}
Representative Tickets: {', '.join(cluster.get('representative_tickets', [])[:3])}
"""

        updates = {}

        # Generate symptoms if missing
        symptoms = cluster.get('symptoms')
        if not symptoms or symptoms is None:
            prompt = f"""Based on this support ticket cluster information, generate a list of 3-5 specific symptoms or indicators that customers report when experiencing this issue.

{context}

Return ONLY a JSON array of symptom strings, like: ["symptom 1", "symptom 2", "symptom 3"]
"""

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": CHAT_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.3},
                    },
                )
                response.raise_for_status()
                llm_response = response.json().get("response", "")

                # Parse JSON array from response
                import json
                try:
                    json_start = llm_response.find("[")
                    json_end = llm_response.rfind("]") + 1
                    if json_start >= 0 and json_end > json_start:
                        symptoms = json.loads(llm_response[json_start:json_end])
                        updates["symptoms"] = symptoms
                    else:
                        symptoms = []
                except json.JSONDecodeError:
                    symptoms = []

        # Generate environment if missing or too short
        environment = cluster.get('environment')
        if not environment or len(environment) < 20:
            prompt = f"""Based on this support ticket cluster information, describe the typical environment or system configuration where this issue occurs. Include software versions, operating systems, hardware, and relevant technical details.

{context}

Return ONLY a single paragraph describing the environment (2-3 sentences).
"""

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": CHAT_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.3},
                    },
                )
                response.raise_for_status()
                environment = response.json().get("response", "").strip()
                if environment:
                    updates["environment"] = environment

        # Generate recommended response if missing or too short
        recommended_response = cluster.get('recommended_response')
        if not recommended_response or len(recommended_response) < 30:
            prompt = f"""Based on this support ticket cluster information, write a professional recommended response for support agents to use when addressing this issue. Include:
1. What to ask the customer for (logs, screenshots, configuration details)
2. What to check or verify
3. Recommended troubleshooting steps
4. When to escalate

{context}

Return ONLY the recommended response text (2-4 sentences).
"""

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": CHAT_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.3},
                    },
                )
                response.raise_for_status()
                recommended_response = response.json().get("response", "").strip()
                if recommended_response:
                    updates["recommended_response"] = recommended_response

        # Update cluster in database if we have any updates
        if updates:
            db.collection("clusters").update({
                "_key": cluster_id,
                **updates
            })

        return {
            "status": "enriched",
            "symptoms": symptoms or cluster.get('symptoms', []),
            "environment": environment or cluster.get('environment', ''),
            "recommended_response": recommended_response or cluster.get('recommended_response', ''),
            "fields_updated": list(updates.keys())
        }

    except Exception as e:
        logger.error(f"Failed to enrich cluster {cluster_id}", error=str(e))
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



# ============================================================================
# Ticket Endpoints
# ============================================================================

@app.get("/api/tickets", response_model=List[TicketSummary])
async def list_tickets(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    cluster_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
):
    """List tickets with optional filtering."""
    try:
        db = get_db()

        # Build dynamic query
        filters = []
        bind_vars = {"offset": offset, "limit": limit}

        if cluster_id:
            filters.append("t.cluster_id == @cluster_id")
            bind_vars["cluster_id"] = cluster_id
        if status:
            filters.append("t.status == @status")
            bind_vars["status"] = status

        where_clause = f"FILTER {' AND '.join(filters)}" if filters else ""

        cursor = db.aql.execute(
            f"""
            FOR t IN tickets
                {where_clause}
                SORT t.created_at DESC
                LIMIT @offset, @limit
                RETURN t
            """,
            bind_vars=bind_vars
        )

        return [
            TicketSummary(
                id=t["_key"],
                subject=t.get("subject", ""),
                status=t.get("status", "unknown"),
                cluster_id=t.get("cluster_id"),
                created_at=t.get("created_at"),
            )
            for t in cursor
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tickets/{ticket_id}", response_model=TicketDetail)
async def get_ticket(ticket_id: str):
    """Get detailed ticket information."""
    try:
        db = get_db()
        ticket = db.collection("tickets").get(ticket_id)

        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")

        return TicketDetail(
            id=ticket["_key"],
            subject=ticket.get("subject", ""),
            description=ticket.get("description", ""),
            status=ticket.get("status", "unknown"),
            priority=ticket.get("priority"),
            tags=ticket.get("tags", []),
            cluster_id=ticket.get("cluster_id"),
            summary=ticket.get("summary"),
            created_at=ticket.get("created_at"),
            updated_at=ticket.get("updated_at"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Search Endpoints (Semantic via Qdrant)
# ============================================================================

@app.post("/api/search", response_model=SearchResult)
async def search_tickets(request: SearchRequest):
    """Semantic search for tickets using embeddings."""
    try:
        # Generate embedding for query
        query_embedding = await embed_text(request.query)

        # Search Qdrant
        qdrant = get_qdrant()

        # Optional cluster filter
        search_filter = None
        if request.cluster_id:
            search_filter = Filter(
                must=[FieldCondition(key="cluster_id", match=MatchValue(value=request.cluster_id))]
            )

        results = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_embedding,
            query_filter=search_filter,
            limit=request.limit,
        ).points

        tickets = [
            TicketSummary(
                id=str(hit.id),
                subject=hit.payload.get("subject", "") if hit.payload else "",
                status=hit.payload.get("status", "unknown") if hit.payload else "unknown",
                cluster_id=hit.payload.get("cluster_id") if hit.payload else None,
                created_at=hit.payload.get("created_at") if hit.payload else None,
                similarity=hit.score,
            )
            for hit in results
        ]

        return SearchResult(
            tickets=tickets,
            query=request.query,
            total=len(tickets),
        )
    except Exception as e:
        logger.error("Search failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Tier-0 Chat Service (RAG-based)
# ============================================================================

TIER0_SYSTEM_PROMPT = """You are a Tier-0 support assistant for a video management system (VMS) product.
Your role is to help classify incoming support tickets and draft initial responses.

IMPORTANT RULES:
1. NEVER reveal customer PII (emails, phone numbers, IP addresses)
2. NEVER fabricate official fixes - only suggest documented solutions
3. When uncertain, ask for more information (logs, screenshots, version numbers)
4. Always cite your sources (cluster summaries, similar tickets)
5. Be concise and professional

You have access to:
- Cluster summaries: common issue patterns and their solutions
- Similar tickets: past cases that match the current issue

Based on the input, provide:
1. The most likely cluster/category for this issue
2. A confidence score (0-1)
3. Recommended actions for the support agent
4. A draft reply to the customer
5. Questions to ask if more information is needed
"""

TIER0_USER_PROMPT = """
NEW TICKET/MESSAGE:
{message}

SIMILAR CLUSTERS FOUND:
{clusters}

SIMILAR PAST TICKETS:
{tickets}

Based on this information, classify the issue and draft a response.
Respond in JSON format:
{{
  "predicted_cluster": "cluster_id or null",
  "cluster_label": "human-readable label",
  "confidence": 0.0-1.0,
  "recommended_actions": ["action1", "action2"],
  "draft_reply": "Professional response to customer",
  "ask_for_info": ["question1", "question2"] // if more info needed
}}
"""


@app.post("/api/chat", response_model=ChatResponse)
async def tier0_chat(request: ChatRequest):
    """
    Tier-0 Chat: Classify ticket and generate draft response using RAG.

    Uses semantic search to find similar tickets and cluster info,
    then generates a response with citations.
    """
    try:
        citations: List[Citation] = []

        # Step 1: Embed the message
        query_embedding = await embed_text(request.message)

        # Step 2: Search for similar tickets
        qdrant = get_qdrant()
        similar_tickets = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_embedding,
            limit=5,
        ).points

        tickets_context = ""
        for hit in similar_tickets:
            if hit.payload:
                tickets_context += f"\n- [{hit.payload.get('subject', 'No subject')}] "
                tickets_context += f"(similarity: {hit.score:.2f}): "
                tickets_context += f"{hit.payload.get('summary', '')[:500]}\n"
                citations.append(Citation(
                    type="ticket",
                    id=str(hit.id),
                    label=hit.payload.get("subject", "")[:50],
                    relevance=hit.score,
                ))

        # Step 3: Get cluster context
        db = get_db()
        clusters_context = ""
        try:
            top_clusters = list(db.aql.execute(
                """
                FOR c IN clusters
                    SORT c.size DESC
                    LIMIT 5
                    RETURN c
                """
            ))
            for c in top_clusters:
                clusters_context += f"\n- **{c.get('label', 'Unknown')}** (size: {c.get('size', 0)}): "
                clusters_context += f"{c.get('issue_description', '')[:300]}\n"
                clusters_context += f"  Response hint: {c.get('recommended_response', '')[:200]}\n"
                citations.append(Citation(
                    type="cluster",
                    id=c["_key"],
                    label=c.get("label", ""),
                    relevance=0.5,  # Default relevance for clusters
                ))
        except Exception:
            pass  # Clusters may not exist yet

        # Step 4: Generate response with LLM
        prompt = TIER0_USER_PROMPT.format(
            message=request.message[:3000],
            clusters=clusters_context or "No cluster data available yet.",
            tickets=tickets_context or "No similar tickets found.",
        )

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": CHAT_MODEL,
                    "system": TIER0_SYSTEM_PROMPT,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3},
                },
            )
            response.raise_for_status()
            llm_response = response.json().get("response", "")

        # Step 5: Parse LLM response
        import json
        try:
            # Try to extract JSON from response
            json_start = llm_response.find("{")
            json_end = llm_response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                parsed = json.loads(llm_response[json_start:json_end])
            else:
                parsed = {}
        except json.JSONDecodeError:
            parsed = {}

        return ChatResponse(
            response=llm_response,
            predicted_cluster=parsed.get("predicted_cluster"),
            cluster_label=parsed.get("cluster_label"),
            confidence=float(parsed.get("confidence", 0)),
            recommended_actions=parsed.get("recommended_actions", []),
            draft_reply=parsed.get("draft_reply"),
            citations=citations[:10] if request.include_citations else [],
            ask_for_info=parsed.get("ask_for_info", []),
        )

    except Exception as e:
        logger.error("Chat failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Cluster Assignment (for new tickets)
# ============================================================================

class AssignClusterRequest(BaseModel):
    ticket_id: str
    text: str  # Ticket subject + description


class AssignClusterResponse(BaseModel):
    ticket_id: str
    cluster_id: Optional[str]
    cluster_label: Optional[str]
    confidence: float
    similar_tickets: List[str]


@app.post("/api/assign-cluster", response_model=AssignClusterResponse)
async def assign_cluster(request: AssignClusterRequest):
    """
    Assign a ticket to the most appropriate cluster based on semantic similarity.
    Used for real-time ticket routing.
    """
    try:
        # Embed the ticket text
        embedding = await embed_text(request.text)

        # Find similar tickets
        qdrant = get_qdrant()
        results = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=embedding,
            limit=10,
        ).points

        if not results:
            return AssignClusterResponse(
                ticket_id=request.ticket_id,
                cluster_id=None,
                cluster_label=None,
                confidence=0.0,
                similar_tickets=[],
            )

        # Vote on cluster based on nearest neighbors
        cluster_votes: dict[str, float] = {}
        similar_ids = []

        for hit in results:
            similar_ids.append(str(hit.id))
            if hit.payload and hit.payload.get("cluster_id"):
                cid = hit.payload["cluster_id"]
                cluster_votes[cid] = cluster_votes.get(cid, 0) + hit.score

        if not cluster_votes:
            return AssignClusterResponse(
                ticket_id=request.ticket_id,
                cluster_id=None,
                cluster_label=None,
                confidence=results[0].score if results else 0.0,
                similar_tickets=similar_ids[:5],
            )

        # Find winning cluster
        best_cluster = max(cluster_votes, key=cluster_votes.get)
        total_votes = sum(cluster_votes.values())
        confidence = cluster_votes[best_cluster] / total_votes if total_votes > 0 else 0

        # Get cluster label
        db = get_db()
        cluster = db.collection("clusters").get(best_cluster)
        cluster_label = cluster.get("label") if cluster else None

        return AssignClusterResponse(
            ticket_id=request.ticket_id,
            cluster_id=best_cluster,
            cluster_label=cluster_label,
            confidence=confidence,
            similar_tickets=similar_ids[:5],
        )

    except Exception as e:
        logger.error("Cluster assignment failed", error=str(e))


# ============================================================================
# Incremental Processing & Webhooks
# ============================================================================

class WebhookPayload(BaseModel):
    """Zendesk webhook payload for ticket events."""
    ticket_id: str
    event_type: str  # "ticket.created", "ticket.updated"
    ticket: Optional[dict] = None


class IncrementalResult(BaseModel):
    processed: int
    skipped: int
    elapsed_seconds: float


@app.post("/api/webhook/zendesk", response_model=IncrementalResult)
async def zendesk_webhook(payload: WebhookPayload):
    """
    Receive Zendesk webhook for real-time ticket processing.
    Processes new/updated tickets immediately.
    """
    try:
        from services.pipeline.incremental import IncrementalPipeline

        pipeline = IncrementalPipeline(
            arango_host=ARANGODB_HOST,
            arango_port=ARANGODB_PORT,
            arango_db=ARANGODB_DB,
            qdrant_host=QDRANT_HOST,
            qdrant_port=QDRANT_PORT,
            ollama_url=OLLAMA_URL,
            embed_model=EMBED_MODEL,
        )

        if payload.ticket:
            result = pipeline.run(tickets=[payload.ticket])
        else:
            # Fetch ticket by ID if not included in payload
            # This would require Zendesk API access
            logger.warning("Webhook received without ticket data", ticket_id=payload.ticket_id)
            result = {"processed": 0, "skipped": 0, "elapsed_seconds": 0}

        return IncrementalResult(**result)

    except Exception as e:
        logger.error("Webhook processing failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


class BatchProcessRequest(BaseModel):
    tickets: list[dict]


@app.post("/api/process/batch", response_model=IncrementalResult)
async def process_batch(request: BatchProcessRequest):
    """
    Process a batch of tickets through the incremental pipeline.
    Used for manual imports or bulk processing.
    """
    try:
        from services.pipeline.incremental import IncrementalPipeline

        pipeline = IncrementalPipeline(
            arango_host=ARANGODB_HOST,
            arango_port=ARANGODB_PORT,
            arango_db=ARANGODB_DB,
            qdrant_host=QDRANT_HOST,
            qdrant_port=QDRANT_PORT,
            ollama_url=OLLAMA_URL,
            embed_model=EMBED_MODEL,
        )

        result = pipeline.run(tickets=request.tickets)
        return IncrementalResult(**result)

    except Exception as e:
        logger.error("Batch processing failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))