"""
ZTI Ingest Service
Pulls tickets from Zendesk MCP server and persists to ArangoDB
"""

import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

import structlog
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from .client import ZendeskMCPClient
from .storage import ArangoStorage

logger = structlog.get_logger()

# Configuration from environment
ZENDESK_MCP_URL = os.getenv("ZENDESK_MCP_URL", "http://192.168.87.79:10005/sse")
ARANGODB_HOST = os.getenv("ARANGODB_HOST", "localhost")
ARANGODB_PORT = int(os.getenv("ARANGODB_PORT", "8529"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize connections on startup"""
    logger.info("Starting ZTI Ingest Service", mcp_url=ZENDESK_MCP_URL)
    app.state.zendesk_client = ZendeskMCPClient(ZENDESK_MCP_URL)
    app.state.storage = ArangoStorage(host=ARANGODB_HOST, port=ARANGODB_PORT)
    # Connect to MCP server
    connected = await app.state.zendesk_client.connect()
    if not connected:
        logger.warning("Failed to connect to MCP server on startup")
    yield
    await app.state.zendesk_client.close()
    logger.info("Shutting down ZTI Ingest Service")


app = FastAPI(
    title="ZTI Ingest Service",
    description="Zendesk Ticket Intelligence - Ingest Service",
    version="0.1.0",
    lifespan=lifespan,
)


class IngestRequest(BaseModel):
    """Request to ingest tickets"""
    days: int = 365
    batch_size: int = 100


class IngestResponse(BaseModel):
    """Response from ingest operation"""
    job_id: str
    status: str
    message: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    agent_reachable: bool
    db_connected: bool


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check health of ingest service and dependencies"""
    agent_ok = await app.state.zendesk_client.check_health()
    db_ok = app.state.storage.check_health()

    return HealthResponse(
        status="healthy" if (agent_ok and db_ok) else "degraded",
        agent_reachable=agent_ok,
        db_connected=db_ok,
    )


@app.post("/ingest", response_model=IngestResponse)
async def start_ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    """Start a ticket ingest job"""
    job_id = str(uuid.uuid4())

    logger.info("Starting ingest job", job_id=job_id, days=request.days)

    background_tasks.add_task(
        run_ingest_job,
        job_id=job_id,
        days=request.days,
        batch_size=request.batch_size,
        client=app.state.zendesk_client,
        storage=app.state.storage,
    )

    return IngestResponse(
        job_id=job_id,
        status="started",
        message=f"Ingest job started for last {request.days} days",
    )


@app.get("/ingest/{job_id}")
async def get_ingest_status(job_id: str):
    """Get status of an ingest job"""
    status = app.state.storage.get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


async def run_ingest_job(
    job_id: str,
    days: int,
    batch_size: int,
    client: ZendeskA2AClient,
    storage: ArangoStorage,
):
    """Background task to run ingest job"""
    logger.info("Running ingest job", job_id=job_id, days=days)

    try:
        storage.update_job_status(job_id, "running", {"started_at": datetime.utcnow().isoformat()})

        # Request tickets from agent
        tickets = await client.fetch_tickets(days=days)

        # Store raw tickets
        count = storage.store_raw_tickets(tickets, job_id=job_id)

        storage.update_job_status(job_id, "completed", {
            "completed_at": datetime.utcnow().isoformat(),
            "tickets_ingested": count,
        })

        logger.info("Ingest job completed", job_id=job_id, tickets=count)

    except Exception as e:
        logger.error("Ingest job failed", job_id=job_id, error=str(e))
        storage.update_job_status(job_id, "failed", {"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

