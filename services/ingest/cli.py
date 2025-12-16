#!/usr/bin/env python3
"""
ZTI Ingest CLI
Command-line tool for ingesting tickets from Zendesk via MCP
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.ingest.client import ZendeskMCPClient
from services.ingest.storage import ArangoStorage


async def test_connection(mcp_url: str) -> bool:
    """Test connection to Zendesk MCP server"""
    print(f"Testing connection to {mcp_url}...")
    client = ZendeskMCPClient(mcp_url)

    if await client.connect():
        print("✅ MCP server is reachable and connected")
        print(f"   Session ID: {client._session_id[:16]}...")
        await client.close()
        return True
    else:
        print("❌ Failed to connect to MCP server")
        await client.close()
        return False


async def fetch_tickets(mcp_url: str, days: int, output_file: str = None) -> list:
    """Fetch tickets from MCP server"""
    print(f"Fetching tickets from last {days} days via MCP...")
    client = ZendeskMCPClient(mcp_url)

    try:
        if not await client.connect():
            print("❌ Failed to connect")
            return []

        tickets = await client.fetch_tickets(days=days)
        print(f"✅ Fetched {len(tickets)} tickets")

        if output_file:
            with open(output_file, "w") as f:
                json.dump(tickets, f, indent=2, default=str)
            print(f"   Saved to {output_file}")

        return tickets
    finally:
        await client.close()


async def run_ingest(
    mcp_url: str,
    days: int,
    arango_host: str,
    arango_port: int,
) -> int:
    """Run full ingest pipeline"""
    print(f"Running ingest for last {days} days...")

    client = ZendeskMCPClient(mcp_url)
    storage = ArangoStorage(host=arango_host, port=arango_port)

    try:
        if not await client.connect():
            print("❌ Failed to connect to MCP server")
            return 0

        # Fetch tickets
        tickets = await client.fetch_tickets(days=days)
        print(f"   Fetched {len(tickets)} tickets")

        # Store tickets
        job_id = f"cli-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        count = storage.store_raw_tickets(tickets, job_id=job_id)
        print(f"✅ Stored {count} tickets (job: {job_id})")

        return count
    finally:
        await client.close()


def main():
    parser = argparse.ArgumentParser(description="ZTI Ingest CLI (MCP)")
    parser.add_argument(
        "--mcp-url",
        default=os.getenv("ZENDESK_MCP_URL", "http://192.168.87.79:10005/sse"),
        help="Zendesk MCP server URL",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Test command
    test_parser = subparsers.add_parser("test", help="Test MCP connection")

    # Fetch command
    fetch_parser = subparsers.add_parser("fetch", help="Fetch tickets to file")
    fetch_parser.add_argument("--days", type=int, default=7, help="Days of history")
    fetch_parser.add_argument("--output", "-o", default="tickets.json", help="Output file")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest tickets to database")
    ingest_parser.add_argument("--days", type=int, default=365, help="Days of history")
    ingest_parser.add_argument(
        "--arango-host",
        default=os.getenv("ARANGODB_HOST", "localhost"),
        help="ArangoDB host",
    )
    ingest_parser.add_argument(
        "--arango-port",
        type=int,
        default=int(os.getenv("ARANGODB_PORT", "8529")),
        help="ArangoDB port",
    )

    args = parser.parse_args()

    if args.command == "test":
        success = asyncio.run(test_connection(args.mcp_url))
        sys.exit(0 if success else 1)

    elif args.command == "fetch":
        asyncio.run(fetch_tickets(args.mcp_url, args.days, args.output))

    elif args.command == "ingest":
        asyncio.run(run_ingest(
            args.mcp_url,
            args.days,
            args.arango_host,
            args.arango_port,
        ))


if __name__ == "__main__":
    main()

