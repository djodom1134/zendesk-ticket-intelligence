#!/usr/bin/env python3
"""
Re-summarize all tickets using MCP for full context and anti-hallucination prompt.
Uses parallel fetching: fetches next ticket while current one is being summarized.

Usage:
    python scripts/resummarize_all.py --mcp-url http://192.168.87.79:10005/sse --ollama-url http://localhost:11434

This script:
1. Fetches all ticket IDs from ArangoDB
2. Parallel pipeline: fetch next ticket while summarizing current one
3. Generates new summaries using the anti-hallucination prompt
4. Stores summaries back to ArangoDB
"""
import argparse
import asyncio
import json
import os
import sys
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.ingest.client import ZendeskMCPClient
from services.embed_cluster.summarizer import TicketSummarizer

# Load from environment
DEFAULT_MCP_URL = os.getenv("MCP_URL", "http://192.168.87.79:10005/sse")
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "nemotron-3-nano:latest")


async def fetch_ticket_full_context(client: ZendeskMCPClient, ticket_id: int) -> dict | None:
    """Fetch a single ticket with full context from MCP."""
    try:
        result = await client.call_tool("get_ticket_full_context", {"ticket_id": ticket_id})
        if result.get("isError"):
            return None

        # Parse the result
        if "content" in result:
            for item in result["content"]:
                if item.get("type") == "text":
                    return json.loads(item["text"])
        return result
    except Exception as e:
        print(f"  ⚠️ Exception fetching {ticket_id}: {e}")
        return None


class ParallelPipeline:
    """
    Parallel pipeline that fetches tickets from MCP while GPU summarizes.
    Uses a queue to decouple fetching from summarization.
    """

    def __init__(
        self,
        mcp_client: ZendeskMCPClient,
        summarizer: TicketSummarizer,
        prefetch_count: int = 3,
    ):
        self.mcp_client = mcp_client
        self.summarizer = summarizer
        self.prefetch_count = prefetch_count
        self.ticket_queue = asyncio.Queue(maxsize=prefetch_count)
        self.results = {}
        self.fetch_errors = 0
        self.summarize_errors = 0

    async def fetch_worker(self, ticket_ids: list[int]):
        """Worker that fetches tickets and puts them in the queue."""
        for ticket_id in ticket_ids:
            ticket = await fetch_ticket_full_context(self.mcp_client, ticket_id)
            if ticket:
                await self.ticket_queue.put((ticket_id, ticket))
            else:
                self.fetch_errors += 1
                await self.ticket_queue.put((ticket_id, None))  # Signal failed fetch

        # Signal end of fetching
        await self.ticket_queue.put((None, None))

    async def process_tickets(
        self,
        ticket_ids: list[int],
        db_collection,
        dry_run: bool = False,
        show_live: bool = True,
    ) -> tuple[int, int]:
        """
        Process tickets with parallel fetch + summarize.

        Returns:
            Tuple of (processed_count, failed_count)
        """
        total = len(ticket_ids)
        processed = 0
        failed = 0
        start_time = time.time()

        # Start fetch worker
        fetch_task = asyncio.create_task(self.fetch_worker(ticket_ids))

        print(f"\n🚀 Starting parallel pipeline (prefetch={self.prefetch_count})")
        print(f"   Fetching from MCP while GPU summarizes...")

        while True:
            # Get next ticket from queue
            ticket_id, ticket = await self.ticket_queue.get()

            # Check for end signal
            if ticket_id is None:
                break

            idx = processed + failed + 1

            if ticket is None:
                # Fetch failed
                failed += 1
                print(f"\n[{idx}/{total}] ❌ Failed to fetch ticket {ticket_id}")
                continue

            # Summarize (this is the slow GPU operation)
            if show_live:
                print(f"\n[{idx}/{total}] 📋 Summarizing ticket {ticket_id}...")

            try:
                summary, metrics = self.summarizer.summarize_full_ticket(
                    ticket=ticket,
                    is_last=(idx == total),
                    show_live=show_live,
                )

                # Store summary
                if not dry_run and db_collection:
                    db_collection.update({
                        "_key": str(ticket_id),
                        "summary": summary,
                        "summary_version": "v2_anti_hallucination",
                    })

                processed += 1

            except Exception as e:
                print(f"  ⚠️ Summarization error: {e}")
                failed += 1

            # Progress stats
            if idx % 10 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed * 3600 if elapsed > 0 else 0
                eta = (total - idx) / (processed / elapsed) if processed > 0 else 0
                print(f"\n📊 Progress: {idx}/{total} | "
                      f"Rate: {rate:.0f}/hr | "
                      f"ETA: {eta/60:.1f}min | "
                      f"Queue: {self.ticket_queue.qsize()}")

        # Wait for fetch worker to complete
        await fetch_task

        return processed, failed


async def main():
    parser = argparse.ArgumentParser(description="Re-summarize all tickets with parallel fetch + summarize")
    parser.add_argument("--mcp-url", default=DEFAULT_MCP_URL, help="MCP server URL")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama URL")
    parser.add_argument("--model", default=DEFAULT_SUMMARY_MODEL, help="Summarization model")
    parser.add_argument("--arango-host", default="localhost", help="ArangoDB host")
    parser.add_argument("--arango-port", type=int, default=8529, help="ArangoDB port")
    parser.add_argument("--prefetch", type=int, default=3, help="Number of tickets to prefetch")
    parser.add_argument("--start-from", type=int, default=0, help="Start from ticket index")
    parser.add_argument("--max-tickets", type=int, default=0, help="Max tickets to process (0=all)")
    parser.add_argument("--dry-run", action="store_true", help="Don't save summaries, just test")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    args = parser.parse_args()

    print("=" * 60)
    print("🔄 PARALLEL RE-SUMMARIZATION PIPELINE")
    print("=" * 60)
    print(f"MCP URL: {args.mcp_url}")
    print(f"Ollama URL: {args.ollama_url}")
    print(f"Model: {args.model}")
    print(f"Prefetch: {args.prefetch} tickets")
    print(f"ArangoDB: {args.arango_host}:{args.arango_port}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Connect to ArangoDB to get ticket IDs
    from arango import ArangoClient
    arango = ArangoClient(hosts=f"http://{args.arango_host}:{args.arango_port}")
    db = arango.db("zti", username="root", password="zti_dev_password")

    # Get all ticket IDs
    tickets_col = db.collection("tickets")
    cursor = db.aql.execute("FOR t IN tickets RETURN t.ticket_id")
    ticket_ids = [int(tid) for tid in cursor if tid]

    print(f"📋 Found {len(ticket_ids)} tickets in ArangoDB")

    # Apply limits
    if args.start_from > 0:
        ticket_ids = ticket_ids[args.start_from:]
        print(f"   Starting from index {args.start_from}")
    if args.max_tickets > 0:
        ticket_ids = ticket_ids[:args.max_tickets]
        print(f"   Limited to {args.max_tickets} tickets")

    print(f"   Processing: {len(ticket_ids)} tickets")
    print()

    # Connect to MCP
    print("🔌 Connecting to MCP server...")
    mcp_client = ZendeskMCPClient(args.mcp_url)
    if not await mcp_client.connect():
        print("❌ Failed to connect to MCP server")
        return 1
    print("✅ Connected to MCP")

    # Initialize summarizer
    summarizer = TicketSummarizer(
        ollama_url=args.ollama_url,
        model=args.model,
        max_output_tokens=2000,
    )

    # Create parallel pipeline
    pipeline = ParallelPipeline(
        mcp_client=mcp_client,
        summarizer=summarizer,
        prefetch_count=args.prefetch,
    )

    start_time = time.time()

    try:
        processed, failed = await pipeline.process_tickets(
            ticket_ids=ticket_ids,
            db_collection=tickets_col if not args.dry_run else None,
            dry_run=args.dry_run,
            show_live=not args.quiet,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        processed, failed = 0, 0
    finally:
        await mcp_client.close()

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✅ Complete!")
    print(f"   Processed: {processed}")
    print(f"   Failed: {failed}")
    print(f"   Time: {elapsed/60:.1f} minutes")
    print(f"   Rate: {processed/elapsed*3600:.0f} tickets/hour" if elapsed > 0 else "")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

