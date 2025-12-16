#!/usr/bin/env python3
"""
Re-summarize all tickets using MCP for full context and anti-hallucination prompt.

Usage:
    python scripts/resummarize_all.py --mcp-url http://192.168.87.79:10005/sse --ollama-url http://192.168.87.134:11434

This script:
1. Fetches all ticket IDs from ArangoDB
2. For each ticket, fetches full context from MCP (with comments, custom fields, agent info)
3. Generates new summaries using the anti-hallucination prompt
4. Stores summaries back to ArangoDB
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.ingest.client import ZendeskMCPClient
from services.embed_cluster.summarizer import TicketSummarizer


async def fetch_ticket_full_context(client: ZendeskMCPClient, ticket_id: int) -> dict | None:
    """Fetch a single ticket with full context from MCP."""
    try:
        result = await client.call_tool("get_ticket_full_context", {"ticket_id": ticket_id})
        if result.get("isError"):
            print(f"  ⚠️ Error fetching {ticket_id}: {result}")
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


async def main():
    parser = argparse.ArgumentParser(description="Re-summarize all tickets with anti-hallucination prompt")
    parser.add_argument("--mcp-url", default="http://192.168.87.79:10005/sse", help="MCP server URL")
    parser.add_argument("--ollama-url", default="http://192.168.87.134:11434", help="Ollama URL")
    parser.add_argument("--arango-host", default="localhost", help="ArangoDB host")
    parser.add_argument("--arango-port", type=int, default=8529, help="ArangoDB port")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for processing")
    parser.add_argument("--start-from", type=int, default=0, help="Start from ticket index")
    parser.add_argument("--max-tickets", type=int, default=0, help="Max tickets to process (0=all)")
    parser.add_argument("--dry-run", action="store_true", help="Don't save summaries, just test")
    args = parser.parse_args()

    print("=" * 60)
    print("🔄 RE-SUMMARIZE ALL TICKETS")
    print("=" * 60)
    print(f"MCP URL: {args.mcp_url}")
    print(f"Ollama URL: {args.ollama_url}")
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
        model="gpt-oss:120b",
        max_output_tokens=2000,  # Shorter summaries for this format
    )

    # Process tickets
    processed = 0
    failed = 0
    
    try:
        for i, ticket_id in enumerate(ticket_ids):
            print(f"\n[{i+1}/{len(ticket_ids)}] Fetching ticket {ticket_id}...")
            
            # Fetch full context from MCP
            ticket = await fetch_ticket_full_context(mcp_client, ticket_id)
            if not ticket:
                failed += 1
                continue
            
            # Generate summary
            summary, metrics = summarizer.summarize_full_ticket(
                ticket=ticket,
                is_last=(i == len(ticket_ids) - 1),
                show_live=True,
            )
            
            # Store summary
            if not args.dry_run:
                tickets_col.update({
                    "_key": str(ticket_id),
                    "summary": summary,
                    "summary_version": "v2_anti_hallucination",
                })
            
            processed += 1
            
            # Progress
            if (processed) % 10 == 0:
                print(f"\n📊 Progress: {processed}/{len(ticket_ids)} processed, {failed} failed")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    finally:
        await mcp_client.close()
    
    print("\n" + "=" * 60)
    print(f"✅ Complete! Processed: {processed}, Failed: {failed}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

