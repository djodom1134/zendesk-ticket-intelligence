#!/usr/bin/env python3
"""
Embed all ticket summaries and store in Qdrant.

This script:
1. Fetches all tickets with summaries from ArangoDB
2. Generates embeddings using qwen3-embedding:8b via Ollama
3. Stores embeddings in Qdrant for similarity search

Usage:
    python scripts/embed_all_summaries.py --ollama-url http://localhost:11434 --qdrant-host localhost
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.embed_cluster.embedder import TicketEmbedder
from services.embed_cluster.vector_store import VectorStore

# Defaults
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
DEFAULT_QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:8b")


def main():
    parser = argparse.ArgumentParser(description="Embed all ticket summaries")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama URL")
    parser.add_argument("--qdrant-host", default=DEFAULT_QDRANT_HOST, help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=DEFAULT_QDRANT_PORT, help="Qdrant port")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Embedding model")
    parser.add_argument("--arango-host", default="localhost", help="ArangoDB host")
    parser.add_argument("--arango-port", type=int, default=8529, help="ArangoDB port")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for embedding")
    parser.add_argument("--start-from", type=int, default=0, help="Start from ticket index")
    parser.add_argument("--max-tickets", type=int, default=0, help="Max tickets to process (0=all)")
    args = parser.parse_args()

    print("=" * 60)
    print("🔢 TICKET EMBEDDING PIPELINE")
    print("=" * 60)
    print(f"Ollama URL: {args.ollama_url}")
    print(f"Embedding Model: {args.embed_model}")
    print(f"Qdrant: {args.qdrant_host}:{args.qdrant_port}")
    print(f"ArangoDB: {args.arango_host}:{args.arango_port}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Connect to ArangoDB
    from arango import ArangoClient
    arango = ArangoClient(hosts=f"http://{args.arango_host}:{args.arango_port}")
    db = arango.db("zti", username="root", password="zti_dev_password")
    tickets_col = db.collection("tickets")

    # Get all tickets with summaries
    print("📋 Fetching tickets with summaries from ArangoDB...")
    query = """
    FOR t IN tickets
        FILTER t.summary != null AND t.summary != ""
        SORT t.ticket_id
        RETURN {
            ticket_id: t.ticket_id,
            summary: t.summary,
            subject: t.subject,
            created_at: t.created_at
        }
    """
    cursor = db.aql.execute(query)
    tickets = list(cursor)

    print(f"   Found {len(tickets)} tickets with summaries")

    # Apply limits
    if args.start_from > 0:
        tickets = tickets[args.start_from:]
        print(f"   Starting from index {args.start_from}")
    if args.max_tickets > 0:
        tickets = tickets[:args.max_tickets]
        print(f"   Limited to {args.max_tickets} tickets")

    print(f"   Processing: {len(tickets)} tickets")
    print()

    if not tickets:
        print("No tickets to process")
        return 0

    # Initialize embedder
    print(f"🔧 Initializing embedder ({args.embed_model})...")
    embedder = TicketEmbedder(
        ollama_url=args.ollama_url,
        model=args.embed_model,
    )

    # Get embedding dimension
    print(f"   Testing embedding dimension...")
    test_emb = embedder.embed("test")
    dimension = len(test_emb)
    print(f"   ✅ Embedding dimension: {dimension}")
    print()

    # Initialize vector store
    print(f"🗄️  Initializing Qdrant vector store...")
    vector_store = VectorStore(
        host=args.qdrant_host,
        port=args.qdrant_port,
        dimension=dimension,
    )

    if not vector_store.check_health():
        print("❌ Qdrant is not healthy")
        return 1

    print(f"   ✅ Qdrant connected")
    print()

    # Process in batches
    total = len(tickets)
    processed = 0
    failed = 0
    start_time = time.time()

    print(f"🚀 Starting embedding pipeline...")
    print(f"   Batch size: {args.batch_size}")
    print()

    for i in range(0, total, args.batch_size):
        batch = tickets[i:i + args.batch_size]
        batch_num = i // args.batch_size + 1
        total_batches = (total + args.batch_size - 1) // args.batch_size

        print(f"[Batch {batch_num}/{total_batches}] Processing {len(batch)} tickets...")

        try:
            # Extract summaries
            summaries = [t["summary"] for t in batch]
            ticket_ids = [str(t["ticket_id"]) for t in batch]

            # Generate embeddings
            batch_start = time.time()
            embeddings = embedder.embed_batch(summaries)
            batch_time = time.time() - batch_start

            # Prepare payloads with metadata
            payloads = [
                {
                    "ticket_id": t["ticket_id"],
                    "subject": t.get("subject", ""),
                    "created_at": t.get("created_at", ""),
                }
                for t in batch
            ]

            # Store in Qdrant
            vector_store.upsert_batch(ticket_ids, embeddings, payloads)

            processed += len(batch)

            # Progress stats
            elapsed = time.time() - start_time
            rate = processed / elapsed * 3600 if elapsed > 0 else 0
            eta = (total - processed) / (processed / elapsed) if processed > 0 else 0

            print(f"   ✅ Batch complete: {batch_time:.1f}s | "
                  f"Progress: {processed}/{total} | "
                  f"Rate: {rate:.0f}/hr | "
                  f"ETA: {eta/60:.1f}min")

        except Exception as e:
            print(f"   ❌ Batch failed: {e}")
            failed += len(batch)
            continue

    # Final stats
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("✅ Embedding complete!")
    print(f"   Processed: {processed}")
    print(f"   Failed: {failed}")
    print(f"   Time: {elapsed/60:.1f} minutes")
    print(f"   Rate: {processed/elapsed*3600:.0f} tickets/hour")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

