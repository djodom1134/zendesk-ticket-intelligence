#!/usr/bin/env python3
"""
ZTI Normalize CLI
Command-line tool for normalizing tickets from raw to canonical format
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from arango import ArangoClient
import structlog

from services.normalize.normalizer import TicketNormalizer
from shared.schemas.ticket import TicketDocument

logger = structlog.get_logger()


def normalize_from_file(
    input_file: str,
    output_file: str = None,
    redact: bool = True,
) -> list[TicketDocument]:
    """Normalize tickets from a JSON file"""
    print(f"Normalizing tickets from {input_file}...")
    
    with open(input_file, "r") as f:
        data = json.load(f)
    
    # Handle both list and {"tickets": [...]} format
    if isinstance(data, dict) and "tickets" in data:
        raw_tickets = data["tickets"]
    elif isinstance(data, list):
        raw_tickets = data
    else:
        print("❌ Invalid input format")
        return []
    
    normalizer = TicketNormalizer()
    normalized = []
    
    for raw in raw_tickets:
        try:
            doc = normalizer.normalize(raw)
            normalized.append(doc)
        except Exception as e:
            logger.warning("Failed to normalize ticket", ticket_id=raw.get("id"), error=str(e))
    
    print(f"✅ Normalized {len(normalized)} tickets")
    
    if output_file:
        with open(output_file, "w") as f:
            json.dump([doc.model_dump(mode="json") for doc in normalized], f, indent=2, default=str)
        print(f"   Saved to {output_file}")
    
    return normalized


def normalize_from_db(
    arango_host: str,
    arango_port: int,
    database: str = "zti",
    batch_size: int = 100,
) -> int:
    """Normalize all raw tickets in database"""
    print(f"Connecting to ArangoDB at {arango_host}:{arango_port}...")
    
    client = ArangoClient(hosts=f"http://{arango_host}:{arango_port}")
    db = client.db(database)
    
    # Ensure normalized collection exists
    if not db.has_collection("normalized_tickets"):
        db.create_collection("normalized_tickets")
    
    raw_collection = db.collection("raw_tickets")
    norm_collection = db.collection("normalized_tickets")
    
    normalizer = TicketNormalizer()
    count = 0
    errors = 0
    
    # Process in batches
    cursor = raw_collection.all()
    batch = []
    
    for doc in cursor:
        raw = doc.get("raw_payload", {})
        try:
            normalized = normalizer.normalize(raw)
            batch.append({
                "_key": str(normalized.ticket_id),
                **normalized.model_dump(mode="json"),
            })
            
            if len(batch) >= batch_size:
                _save_batch(norm_collection, batch)
                count += len(batch)
                print(f"   Processed {count} tickets...")
                batch = []
                
        except Exception as e:
            errors += 1
            logger.warning("Failed to normalize", ticket_id=raw.get("id"), error=str(e))
    
    # Save remaining batch
    if batch:
        _save_batch(norm_collection, batch)
        count += len(batch)
    
    print(f"✅ Normalized {count} tickets ({errors} errors)")
    return count


def _save_batch(collection, batch: list[dict]):
    """Save a batch of documents to collection"""
    for doc in batch:
        key = doc.get("_key")
        if collection.has(key):
            collection.update(doc)
        else:
            collection.insert(doc)


def main():
    parser = argparse.ArgumentParser(description="ZTI Normalize CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # File command
    file_parser = subparsers.add_parser("file", help="Normalize from JSON file")
    file_parser.add_argument("input", help="Input JSON file")
    file_parser.add_argument("--output", "-o", help="Output JSON file")
    file_parser.add_argument("--no-redact", action="store_true", help="Disable PII redaction")
    
    # Database command
    db_parser = subparsers.add_parser("db", help="Normalize from ArangoDB")
    db_parser.add_argument("--host", default=os.getenv("ARANGODB_HOST", "localhost"))
    db_parser.add_argument("--port", type=int, default=int(os.getenv("ARANGODB_PORT", "8529")))
    db_parser.add_argument("--database", default="zti")
    db_parser.add_argument("--batch-size", type=int, default=100)
    
    args = parser.parse_args()
    
    if args.command == "file":
        normalize_from_file(args.input, args.output, redact=not args.no_redact)
    elif args.command == "db":
        normalize_from_db(args.host, args.port, args.database, args.batch_size)


if __name__ == "__main__":
    main()

