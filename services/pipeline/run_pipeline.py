#!/usr/bin/env python3
"""
ZTI Pipeline Runner - Runs full pipeline: normalize → embed → cluster → label
All steps run inside the pipeline container on the GPU machine.
"""

import json
import os
import sys
import click
import structlog

log = structlog.get_logger()

# Add app root to path for imports
sys.path.insert(0, "/app")

from services.normalize.normalizer import TicketNormalizer
from services.normalize.pii_redactor import PIIRedactor
from services.embed.embedder import TicketEmbedder
from services.cluster.clusterer import TicketClusterer
from services.cluster_label.labeler import ClusterLabeler


@click.command()
@click.option("--input", "-i", "input_file", required=True, help="Input JSON file with raw tickets")
@click.option("--output-dir", "-o", default="/data/zti", help="Output directory for results")
@click.option("--skip-normalize", is_flag=True, help="Skip normalization step")
@click.option("--skip-embed", is_flag=True, help="Skip embedding step")
@click.option("--skip-cluster", is_flag=True, help="Skip clustering step")
@click.option("--skip-label", is_flag=True, help="Skip labeling step")
def main(input_file: str, output_dir: str, skip_normalize: bool, skip_embed: bool,
         skip_cluster: bool, skip_label: bool):
    """Run the ZTI pipeline on ticket data."""

    os.makedirs(output_dir, exist_ok=True)

    # Load input
    log.info("Loading input tickets", file=input_file)
    with open(input_file) as f:
        tickets = json.load(f)
    log.info("Loaded tickets", count=len(tickets))

    # Step 1: Normalize
    if not skip_normalize:
        log.info("Step 1: Normalizing tickets...")
        normalizer = TicketNormalizer()
        redactor = PIIRedactor()

        normalized = []
        for ticket in tickets:
            norm = normalizer.normalize(ticket)
            norm = redactor.redact(norm)
            normalized.append(norm)

        norm_file = os.path.join(output_dir, "tickets_normalized.json")
        with open(norm_file, "w") as f:
            json.dump(normalized, f, indent=2, default=str)
        log.info("Normalization complete", output=norm_file, count=len(normalized))
        tickets = normalized
    else:
        log.info("Skipping normalization")

    # Step 2: Embed
    embeddings = None
    if not skip_embed:
        log.info("Step 2: Generating embeddings...")
        embedder = TicketEmbedder()

        texts = [t.get("combined_text", "") or f"{t.get('subject', '')} {t.get('description', '')}"
                 for t in tickets]
        embeddings = embedder.embed_batch(texts)

        embed_file = os.path.join(output_dir, "embeddings.json")
        with open(embed_file, "w") as f:
            json.dump({"embeddings": [e.tolist() for e in embeddings], "ticket_ids": [t.get("id") for t in tickets]}, f)
        log.info("Embedding complete", output=embed_file, vectors=len(embeddings))
    else:
        log.info("Skipping embedding")

    # Step 3: Cluster
    clusters = None
    if not skip_cluster and embeddings is not None:
        log.info("Step 3: Clustering tickets...")
        clusterer = TicketClusterer()

        cluster_result = clusterer.cluster(embeddings, tickets)
        clusters = cluster_result["clusters"]

        cluster_file = os.path.join(output_dir, "clusters.json")
        with open(cluster_file, "w") as f:
            json.dump(cluster_result, f, indent=2, default=str)
        log.info("Clustering complete", output=cluster_file, num_clusters=len(clusters))
    else:
        log.info("Skipping clustering")

    # Step 4: Label
    if not skip_label and clusters is not None:
        log.info("Step 4: Labeling clusters...")
        labeler = ClusterLabeler()

        labeled = labeler.label_clusters(clusters, tickets)

        label_file = os.path.join(output_dir, "clusters_labeled.json")
        with open(label_file, "w") as f:
            json.dump(labeled, f, indent=2)
        log.info("Labeling complete", output=label_file)
    else:
        log.info("Skipping labeling")

    log.info("Pipeline complete!", output_dir=output_dir)
    print(f"\n✅ Pipeline complete! Results in {output_dir}")


if __name__ == "__main__":
    main()

