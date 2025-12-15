PRD — Zendesk Ticket Intelligence (ZTI) built on NVIDIA txt2kg

1) Overview

Build a local-first system that:
	1.	Ingests Zendesk tickets (batch + real-time) from your locally hosted Zendesk agent (http://192.168.87.79:10004).
	2.	Converts tickets into a knowledge graph + embeddings, then clusters them into “problem families.”
	3.	Ships an interactive visualization + analytics UI to explore clusters, trends, duplicates, root-causes, and “fix-the-docs / fix-the-product” opportunities.
	4.	Prototypes a Tier-0 chatbot using a local Ollama LLM that answers “what is this ticket?” and “how do we solve it?” using the clusters/KB as grounding.

We will use NVIDIA’s DGX Spark playbook dgx-spark-playbooks/nvidia/txt2kg as the skeleton (dockerized services + UI pattern), but tailor the pipeline to Zendesk and to clustering/ticket reduction. Note: NVIDIA forum threads show multiple operational gotchas we must bake into setup/acceptance checks (GPU fallback, path typos, optional/complete stack drift, Qdrant switch, Ollama container healthchecks).  ￼

⸻

2) Context + why now (your numbers)

You already proved bulk export works for 90 days:
	•	1,412 tickets (2025-09-16 → 2025-12-15), 13 weekly batches, 15.92s, 88.72 tickets/sec, 0 capped chunks
	•	A 365-day pull estimated: ~5,600 tickets, ~60–90s, ~75–100 calls

This PRD assumes those export tools are “production-ready” and shifts focus to organizing → clustering → visualization → reduction → Tier-0.

⸻

3) Goals and success metrics

Primary outcomes
	•	Cluster every ticket into one (or a small set) of stable issue clusters.
	•	Provide a visual dashboard that makes it obvious:
	•	Top clusters by volume, severity, cost, and trend
	•	Top “duplicate drivers”
	•	Which clusters are best solved by Docs, Product fix, Support macro, Known-issue banner, or Automation
	•	Prototype Tier-0 that can:
	•	Route new tickets to a cluster
	•	Suggest best next steps (KB links, macros, “collect these logs,” “upgrade path,” “known bug”)
	•	Draft a first response (optional toggle)

Quantitative success (Phase 1–2)
	•	Batch ingest 365 days end-to-end (ingest → preprocess → embeddings → clustering → UI ready) in ≤ 2 hours on your hardware (initial run), incremental updates ≤ 2 minutes/day.
	•	Cluster coverage: ≥ 95% of tickets assigned to a cluster; remainder marked “needs triage.”
	•	Duplicate detection precision: top-20 “closest neighbors” contains a true duplicate/same-root-cause ≥ 60% of the time (baseline), improving over time with feedback.

Long-term success (Phase 3)
	•	Demonstrate a path to ticket reduction (target 75%) by shipping:
	•	cluster-to-doc updates
	•	cluster-to-product fixes
	•	cluster-driven Tier-0 deflection

⸻

4) Non-goals (for this project)
	•	Replacing Zendesk workflows, permissions, or agent UI.
	•	Training a bespoke LLM from scratch.
	•	Perfect “single truth” root cause inference. (We aim for useful clustering and routing, then refine.)

⸻

5) Key constraints & known txt2kg integration issues (bake into build)

Your coding model must clone and inspect the repo directly, but also implement these guardrails based on NVIDIA forum-confirmed issues:
	1.	Doc/path typo: Step instructions have been seen missing the dgx-spark-playbooks/ directory segment (fix your setup docs + scripts).  ￼
	2.	GPU fallback fix: txt2kg may run Ollama on CPU unless OLLAMA_LLM_LIBRARY is set to the correct CUDA backend (reported fix: cuda_v13). Add an automated startup self-test that fails loudly if Ollama is not using GPU.  ￼
	3.	Complete/optional stack drift: NVIDIA forums indicate “complete stack” behavior has been inconsistent; Pinecone was not ARM-friendly and the author switched to Qdrant (but compose/scripts may be out-of-sync). Our project must explicitly define the vector store and ensure compose files/scripts match.  ￼
	4.	Ollama container healthcheck pitfalls: Some users hit an unhealthy Ollama container due to missing curl during healthcheck exec. Our Docker images/healthchecks must be validated.  ￼

⸻

6) Users / personas
	•	Support Ops / Support Lead: wants top issues, routing rules, macro opportunities.
	•	Product / Engineering: wants reproducible clusters, top root causes, trend spikes.
	•	Docs: wants “clusters that can be deflected by documentation.”
	•	Executives: wants a clean narrative + dashboard: “ticket drivers + reduction plan.”

⸻

7) System architecture (high-level)

A) Batch pipeline (historical)
	1.	Zendesk Agent (existing): GET /tickets?days=365 (or equivalent)
	2.	zti-ingest (new): pulls tickets/comments incrementally → writes raw JSON to disk + zti_raw table/collection
	3.	zti-normalize (new): canonicalizes into TicketDocument objects (see schema)
	4.	zti-kg-extract (adapted from txt2kg): LLM extracts entities/relations → writes graph triples
	5.	zti-embed (new or from playbook optional stack): embeds tickets + cluster summaries
	6.	Stores:
	•	Graph DB: ArangoDB (aligns with playbook UI pattern)
	•	Vector DB: Qdrant (recommended given ARM/DGX Spark constraints surfaced in forum)  ￼
	7.	zti-ui (adapted from txt2kg UI): graph explorer + cluster dashboards + ticket drilldown

B) Real-time pipeline (incremental)
	•	Scheduled pull (every N minutes) or webhook receiver (if your local agent supports it)
	•	Runs the same normalize → embed → cluster-assign → KG updates
	•	Produces “New ticket → cluster → Tier-0 suggestion”

C) Tier-0 prototype
	•	zti-chat service (new): local Ollama model + retrieval (Qdrant + cluster metadata)
	•	Outputs: cluster assignment, recommended reply, KB links, “ask-for-more-info” checklist.

⸻

8) Data contracts / schemas

8.1 Raw ticket record (persist exactly)
Store the raw Zendesk payload you already export (tickets, comments, metadata). Keep immutable.

8.2 Normalized document (the canonical unit for embeddings + LLM extraction)
TicketDocument (JSON):
	•	ticket_id (string/int)
	•	created_at, updated_at, solved_at
	•	status, priority, type
	•	brand, product_line (if derivable), platform (Win/Linux/Mac/ARM, etc)
	•	subject
	•	description
	•	comments[]: {author_role, created_at, body_text}
	•	tags[]
	•	custom_fields{} (flattened)
	•	attachments[] (metadata only; no binaries unless explicitly enabled)
	•	pii_redacted_text (the text that downstream models see)
	•	source_url (zendesk link if allowed; otherwise internal ref)

8.3 Knowledge graph model (minimum viable)
Nodes:
	•	Ticket
	•	IssueCluster
	•	Component (e.g., Media Server, Client, Cloud, Mobile, Storage, SSO…)
	•	ErrorSignature (error codes / log phrases)
	•	Environment (OS/GPU/driver/version)
	•	Feature / Workflow
	•	CustomerOrg (optional; hashed/pseudonymous)
	•	Fix (release, workaround, KB article)

Edges:
	•	Ticket -> mentions -> Component|Feature|ErrorSignature|Environment
	•	Ticket -> belongs_to -> IssueCluster
	•	IssueCluster -> caused_by -> Component|Feature
	•	IssueCluster -> mitigated_by -> Fix
	•	Ticket -> duplicate_of -> Ticket (optional, inferred)
	•	IssueCluster -> related_to -> IssueCluster (optional)

⸻

9) Clustering strategy (the “ticket reduction engine”)

You will implement two clustering signals and fuse them:
	1.	Embedding clustering (semantic):

	•	Create embeddings for:
	•	ticket_fulltext (subject + description + key comments)
	•	error_signature_text (extracted)
	•	Run UMAP + HDBSCAN (or BERTopic) to form clusters.
	•	Persist: cluster_id, cluster_label, cluster_keywords, representative_tickets[].

	2.	Graph community clustering (structural):

	•	Build a ticket-entity graph (tickets connected via shared entities/components/error signatures).
	•	Run community detection to find dense communities.
	•	Use as a “stabilizer” for semantic clusters.

	3.	Cluster labeling & summarization

	•	For each cluster:
	•	Generate: “What is this issue?”, “how to reproduce?”, “common environment?”, “recommended response/macro?”, “best deflection path (Docs/Product/Support)”
	•	Store cluster summary as a first-class object (also embedded for retrieval).

	4.	Feedback loop (must-have)

	•	UI allows a human to:
	•	merge/split clusters
	•	mark a ticket misclustered
	•	mark duplicates
	•	These edits create a small “training signal” (rules + exemplars) for improved routing.

⸻

10) UI / UX requirements

10.1 Core screens
	1.	Executive Overview

	•	KPIs: tickets/week, solved time trend, top clusters, new clusters, deflection potential
	•	“Reduction backlog”: clusters ranked by ROI

	2.	Cluster Explorer

	•	Table + filters (date range, product/component, priority)
	•	Sparkline trends per cluster
	•	Click into cluster → summary, keywords, representative tickets, suggested macro/KB

	3.	Graph View

	•	Interactive knowledge graph:
	•	nodes: clusters, components, errors, environments
	•	edges weighted by frequency
	•	Selecting a node filters the ticket list

	4.	Ticket Drilldown

	•	Shows normalized text, extracted entities, nearest neighbors, predicted cluster, confidence
	•	Buttons: “mark duplicate,” “move to cluster,” “create new cluster”

	5.	Tier-0 Playground

	•	Paste a ticket / pick an existing ticket → see:
	•	cluster assignment
	•	“first response draft”
	•	“ask-for-more-info checklist”
	•	citations: which cluster summary + which representative tickets were used

⸻

11) Services / repo structure (what the coding model must build)

Monorepo layout (recommended):
	•	docker/
	•	docker-compose.yml (single source of truth)
	•	docker-compose.dev.yml
	•	services/
	•	ingest/ (FastAPI + scheduler)
	•	normalize/ (library + CLI)
	•	kg_extract/ (wrap/adapt txt2kg extraction step)
	•	embed_cluster/ (embeddings, clustering, persistence)
	•	chat/ (Tier-0 prototype)
	•	ui/ (fork/adapt txt2kg UI or integrate into it)
	•	shared/
	•	schemas/
	•	prompts/
	•	config/
	•	utils/
	•	scripts/
	•	bootstrap.sh (clone/playbook wiring, env checks)
	•	healthcheck.sh (GPU + db + vector store + UI checks)

Hard requirement: the coding model must vendor or reference NVIDIA’s txt2kg playbook as a dependency (git submodule or documented clone step), and clearly document the correct pathing (forum confirmed the doc line can be wrong).  ￼

⸻

12) Deployment requirements (Docker-first)
	•	Must run on your local environment (DGX Spark or similar) with:
	•	Ollama container configured for GPU (and verified)  ￼
	•	ArangoDB container
	•	Qdrant container (recommended due ARM issues with Pinecone mentioned in the community thread)  ￼
	•	UI container
	•	Worker containers (ingest, extract, embed/cluster)

Startup self-tests (blocker if failing):
	•	ollama ps shows PROCESSOR = GPU for the chosen model (or equivalent)
	•	Qdrant health endpoint OK
	•	ArangoDB reachable
	•	UI reachable remotely (LAN), not just localhost

⸻

13) Tier-0 chatbot requirements (prototype)

Inputs
	•	A new ticket payload (subject/description/comments/tags/fields)

Process
	•	Normalize + redact
	•	Embed and retrieve:
	•	nearest clusters (top K)
	•	top representative tickets
	•	cluster “known fixes / macros / KB”
	•	Generate response with:
	•	predicted_cluster
	•	confidence
	•	recommended_actions
	•	draft_reply (optional toggle)
	•	citations (IDs/links to internal cluster/tickets)

Safety / policy
	•	Never reveal customer PII.
	•	Never fabricate “official fixes.” If uncertain, it must ask for logs / details.

⸻

14) Testing & evaluation

Unit tests
	•	Ticket normalization (handles missing fields, large comment threads)
	•	Redaction (PII patterns)
	•	Deterministic chunking

Integration tests
	•	Pull 7 days from your local agent
	•	Run end-to-end pipeline
	•	Assert: clusters created, graph populated, UI loads

Quality checks
	•	Manual evaluation set: 50–100 tickets labeled by humans into ~10–20 clusters
	•	Measure cluster purity + duplicate hit-rate
	•	Track improvements after feedback merges/splits

⸻

15) Milestones (phased delivery)

Phase 0 — Skeleton runs (1–2 days)
	•	Clone/attach txt2kg
	•	Compose boots, UI opens
	•	GPU verification + env var fix baked into scripts (avoid CPU fallback)  ￼

Phase 1 — Historical pipeline (3–7 days)
	•	Ingest 365 days from local agent
	•	Normalize → embed → cluster → UI cluster explorer
	•	Export cluster report (CSV/JSON)

Phase 2 — Knowledge graph + fusion clustering (1–2 weeks)
	•	LLM extraction into graph
	•	Graph view in UI
	•	Community detection + fused clusters
	•	Human feedback actions (merge/split/move)

Phase 3 — Real-time + Tier-0 (1–2 weeks)
	•	Incremental ingest job
	•	Ticket-to-cluster routing in near-real-time
	•	Tier-0 playground with citations

⸻

16) Open decisions (make these explicit in code/config)
	1.	Vector DB: Qdrant (recommended) vs ArangoSearch-only. (Community reports indicate Pinecone was problematic on ARM and a switch to Qdrant was made.)  ￼
	2.	LLM choice for extraction: local Ollama model(s) (fast) vs optional vLLM microservice (if you want throughput later).
	3.	Redaction policy: strict (default) vs allow internal org names in a “private mode.”
