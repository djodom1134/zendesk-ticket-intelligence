/**
 * API client for ZTI backend
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Cluster {
  id: string;
  label: string;
  size: number;
  priority: string;
  confidence: number;
  keywords: string[];
  issue_description?: string;
  created_at?: string;
}

export interface ClusterDetail extends Cluster {
  environment?: string;
  symptoms?: string[];
  recommended_response?: string;
  deflection_path?: string;
  representative_tickets: string[];
  trend: number[];
}

export interface Ticket {
  id: string;
  subject: string;
  status: string;
  cluster_id?: string;
  created_at?: string;
  similarity?: number;
}

export interface TicketDetail extends Ticket {
  description: string;
  priority?: string;
  tags: string[];
  summary?: string;
  updated_at?: string;
}

export interface SearchResult {
  tickets: Ticket[];
  query: string;
  total: number;
}

export interface Citation {
  type: string;
  id: string;
  label: string;
  relevance: number;
}

export interface ChatResponse {
  response: string;
  predicted_cluster?: string;
  cluster_label?: string;
  confidence: number;
  recommended_actions: string[];
  draft_reply?: string;
  citations: Citation[];
  ask_for_info: string[];
}

export interface Stats {
  total_clusters: number;
  total_tickets: number;
  avg_confidence: number;
  trending_up: number;
  tickets_last_7d?: number;
  top_clusters?: { id: string; label: string; size: number }[];
}

// API functions
export async function fetchClusters(limit = 50, offset = 0): Promise<Cluster[]> {
  const res = await fetch(`${API_URL}/api/clusters?limit=${limit}&offset=${offset}`);
  if (!res.ok) throw new Error(`Failed to fetch clusters: ${res.status}`);
  return res.json();
}

export async function fetchCluster(id: string): Promise<ClusterDetail> {
  const res = await fetch(`${API_URL}/api/clusters/${id}`);
  if (!res.ok) throw new Error(`Failed to fetch cluster: ${res.status}`);
  return res.json();
}

export async function fetchStats(): Promise<Stats> {
  const res = await fetch(`${API_URL}/api/stats`);
  if (!res.ok) throw new Error(`Failed to fetch stats: ${res.status}`);
  return res.json();
}

export async function fetchTickets(options?: {
  limit?: number;
  offset?: number;
  cluster_id?: string;
  status?: string;
}): Promise<Ticket[]> {
  const params = new URLSearchParams();
  if (options?.limit) params.set("limit", String(options.limit));
  if (options?.offset) params.set("offset", String(options.offset));
  if (options?.cluster_id) params.set("cluster_id", options.cluster_id);
  if (options?.status) params.set("status", options.status);
  
  const res = await fetch(`${API_URL}/api/tickets?${params}`);
  if (!res.ok) throw new Error(`Failed to fetch tickets: ${res.status}`);
  return res.json();
}

export async function fetchTicket(id: string): Promise<TicketDetail> {
  const res = await fetch(`${API_URL}/api/tickets/${id}`);
  if (!res.ok) throw new Error(`Failed to fetch ticket: ${res.status}`);
  return res.json();
}

export async function searchTickets(query: string, limit = 10, cluster_id?: string): Promise<SearchResult> {
  const res = await fetch(`${API_URL}/api/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, limit, cluster_id }),
  });
  if (!res.ok) throw new Error(`Search failed: ${res.status}`);
  return res.json();
}

export async function sendChatMessage(message: string, includeCitations = true): Promise<ChatResponse> {
  const res = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, include_citations: includeCitations }),
  });
  if (!res.ok) throw new Error(`Chat failed: ${res.status}`);
  return res.json();
}

