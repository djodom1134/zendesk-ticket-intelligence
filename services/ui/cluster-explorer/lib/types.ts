/**
 * Shared types for the ZTI Cluster Explorer
 */

// API response type
export interface ClusterAPIResponse {
  id: string;
  label: string;
  size: number;
  priority: string;
  confidence: number;
  keywords: string[];
  issue_description?: string;
  created_at?: string;
  environment?: string;
  recommended_response?: string;
  deflection_path?: string;
  representative_tickets?: string[];
  x?: number;  // 2D UMAP x-coordinate
  y?: number;  // 2D UMAP y-coordinate
}

// UI type with computed fields
export interface Cluster {
  id: string;
  label: string;
  size: number;
  trend: number[];
  priority: string;
  keywords: string[];
  issueDescription: string;
  environment?: string;
  recommendedResponse?: string;
  deflectionPath?: string;
  confidence: number;
  representativeTickets?: string[];
  createdAt: string;
  x?: number;  // 2D UMAP x-coordinate
  y?: number;  // 2D UMAP y-coordinate
}

// Ticket position for scatter plot visualization
export interface TicketPosition {
  ticket_id: string;
  cluster_id?: string;
  cluster_label?: string;
  subject?: string;
  x: number;
  y: number;
  z?: number;
}

// Transform API response to UI format
export function transformCluster(apiCluster: ClusterAPIResponse): Cluster {
  return {
    id: apiCluster.id,
    label: apiCluster.label,
    size: apiCluster.size,
    trend: Array(7).fill(0).map((_, i) => Math.max(1, apiCluster.size - (6 - i) * 2)), // Generate fake trend
    priority: apiCluster.priority || 'medium',
    keywords: apiCluster.keywords || [],
    issueDescription: apiCluster.issue_description || '',
    environment: apiCluster.environment,
    recommendedResponse: apiCluster.recommended_response,
    deflectionPath: apiCluster.deflection_path,
    confidence: apiCluster.confidence || 0,
    representativeTickets: apiCluster.representative_tickets,
    createdAt: apiCluster.created_at || new Date().toISOString(),
    x: apiCluster.x,
    y: apiCluster.y,
  };
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

