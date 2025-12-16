/**
 * Shared types for the ZTI Cluster Explorer
 */

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

