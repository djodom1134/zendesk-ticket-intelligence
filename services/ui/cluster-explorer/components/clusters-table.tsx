"use client";

import { useState } from "react";
import { 
  AlertCircle, 
  TrendingUp, 
  Eye, 
  Tag,
  Sparkles 
} from "lucide-react";

interface Cluster {
  id: string;
  label: string;
  size: number;
  trend: number[];
  priority: string;
  keywords: string[];
  issueDescription: string;
  confidence: number;
  createdAt: string;
}

interface ClustersTableProps {
  clusters: Cluster[];
  onSelectCluster: (cluster: Cluster) => void;
}

export function ClustersTable({ clusters, onSelectCluster }: ClustersTableProps) {
  const [sortBy, setSortBy] = useState<"size" | "confidence" | "createdAt">("size");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");

  const sortedClusters = [...clusters].sort((a, b) => {
    const order = sortOrder === "asc" ? 1 : -1;
    if (sortBy === "size") return (a.size - b.size) * order;
    if (sortBy === "confidence") return (a.confidence - b.confidence) * order;
    return a.createdAt.localeCompare(b.createdAt) * order;
  });

  const getPriorityBadge = (priority: string) => {
    const styles = {
      critical: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400",
      high: "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-400",
      medium: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400",
      low: "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400",
    };
    return styles[priority as keyof typeof styles] || styles.medium;
  };

  // Simple sparkline using SVG
  const Sparkline = ({ data }: { data: number[] }) => {
    const max = Math.max(...data);
    const min = Math.min(...data);
    const range = max - min || 1;
    const width = 80;
    const height = 24;
    const points = data.map((v, i) => {
      const x = (i / (data.length - 1)) * width;
      const y = height - ((v - min) / range) * height;
      return `${x},${y}`;
    }).join(" ");
    
    return (
      <svg width={width} height={height} className="inline-block">
        <polyline
          points={points}
          fill="none"
          stroke="#76b900"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    );
  };

  return (
    <div className="nvidia-build-card p-0 overflow-hidden">
      {/* Header */}
      <div className="flex justify-between items-center p-6 bg-muted/10 border-b border-border/20">
        <div className="flex items-center gap-4">
          <span className="text-lg font-semibold">{clusters.length} Clusters</span>
          <span className="text-sm text-muted-foreground">Click a row to view details</span>
        </div>
        <div className="flex items-center gap-2">
          <select 
            className="text-sm bg-background border border-border rounded-lg px-3 py-1.5"
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
          >
            <option value="size">Sort by Size</option>
            <option value="confidence">Sort by Confidence</option>
            <option value="createdAt">Sort by Date</option>
          </select>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-border/20 bg-muted/5">
              <th className="text-xs font-semibold text-muted-foreground uppercase tracking-wider text-left py-3 px-6">Cluster</th>
              <th className="text-xs font-semibold text-muted-foreground uppercase tracking-wider text-left py-3">Priority</th>
              <th className="text-xs font-semibold text-muted-foreground uppercase tracking-wider text-right py-3">Size</th>
              <th className="text-xs font-semibold text-muted-foreground uppercase tracking-wider text-center py-3">Trend</th>
              <th className="text-xs font-semibold text-muted-foreground uppercase tracking-wider text-center py-3">Confidence</th>
              <th className="text-xs font-semibold text-muted-foreground uppercase tracking-wider text-left py-3">Keywords</th>
              <th className="text-xs font-semibold text-muted-foreground uppercase tracking-wider text-center py-3 pr-6">Actions</th>
            </tr>
          </thead>
          <tbody>
            {sortedClusters.map((cluster) => (
              <tr 
                key={cluster.id}
                className="transition-all duration-200 hover:bg-[#76b900]/5 cursor-pointer group border-b border-border/10 last:border-b-0 hover:border-l-4 hover:border-l-[#76b900]"
                onClick={() => onSelectCluster(cluster)}
              >
                <td className="py-4 px-6">
                  <div className="flex items-center gap-3">
                    <div className="h-8 w-8 rounded-lg bg-[#76b900]/15 flex items-center justify-center">
                      <Sparkles className="h-4 w-4 text-[#76b900]" />
                    </div>
                    <div>
                      <span className="text-sm font-medium">{cluster.label}</span>
                      <p className="text-xs text-muted-foreground truncate max-w-[250px]">{cluster.issueDescription}</p>
                    </div>
                  </div>
                </td>
                <td className="py-4">
                  <span className={`text-xs font-medium px-2.5 py-1 rounded-full capitalize ${getPriorityBadge(cluster.priority)}`}>
                    {cluster.priority}
                  </span>
                </td>
                <td className="py-4 text-right">
                  <span className="text-sm font-mono bg-muted/50 px-2 py-1 rounded">{cluster.size}</span>
                </td>
                <td className="py-4 text-center">
                  <Sparkline data={cluster.trend} />
                </td>
                <td className="py-4 text-center">
                  <span className={`text-xs font-bold px-2.5 py-1 rounded-full ${
                    cluster.confidence >= 0.8 ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400" :
                    cluster.confidence >= 0.6 ? "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400" :
                    "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400"
                  }`}>
                    {Math.round(cluster.confidence * 100)}%
                  </span>
                </td>
                <td className="py-4">
                  <div className="flex gap-1 flex-wrap max-w-[200px]">
                    {cluster.keywords.slice(0, 3).map((kw) => (
                      <span key={kw} className="text-xs bg-muted px-2 py-0.5 rounded">{kw}</span>
                    ))}
                  </div>
                </td>
                <td className="py-4 pr-6">
                  <button className="p-2 text-muted-foreground hover:text-[#76b900] hover:bg-[#76b900]/10 rounded-lg transition-colors opacity-0 group-hover:opacity-100">
                    <Eye className="h-4 w-4" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

