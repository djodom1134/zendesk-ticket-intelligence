"use client";

import { useEffect, useState } from "react";
import { 
  TrendingUp, TrendingDown, Ticket, Layers, Clock, Target,
  AlertTriangle, CheckCircle2, BarChart3, Loader2
} from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Stats {
  total_clusters: number;
  total_tickets: number;
  avg_confidence: number;
  trending_up: number;
  tickets_last_7d: number;
  top_clusters: { id: string; label: string; size: number }[];
}

interface Cluster {
  id: string;
  label: string;
  size: number;
  priority: string;
  confidence: number;
  keywords: string[];
}

function KPICard({ 
  title, value, subtitle, icon: Icon, trend, trendUp 
}: { 
  title: string; 
  value: string | number; 
  subtitle?: string;
  icon: React.ElementType;
  trend?: string;
  trendUp?: boolean;
}) {
  return (
    <div className="bg-card border border-border rounded-xl p-6 shadow-sm">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-muted-foreground">{title}</p>
          <p className="text-3xl font-bold mt-1">{value}</p>
          {subtitle && <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>}
        </div>
        <div className="h-10 w-10 rounded-lg bg-[#76b900]/10 flex items-center justify-center">
          <Icon className="h-5 w-5 text-[#76b900]" />
        </div>
      </div>
      {trend && (
        <div className={`flex items-center gap-1 mt-3 text-sm ${trendUp ? 'text-green-500' : 'text-red-500'}`}>
          {trendUp ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
          <span>{trend}</span>
        </div>
      )}
    </div>
  );
}

export default function OverviewPage() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [clusters, setClusters] = useState<Cluster[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const [statsRes, clustersRes] = await Promise.all([
          fetch(`${API_URL}/api/stats`),
          fetch(`${API_URL}/api/clusters?limit=10`),
        ]);
        if (!statsRes.ok || !clustersRes.ok) throw new Error("Failed to fetch data");
        setStats(await statsRes.json());
        setClusters(await clustersRes.json());
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto px-4 py-6">
        <div className="p-4 bg-destructive/10 border border-destructive/30 rounded-lg text-destructive">
          {error}
        </div>
      </div>
    );
  }

  const deflectionPotential = clusters.filter(c => c.priority === "high").reduce((sum, c) => sum + c.size, 0);

  return (
    <main className="container mx-auto px-4 py-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold gradient-text mb-2">Executive Overview</h1>
        <p className="text-muted-foreground">Key metrics and insights from ticket intelligence</p>
      </div>

      {/* KPI Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <KPICard 
          title="Total Tickets" 
          value={stats?.total_tickets.toLocaleString() || 0}
          subtitle="Last 90 days"
          icon={Ticket}
          trend="+12% vs last period"
          trendUp={true}
        />
        <KPICard 
          title="Active Clusters" 
          value={stats?.total_clusters || 0}
          subtitle="Unique issue patterns"
          icon={Layers}
        />
        <KPICard 
          title="Avg Confidence" 
          value={`${((stats?.avg_confidence || 0) * 100).toFixed(0)}%`}
          subtitle="Cluster assignment accuracy"
          icon={Target}
        />
        <KPICard 
          title="Deflection Potential" 
          value={deflectionPotential}
          subtitle="High-priority tickets automatable"
          icon={CheckCircle2}
        />
      </div>

      {/* Top Clusters Table */}
      <div className="bg-card border border-border rounded-xl p-6 shadow-sm">
        <div className="flex items-center gap-2 mb-4">
          <BarChart3 className="h-5 w-5 text-[#76b900]" />
          <h2 className="text-lg font-semibold">Top Clusters by Volume</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-left text-muted-foreground">
                <th className="pb-3 font-medium">Cluster</th>
                <th className="pb-3 font-medium text-right">Tickets</th>
                <th className="pb-3 font-medium text-right">Priority</th>
                <th className="pb-3 font-medium text-right">Confidence</th>
              </tr>
            </thead>
            <tbody>
              {clusters.map((c, i) => (
                <tr key={c.id} className="border-b border-border/50 hover:bg-muted/30">
                  <td className="py-3">
                    <span className="font-medium">{c.label}</span>
                  </td>
                  <td className="py-3 text-right">{c.size}</td>
                  <td className="py-3 text-right">
                    <span className={`px-2 py-0.5 rounded text-xs ${
                      c.priority === 'high' ? 'bg-red-500/20 text-red-500' :
                      c.priority === 'medium' ? 'bg-yellow-500/20 text-yellow-500' :
                      'bg-green-500/20 text-green-500'
                    }`}>
                      {c.priority}
                    </span>
                  </td>
                  <td className="py-3 text-right">{(c.confidence * 100).toFixed(0)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </main>
  );
}

