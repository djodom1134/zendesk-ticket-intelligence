"use client";

import { useEffect, useState } from "react";
import {
  TrendingUp, TrendingDown, Ticket, Layers, Clock, Target,
  AlertTriangle, CheckCircle2, BarChart3, Loader2, Zap, Sparkles
} from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface WeeklyTrend {
  week_start: string;
  count: number;
  change_pct: number | null;
}

interface ClusterGrowth {
  cluster_id: string;
  label: string;
  current_size: number;
  previous_size: number;
  growth_rate: number;
  is_new: boolean;
}

interface ResolutionMetrics {
  avg_hours: number;
  median_hours: number;
  p90_hours: number;
  trend_pct: number;
}

interface DeflectionMetrics {
  total_deflectable: number;
  deflection_rate: number;
  top_deflectable_clusters: { cluster_id: string; label: string; size: number; confidence: number }[];
  estimated_hours_saved: number;
}

interface Stats {
  total_clusters: number;
  total_tickets: number;
  avg_confidence: number;
  trending_up: number;
  tickets_this_week: number;
  tickets_last_week: number;
  week_over_week_change: number;
  weekly_trend: WeeklyTrend[];
  new_clusters_this_week: number;
  growing_clusters: ClusterGrowth[];
  resolution: ResolutionMetrics | null;
  deflection: DeflectionMetrics | null;
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

  const wowChange = stats?.week_over_week_change || 0;
  const deflection = stats?.deflection;
  const resolution = stats?.resolution;

  return (
    <main className="container mx-auto px-4 py-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold gradient-text mb-2">Executive Overview</h1>
        <p className="text-muted-foreground">Real-time KPIs calculated from ticket data</p>
      </div>

      {/* Primary KPI Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <KPICard
          title="Tickets This Week"
          value={stats?.tickets_this_week?.toLocaleString() || 0}
          subtitle={`Last week: ${stats?.tickets_last_week?.toLocaleString() || 0}`}
          icon={Ticket}
          trend={`${wowChange >= 0 ? '+' : ''}${wowChange.toFixed(1)}% vs last week`}
          trendUp={wowChange <= 0}  // Less tickets = good
        />
        <KPICard
          title="Active Clusters"
          value={stats?.total_clusters || 0}
          subtitle={stats?.new_clusters_this_week ? `${stats.new_clusters_this_week} new this week` : "Unique issue patterns"}
          icon={Layers}
          trend={stats?.new_clusters_this_week ? `+${stats.new_clusters_this_week} new` : undefined}
          trendUp={false}
        />
        <KPICard
          title="Avg Resolution Time"
          value={resolution ? `${resolution.avg_hours.toFixed(0)}h` : "N/A"}
          subtitle={resolution ? `Median: ${resolution.median_hours.toFixed(0)}h, P90: ${resolution.p90_hours.toFixed(0)}h` : "No resolved tickets"}
          icon={Clock}
        />
        <KPICard
          title="Deflection Potential"
          value={deflection ? `${deflection.deflection_rate.toFixed(0)}%` : "0%"}
          subtitle={deflection ? `${deflection.total_deflectable} tickets (${deflection.estimated_hours_saved.toFixed(0)}h saved)` : "Calculate from clusters"}
          icon={Zap}
          trend={deflection ? `${deflection.total_deflectable} automatable` : undefined}
          trendUp={true}
        />
      </div>

      {/* Secondary KPIs */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <KPICard
          title="Total Tickets"
          value={stats?.total_tickets.toLocaleString() || 0}
          subtitle="All time"
          icon={BarChart3}
        />
        <KPICard
          title="Avg Confidence"
          value={`${((stats?.avg_confidence || 0) * 100).toFixed(0)}%`}
          subtitle="Cluster assignment accuracy"
          icon={Target}
        />
        <KPICard
          title="Trending Up"
          value={stats?.trending_up || 0}
          subtitle="Clusters growing this week"
          icon={TrendingUp}
        />
      </div>

      {/* Weekly Trend Chart (simple bar representation) */}
      {stats?.weekly_trend && stats.weekly_trend.length > 0 && (
        <div className="bg-card border border-border rounded-xl p-6 shadow-sm mb-6">
          <div className="flex items-center gap-2 mb-4">
            <BarChart3 className="h-5 w-5 text-[#76b900]" />
            <h2 className="text-lg font-semibold">Weekly Ticket Trend</h2>
          </div>
          <div className="flex items-end gap-2 h-32">
            {stats.weekly_trend.slice().reverse().map((week, i) => {
              const maxCount = Math.max(...stats.weekly_trend.map(w => w.count));
              const height = maxCount > 0 ? (week.count / maxCount) * 100 : 0;
              return (
                <div key={i} className="flex-1 flex flex-col items-center">
                  <div
                    className="w-full bg-[#76b900]/70 rounded-t transition-all hover:bg-[#76b900]"
                    style={{ height: `${height}%`, minHeight: week.count > 0 ? '4px' : '0' }}
                    title={`${week.week_start}: ${week.count} tickets`}
                  />
                  <span className="text-xs text-muted-foreground mt-1">{week.count}</span>
                  <span className="text-xs text-muted-foreground">{week.week_start.slice(5)}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Growing Clusters */}
      {stats?.growing_clusters && stats.growing_clusters.length > 0 && (
        <div className="bg-card border border-border rounded-xl p-6 shadow-sm mb-6">
          <div className="flex items-center gap-2 mb-4">
            <Sparkles className="h-5 w-5 text-[#76b900]" />
            <h2 className="text-lg font-semibold">Cluster Growth</h2>
          </div>
          <div className="space-y-2">
            {stats.growing_clusters.map((c) => (
              <div key={c.cluster_id} className="flex items-center justify-between p-2 bg-muted/30 rounded">
                <div className="flex items-center gap-2">
                  {c.is_new && <span className="px-1.5 py-0.5 text-xs bg-green-500/20 text-green-500 rounded">NEW</span>}
                  <span className="font-medium">{c.label}</span>
                </div>
                <div className="flex items-center gap-4 text-sm">
                  <span className="text-muted-foreground">{c.previous_size} → {c.current_size}</span>
                  <span className={`flex items-center gap-1 ${c.growth_rate >= 0 ? 'text-red-500' : 'text-green-500'}`}>
                    {c.growth_rate >= 0 ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
                    {c.growth_rate >= 0 ? '+' : ''}{c.growth_rate.toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Deflection Potential Details */}
      {deflection && deflection.top_deflectable_clusters.length > 0 && (
        <div className="bg-card border border-border rounded-xl p-6 shadow-sm mb-6">
          <div className="flex items-center gap-2 mb-4">
            <Zap className="h-5 w-5 text-[#76b900]" />
            <h2 className="text-lg font-semibold">Top Deflectable Clusters</h2>
            <span className="ml-auto text-sm text-muted-foreground">
              Est. {deflection.estimated_hours_saved.toFixed(0)} hours saved via automation
            </span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left text-muted-foreground">
                  <th className="pb-3 font-medium">Cluster</th>
                  <th className="pb-3 font-medium text-right">Tickets</th>
                  <th className="pb-3 font-medium text-right">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {deflection.top_deflectable_clusters.map((c) => (
                  <tr key={c.cluster_id} className="border-b border-border/50">
                    <td className="py-3 font-medium">{c.label}</td>
                    <td className="py-3 text-right">{c.size}</td>
                    <td className="py-3 text-right">{(c.confidence * 100).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

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

