"use client";

import { useState, useEffect } from "react";
import { ClustersTable } from "@/components/clusters-table";
import { ClusterDetails } from "@/components/cluster-details";
import { TicketScatterPlot } from "@/components/ticket-scatter-plot";
import { ClusterSearch } from "@/components/cluster-search";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { LayoutGrid, TrendingUp, FileText, Network, Search } from "lucide-react";
import { Cluster, ClusterAPIResponse, transformCluster } from "@/lib/types";

// Mock data - will be replaced with API calls
const mockClusters = [
  {
    id: "cluster-0",
    label: "Video Data Compatibility Issues",
    size: 45,
    trend: [12, 15, 18, 22, 28, 35, 45],
    priority: "high",
    keywords: ["video", "recording", "compatibility", "version", "upgrade"],
    issueDescription: "Users experiencing issues with video data compatibility between different Store Recorder versions",
    environment: "Store Recorder 5.0, 4.0, 3.x versions",
    recommendedResponse: "We understand you're experiencing compatibility issues with video recordings...",
    deflectionPath: "docs/video-compatibility-guide",
    confidence: 0.85,
    representativeTickets: ["TKT-1234", "TKT-1256", "TKT-1289"],
    createdAt: "2024-12-15",
  },
  {
    id: "cluster-1",
    label: "Login Authentication Failures",
    size: 32,
    trend: [8, 10, 12, 18, 22, 28, 32],
    priority: "critical",
    keywords: ["login", "authentication", "password", "SSO", "timeout"],
    issueDescription: "Users unable to authenticate or experiencing frequent session timeouts",
    environment: "Web portal, SSO integration, Active Directory",
    recommendedResponse: "We apologize for the login difficulties. Let's troubleshoot this together...",
    deflectionPath: "docs/login-troubleshooting",
    confidence: 0.92,
    representativeTickets: ["TKT-1301", "TKT-1315", "TKT-1322"],
    createdAt: "2024-12-14",
  },
  {
    id: "cluster-2",
    label: "Report Generation Errors",
    size: 18,
    trend: [5, 6, 8, 10, 12, 15, 18],
    priority: "medium",
    keywords: ["report", "export", "PDF", "timeout", "generation"],
    issueDescription: "Reports failing to generate or timing out during export",
    environment: "Reporting module, PDF exports, large datasets",
    recommendedResponse: "Report generation issues can occur with large datasets. Try these steps...",
    deflectionPath: "docs/report-optimization",
    confidence: 0.78,
    representativeTickets: ["TKT-1401", "TKT-1418"],
    createdAt: "2024-12-13",
  },
];

export default function Home() {
  const [selectedCluster, setSelectedCluster] = useState<Cluster | null>(null);
  const [activeTab, setActiveTab] = useState("clusters");
  const [clusters, setClusters] = useState<Cluster[]>([]);
  const [stats, setStats] = useState({ totalClusters: 0, totalTickets: 0, avgConfidence: 0 });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch clusters from API with 2D positions
    const fetchClusters = async () => {
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
        const response = await fetch(`${apiUrl}/api/clusters?limit=100&include_positions=true`);
        if (response.ok) {
          const data: ClusterAPIResponse[] = await response.json();
          // API returns array directly, not wrapped in {clusters: [...]}
          if (Array.isArray(data) && data.length > 0) {
            const transformedClusters = data.map(transformCluster);
            setClusters(transformedClusters);
            setStats({
              totalClusters: transformedClusters.length,
              totalTickets: transformedClusters.reduce((sum: number, c: Cluster) => sum + c.size, 0),
              avgConfidence: Math.round(transformedClusters.reduce((sum: number, c: Cluster) => sum + c.confidence, 0) / transformedClusters.length * 100),
            });
          }
        }
      } catch (error) {
        console.error('Failed to fetch clusters:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchClusters();
  }, []);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <main className="container mx-auto px-6 py-8">
        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <StatCard title="Total Clusters" value={stats.totalClusters.toString()} icon={<LayoutGrid className="h-5 w-5" />} />
          <StatCard title="Active Tickets" value={stats.totalTickets.toLocaleString()} icon={<FileText className="h-5 w-5" />} />
          <StatCard title="Avg Confidence" value={`${stats.avgConfidence}%`} icon={<TrendingUp className="h-5 w-5" />} />
          <StatCard title="Status" value={loading ? "Loading..." : "Ready"} icon={<Network className="h-5 w-5" />} />
        </div>

        <Tabs defaultValue="clusters" className="w-full" onValueChange={setActiveTab}>
          <TabsList className="nvidia-build-tabs mb-6">
            <TabsTrigger value="clusters" className="nvidia-build-tab">
              <div className="nvidia-build-tab-icon">
                <LayoutGrid className="h-3 w-3 text-[#76b900]" />
              </div>
              <span>Cluster Table</span>
            </TabsTrigger>
            <TabsTrigger value="graph" className="nvidia-build-tab">
              <div className="nvidia-build-tab-icon">
                <Network className="h-3 w-3 text-[#76b900]" />
              </div>
              <span>3D Graph</span>
            </TabsTrigger>
            <TabsTrigger value="search" className="nvidia-build-tab">
              <div className="nvidia-build-tab-icon">
                <Search className="h-3 w-3 text-[#76b900]" />
              </div>
              <span>Search</span>
            </TabsTrigger>
            <TabsTrigger value="details" className="nvidia-build-tab" disabled={!selectedCluster}>
              <div className="nvidia-build-tab-icon">
                <FileText className="h-3 w-3 text-[#76b900]" />
              </div>
              <span>Cluster Details</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="clusters">
            <ClustersTable
              clusters={clusters}
              onSelectCluster={(cluster) => {
                setSelectedCluster(cluster);
                setActiveTab("details");
              }}
            />
          </TabsContent>

          <TabsContent value="graph">
            <div className="h-[800px]">
              <TicketScatterPlot
                apiUrl={process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}
                initialMode="3d"
                xDim={0}
                yDim={1}
                zDim={2}
              />
            </div>
          </TabsContent>

          <TabsContent value="search">
            <div className="nvidia-build-card p-6">
              <h2 className="text-xl font-bold mb-4">Search Clusters</h2>
              <ClusterSearch
                clusters={clusters}
                onSelectCluster={(cluster) => {
                  setSelectedCluster(cluster);
                  setActiveTab("details");
                }}
              />
            </div>
          </TabsContent>

          <TabsContent value="details">
            {selectedCluster && (
              <ClusterDetails
                cluster={selectedCluster}
                onBack={() => setActiveTab("clusters")}
              />
            )}
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}

function StatCard({
  title,
  value,
  icon,
  trend
}: {
  title: string;
  value: string;
  icon: React.ReactNode;
  trend?: string;
}) {
  return (
    <div className="nvidia-build-card p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm text-muted-foreground">{title}</span>
        <div className="text-[#76b900]">{icon}</div>
      </div>
      <div className="flex items-end gap-2">
        <span className="text-2xl font-bold">{value}</span>
        {trend && (
          <span className="text-sm text-green-500 mb-1">{trend}</span>
        )}
      </div>
    </div>
  );
}
