"use client";

import { useState } from "react";
import { ClustersTable } from "@/components/clusters-table";
import { ClusterDetails } from "@/components/cluster-details";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { LayoutGrid, TrendingUp, FileText } from "lucide-react";

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
  const [selectedCluster, setSelectedCluster] = useState<typeof mockClusters[0] | null>(null);
  const [activeTab, setActiveTab] = useState("clusters");

  return (
    <div className="min-h-screen bg-background text-foreground">
      <main className="container mx-auto px-6 py-8">
        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <StatCard title="Total Clusters" value="24" icon={<LayoutGrid className="h-5 w-5" />} />
          <StatCard title="Active Tickets" value="1,247" icon={<FileText className="h-5 w-5" />} />
          <StatCard title="Trending Up" value="8" icon={<TrendingUp className="h-5 w-5" />} trend="+23%" />
          <StatCard title="Avg Confidence" value="84%" icon={<LayoutGrid className="h-5 w-5" />} />
        </div>

        <Tabs defaultValue="clusters" className="w-full" onValueChange={setActiveTab}>
          <TabsList className="nvidia-build-tabs mb-6">
            <TabsTrigger value="clusters" className="nvidia-build-tab">
              <div className="nvidia-build-tab-icon">
                <LayoutGrid className="h-3 w-3 text-[#76b900]" />
              </div>
              <span>Cluster Overview</span>
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
              clusters={mockClusters}
              onSelectCluster={(cluster) => {
                setSelectedCluster(cluster);
                setActiveTab("details");
              }}
            />
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
