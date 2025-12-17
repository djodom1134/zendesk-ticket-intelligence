"use client";

import { ArrowLeft, Tag, FileText, MessageSquare, ExternalLink, Copy, Check, Sparkles, AlertCircle } from "lucide-react";
import { useState } from "react";
import { Cluster } from "../lib/types";

interface ClusterDetailsProps {
  cluster: Cluster;
  onBack: () => void;
}

export function ClusterDetails({ cluster, onBack }: ClusterDetailsProps) {
  const [copied, setCopied] = useState(false);
  const [enriching, setEnriching] = useState(false);
  const [symptoms, setSymptoms] = useState<string[]>([]);
  const [environment, setEnvironment] = useState(cluster.environment || '');
  const [recommendedResponse, setRecommendedResponse] = useState(cluster.recommendedResponse || '');

  const copyResponse = () => {
    const textToCopy = recommendedResponse || cluster.recommendedResponse;
    if (textToCopy) {
      navigator.clipboard.writeText(textToCopy);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const enrichCluster = async () => {
    setEnriching(true);
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/clusters/${cluster.id}/enrich`, {
        method: 'POST',
      });
      if (response.ok) {
        const data = await response.json();
        setSymptoms(data.symptoms || []);
        setEnvironment(data.environment || cluster.environment || '');
        setRecommendedResponse(data.recommended_response || cluster.recommendedResponse || '');
      }
    } catch (error) {
      console.error('Failed to enrich cluster:', error);
    } finally {
      setEnriching(false);
    }
  };

  const needsEnrichment = !symptoms.length || !environment || environment.length < 20 || !recommendedResponse || recommendedResponse.length < 30;

  const getPriorityColor = (priority: string) => {
    const colors = {
      critical: "text-red-500",
      high: "text-orange-500",
      medium: "text-yellow-500",
      low: "text-green-500",
    };
    return colors[priority as keyof typeof colors] || colors.medium;
  };

  return (
    <div className="space-y-6">
      {/* Back Button */}
      <button
        onClick={onBack}
        className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to Clusters
      </button>

      {/* Header */}
      <div className="nvidia-build-card">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="nvidia-build-h3 mb-2">{cluster.label}</h2>
            <p className="text-muted-foreground">{cluster.issueDescription}</p>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-right">
              <div className="text-2xl font-bold">{cluster.size}</div>
              <div className="text-sm text-muted-foreground">Tickets</div>
            </div>
            <div className="text-right">
              <div className={`text-2xl font-bold ${cluster.confidence >= 0.8 ? "text-green-500" : "text-yellow-500"}`}>
                {Math.round(cluster.confidence * 100)}%
              </div>
              <div className="text-sm text-muted-foreground">Confidence</div>
            </div>
          </div>
        </div>

        {/* Keywords */}
        <div className="mt-6 flex flex-wrap gap-2">
          {cluster.keywords.map((keyword) => (
            <span
              key={keyword}
              className="nvidia-build-tag"
            >
              <Tag className="h-3 w-3 mr-1" />
              {keyword}
            </span>
          ))}
        </div>
      </div>

      {/* Enrich Button */}
      {needsEnrichment && (
        <div className="nvidia-build-card">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold mb-1">AI-Powered Enrichment</h3>
              <p className="text-sm text-muted-foreground">Generate missing details using AI</p>
            </div>
            <button
              onClick={enrichCluster}
              disabled={enriching}
              className="flex items-center gap-2 px-4 py-2 bg-[#76b900] hover:bg-[#76b900]/90 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Sparkles className="h-4 w-4" />
              {enriching ? "Generating..." : "Enrich with AI"}
            </button>
          </div>
        </div>
      )}

      {/* Details Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Environment */}
        <div className="nvidia-build-card">
          <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3">Environment</h3>
          <p className="text-foreground">{environment || cluster.environment || 'No environment data available'}</p>
        </div>

        {/* Priority & Date */}
        <div className="nvidia-build-card">
          <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3">Metadata</h3>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Priority</span>
              <span className={`font-medium capitalize ${getPriorityColor(cluster.priority)}`}>{cluster.priority}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Created</span>
              <span className="font-medium">{cluster.createdAt}</span>
            </div>
          </div>
        </div>

        {/* Recommended Response */}
        <div className="nvidia-build-card md:col-span-2">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
              <MessageSquare className="h-4 w-4" />
              Recommended Response
            </h3>
            {(recommendedResponse || cluster.recommendedResponse) && (
              <button
                onClick={copyResponse}
                className="flex items-center gap-1 text-sm text-[#76b900] hover:text-[#76b900]/80 transition-colors"
              >
                {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                {copied ? "Copied!" : "Copy"}
              </button>
            )}
          </div>
          <p className="text-foreground bg-muted/30 p-4 rounded-lg">
            {recommendedResponse || cluster.recommendedResponse || 'No recommended response available'}
          </p>
        </div>

        {/* Symptoms */}
        <div className="nvidia-build-card md:col-span-2">
          <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3 flex items-center gap-2">
            <AlertCircle className="h-4 w-4" />
            Common Symptoms
          </h3>
          {symptoms.length > 0 ? (
            <ul className="space-y-2">
              {symptoms.map((symptom, idx) => (
                <li key={idx} className="flex items-start gap-2">
                  <span className="text-[#76b900] mt-1">•</span>
                  <span className="text-foreground">{symptom}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-muted-foreground italic">No symptoms data available. Use "Enrich with AI" button above to generate.</p>
          )}
        </div>

        {/* Deflection Path */}
        <div className="nvidia-build-card">
          <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3 flex items-center gap-2">
            <ExternalLink className="h-4 w-4" />
            Deflection Path
          </h3>
          <a href={`/${cluster.deflectionPath}`} className="text-[#76b900] hover:underline">{cluster.deflectionPath}</a>
        </div>

        {/* Representative Tickets */}
        {cluster.representativeTickets && cluster.representativeTickets.length > 0 && (
          <div className="nvidia-build-card">
            <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3 flex items-center gap-2">
              <FileText className="h-4 w-4" />
              Representative Tickets
            </h3>
            <div className="space-y-2">
              {cluster.representativeTickets.map((ticket) => (
                <div key={ticket} className="flex items-center gap-2 text-sm">
                  <span className="font-mono bg-muted px-2 py-1 rounded">{ticket}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

