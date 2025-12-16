"use client";

import { useState } from "react";
import { Send, Bot, User, Loader2, AlertCircle, CheckCircle, FileText, Tag } from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Citation {
  type: string;
  id: string;
  label: string;
  relevance: number;
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  predictedCluster?: string;
  clusterLabel?: string;
  confidence?: number;
  recommendedActions?: string[];
  draftReply?: string;
  citations?: Citation[];
  askForInfo?: string[];
  timestamp: Date;
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  return (
    <div className={`flex gap-3 ${isUser ? "justify-end" : "justify-start"}`}>
      {!isUser && (
        <div className="h-8 w-8 rounded-full bg-[#76b900] flex items-center justify-center flex-shrink-0">
          <Bot className="h-5 w-5 text-white" />
        </div>
      )}
      <div className={`max-w-[80%] ${isUser ? "bg-primary text-primary-foreground" : "bg-muted"} rounded-lg p-4`}>
        {/* Main content */}
        <p className="whitespace-pre-wrap text-sm">{message.content}</p>

        {/* Cluster prediction */}
        {message.clusterLabel && (
          <div className="mt-3 p-2 bg-background/50 rounded border border-border">
            <div className="flex items-center gap-2 text-xs font-medium">
              <Tag className="h-3 w-3" />
              <span>Predicted Cluster: {message.clusterLabel}</span>
              {message.confidence !== undefined && (
                <span className="ml-auto text-muted-foreground">
                  {(message.confidence * 100).toFixed(0)}% confidence
                </span>
              )}
            </div>
          </div>
        )}

        {/* Draft reply */}
        {message.draftReply && (
          <div className="mt-3 p-3 bg-green-500/10 border border-green-500/30 rounded">
            <div className="flex items-center gap-2 text-xs font-medium text-green-600 dark:text-green-400 mb-2">
              <CheckCircle className="h-3 w-3" />
              <span>Draft Reply</span>
            </div>
            <p className="text-sm whitespace-pre-wrap">{message.draftReply}</p>
          </div>
        )}

        {/* Recommended actions */}
        {message.recommendedActions && message.recommendedActions.length > 0 && (
          <div className="mt-3">
            <p className="text-xs font-medium mb-1">Recommended Actions:</p>
            <ul className="text-xs space-y-1">
              {message.recommendedActions.map((action, i) => (
                <li key={i} className="flex items-start gap-1">
                  <span className="text-[#76b900]">•</span>
                  <span>{action}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Ask for info */}
        {message.askForInfo && message.askForInfo.length > 0 && (
          <div className="mt-3 p-2 bg-yellow-500/10 border border-yellow-500/30 rounded">
            <p className="text-xs font-medium text-yellow-600 dark:text-yellow-400 mb-1">
              Questions to ask customer:
            </p>
            <ul className="text-xs space-y-1">
              {message.askForInfo.map((q, i) => (
                <li key={i}>• {q}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Citations */}
        {message.citations && message.citations.length > 0 && (
          <div className="mt-3 border-t border-border pt-2">
            <p className="text-xs font-medium mb-1 flex items-center gap-1">
              <FileText className="h-3 w-3" />
              Sources
            </p>
            <div className="flex flex-wrap gap-1">
              {message.citations.slice(0, 5).map((cite, i) => (
                <span
                  key={i}
                  className="text-xs px-2 py-0.5 bg-background rounded border border-border"
                  title={`${cite.type}: ${cite.id} (${(cite.relevance * 100).toFixed(0)}% relevant)`}
                >
                  {cite.label.slice(0, 30)}...
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
      {isUser && (
        <div className="h-8 w-8 rounded-full bg-muted flex items-center justify-center flex-shrink-0">
          <User className="h-5 w-5" />
        </div>
      )}
    </div>
  );
}

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage: ChatMessage = {
      role: "user",
      content: input,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: input, include_citations: true }),
      });

      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const data = await res.json();

      const assistantMessage: ChatMessage = {
        role: "assistant",
        content: data.response,
        predictedCluster: data.predicted_cluster,
        clusterLabel: data.cluster_label,
        confidence: data.confidence,
        recommendedActions: data.recommended_actions,
        draftReply: data.draft_reply,
        citations: data.citations,
        askForInfo: data.ask_for_info,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to send message");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="container mx-auto px-4 py-6 max-w-4xl">
      <div className="mb-6">
        <h1 className="text-3xl font-bold gradient-text mb-2">Tier-0 Support Assistant</h1>
        <p className="text-muted-foreground">
          Paste a ticket description to get classification, similar cases, and a draft response.
        </p>
      </div>

      {/* Chat Messages */}
      <div className="space-y-4 mb-4 min-h-[400px] max-h-[600px] overflow-y-auto">
        {messages.length === 0 && (
          <div className="text-center text-muted-foreground py-20">
            <Bot className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>Paste a support ticket or describe an issue to get started.</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <MessageBubble key={i} message={msg} />
        ))}
        {loading && (
          <div className="flex items-center gap-2 text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span>Analyzing ticket...</span>
          </div>
        )}
      </div>

      {error && (
        <div className="mb-4 p-3 bg-destructive/10 border border-destructive/30 rounded-lg flex items-center gap-2 text-destructive">
          <AlertCircle className="h-4 w-4" />
          <span>{error}</span>
        </div>
      )}

      {/* Input */}
      <div className="flex gap-2">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && (e.preventDefault(), sendMessage())}
          placeholder="Paste ticket content or describe the issue..."
          className="flex-1 min-h-[100px] p-3 rounded-lg border border-border bg-background resize-none focus:outline-none focus:ring-2 focus:ring-[#76b900]/50"
          disabled={loading}
        />
        <button
          onClick={sendMessage}
          disabled={loading || !input.trim()}
          className="px-4 py-2 bg-[#76b900] text-white rounded-lg hover:bg-[#76b900]/90 disabled:opacity-50 disabled:cursor-not-allowed self-end"
        >
          <Send className="h-5 w-5" />
        </button>
      </div>
    </main>
  );
}

