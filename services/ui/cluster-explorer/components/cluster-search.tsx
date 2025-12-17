"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Search, Loader2, X } from "lucide-react"
import { Cluster } from "@/lib/types"

interface ClusterSearchProps {
  clusters: Cluster[]
  onSelectCluster: (cluster: Cluster) => void
}

export function ClusterSearch({ clusters, onSelectCluster }: ClusterSearchProps) {
  const [query, setQuery] = useState("")
  const [results, setResults] = useState<Cluster[]>([])
  const [isSearching, setIsSearching] = useState(false)

  const handleSearch = async () => {
    if (!query.trim()) {
      setResults([])
      return
    }

    setIsSearching(true)
    
    // Simple client-side search for now
    // TODO: Replace with backend RAG search
    const searchResults = clusters.filter(cluster => 
      cluster.label.toLowerCase().includes(query.toLowerCase()) ||
      cluster.issueDescription?.toLowerCase().includes(query.toLowerCase()) ||
      cluster.keywords?.some(k => k.toLowerCase().includes(query.toLowerCase()))
    )

    setResults(searchResults)
    setIsSearching(false)
  }

  const handleClear = () => {
    setQuery("")
    setResults([])
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch()
    }
  }

  return (
    <div className="space-y-4">
      {/* Search Input */}
      <div className="flex gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            type="text"
            placeholder="Search clusters by label, description, or keywords..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            className="pl-10"
          />
        </div>
        <Button onClick={handleSearch} disabled={isSearching}>
          {isSearching ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Searching
            </>
          ) : (
            "Search"
          )}
        </Button>
        {query && (
          <Button variant="outline" onClick={handleClear}>
            <X className="h-4 w-4" />
          </Button>
        )}
      </div>

      {/* Search Results */}
      {results.length > 0 && (
        <div className="space-y-2">
          <p className="text-sm text-muted-foreground">
            Found {results.length} cluster{results.length !== 1 ? 's' : ''}
          </p>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {results.map((cluster) => (
              <Card
                key={cluster.id}
                className="cursor-pointer hover:bg-accent transition-colors"
                onClick={() => onSelectCluster(cluster)}
              >
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">
                    {cluster.label}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-xs text-muted-foreground line-clamp-2">
                    {cluster.issueDescription || "No description available"}
                  </p>
                  <div className="flex gap-1 mt-2 flex-wrap">
                    {cluster.keywords?.slice(0, 3).map((keyword, idx) => (
                      <span
                        key={idx}
                        className="text-xs px-2 py-0.5 bg-secondary rounded-full"
                      >
                        {keyword}
                      </span>
                    ))}
                  </div>
                  <div className="flex justify-between items-center mt-2 text-xs text-muted-foreground">
                    <span>{cluster.size} tickets</span>
                    <span className="capitalize">{cluster.priority} priority</span>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}

      {query && results.length === 0 && !isSearching && (
        <p className="text-sm text-muted-foreground text-center py-8">
          No clusters found matching "{query}"
        </p>
      )}
    </div>
  )
}

