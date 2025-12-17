"use client"

import { useState } from "react"
import { FallbackGraph } from "./fallback-graph"
import { ForceGraphWrapper } from "./force-graph-wrapper"
import { Button } from "@/components/ui/button"
import { CuboidIcon as Cube, LayoutGrid } from "lucide-react"
import type { Triple } from "@/utils/text-processing"

interface ClusterGraphViewProps {
  triples: Triple[]
  fullscreen?: boolean
  highlightedNodes?: string[]
  layoutType?: string
  initialMode?: '2d' | '3d'
}

export function ClusterGraphView({ 
  triples, 
  fullscreen = false,
  highlightedNodes = [],
  layoutType = "force",
  initialMode = '2d'
}: ClusterGraphViewProps) {
  const [use3D, setUse3D] = useState(initialMode === '3d')

  // Convert triples to graph data format
  const graphData = convertTriplesToGraphData(triples)

  return (
    <div className="relative w-full h-full">
      {/* 2D/3D Toggle */}
      <div className="absolute top-4 right-4 z-10 flex gap-2">
        <Button
          size="sm"
          variant={use3D ? "outline" : "default"}
          onClick={() => setUse3D(false)}
        >
          <LayoutGrid className="h-4 w-4 mr-2" />
          2D
        </Button>
        <Button
          size="sm"
          variant={use3D ? "default" : "outline"}
          onClick={() => setUse3D(true)}
        >
          <Cube className="h-4 w-4 mr-2" />
          3D
        </Button>
      </div>

      {/* Graph Display */}
      {use3D ? (
        <ForceGraphWrapper
          jsonData={graphData}
          fullscreen={fullscreen}
          layoutType={layoutType}
          highlightedNodes={highlightedNodes}
          enableClustering={false}
          enableClusterColors={true}
        />
      ) : (
        <FallbackGraph
          triples={triples}
          fullscreen={fullscreen}
          highlightedNodes={highlightedNodes}
          layoutType={layoutType}
        />
      )}
    </div>
  )
}

function convertTriplesToGraphData(triples: Triple[]) {
  const nodesMap = new Map<string, any>()
  const links: any[] = []

  // Extract unique nodes from triples
  triples.forEach(triple => {
    // Add subject node
    if (!nodesMap.has(triple.subject)) {
      nodesMap.set(triple.subject, {
        id: triple.subject,
        name: triple.subject,
        val: 10, // Default size
        color: getNodeColor(triple.subject)
      })
    }

    // Add object node if it's not a property value
    if (!triple.predicate.startsWith('has_')) {
      if (!nodesMap.has(triple.object)) {
        nodesMap.set(triple.object, {
          id: triple.object,
          name: triple.object,
          val: 10,
          color: getNodeColor(triple.object)
        })
      }

      // Add link
      links.push({
        source: triple.subject,
        target: triple.object,
        name: triple.predicate,
        color: '#666666'
      })
    } else {
      // Update node properties based on predicates
      const node = nodesMap.get(triple.subject)
      if (node) {
        if (triple.predicate === 'has_size') {
          const size = parseInt(triple.object.split(' ')[0])
          node.val = Math.max(5, Math.min(50, size / 2)) // Scale node size
        }
        if (triple.predicate === 'has_priority') {
          node.color = getPriorityColor(triple.object)
        }
      }
    }
  })

  return {
    nodes: Array.from(nodesMap.values()),
    links: links
  }
}

function getNodeColor(nodeName: string): string {
  // Default colors - will be overridden by priority if available
  const hash = nodeName.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0)
  const colors = ['#76b900', '#00a8e0', '#9370db', '#ff6b6b', '#4ecdc4']
  return colors[hash % colors.length]
}

function getPriorityColor(priority: string): string {
  switch (priority.toLowerCase()) {
    case 'high':
    case 'critical':
      return '#ff4444'
    case 'medium':
      return '#76b900'
    case 'low':
      return '#00a8e0'
    default:
      return '#9370db'
  }
}

