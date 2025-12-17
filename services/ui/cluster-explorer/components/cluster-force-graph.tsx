"use client"

import React, { useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Maximize2, Minimize2, Pause, Play, RefreshCw, ZoomIn, ZoomOut } from "lucide-react"

// Dynamic import for 3d-force-graph to avoid SSR issues
let ForceGraph3D: any = null
if (typeof window !== 'undefined') {
  import('3d-force-graph').then((mod) => {
    ForceGraph3D = mod.default
  })
}

interface ClusterNode {
  id: string
  label: string
  size: number
  group?: string
  color?: string
}

interface ClusterLink {
  source: string
  target: string
  value?: number
}

interface ClusterForceGraphProps {
  clusters: ClusterNode[]
  onClusterClick?: (clusterId: string) => void
}

export function ClusterForceGraph({ clusters, onClusterClick }: ClusterForceGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const graphRef = useRef<any>(null)
  const [is3D, setIs3D] = useState(true)
  const [isPaused, setIsPaused] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(false)

  useEffect(() => {
    if (!containerRef.current || !ForceGraph3D || clusters.length === 0) return

    // Convert clusters to graph data
    const nodes = clusters.map(c => ({
      id: c.id,
      name: c.label,
      val: c.size,
      color: getClusterColor(c.size),
      group: c.group || 'default'
    }))

    // Create links based on cluster similarity (simplified - connect nearby clusters)
    const links: ClusterLink[] = []
    for (let i = 0; i < Math.min(clusters.length, 20); i++) {
      const targetIdx = (i + 1) % clusters.length
      links.push({
        source: clusters[i].id,
        target: clusters[targetIdx].id,
        value: 1
      })
    }

    const graphData = { nodes, links }

    // Initialize 3D force graph
    if (!graphRef.current) {
      graphRef.current = ForceGraph3D()(containerRef.current)
        .graphData(graphData)
        .nodeLabel('name')
        .nodeVal('val')
        .nodeColor('color')
        .nodeAutoColorBy('group')
        .linkDirectionalParticles(2)
        .linkDirectionalParticleWidth(2)
        .onNodeClick((node: any) => {
          if (onClusterClick) {
            onClusterClick(node.id)
          }
        })
        .backgroundColor('#0a0a0a')
        .width(containerRef.current.clientWidth)
        .height(containerRef.current.clientHeight)
    } else {
      graphRef.current.graphData(graphData)
    }

    return () => {
      if (graphRef.current) {
        graphRef.current._destructor()
        graphRef.current = null
      }
    }
  }, [clusters, onClusterClick])

  const getClusterColor = (size: number) => {
    if (size > 100) return '#76b900' // NVIDIA green for large clusters
    if (size > 50) return '#00a8e0' // Blue for medium
    return '#9370db' // Purple for small
  }

  const handleZoomIn = () => {
    if (graphRef.current) {
      const camera = graphRef.current.camera()
      camera.position.multiplyScalar(0.8)
    }
  }

  const handleZoomOut = () => {
    if (graphRef.current) {
      const camera = graphRef.current.camera()
      camera.position.multiplyScalar(1.2)
    }
  }

  const handleReset = () => {
    if (graphRef.current) {
      graphRef.current.zoomToFit(400)
    }
  }

  const togglePause = () => {
    if (graphRef.current) {
      if (isPaused) {
        graphRef.current.resumeAnimation()
      } else {
        graphRef.current.pauseAnimation()
      }
      setIsPaused(!isPaused)
    }
  }

  return (
    <div className="relative w-full h-full">
      {/* Toolbar */}
      <div className="absolute top-4 right-4 z-10 flex gap-2">
        <Button size="sm" variant="secondary" onClick={handleZoomIn}>
          <ZoomIn className="h-4 w-4" />
        </Button>
        <Button size="sm" variant="secondary" onClick={handleZoomOut}>
          <ZoomOut className="h-4 w-4" />
        </Button>
        <Button size="sm" variant="secondary" onClick={handleReset}>
          <RefreshCw className="h-4 w-4" />
        </Button>
        <Button size="sm" variant="secondary" onClick={togglePause}>
          {isPaused ? <Play className="h-4 w-4" /> : <Pause className="h-4 w-4" />}
        </Button>
      </div>

      {/* Graph container */}
      <div ref={containerRef} className="w-full h-full" />
    </div>
  )
}

