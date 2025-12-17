"use client"

import { useState, useEffect } from "react"
import { ForceGraphWrapper } from "./force-graph-wrapper"
import { FallbackGraph } from "./fallback-graph"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Loader2, Link as LinkIcon, Box, Square } from "lucide-react"
import type { TicketPosition } from "@/lib/types"

interface TicketScatterPlotProps {
  apiUrl: string
  initialMode?: '2d' | '3d'
  xDim?: number
  yDim?: number
  zDim?: number
}

export function TicketScatterPlot({
  apiUrl,
  initialMode = '2d',
  xDim: initialXDim = 0,
  yDim: initialYDim = 1,
  zDim: initialZDim = 2
}: TicketScatterPlotProps) {
  const [xDim, setXDim] = useState(initialXDim)
  const [yDim, setYDim] = useState(initialYDim)
  const [zDim, setZDim] = useState(initialZDim)
  const [showLinks, setShowLinks] = useState(false)
  const [nodeSize, setNodeSize] = useState(0.005) // Default: 0.005
  const [viewMode, setViewMode] = useState<'2d' | '3d'>('3d') // 2D or 3D view
  const [tickets, setTickets] = useState<TicketPosition[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchTickets = async () => {
      try {
        setLoading(true)
        setError(null)

        const params = new URLSearchParams({
          x_dim: xDim.toString(),
          y_dim: yDim.toString(),
          z_dim: zDim.toString(),
          use_3d: 'true', // Always use 3D
          limit: '2000'
        })

        const response = await fetch(`${apiUrl}/api/tickets/positions?${params}`)
        if (!response.ok) throw new Error('Failed to fetch ticket positions')

        const data: TicketPosition[] = await response.json()
        setTickets(data)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error')
      } finally {
        setLoading(false)
      }
    }

    fetchTickets()
  }, [apiUrl, xDim, yDim, zDim])

  // Convert tickets to graph data format
  const graphData = convertTicketsToGraphData(tickets, showLinks, nodeSize)

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="h-8 w-8 animate-spin text-[#76b900]" />
        <span className="ml-2">Loading {tickets.length > 0 ? tickets.length : ''} tickets...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full text-red-500">
        Error: {error}
      </div>
    )
  }

  return (
    <div className="relative w-full h-full">
      {/* Controls */}
      <div className="absolute top-4 left-4 right-4 z-10 flex flex-col gap-2">
        {/* Top row: Dimension selectors and controls */}
        <div className="flex items-center justify-between gap-4">
          {/* Left side: Dimension selectors */}
          <div className="flex items-center gap-2 bg-background/80 backdrop-blur-sm px-3 py-2 rounded-md">
            <span className="text-sm font-medium">Dimensions:</span>

          {/* X Dimension */}
          <div className="flex items-center gap-1">
            <span className="text-xs text-muted-foreground">X:</span>
            <Select value={xDim.toString()} onValueChange={(v) => setXDim(parseInt(v))}>
              <SelectTrigger className="h-7 w-16 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(i => (
                  <SelectItem key={i} value={i.toString()}>C{i}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Y Dimension */}
          <div className="flex items-center gap-1">
            <span className="text-xs text-muted-foreground">Y:</span>
            <Select value={yDim.toString()} onValueChange={(v) => setYDim(parseInt(v))}>
              <SelectTrigger className="h-7 w-16 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(i => (
                  <SelectItem key={i} value={i.toString()}>C{i}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Z Dimension */}
          <div className="flex items-center gap-1">
            <span className="text-xs text-muted-foreground">Z:</span>
            <Select value={zDim.toString()} onValueChange={(v) => setZDim(parseInt(v))}>
              <SelectTrigger className="h-7 w-16 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(i => (
                  <SelectItem key={i} value={i.toString()}>C{i}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

          {/* Right side: Links toggle and ticket count */}
          <div className="flex items-center gap-2">
            <div className="bg-background/80 backdrop-blur-sm px-3 py-1 rounded-md text-sm">
              {tickets.length} tickets
            </div>
            <Button
              size="sm"
              variant={showLinks ? "default" : "outline"}
              onClick={() => setShowLinks(!showLinks)}
              title="Toggle cluster connections"
            >
              <LinkIcon className="h-4 w-4 mr-2" />
              Links
            </Button>
          </div>
        </div>

        {/* Bottom row: Node size slider and 2D/3D toggle */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-3 bg-background/80 backdrop-blur-sm px-3 py-2 rounded-md">
            <span className="text-sm font-medium whitespace-nowrap">Node Size:</span>
            <Slider
              value={[nodeSize * 1000]} // Scale to 0-10 range for slider
              onValueChange={(values) => setNodeSize(values[0] / 1000)}
              min={0.1}
              max={10}
              step={0.1}
              className="w-32"
            />
            <span className="text-xs text-muted-foreground w-12 text-right">
              {(nodeSize * 1000).toFixed(1)}
            </span>
          </div>

          {/* 2D/3D Toggle */}
          <div className="flex items-center gap-2 bg-background/80 backdrop-blur-sm px-3 py-2 rounded-md">
            <span className="text-sm font-medium whitespace-nowrap">View:</span>
            <Button
              size="sm"
              variant={viewMode === '2d' ? "default" : "outline"}
              onClick={() => setViewMode('2d')}
              className="h-7"
            >
              <Square className="h-3 w-3 mr-1" />
              2D
            </Button>
            <Button
              size="sm"
              variant={viewMode === '3d' ? "default" : "outline"}
              onClick={() => setViewMode('3d')}
              className="h-7"
            >
              <Box className="h-3 w-3 mr-1" />
              3D
            </Button>
          </div>
        </div>
      </div>

      {/* Graph Display */}
      {loading ? (
        <div className="flex items-center justify-center h-full">
          <Loader2 className="h-8 w-8 animate-spin text-[#76b900]" />
          <span className="ml-2">Loading tickets...</span>
        </div>
      ) : tickets.length > 0 ? (
        <ForceGraphWrapper
          key="ticket-scatter-graph" // Stable key to prevent remounting
          jsonData={graphData}
          fullscreen={false}
          layoutType="force"
          highlightedNodes={[]}
          enableClustering={false}
          enableClusterColors={true}
          viewMode={viewMode}
        />
      ) : (
        <div className="flex items-center justify-center h-full text-muted-foreground">
          No tickets to display
        </div>
      )}
    </div>
  )
}

function convertTicketsToGraphData(tickets: TicketPosition[], showLinks: boolean = false, nodeSize: number = 0.005) {
  // Create a color map for clusters
  const clusterColors = new Map<string, string>()

  // Base color palette (10 distinct colors)
  const baseColors = [
    '#76b900', // NVIDIA green
    '#00a8e0', // Blue
    '#9370db', // Purple
    '#ff6b6b', // Red
    '#4ecdc4', // Teal
    '#f9ca24', // Yellow
    '#6c5ce7', // Indigo
    '#fd79a8', // Pink
    '#00b894', // Emerald
    '#fdcb6e', // Orange
  ]

  // Generate 63 colors by interpolating between base colors
  const generateColorGradient = (numColors: number): string[] => {
    const colors: string[] = []
    const segmentSize = numColors / baseColors.length

    for (let i = 0; i < numColors; i++) {
      const segmentIndex = Math.floor(i / segmentSize)
      const nextSegmentIndex = (segmentIndex + 1) % baseColors.length
      const t = (i % segmentSize) / segmentSize

      const color1 = baseColors[segmentIndex]
      const color2 = baseColors[nextSegmentIndex]

      // Interpolate between color1 and color2
      const r1 = parseInt(color1.slice(1, 3), 16)
      const g1 = parseInt(color1.slice(3, 5), 16)
      const b1 = parseInt(color1.slice(5, 7), 16)

      const r2 = parseInt(color2.slice(1, 3), 16)
      const g2 = parseInt(color2.slice(3, 5), 16)
      const b2 = parseInt(color2.slice(5, 7), 16)

      const r = Math.round(r1 + (r2 - r1) * t)
      const g = Math.round(g1 + (g2 - g1) * t)
      const b = Math.round(b1 + (b2 - b1) * t)

      colors.push(`#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`)
    }

    return colors
  }

  // Assign colors to clusters
  const uniqueClusters = [...new Set(tickets.map(t => t.cluster_id).filter(Boolean))]
  const colorGradient = generateColorGradient(Math.max(uniqueClusters.length, 63))

  uniqueClusters.forEach((clusterId, idx) => {
    clusterColors.set(clusterId!, colorGradient[idx])
  })

  // Convert tickets to nodes
  const nodes = tickets.map(ticket => ({
    id: ticket.ticket_id,
    name: ticket.subject || ticket.ticket_id,
    val: nodeSize, // User-adjustable node size
    color: ticket.cluster_id ? clusterColors.get(ticket.cluster_id) : '#666666', // Color by cluster assignment
    x: ticket.x,
    y: ticket.y,
    z: ticket.z,
    fx: ticket.x, // Fix position
    fy: ticket.y,
    fz: ticket.z,
    cluster: ticket.cluster_label || 'Unclustered',
    cluster_id: ticket.cluster_id
  }))

  // Create links between tickets in the same cluster (if enabled)
  let links: any[] = []

  if (showLinks) {
    const maxLinksPerNode = 3 // Limit links to avoid clutter

    // Group tickets by cluster
    const clusterGroups = new Map<string, typeof nodes>()
    nodes.forEach(node => {
      if (node.cluster_id) {
        if (!clusterGroups.has(node.cluster_id)) {
          clusterGroups.set(node.cluster_id, [])
        }
        clusterGroups.get(node.cluster_id)!.push(node)
      }
    })

    // For each cluster, create links between nearby tickets
    clusterGroups.forEach((clusterNodes, clusterId) => {
      clusterNodes.forEach((node, idx) => {
        // Connect to next few nodes in the same cluster (circular)
        for (let i = 1; i <= maxLinksPerNode && i < clusterNodes.length; i++) {
          const targetIdx = (idx + i) % clusterNodes.length
          const target = clusterNodes[targetIdx]

          links.push({
            source: node.id,
            target: target.id,
            color: clusterColors.get(clusterId) || '#666666',
            name: `Same cluster: ${node.cluster}`
          })
        }
      })
    })
  }

  return {
    nodes,
    links
  }
}

