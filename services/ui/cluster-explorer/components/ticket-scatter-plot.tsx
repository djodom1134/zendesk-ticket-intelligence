"use client"

import { useState, useEffect } from "react"
import { ForceGraphWrapper } from "./force-graph-wrapper"
import { FallbackGraph } from "./fallback-graph"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { CuboidIcon as Cube, LayoutGrid, Loader2 } from "lucide-react"
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
  const [use3D, setUse3D] = useState(initialMode === '3d')
  const [xDim, setXDim] = useState(initialXDim)
  const [yDim, setYDim] = useState(initialYDim)
  const [zDim, setZDim] = useState(initialZDim)
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
          use_3d: use3D.toString(),
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
  }, [apiUrl, xDim, yDim, zDim, use3D])

  // Convert tickets to graph data format
  const graphData = convertTicketsToGraphData(tickets)

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
      <div className="absolute top-4 left-4 right-4 z-10 flex items-center justify-between gap-4">
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

          {/* Z Dimension (only in 3D mode) */}
          {use3D && (
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
          )}
        </div>

        {/* Right side: 2D/3D toggle and ticket count */}
        <div className="flex items-center gap-2">
          <div className="bg-background/80 backdrop-blur-sm px-3 py-1 rounded-md text-sm">
            {tickets.length} tickets
          </div>
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
      </div>

      {/* Graph Display */}
      <ForceGraphWrapper
        jsonData={graphData}
        fullscreen={false}
        layoutType="force"
        highlightedNodes={[]}
        enableClustering={false}
        enableClusterColors={true}
        initialMode={use3D ? '3d' : '2d'}
      />
    </div>
  )
}

function convertTicketsToGraphData(tickets: TicketPosition[]) {
  // Create a color map for clusters
  const clusterColors = new Map<string, string>()
  const colorPalette = [
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

  // Assign colors to clusters
  const uniqueClusters = [...new Set(tickets.map(t => t.cluster_id).filter(Boolean))]
  uniqueClusters.forEach((clusterId, idx) => {
    clusterColors.set(clusterId!, colorPalette[idx % colorPalette.length])
  })

  // Convert tickets to nodes
  const nodes = tickets.map(ticket => ({
    id: ticket.ticket_id,
    name: ticket.subject || ticket.ticket_id,
    val: 0.5, // Very small dots for scatter plot
    color: ticket.cluster_id ? clusterColors.get(ticket.cluster_id) : '#666666',
    x: ticket.x,
    y: ticket.y,
    z: ticket.z,
    fx: ticket.x, // Fix position
    fy: ticket.y,
    fz: ticket.z,
    cluster: ticket.cluster_label || 'Unclustered'
  }))

  return {
    nodes,
    links: [] // No links for scatter plot
  }
}

