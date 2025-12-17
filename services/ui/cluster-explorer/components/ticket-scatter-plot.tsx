"use client"

import { useState, useEffect } from "react"
import { ForceGraphWrapper } from "./force-graph-wrapper"
import { FallbackGraph } from "./fallback-graph"
import { Button } from "@/components/ui/button"
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
  xDim = 0,
  yDim = 1,
  zDim = 2
}: TicketScatterPlotProps) {
  const [use3D, setUse3D] = useState(initialMode === '3d')
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
      {/* 2D/3D Toggle */}
      <div className="absolute top-4 right-4 z-10 flex gap-2">
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

      {/* Graph Display */}
      {use3D ? (
        <ForceGraphWrapper
          jsonData={graphData}
          fullscreen={false}
          layoutType="force"
          highlightedNodes={[]}
          enableClustering={false}
          enableClusterColors={true}
        />
      ) : (
        <div className="w-full h-full">
          {/* TODO: Create 2D scatter plot component */}
          <div className="flex items-center justify-center h-full">
            2D scatter plot coming soon - use 3D view for now
          </div>
        </div>
      )}
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
    val: 2, // Small dots
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

