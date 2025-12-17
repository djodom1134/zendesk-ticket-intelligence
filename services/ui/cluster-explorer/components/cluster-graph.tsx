"use client"

import React, { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'

interface ClusterNode {
  id: string
  label: string
  size: number
  x?: number
  y?: number
  fx?: number | null
  fy?: number | null
}

interface ClusterLink {
  source: string | ClusterNode
  target: string | ClusterNode
  value: number
}

interface ClusterGraphProps {
  clusters: Array<{
    cluster_id: string
    label: string
    size: number
  }>
  onClusterClick?: (clusterId: string) => void
}

export function ClusterGraph({ clusters, onClusterClick }: ClusterGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 })

  useEffect(() => {
    if (!svgRef.current || !clusters || clusters.length === 0) return

    // Clear previous content
    d3.select(svgRef.current).selectAll('*').remove()

    const svg = d3.select(svgRef.current)
    const width = dimensions.width
    const height = dimensions.height

    // Create nodes from clusters
    const nodes: ClusterNode[] = clusters.map(c => ({
      id: c.cluster_id,
      label: c.label,
      size: c.size,
    }))

    // Create links (for now, connect clusters with similar sizes)
    const links: ClusterLink[] = []
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < Math.min(i + 4, nodes.length); j++) {
        links.push({
          source: nodes[i].id,
          target: nodes[j].id,
          value: 1,
        })
      }
    }

    // Create force simulation
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id((d: any) => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius((d: any) => Math.sqrt(d.size) * 2 + 10))

    // Create container group
    const g = svg.append('g')

    // Add zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform)
      })

    svg.call(zoom)

    // Draw links
    const link = g.append('g')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.3)
      .attr('stroke-width', 1)

    // Draw nodes
    const node = g.append('g')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .call(d3.drag<SVGGElement, ClusterNode>()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended) as any)

    // Add circles
    node.append('circle')
      .attr('r', (d) => Math.sqrt(d.size) * 2 + 5)
      .attr('fill', (d, i) => d3.schemeCategory10[i % 10])
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .on('click', (event, d) => {
        if (onClusterClick) {
          onClusterClick(d.id)
        }
      })

    // Add labels
    node.append('text')
      .text((d) => `${d.label} (${d.size})`)
      .attr('x', 0)
      .attr('y', (d) => Math.sqrt(d.size) * 2 + 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', '#333')
      .style('pointer-events', 'none')

    // Update positions on tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y)

      node.attr('transform', (d) => `translate(${d.x},${d.y})`)
    })

    // Drag functions
    function dragstarted(event: any, d: ClusterNode) {
      if (!event.active) simulation.alphaTarget(0.3).restart()
      d.fx = d.x
      d.fy = d.y
    }

    function dragged(event: any, d: ClusterNode) {
      d.fx = event.x
      d.fy = event.y
    }

    function dragended(event: any, d: ClusterNode) {
      if (!event.active) simulation.alphaTarget(0)
      d.fx = null
      d.fy = null
    }

    return () => {
      simulation.stop()
    }
  }, [clusters, dimensions, onClusterClick])

  return (
    <div className="w-full h-full">
      <svg
        ref={svgRef}
        width={dimensions.width}
        height={dimensions.height}
        className="border border-gray-200 rounded-lg"
      />
    </div>
  )
}

