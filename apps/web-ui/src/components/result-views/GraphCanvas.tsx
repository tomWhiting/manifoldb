import { useRef, useEffect, useCallback, useState } from 'react'
import { useAppStore } from '../../stores/app-store'
import { useInspectorStore } from '../../stores/inspector-store'
import { useForceLayout } from '../../hooks/useForceLayout'
import {
  type LayoutNode,
  type LayoutEdge,
  type ViewTransform,
  labelToColor,
  labelToLightColor,
  getNodeRadius,
  screenToWorld,
  worldToScreen,
  findNodeAtPosition,
  findEdgeAtPosition,
} from '../../utils/graph-layout'
import { RotateCcw, ZoomIn, ZoomOut, Maximize2 } from 'lucide-react'
import { InspectorContainer, ContextMenu, type ContextMenuTarget } from '../inspector'

interface InteractionState {
  mode: 'idle' | 'panning' | 'dragging'
  startX: number
  startY: number
  draggedNodeId: string | null
}

import type { QueryResult } from '../../types'

interface GraphCanvasProps {
  result?: QueryResult
}

export function GraphCanvas({ result: propResult }: GraphCanvasProps = {}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 })
  const [selectedNodes, setSelectedNodes] = useState<Set<string>>(new Set())
  const [hoveredNode, setHoveredNode] = useState<LayoutNode | null>(null)
  const [hoveredEdge, setHoveredEdge] = useState<LayoutEdge | null>(null)
  const [contextMenu, setContextMenu] = useState<{
    target: ContextMenuTarget
    position: { x: number; y: number }
  } | null>(null)
  const interactionRef = useRef<InteractionState>({
    mode: 'idle',
    startX: 0,
    startY: 0,
    draggedNodeId: null,
  })

  // Inspector store
  const openNodeInspector = useInspectorStore((s) => s.openNodeInspector)
  const openEdgeInspector = useInspectorStore((s) => s.openEdgeInspector)

  const tabs = useAppStore((s) => s.tabs)
  const activeTabId = useAppStore((s) => s.activeTabId)
  const activeTab = tabs.find((t) => t.id === activeTabId)
  const result = propResult ?? activeTab?.result

  const {
    layout,
    transform,
    isSimulating,
    resetLayout,
    setTransform,
    pan,
    zoom,
    fixNode,
    releaseNode,
    reheat,
  } = useForceLayout({
    nodes: result?.nodes || [],
    edges: result?.edges || [],
    width: dimensions.width,
    height: dimensions.height,
    preservePositions: true,
  })

  // Handle resize
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const resizeObserver = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect
      setDimensions({ width, height })
    })
    resizeObserver.observe(container)

    return () => resizeObserver.disconnect()
  }, [])

  // Set up canvas for high-DPI
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || dimensions.width === 0) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = dimensions.width * dpr
    canvas.height = dimensions.height * dpr
    canvas.style.width = `${dimensions.width}px`
    canvas.style.height = `${dimensions.height}px`
  }, [dimensions])

  // Render function
  const render = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const width = dimensions.width
    const height = dimensions.height

    // Reset transform and clear
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.fillStyle = '#0a0a0a'
    ctx.fillRect(0, 0, width, height)

    // Draw grid
    drawGrid(ctx, width, height, transform)

    if (!layout || layout.nodes.size === 0) {
      // No data message
      ctx.fillStyle = '#525252'
      ctx.font = '14px system-ui'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText('Run a query to visualize results', width / 2, height / 2)
      return
    }

    // Apply view transform
    ctx.save()
    ctx.translate(transform.x, transform.y)
    ctx.scale(transform.scale, transform.scale)

    // Draw edges first (below nodes)
    for (const edge of layout.edges) {
      drawEdge(ctx, edge, layout, transform, selectedNodes, hoveredEdge)
    }

    // Draw nodes
    for (const node of layout.nodes.values()) {
      drawNode(ctx, node, selectedNodes.has(node.id), hoveredNode?.id === node.id)
    }

    ctx.restore()

    // Draw edge label on hover (in screen space)
    if (hoveredEdge) {
      drawEdgeLabel(ctx, hoveredEdge, layout, transform)
    }
  }, [layout, transform, selectedNodes, hoveredNode, hoveredEdge, dimensions])

  // Render on changes
  useEffect(() => {
    render()
  }, [render])

  // Mouse event handlers
  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!layout) return

      const rect = canvasRef.current?.getBoundingClientRect()
      if (!rect) return

      const x = e.clientX - rect.left
      const y = e.clientY - rect.top

      const node = findNodeAtPosition(x, y, layout, transform)

      if (node) {
        // Start dragging node
        interactionRef.current = {
          mode: 'dragging',
          startX: x,
          startY: y,
          draggedNodeId: node.id,
        }
        const worldPos = screenToWorld(x, y, transform)
        fixNode(node.id, worldPos.x, worldPos.y)

        // Handle selection
        if (e.shiftKey) {
          // Multi-select toggle
          setSelectedNodes((prev) => {
            const next = new Set(prev)
            if (next.has(node.id)) {
              next.delete(node.id)
            } else {
              next.add(node.id)
            }
            return next
          })
        } else if (!selectedNodes.has(node.id)) {
          // Single select
          setSelectedNodes(new Set([node.id]))
        }
      } else {
        // Start panning
        interactionRef.current = {
          mode: 'panning',
          startX: x,
          startY: y,
          draggedNodeId: null,
        }

        // Clear selection on canvas click (unless shift held)
        if (!e.shiftKey) {
          setSelectedNodes(new Set())
        }
      }
    },
    [layout, transform, selectedNodes, fixNode]
  )

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!layout) return

      const rect = canvasRef.current?.getBoundingClientRect()
      if (!rect) return

      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      const interaction = interactionRef.current

      if (interaction.mode === 'panning') {
        const dx = x - interaction.startX
        const dy = y - interaction.startY
        pan(dx, dy)
        interaction.startX = x
        interaction.startY = y
      } else if (interaction.mode === 'dragging' && interaction.draggedNodeId) {
        const worldPos = screenToWorld(x, y, transform)
        fixNode(interaction.draggedNodeId, worldPos.x, worldPos.y)
        reheat()
      } else {
        // Update hover state
        const node = findNodeAtPosition(x, y, layout, transform)
        setHoveredNode(node)

        if (!node) {
          const edge = findEdgeAtPosition(x, y, layout, transform)
          setHoveredEdge(edge)
        } else {
          setHoveredEdge(null)
        }

        // Update cursor
        if (canvasRef.current) {
          canvasRef.current.style.cursor = node ? 'grab' : 'default'
        }
      }
    },
    [layout, transform, pan, fixNode, reheat]
  )

  const handleMouseUp = useCallback(() => {
    const interaction = interactionRef.current

    if (interaction.mode === 'dragging' && interaction.draggedNodeId) {
      releaseNode(interaction.draggedNodeId)
      reheat()
    }

    interactionRef.current = {
      mode: 'idle',
      startX: 0,
      startY: 0,
      draggedNodeId: null,
    }

    if (canvasRef.current) {
      canvasRef.current.style.cursor = hoveredNode ? 'grab' : 'default'
    }
  }, [releaseNode, reheat, hoveredNode])

  const handleMouseLeave = useCallback(() => {
    handleMouseUp()
    setHoveredNode(null)
    setHoveredEdge(null)
  }, [handleMouseUp])

  const handleWheel = useCallback(
    (e: React.WheelEvent<HTMLCanvasElement>) => {
      e.preventDefault()

      const rect = canvasRef.current?.getBoundingClientRect()
      if (!rect) return

      const x = e.clientX - rect.left
      const y = e.clientY - rect.top

      // Zoom in/out based on wheel direction
      const factor = e.deltaY < 0 ? 1.1 : 0.9
      zoom(factor, x, y)
    },
    [zoom]
  )

  const handleDoubleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!layout) return

      const rect = canvasRef.current?.getBoundingClientRect()
      if (!rect) return

      const x = e.clientX - rect.left
      const y = e.clientY - rect.top

      const node = findNodeAtPosition(x, y, layout, transform)

      if (node) {
        // Open node inspector on double-click
        const screenPos = worldToScreen(node.x, node.y, transform)
        openNodeInspector(node, { x: screenPos.x + 50, y: screenPos.y })
        setSelectedNodes(new Set([node.id]))
      } else {
        // Check for edge double-click
        const edge = findEdgeAtPosition(x, y, layout, transform)
        if (edge) {
          const sourceNode = layout.nodes.get(edge.source)
          const targetNode = layout.nodes.get(edge.target)
          if (sourceNode && targetNode) {
            openEdgeInspector(edge, sourceNode, targetNode, { x: e.clientX, y: e.clientY })
          }
        }
      }
    },
    [layout, transform, openNodeInspector, openEdgeInspector]
  )

  // Handle right-click context menu
  const handleContextMenu = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      e.preventDefault()
      if (!layout) return

      const rect = canvasRef.current?.getBoundingClientRect()
      if (!rect) return

      const x = e.clientX - rect.left
      const y = e.clientY - rect.top

      // Check for node first
      const node = findNodeAtPosition(x, y, layout, transform)
      if (node) {
        setContextMenu({
          target: { type: 'node', node },
          position: { x: e.clientX, y: e.clientY },
        })
        return
      }

      // Check for edge
      const edge = findEdgeAtPosition(x, y, layout, transform)
      if (edge) {
        const sourceNode = layout.nodes.get(edge.source)
        const targetNode = layout.nodes.get(edge.target)
        if (sourceNode && targetNode) {
          setContextMenu({
            target: { type: 'edge', edge, sourceNode, targetNode },
            position: { x: e.clientX, y: e.clientY },
          })
        }
        return
      }

      // Close context menu if clicked on empty canvas
      setContextMenu(null)
    },
    [layout, transform]
  )

  // Handler for inspecting from context menu
  const handleInspect = useCallback(() => {
    if (!contextMenu?.target) return

    if (contextMenu.target.type === 'node') {
      openNodeInspector(contextMenu.target.node, {
        x: contextMenu.position.x + 20,
        y: contextMenu.position.y,
      })
    } else if (contextMenu.target.type === 'edge') {
      openEdgeInspector(
        contextMenu.target.edge,
        contextMenu.target.sourceNode,
        contextMenu.target.targetNode,
        { x: contextMenu.position.x + 20, y: contextMenu.position.y }
      )
    }
  }, [contextMenu, openNodeInspector, openEdgeInspector])

  // Handler for navigating to a node (from edge inspector or context menu)
  const handleNavigateToNode = useCallback(
    (nodeId: string) => {
      if (!layout) return

      const node = layout.nodes.get(nodeId)
      if (!node) return

      // Select the node
      setSelectedNodes(new Set([nodeId]))

      // Open inspector for the node
      const screenPos = worldToScreen(node.x, node.y, transform)
      openNodeInspector(node, { x: screenPos.x + 50, y: screenPos.y })
    },
    [layout, transform, openNodeInspector]
  )

  // Toolbar handlers
  const handleZoomIn = useCallback(() => {
    zoom(1.25, dimensions.width / 2, dimensions.height / 2)
  }, [zoom, dimensions])

  const handleZoomOut = useCallback(() => {
    zoom(0.8, dimensions.width / 2, dimensions.height / 2)
  }, [zoom, dimensions])

  const handleFitView = useCallback(() => {
    if (!layout || layout.nodes.size === 0) return

    // Calculate bounding box of all nodes
    let minX = Infinity,
      minY = Infinity,
      maxX = -Infinity,
      maxY = -Infinity

    for (const node of layout.nodes.values()) {
      const r = getNodeRadius(node.connectionCount)
      minX = Math.min(minX, node.x - r)
      minY = Math.min(minY, node.y - r)
      maxX = Math.max(maxX, node.x + r)
      maxY = Math.max(maxY, node.y + r)
    }

    const graphWidth = maxX - minX
    const graphHeight = maxY - minY
    const padding = 50

    const scaleX = (dimensions.width - padding * 2) / graphWidth
    const scaleY = (dimensions.height - padding * 2) / graphHeight
    const scale = Math.min(scaleX, scaleY, 2) // Cap at 2x zoom

    const centerX = (minX + maxX) / 2
    const centerY = (minY + maxY) / 2

    setTransform({
      x: dimensions.width / 2 - centerX * scale,
      y: dimensions.height / 2 - centerY * scale,
      scale,
    })
  }, [layout, dimensions, setTransform])

  const handleReset = useCallback(() => {
    resetLayout()
    setSelectedNodes(new Set())
  }, [resetLayout])

  return (
    <div ref={containerRef} className="relative w-full h-full">
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        onWheel={handleWheel}
        onDoubleClick={handleDoubleClick}
        onContextMenu={handleContextMenu}
      />

      {/* Toolbar */}
      <div className="absolute top-2 right-2 flex gap-1 bg-bg-secondary/80 backdrop-blur-sm rounded-md p-1 border border-border">
        <button
          onClick={handleZoomIn}
          className="p-1.5 rounded hover:bg-bg-tertiary transition-colors"
          title="Zoom in"
        >
          <ZoomIn size={16} className="text-text-muted" />
        </button>
        <button
          onClick={handleZoomOut}
          className="p-1.5 rounded hover:bg-bg-tertiary transition-colors"
          title="Zoom out"
        >
          <ZoomOut size={16} className="text-text-muted" />
        </button>
        <button
          onClick={handleFitView}
          className="p-1.5 rounded hover:bg-bg-tertiary transition-colors"
          title="Fit to view"
        >
          <Maximize2 size={16} className="text-text-muted" />
        </button>
        <div className="w-px bg-border mx-0.5" />
        <button
          onClick={handleReset}
          className="p-1.5 rounded hover:bg-bg-tertiary transition-colors"
          title="Reset layout"
        >
          <RotateCcw size={16} className="text-text-muted" />
        </button>
      </div>

      {/* Simulation indicator */}
      {isSimulating && (
        <div className="absolute bottom-2 left-2 flex items-center gap-2 text-xs text-text-muted bg-bg-secondary/80 backdrop-blur-sm rounded px-2 py-1 border border-border">
          <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
          Layout settling...
        </div>
      )}

      {/* Zoom indicator */}
      <div className="absolute bottom-2 right-2 text-xs text-text-muted bg-bg-secondary/80 backdrop-blur-sm rounded px-2 py-1 border border-border">
        {Math.round(transform.scale * 100)}%
      </div>

      {/* Context menu */}
      {contextMenu && (
        <ContextMenu
          target={contextMenu.target}
          position={contextMenu.position}
          onClose={() => setContextMenu(null)}
          onInspect={handleInspect}
          onNavigateToSource={
            contextMenu.target?.type === 'edge'
              ? () => {
                  const target = contextMenu.target
                  if (target?.type === 'edge') {
                    handleNavigateToNode(target.sourceNode.id)
                  }
                }
              : undefined
          }
          onNavigateToTarget={
            contextMenu.target?.type === 'edge'
              ? () => {
                  const target = contextMenu.target
                  if (target?.type === 'edge') {
                    handleNavigateToNode(target.targetNode.id)
                  }
                }
              : undefined
          }
        />
      )}

      {/* Inspector windows */}
      <InspectorContainer onNavigateToNode={handleNavigateToNode} />
    </div>
  )
}

// Drawing helper functions

function drawGrid(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  transform: ViewTransform
) {
  const gridSize = 40 * transform.scale
  const offsetX = transform.x % gridSize
  const offsetY = transform.y % gridSize

  ctx.strokeStyle = '#1a1a1a'
  ctx.lineWidth = 1

  for (let x = offsetX; x < width; x += gridSize) {
    ctx.beginPath()
    ctx.moveTo(x, 0)
    ctx.lineTo(x, height)
    ctx.stroke()
  }

  for (let y = offsetY; y < height; y += gridSize) {
    ctx.beginPath()
    ctx.moveTo(0, y)
    ctx.lineTo(width, y)
    ctx.stroke()
  }
}

function drawNode(
  ctx: CanvasRenderingContext2D,
  node: LayoutNode,
  isSelected: boolean,
  isHovered: boolean
) {
  const radius = getNodeRadius(node.connectionCount)
  const color = labelToColor(node.label)
  const lightColor = labelToLightColor(node.label)

  // Glow effect for selected/hovered
  if (isSelected || isHovered) {
    ctx.beginPath()
    ctx.arc(node.x, node.y, radius + 6, 0, Math.PI * 2)
    ctx.fillStyle = isSelected ? 'rgba(59, 130, 246, 0.3)' : 'rgba(255, 255, 255, 0.15)'
    ctx.fill()
  }

  // Node shadow
  ctx.beginPath()
  ctx.arc(node.x + 2, node.y + 2, radius, 0, Math.PI * 2)
  ctx.fillStyle = 'rgba(0, 0, 0, 0.3)'
  ctx.fill()

  // Node fill
  ctx.beginPath()
  ctx.arc(node.x, node.y, radius, 0, Math.PI * 2)
  const gradient = ctx.createRadialGradient(
    node.x - radius * 0.3,
    node.y - radius * 0.3,
    0,
    node.x,
    node.y,
    radius
  )
  gradient.addColorStop(0, lightColor)
  gradient.addColorStop(1, color)
  ctx.fillStyle = gradient
  ctx.fill()

  // Node border
  ctx.strokeStyle = isSelected ? '#3b82f6' : isHovered ? '#fff' : color
  ctx.lineWidth = isSelected ? 3 : isHovered ? 2 : 1.5
  ctx.stroke()

  // Node label
  ctx.fillStyle = '#fff'
  ctx.font = `bold ${Math.max(10, radius * 0.6)}px system-ui`
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'

  // Truncate label if too long
  const maxChars = Math.floor(radius / 4)
  const label = node.label.length > maxChars ? node.label.slice(0, maxChars) + '…' : node.label
  ctx.fillText(label, node.x, node.y)
}

function drawEdge(
  ctx: CanvasRenderingContext2D,
  edge: LayoutEdge,
  layout: { nodes: Map<string, LayoutNode> },
  _transform: ViewTransform,
  selectedNodes: Set<string>,
  hoveredEdge: LayoutEdge | null
) {
  const source = layout.nodes.get(edge.source)
  const target = layout.nodes.get(edge.target)
  if (!source || !target) return

  const isHighlighted =
    hoveredEdge?.id === edge.id ||
    selectedNodes.has(edge.source) ||
    selectedNodes.has(edge.target)

  // Calculate control point for curve (for multiple edges)
  const dx = target.x - source.x
  const dy = target.y - source.y
  const dist = Math.sqrt(dx * dx + dy * dy) || 1

  // Perpendicular offset for curved edges
  let offsetX = 0,
    offsetY = 0
  if (edge.curveTotal > 1) {
    const curveOffset = ((edge.curveIndex - (edge.curveTotal - 1) / 2) * 30)
    offsetX = (-dy / dist) * curveOffset
    offsetY = (dx / dist) * curveOffset
  }

  const midX = (source.x + target.x) / 2 + offsetX
  const midY = (source.y + target.y) / 2 + offsetY

  // Draw edge line
  ctx.beginPath()
  ctx.moveTo(source.x, source.y)

  if (edge.curveTotal > 1) {
    ctx.quadraticCurveTo(midX, midY, target.x, target.y)
  } else {
    ctx.lineTo(target.x, target.y)
  }

  ctx.strokeStyle = isHighlighted ? '#6b7280' : '#3f3f46'
  ctx.lineWidth = isHighlighted ? 2 : 1
  ctx.stroke()

  // Draw arrowhead
  const sourceRadius = getNodeRadius(source.connectionCount)
  const targetRadius = getNodeRadius(target.connectionCount)

  // Calculate arrow position (at edge of target node)
  let arrowX: number, arrowY: number, angle: number

  if (edge.curveTotal > 1) {
    // For curved edges, calculate tangent at target
    const t = 0.95
    const ax = source.x * (1 - t) * (1 - t) + midX * 2 * (1 - t) * t + target.x * t * t
    const ay = source.y * (1 - t) * (1 - t) + midY * 2 * (1 - t) * t + target.y * t * t
    angle = Math.atan2(target.y - ay, target.x - ax)
    arrowX = target.x - Math.cos(angle) * targetRadius
    arrowY = target.y - Math.sin(angle) * targetRadius
  } else {
    angle = Math.atan2(dy, dx)
    arrowX = target.x - Math.cos(angle) * targetRadius
    arrowY = target.y - Math.sin(angle) * targetRadius
  }

  // Suppress unused variable warning
  void sourceRadius

  const arrowSize = isHighlighted ? 10 : 8
  ctx.beginPath()
  ctx.moveTo(arrowX, arrowY)
  ctx.lineTo(
    arrowX - arrowSize * Math.cos(angle - Math.PI / 6),
    arrowY - arrowSize * Math.sin(angle - Math.PI / 6)
  )
  ctx.lineTo(
    arrowX - arrowSize * Math.cos(angle + Math.PI / 6),
    arrowY - arrowSize * Math.sin(angle + Math.PI / 6)
  )
  ctx.closePath()
  ctx.fillStyle = isHighlighted ? '#6b7280' : '#3f3f46'
  ctx.fill()
}

function drawEdgeLabel(
  ctx: CanvasRenderingContext2D,
  edge: LayoutEdge,
  layout: { nodes: Map<string, LayoutNode> },
  transform: ViewTransform
) {
  const source = layout.nodes.get(edge.source)
  const target = layout.nodes.get(edge.target)
  if (!source || !target) return

  // Calculate midpoint in world coordinates
  const dx = target.x - source.x
  const dy = target.y - source.y
  const dist = Math.sqrt(dx * dx + dy * dy) || 1

  let midX = (source.x + target.x) / 2
  let midY = (source.y + target.y) / 2

  if (edge.curveTotal > 1) {
    const curveOffset = ((edge.curveIndex - (edge.curveTotal - 1) / 2) * 30)
    midX += (-dy / dist) * curveOffset
    midY += (dx / dist) * curveOffset
  }

  // Convert to screen coordinates
  const screenX = midX * transform.scale + transform.x
  const screenY = midY * transform.scale + transform.y

  // Draw label background
  ctx.font = '11px system-ui'
  const text = edge.type
  const metrics = ctx.measureText(text)
  const padding = 4

  ctx.fillStyle = 'rgba(0, 0, 0, 0.8)'
  ctx.fillRect(
    screenX - metrics.width / 2 - padding,
    screenY - 7 - padding,
    metrics.width + padding * 2,
    14 + padding * 2
  )

  ctx.fillStyle = '#fff'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText(text, screenX, screenY)
}
