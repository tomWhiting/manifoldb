import { useState, useCallback, useMemo, useRef } from 'react'
import { ZoomIn, ZoomOut, Maximize2 } from 'lucide-react'
import type { SchemaLabel, SchemaRelationship } from './SchemaEditor'

interface SchemaDiagramProps {
  labels: SchemaLabel[]
  relationships: SchemaRelationship[]
  onSelectLabel: (name: string | null) => void
  onSelectRelationship: (name: string | null) => void
  selectedLabel: string | null
  selectedRelationship: string | null
}

interface NodePosition {
  x: number
  y: number
  width: number
  height: number
}

interface DragState {
  nodeId: string | null
  startX: number
  startY: number
  nodeStartX: number
  nodeStartY: number
}

const NODE_WIDTH = 160
const NODE_HEIGHT = 60
const NODE_PADDING = 40

function calculateDefaultPosition(index: number, totalCount: number): NodePosition {
  const cols = Math.ceil(Math.sqrt(totalCount)) || 1
  const col = index % cols
  const row = Math.floor(index / cols)
  return {
    x: 100 + col * (NODE_WIDTH + NODE_PADDING),
    y: 100 + row * (NODE_HEIGHT + NODE_PADDING * 2),
    width: NODE_WIDTH,
    height: NODE_HEIGHT,
  }
}

// Get the point where a line from center at given angle intersects the rectangle edge
function getEdgePoint(rect: NodePosition, angle: number): { x: number; y: number } {
  const centerX = rect.x + rect.width / 2
  const centerY = rect.y + rect.height / 2
  const cos = Math.cos(angle)
  const sin = Math.sin(angle)

  // Calculate intersection with rectangle edges
  const tRight = (rect.width / 2) / Math.abs(cos)
  const tBottom = (rect.height / 2) / Math.abs(sin)
  const t = Math.min(tRight, tBottom)

  return {
    x: centerX + cos * t * Math.sign(cos),
    y: centerY + sin * t * Math.sign(sin),
  }
}

export function SchemaDiagram({
  labels,
  relationships,
  onSelectLabel,
  onSelectRelationship,
  selectedLabel,
  selectedRelationship,
}: SchemaDiagramProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [isPanning, setIsPanning] = useState(false)
  const [panStart, setPanStart] = useState({ x: 0, y: 0 })
  const [dragState, setDragState] = useState<DragState>({
    nodeId: null,
    startX: 0,
    startY: 0,
    nodeStartX: 0,
    nodeStartY: 0,
  })

  // Store only custom (dragged) positions - positions that differ from defaults
  const [customPositions, setCustomPositions] = useState<Record<string, NodePosition>>({})

  // Compute effective node positions by merging default positions with custom ones
  const nodePositions = useMemo(() => {
    const positions: Record<string, NodePosition> = {}
    labels.forEach((label, index) => {
      if (customPositions[label.name]) {
        positions[label.name] = customPositions[label.name]
      } else {
        positions[label.name] = calculateDefaultPosition(index, labels.length)
      }
    })
    return positions
  }, [labels, customPositions])

  // Calculate edge paths between labels
  const edgePaths = useMemo(() => {
    // Group relationships by source-target pair
    const relationshipGroups = new Map<string, SchemaRelationship[]>()

    relationships.forEach(rel => {
      // Since we don't have specific source/target labels from the schema,
      // we'll just draw edges from each label to others based on relationships
      const key = `all`
      const group = relationshipGroups.get(key) ?? []
      group.push(rel)
      relationshipGroups.set(key, group)
    })

    return relationships.map((rel, index) => {
      // For visualization, distribute relationship edges around the diagram
      // This is a simplified view since we don't have source/target label data
      const labelNames = Object.keys(nodePositions)
      if (labelNames.length < 2) return null

      const sourceIndex = index % labelNames.length
      const targetIndex = (index + 1) % labelNames.length
      const sourceName = labelNames[sourceIndex]
      const targetName = labelNames[targetIndex]

      const source = nodePositions[sourceName]
      const target = nodePositions[targetName]

      if (!source || !target) return null

      // Calculate edge path
      const sourceCenter = {
        x: source.x + source.width / 2,
        y: source.y + source.height / 2,
      }
      const targetCenter = {
        x: target.x + target.width / 2,
        y: target.y + target.height / 2,
      }

      // If source and target are the same, draw a self-loop
      if (sourceName === targetName) {
        const loopSize = 30
        return {
          rel,
          path: `M ${source.x + source.width} ${source.y + source.height / 2}
                 C ${source.x + source.width + loopSize * 2} ${source.y + source.height / 2 - loopSize},
                   ${source.x + source.width + loopSize * 2} ${source.y + source.height / 2 + loopSize},
                   ${source.x + source.width} ${source.y + source.height / 2 + 10}`,
          labelPos: {
            x: source.x + source.width + loopSize * 1.5,
            y: source.y + source.height / 2,
          },
          arrowAngle: 90,
          isSelected: selectedRelationship === rel.name,
        }
      }

      // Calculate the direction vector
      const dx = targetCenter.x - sourceCenter.x
      const dy = targetCenter.y - sourceCenter.y
      const angle = Math.atan2(dy, dx)

      // Calculate points on the edge of the rectangles
      const sourceEdge = getEdgePoint(source, angle)
      const targetEdge = getEdgePoint(target, angle + Math.PI)

      // Create a curved path
      const midX = (sourceEdge.x + targetEdge.x) / 2
      const midY = (sourceEdge.y + targetEdge.y) / 2
      const curvature = 0.2
      const perpX = -dy * curvature
      const perpY = dx * curvature

      return {
        rel,
        path: `M ${sourceEdge.x} ${sourceEdge.y} Q ${midX + perpX} ${midY + perpY} ${targetEdge.x} ${targetEdge.y}`,
        labelPos: {
          x: midX + perpX / 2,
          y: midY + perpY / 2,
        },
        arrowAngle: Math.atan2(targetEdge.y - (midY + perpY), targetEdge.x - (midX + perpX)) * 180 / Math.PI,
        isSelected: selectedRelationship === rel.name,
      }
    }).filter(Boolean)
  }, [relationships, nodePositions, selectedRelationship])

  const handleZoomIn = () => setZoom(z => Math.min(z * 1.2, 3))
  const handleZoomOut = () => setZoom(z => Math.max(z / 1.2, 0.3))
  const handleResetView = () => {
    setZoom(1)
    setPan({ x: 0, y: 0 })
  }

  const handleWheel = useCallback((e: React.WheelEvent) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault()
      const delta = e.deltaY > 0 ? 0.9 : 1.1
      setZoom(z => Math.max(0.3, Math.min(3, z * delta)))
    } else {
      setPan(p => ({
        x: p.x - e.deltaX,
        y: p.y - e.deltaY,
      }))
    }
  }, [])

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.target === containerRef.current || (e.target as HTMLElement).classList.contains('diagram-canvas')) {
      setIsPanning(true)
      setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y })
    }
  }, [pan])

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (isPanning) {
      setPan({
        x: e.clientX - panStart.x,
        y: e.clientY - panStart.y,
      })
    } else if (dragState.nodeId) {
      const dx = (e.clientX - dragState.startX) / zoom
      const dy = (e.clientY - dragState.startY) / zoom
      const nodeId = dragState.nodeId
      const currentPos = nodePositions[nodeId]

      if (currentPos) {
        setCustomPositions(prev => ({
          ...prev,
          [nodeId]: {
            ...currentPos,
            x: dragState.nodeStartX + dx,
            y: dragState.nodeStartY + dy,
          },
        }))
      }
    }
  }, [isPanning, panStart, dragState, zoom, nodePositions])

  const handleMouseUp = useCallback(() => {
    setIsPanning(false)
    setDragState({
      nodeId: null,
      startX: 0,
      startY: 0,
      nodeStartX: 0,
      nodeStartY: 0,
    })
  }, [])

  const handleNodeMouseDown = useCallback((e: React.MouseEvent, labelName: string) => {
    e.stopPropagation()
    const pos = nodePositions[labelName]
    if (pos) {
      setDragState({
        nodeId: labelName,
        startX: e.clientX,
        startY: e.clientY,
        nodeStartX: pos.x,
        nodeStartY: pos.y,
      })
    }
  }, [nodePositions])

  const handleNodeClick = useCallback((e: React.MouseEvent, labelName: string) => {
    e.stopPropagation()
    onSelectLabel(selectedLabel === labelName ? null : labelName)
    onSelectRelationship(null)
  }, [onSelectLabel, onSelectRelationship, selectedLabel])

  const handleEdgeClick = useCallback((e: React.MouseEvent, relName: string) => {
    e.stopPropagation()
    onSelectRelationship(selectedRelationship === relName ? null : relName)
    onSelectLabel(null)
  }, [onSelectLabel, onSelectRelationship, selectedRelationship])

  const handleCanvasClick = useCallback(() => {
    onSelectLabel(null)
    onSelectRelationship(null)
  }, [onSelectLabel, onSelectRelationship])

  if (labels.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-text-muted">
        <div className="text-center">
          <p className="mb-2">No labels in the schema</p>
          <p className="text-sm">Create a label to get started</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Zoom controls */}
      <div className="absolute top-20 right-4 z-10 flex flex-col gap-1 bg-bg-secondary border border-border rounded-md p-1">
        <button
          onClick={handleZoomIn}
          className="p-2 hover:bg-bg-tertiary rounded text-text-muted hover:text-text-primary transition-colors"
          title="Zoom in"
        >
          <ZoomIn size={16} />
        </button>
        <button
          onClick={handleZoomOut}
          className="p-2 hover:bg-bg-tertiary rounded text-text-muted hover:text-text-primary transition-colors"
          title="Zoom out"
        >
          <ZoomOut size={16} />
        </button>
        <div className="border-t border-border my-1" />
        <button
          onClick={handleResetView}
          className="p-2 hover:bg-bg-tertiary rounded text-text-muted hover:text-text-primary transition-colors"
          title="Reset view"
        >
          <Maximize2 size={16} />
        </button>
        <div className="px-2 py-1 text-xs text-text-muted text-center">
          {Math.round(zoom * 100)}%
        </div>
      </div>

      {/* Diagram canvas */}
      <div
        ref={containerRef}
        className="flex-1 overflow-hidden cursor-grab active:cursor-grabbing diagram-canvas"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onClick={handleCanvasClick}
        style={{ background: 'var(--bg-primary)' }}
      >
        <svg
          className="w-full h-full"
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: '0 0',
          }}
        >
          {/* Grid pattern */}
          <defs>
            <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
              <path
                d="M 20 0 L 0 0 0 20"
                fill="none"
                stroke="var(--border)"
                strokeWidth="0.5"
                opacity="0.3"
              />
            </pattern>
            <marker
              id="arrowhead"
              markerWidth="10"
              markerHeight="7"
              refX="9"
              refY="3.5"
              orient="auto"
            >
              <polygon points="0 0, 10 3.5, 0 7" fill="var(--text-muted)" />
            </marker>
            <marker
              id="arrowhead-selected"
              markerWidth="10"
              markerHeight="7"
              refX="9"
              refY="3.5"
              orient="auto"
            >
              <polygon points="0 0, 10 3.5, 0 7" fill="var(--accent)" />
            </marker>
          </defs>

          <rect width="3000" height="2000" fill="url(#grid)" x="-500" y="-500" />

          {/* Edges */}
          <g className="edges">
            {edgePaths.map((edge, index) => edge && (
              <g key={`edge-${index}`}>
                <path
                  d={edge.path}
                  fill="none"
                  stroke={edge.isSelected ? 'var(--accent)' : 'var(--text-muted)'}
                  strokeWidth={edge.isSelected ? 2 : 1.5}
                  markerEnd={edge.isSelected ? 'url(#arrowhead-selected)' : 'url(#arrowhead)'}
                  className="cursor-pointer hover:stroke-accent"
                  onClick={(e) => handleEdgeClick(e, edge.rel.name)}
                />
                <text
                  x={edge.labelPos.x}
                  y={edge.labelPos.y}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  className={`text-xs cursor-pointer ${edge.isSelected ? 'fill-accent' : 'fill-text-muted'}`}
                  style={{ fontSize: '10px', pointerEvents: 'all' }}
                  onClick={(e) => handleEdgeClick(e, edge.rel.name)}
                >
                  {edge.rel.name}
                </text>
              </g>
            ))}
          </g>

          {/* Nodes */}
          <g className="nodes">
            {labels.map(label => {
              const pos = nodePositions[label.name]
              if (!pos) return null

              const isSelected = selectedLabel === label.name

              return (
                <g
                  key={label.name}
                  transform={`translate(${pos.x}, ${pos.y})`}
                  className="cursor-pointer"
                  onMouseDown={(e) => handleNodeMouseDown(e, label.name)}
                  onClick={(e) => handleNodeClick(e, label.name)}
                >
                  {/* Node background */}
                  <rect
                    width={pos.width}
                    height={pos.height}
                    rx="8"
                    ry="8"
                    fill="var(--bg-secondary)"
                    stroke={isSelected ? 'var(--accent)' : 'var(--border)'}
                    strokeWidth={isSelected ? 2 : 1}
                    className="hover:stroke-accent transition-colors"
                  />

                  {/* Label name */}
                  <text
                    x={pos.width / 2}
                    y={pos.height / 2 - 6}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    className={`text-sm font-medium ${isSelected ? 'fill-accent' : 'fill-text-primary'}`}
                    style={{ fontSize: '12px' }}
                  >
                    {label.name}
                  </text>

                  {/* Node count */}
                  <text
                    x={pos.width / 2}
                    y={pos.height / 2 + 12}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    className="fill-text-muted"
                    style={{ fontSize: '10px' }}
                  >
                    {label.count.toLocaleString()} nodes
                  </text>
                </g>
              )
            })}
          </g>
        </svg>
      </div>

      {/* Selection info */}
      {(selectedLabel || selectedRelationship) && (
        <div className="absolute bottom-4 left-4 bg-bg-secondary border border-border rounded-md px-3 py-2 text-sm">
          {selectedLabel && (
            <span className="text-text-primary">
              Selected: <span className="text-accent font-medium">:{selectedLabel}</span>
            </span>
          )}
          {selectedRelationship && (
            <span className="text-text-primary">
              Selected: <span className="text-accent font-medium">[:{selectedRelationship}]</span>
            </span>
          )}
        </div>
      )}
    </div>
  )
}
