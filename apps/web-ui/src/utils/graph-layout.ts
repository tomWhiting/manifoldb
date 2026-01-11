import type { GraphNode, GraphEdge } from '../types'

// Types for layout nodes with positions and physics properties
export interface LayoutNode {
  id: string
  x: number
  y: number
  vx: number
  vy: number
  fx: number | null // Fixed x position (when dragging)
  fy: number | null // Fixed y position (when dragging)
  label: string
  labels: string[]
  properties: Record<string, unknown>
  connectionCount: number
}

export interface LayoutEdge {
  id: string
  source: string
  target: string
  type: string
  properties: Record<string, unknown>
  curveIndex: number // For multiple edges between same nodes
  curveTotal: number // Total edges between same nodes
}

export interface GraphLayout {
  nodes: Map<string, LayoutNode>
  edges: LayoutEdge[]
  width: number
  height: number
}

export interface ViewTransform {
  x: number // Pan offset X
  y: number // Pan offset Y
  scale: number // Zoom level
}

// Force simulation parameters
export interface ForceParams {
  repulsion: number // Node repulsion strength
  attraction: number // Edge attraction strength
  damping: number // Velocity damping
  centerForce: number // Force pulling nodes toward center
  idealEdgeLength: number // Target edge length
}

export const DEFAULT_FORCE_PARAMS: ForceParams = {
  repulsion: 5000,
  attraction: 0.05,
  damping: 0.9,
  centerForce: 0.01,
  idealEdgeLength: 100,
}

// Hash a string to a color
export function labelToColor(label: string): string {
  let hash = 0
  for (let i = 0; i < label.length; i++) {
    hash = label.charCodeAt(i) + ((hash << 5) - hash)
    hash = hash & hash
  }
  // Generate HSL color with good saturation and lightness
  const h = Math.abs(hash) % 360
  return `hsl(${h}, 65%, 55%)`
}

// Hash a string to a lighter color (for fills)
export function labelToLightColor(label: string): string {
  let hash = 0
  for (let i = 0; i < label.length; i++) {
    hash = label.charCodeAt(i) + ((hash << 5) - hash)
    hash = hash & hash
  }
  const h = Math.abs(hash) % 360
  return `hsl(${h}, 60%, 70%)`
}

// Calculate node radius based on connection count
export function getNodeRadius(connectionCount: number): number {
  const baseRadius = 20
  const maxRadius = 40
  const scale = Math.log2(connectionCount + 1) * 5
  return Math.min(baseRadius + scale, maxRadius)
}

// Initialize layout from graph data
export function initializeLayout(
  nodes: GraphNode[],
  edges: GraphEdge[],
  width: number,
  height: number,
  existingPositions?: Map<string, { x: number; y: number }>
): GraphLayout {
  const layoutNodes = new Map<string, LayoutNode>()

  // Count connections for each node
  const connectionCounts = new Map<string, number>()
  for (const edge of edges) {
    connectionCounts.set(edge.sourceId, (connectionCounts.get(edge.sourceId) || 0) + 1)
    connectionCounts.set(edge.targetId, (connectionCounts.get(edge.targetId) || 0) + 1)
  }

  // Create layout nodes
  const centerX = width / 2
  const centerY = height / 2
  nodes.forEach((node, i) => {
    // Use existing position or initialize in a circle
    const existing = existingPositions?.get(node.id)
    let x: number, y: number

    if (existing) {
      x = existing.x
      y = existing.y
    } else {
      // Random position around center with some spread
      const angle = (i / nodes.length) * Math.PI * 2
      const radius = Math.min(width, height) * 0.3 + Math.random() * 50
      x = centerX + Math.cos(angle) * radius + (Math.random() - 0.5) * 50
      y = centerY + Math.sin(angle) * radius + (Math.random() - 0.5) * 50
    }

    layoutNodes.set(node.id, {
      id: node.id,
      x,
      y,
      vx: 0,
      vy: 0,
      fx: null,
      fy: null,
      label: node.labels[0] || node.id.slice(0, 8),
      labels: node.labels,
      properties: node.properties,
      connectionCount: connectionCounts.get(node.id) || 0,
    })
  })

  // Process edges, handling multiple edges between same nodes
  const edgeCounts = new Map<string, number>()
  const edgeIndices = new Map<string, number>()

  // First pass: count edges between each pair
  for (const edge of edges) {
    const key = [edge.sourceId, edge.targetId].sort().join('-')
    edgeCounts.set(key, (edgeCounts.get(key) || 0) + 1)
  }

  // Second pass: create layout edges with curve indices
  const layoutEdges: LayoutEdge[] = []
  for (const edge of edges) {
    const key = [edge.sourceId, edge.targetId].sort().join('-')
    const total = edgeCounts.get(key) || 1
    const index = edgeIndices.get(key) || 0
    edgeIndices.set(key, index + 1)

    layoutEdges.push({
      id: edge.id,
      source: edge.sourceId,
      target: edge.targetId,
      type: edge.type,
      properties: edge.properties,
      curveIndex: index,
      curveTotal: total,
    })
  }

  return {
    nodes: layoutNodes,
    edges: layoutEdges,
    width,
    height,
  }
}

// Apply one step of force-directed simulation
export function simulateForces(
  layout: GraphLayout,
  params: ForceParams = DEFAULT_FORCE_PARAMS,
  alpha: number = 1
): number {
  const nodes = Array.from(layout.nodes.values())
  const centerX = layout.width / 2
  const centerY = layout.height / 2

  // Reset forces
  for (const node of nodes) {
    if (node.fx !== null) {
      node.x = node.fx
      node.vx = 0
    }
    if (node.fy !== null) {
      node.y = node.fy
      node.vy = 0
    }
  }

  // Apply repulsion between all node pairs
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const nodeA = nodes[i]
      const nodeB = nodes[j]

      const dx = nodeB.x - nodeA.x
      const dy = nodeB.y - nodeA.y
      const distSq = dx * dx + dy * dy
      const dist = Math.sqrt(distSq) || 1

      // Coulomb's law repulsion
      const force = (params.repulsion * alpha) / distSq
      const fx = (dx / dist) * force
      const fy = (dy / dist) * force

      if (nodeA.fx === null) nodeA.vx -= fx
      if (nodeA.fy === null) nodeA.vy -= fy
      if (nodeB.fx === null) nodeB.vx += fx
      if (nodeB.fy === null) nodeB.vy += fy
    }
  }

  // Apply attraction along edges (spring force)
  for (const edge of layout.edges) {
    const source = layout.nodes.get(edge.source)
    const target = layout.nodes.get(edge.target)
    if (!source || !target) continue

    const dx = target.x - source.x
    const dy = target.y - source.y
    const dist = Math.sqrt(dx * dx + dy * dy) || 1

    // Hooke's law attraction
    const displacement = dist - params.idealEdgeLength
    const force = displacement * params.attraction * alpha
    const fx = (dx / dist) * force
    const fy = (dy / dist) * force

    if (source.fx === null) source.vx += fx
    if (source.fy === null) source.vy += fy
    if (target.fx === null) target.vx -= fx
    if (target.fy === null) target.vy -= fy
  }

  // Apply center gravity
  for (const node of nodes) {
    if (node.fx === null) {
      node.vx += (centerX - node.x) * params.centerForce * alpha
    }
    if (node.fy === null) {
      node.vy += (centerY - node.y) * params.centerForce * alpha
    }
  }

  // Apply velocity and damping
  let totalMovement = 0
  for (const node of nodes) {
    if (node.fx === null) {
      node.vx *= params.damping
      node.x += node.vx
      totalMovement += Math.abs(node.vx)
    }
    if (node.fy === null) {
      node.vy *= params.damping
      node.y += node.vy
      totalMovement += Math.abs(node.vy)
    }
  }

  return totalMovement
}

// Transform screen coordinates to world coordinates
export function screenToWorld(
  screenX: number,
  screenY: number,
  transform: ViewTransform
): { x: number; y: number } {
  return {
    x: (screenX - transform.x) / transform.scale,
    y: (screenY - transform.y) / transform.scale,
  }
}

// Transform world coordinates to screen coordinates
export function worldToScreen(
  worldX: number,
  worldY: number,
  transform: ViewTransform
): { x: number; y: number } {
  return {
    x: worldX * transform.scale + transform.x,
    y: worldY * transform.scale + transform.y,
  }
}

// Find node at screen position
export function findNodeAtPosition(
  screenX: number,
  screenY: number,
  layout: GraphLayout,
  transform: ViewTransform
): LayoutNode | null {
  const world = screenToWorld(screenX, screenY, transform)

  for (const node of layout.nodes.values()) {
    const radius = getNodeRadius(node.connectionCount)
    const dx = world.x - node.x
    const dy = world.y - node.y
    if (dx * dx + dy * dy <= radius * radius) {
      return node
    }
  }

  return null
}

// Find edge at screen position (for hover)
export function findEdgeAtPosition(
  screenX: number,
  screenY: number,
  layout: GraphLayout,
  transform: ViewTransform,
  threshold: number = 8
): LayoutEdge | null {
  const world = screenToWorld(screenX, screenY, transform)
  const thresholdWorld = threshold / transform.scale

  for (const edge of layout.edges) {
    const source = layout.nodes.get(edge.source)
    const target = layout.nodes.get(edge.target)
    if (!source || !target) continue

    // Point-to-line-segment distance
    const dx = target.x - source.x
    const dy = target.y - source.y
    const lengthSq = dx * dx + dy * dy

    if (lengthSq === 0) {
      // Source and target are the same point
      const dist = Math.sqrt((world.x - source.x) ** 2 + (world.y - source.y) ** 2)
      if (dist <= thresholdWorld) return edge
      continue
    }

    // Project point onto line segment
    let t = ((world.x - source.x) * dx + (world.y - source.y) * dy) / lengthSq
    t = Math.max(0, Math.min(1, t))

    const nearestX = source.x + t * dx
    const nearestY = source.y + t * dy
    const dist = Math.sqrt((world.x - nearestX) ** 2 + (world.y - nearestY) ** 2)

    if (dist <= thresholdWorld) return edge
  }

  return null
}
