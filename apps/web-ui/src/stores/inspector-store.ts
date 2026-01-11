import { create } from 'zustand'
import type { LayoutNode, LayoutEdge } from '../utils/graph-layout'

export type InspectorType = 'node' | 'edge'

export interface InspectorPosition {
  x: number
  y: number
}

export interface InspectorSize {
  width: number
  height: number
}

interface BaseInspector {
  id: string
  type: InspectorType
  position: InspectorPosition
  size: InspectorSize
  minimized: boolean
  zIndex: number
}

export interface NodeInspector extends BaseInspector {
  type: 'node'
  node: LayoutNode
}

export interface EdgeInspector extends BaseInspector {
  type: 'edge'
  edge: LayoutEdge
  sourceNode: LayoutNode
  targetNode: LayoutNode
}

export type Inspector = NodeInspector | EdgeInspector

interface InspectorState {
  inspectors: Inspector[]
  maxZIndex: number

  // Actions
  openNodeInspector: (node: LayoutNode, position?: InspectorPosition) => void
  openEdgeInspector: (
    edge: LayoutEdge,
    sourceNode: LayoutNode,
    targetNode: LayoutNode,
    position?: InspectorPosition
  ) => void
  closeInspector: (id: string) => void
  closeAllInspectors: () => void
  bringToFront: (id: string) => void
  updatePosition: (id: string, position: InspectorPosition) => void
  updateSize: (id: string, size: InspectorSize) => void
  toggleMinimized: (id: string) => void
  updateNodeInspector: (id: string, node: LayoutNode) => void
}

const DEFAULT_SIZE: InspectorSize = { width: 320, height: 400 }
const MIN_SIZE: InspectorSize = { width: 240, height: 200 }
const MAX_SIZE: InspectorSize = { width: 600, height: 800 }

let inspectorIdCounter = 0
const generateInspectorId = () => `inspector-${++inspectorIdCounter}`

// Calculate a default position for new inspectors
const getDefaultPosition = (existingCount: number): InspectorPosition => {
  const baseX = 20
  const baseY = 60
  const offset = existingCount * 30
  return { x: baseX + offset, y: baseY + offset }
}

export const useInspectorStore = create<InspectorState>((set, get) => ({
  inspectors: [],
  maxZIndex: 100,

  openNodeInspector: (node, position) => {
    const state = get()

    // Check if inspector for this node already exists
    const existing = state.inspectors.find(
      (i) => i.type === 'node' && (i as NodeInspector).node.id === node.id
    )

    if (existing) {
      // Bring existing inspector to front
      get().bringToFront(existing.id)
      return
    }

    const newInspector: NodeInspector = {
      id: generateInspectorId(),
      type: 'node',
      node,
      position: position ?? getDefaultPosition(state.inspectors.length),
      size: { ...DEFAULT_SIZE },
      minimized: false,
      zIndex: state.maxZIndex + 1,
    }

    set({
      inspectors: [...state.inspectors, newInspector],
      maxZIndex: state.maxZIndex + 1,
    })
  },

  openEdgeInspector: (edge, sourceNode, targetNode, position) => {
    const state = get()

    // Check if inspector for this edge already exists
    const existing = state.inspectors.find(
      (i) => i.type === 'edge' && (i as EdgeInspector).edge.id === edge.id
    )

    if (existing) {
      // Bring existing inspector to front
      get().bringToFront(existing.id)
      return
    }

    const newInspector: EdgeInspector = {
      id: generateInspectorId(),
      type: 'edge',
      edge,
      sourceNode,
      targetNode,
      position: position ?? getDefaultPosition(state.inspectors.length),
      size: { width: 320, height: 320 },
      minimized: false,
      zIndex: state.maxZIndex + 1,
    }

    set({
      inspectors: [...state.inspectors, newInspector],
      maxZIndex: state.maxZIndex + 1,
    })
  },

  closeInspector: (id) => {
    set((state) => ({
      inspectors: state.inspectors.filter((i) => i.id !== id),
    }))
  },

  closeAllInspectors: () => {
    set({ inspectors: [] })
  },

  bringToFront: (id) => {
    set((state) => {
      const newMaxZ = state.maxZIndex + 1
      return {
        inspectors: state.inspectors.map((i) =>
          i.id === id ? { ...i, zIndex: newMaxZ } : i
        ),
        maxZIndex: newMaxZ,
      }
    })
  },

  updatePosition: (id, position) => {
    set((state) => ({
      inspectors: state.inspectors.map((i) =>
        i.id === id ? { ...i, position } : i
      ),
    }))
  },

  updateSize: (id, size) => {
    // Clamp size to min/max
    const clampedSize: InspectorSize = {
      width: Math.max(MIN_SIZE.width, Math.min(MAX_SIZE.width, size.width)),
      height: Math.max(MIN_SIZE.height, Math.min(MAX_SIZE.height, size.height)),
    }

    set((state) => ({
      inspectors: state.inspectors.map((i) =>
        i.id === id ? { ...i, size: clampedSize } : i
      ),
    }))
  },

  toggleMinimized: (id) => {
    set((state) => ({
      inspectors: state.inspectors.map((i) =>
        i.id === id ? { ...i, minimized: !i.minimized } : i
      ),
    }))
  },

  updateNodeInspector: (id, node) => {
    set((state) => ({
      inspectors: state.inspectors.map((i) =>
        i.id === id && i.type === 'node' ? { ...i, node } : i
      ),
    }))
  },
}))

export { MIN_SIZE, MAX_SIZE }
