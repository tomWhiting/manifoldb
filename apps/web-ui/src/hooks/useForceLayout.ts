import { useRef, useCallback, useEffect, useState } from 'react'
import type { GraphNode, GraphEdge } from '../types'
import {
  type GraphLayout,
  type ViewTransform,
  type ForceParams,
  DEFAULT_FORCE_PARAMS,
  initializeLayout,
  simulateForces,
} from '../utils/graph-layout'

export interface UseForceLayoutOptions {
  nodes: GraphNode[]
  edges: GraphEdge[]
  width: number
  height: number
  params?: ForceParams
  onTick?: (layout: GraphLayout) => void
  preservePositions?: boolean
}

export interface UseForceLayoutReturn {
  layout: GraphLayout | null
  transform: ViewTransform
  isSimulating: boolean
  alpha: number
  // Actions
  startSimulation: () => void
  stopSimulation: () => void
  resetLayout: () => void
  setTransform: (transform: ViewTransform) => void
  pan: (dx: number, dy: number) => void
  zoom: (factor: number, centerX: number, centerY: number) => void
  fixNode: (nodeId: string, x: number, y: number) => void
  releaseNode: (nodeId: string) => void
  reheat: () => void
}

const MIN_ZOOM = 0.1
const MAX_ZOOM = 4
const ALPHA_DECAY = 0.0228 // Same as D3 default
const ALPHA_MIN = 0.001

export function useForceLayout({
  nodes,
  edges,
  width,
  height,
  params = DEFAULT_FORCE_PARAMS,
  onTick,
  preservePositions = true,
}: UseForceLayoutOptions): UseForceLayoutReturn {
  const [layout, setLayout] = useState<GraphLayout | null>(null)
  const [transform, setTransform] = useState<ViewTransform>({ x: 0, y: 0, scale: 1 })
  const [isSimulating, setIsSimulating] = useState(false)
  const [alpha, setAlpha] = useState(1)

  const animationFrameRef = useRef<number | null>(null)
  const alphaRef = useRef(1)
  const paramsRef = useRef(params)
  const onTickRef = useRef(onTick)
  const previousNodesRef = useRef<Map<string, { x: number; y: number }>>(new Map())

  // Keep refs updated
  useEffect(() => {
    paramsRef.current = params
  }, [params])

  useEffect(() => {
    onTickRef.current = onTick
  }, [onTick])

  // Initialize or update layout when inputs change
  useEffect(() => {
    // Preserve positions from previous layout if enabled
    setLayout((prevLayout) => {
      // Handle empty data case
      if (nodes.length === 0 || width === 0 || height === 0) {
        return null
      }

      let existingPositions: Map<string, { x: number; y: number }> | undefined
      if (preservePositions && prevLayout) {
        existingPositions = new Map()
        for (const [id, node] of prevLayout.nodes) {
          existingPositions.set(id, { x: node.x, y: node.y })
        }
        // Also merge with previously saved positions
        for (const [id, pos] of previousNodesRef.current) {
          if (!existingPositions.has(id)) {
            existingPositions.set(id, pos)
          }
        }
      }

      const newLayout = initializeLayout(nodes, edges, width, height, existingPositions)

      // Save positions for future use
      if (preservePositions) {
        for (const [id, node] of newLayout.nodes) {
          previousNodesRef.current.set(id, { x: node.x, y: node.y })
        }
      }

      return newLayout
    })

    // Start simulation for new data (only if we have data)
    if (nodes.length > 0 && width > 0 && height > 0) {
      alphaRef.current = 1
      setAlpha(1)
      setIsSimulating(true)
    }
  }, [nodes, edges, width, height, preservePositions])

  // Animation loop
  useEffect(() => {
    if (!isSimulating) {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current)
        animationFrameRef.current = null
      }
      return
    }

    const tick = () => {
      if (alphaRef.current < ALPHA_MIN) {
        setIsSimulating(false)
        return
      }

      setLayout((currentLayout) => {
        if (!currentLayout) {
          setIsSimulating(false)
          return null
        }

        // Run simulation step - mutates currentLayout in place
        simulateForces(currentLayout, paramsRef.current, alphaRef.current)

        // Decay alpha
        alphaRef.current *= 1 - ALPHA_DECAY
        setAlpha(alphaRef.current)

        // Call onTick callback
        onTickRef.current?.(currentLayout)

        // Return a new object reference to trigger re-render
        return { ...currentLayout }
      })

      // Schedule next frame
      animationFrameRef.current = requestAnimationFrame(tick)
    }

    animationFrameRef.current = requestAnimationFrame(tick)

    return () => {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current)
        animationFrameRef.current = null
      }
    }
  }, [isSimulating])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [])

  const startSimulation = useCallback(() => {
    alphaRef.current = 1
    setAlpha(1)
    setIsSimulating(true)
  }, [])

  const stopSimulation = useCallback(() => {
    setIsSimulating(false)
  }, [])

  const resetLayout = useCallback(() => {
    previousNodesRef.current.clear()
    if (nodes.length === 0 || width === 0 || height === 0) return

    const newLayout = initializeLayout(nodes, edges, width, height)
    setLayout(newLayout)
    setTransform({ x: 0, y: 0, scale: 1 })

    // Start fresh simulation
    alphaRef.current = 1
    setAlpha(1)
    setIsSimulating(true)
  }, [nodes, edges, width, height])

  const pan = useCallback((dx: number, dy: number) => {
    setTransform((t) => ({
      ...t,
      x: t.x + dx,
      y: t.y + dy,
    }))
  }, [])

  const zoom = useCallback((factor: number, centerX: number, centerY: number) => {
    setTransform((t) => {
      const newScale = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, t.scale * factor))
      const scaleChange = newScale / t.scale

      // Zoom toward cursor position
      return {
        x: centerX - (centerX - t.x) * scaleChange,
        y: centerY - (centerY - t.y) * scaleChange,
        scale: newScale,
      }
    })
  }, [])

  const fixNode = useCallback((nodeId: string, x: number, y: number) => {
    setLayout((currentLayout) => {
      if (!currentLayout) return null
      const node = currentLayout.nodes.get(nodeId)
      if (node) {
        node.fx = x
        node.fy = y
        node.x = x
        node.y = y
        return { ...currentLayout }
      }
      return currentLayout
    })
  }, [])

  const releaseNode = useCallback((nodeId: string) => {
    setLayout((currentLayout) => {
      if (!currentLayout) return null
      const node = currentLayout.nodes.get(nodeId)
      if (node) {
        node.fx = null
        node.fy = null
        return { ...currentLayout }
      }
      return currentLayout
    })
  }, [])

  const reheat = useCallback(() => {
    alphaRef.current = 0.3
    setAlpha(0.3)
    setIsSimulating(true)
  }, [])

  return {
    layout,
    transform,
    isSimulating,
    alpha,
    startSimulation,
    stopSimulation,
    resetLayout,
    setTransform,
    pan,
    zoom,
    fixNode,
    releaseNode,
    reheat,
  }
}
