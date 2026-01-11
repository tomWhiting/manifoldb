import { useState, useCallback, useMemo, useEffect } from 'react'
import {
  type ProjectionMethod,
  type ProjectionResult,
  type ClusterResult,
  projectVectors,
  calculateSimilarities,
  clusterPoints,
  similarityToColor,
  clusterToColor,
} from '../utils/vector-projection'

export type VectorOverlayMode = 'none' | 'similarity' | 'clusters' | 'projection'

export interface VectorOverlayState {
  enabled: boolean
  mode: VectorOverlayMode
  projectionMethod: ProjectionMethod
  referenceNodeId: string | null
  clusterCount: number
  showClusterBoundaries: boolean
}

export interface VectorData {
  nodeId: string
  vector: number[]
}

export interface UseVectorOverlayOptions {
  nodeVectors: VectorData[]
}

export interface UseVectorOverlayReturn {
  // State
  state: VectorOverlayState
  isComputing: boolean

  // Computed results
  projection: ProjectionResult | null
  clusters: ClusterResult | null
  similarities: Map<string, number> | null

  // Actions
  setEnabled: (enabled: boolean) => void
  setMode: (mode: VectorOverlayMode) => void
  setProjectionMethod: (method: ProjectionMethod) => void
  setReferenceNode: (nodeId: string | null) => void
  setClusterCount: (count: number) => void
  toggleClusterBoundaries: () => void
  recompute: () => void

  // Helpers
  getNodeColor: (nodeId: string, defaultColor: string) => string
  getNodeOpacity: (nodeId: string) => number
}

const DEFAULT_STATE: VectorOverlayState = {
  enabled: false,
  mode: 'similarity',
  projectionMethod: 'pca',
  referenceNodeId: null,
  clusterCount: 5,
  showClusterBoundaries: true,
}

export function useVectorOverlay({
  nodeVectors,
}: UseVectorOverlayOptions): UseVectorOverlayReturn {
  const [state, setState] = useState<VectorOverlayState>(DEFAULT_STATE)
  const [isComputing, setIsComputing] = useState(false)
  const [projection, setProjection] = useState<ProjectionResult | null>(null)
  const [clusters, setClusters] = useState<ClusterResult | null>(null)
  const [similarities, setSimilarities] = useState<Map<string, number> | null>(null)

  // Compute projection when needed
  const computeProjection = useCallback(
    () => {
      if (nodeVectors.length === 0) {
        setProjection(null)
        return null
      }

      setIsComputing(true)

      try {
        const vectors = nodeVectors.map((v) => ({
          id: v.nodeId,
          vector: v.vector,
        }))

        const result = projectVectors(vectors, { method: state.projectionMethod })
        setProjection(result)
        return result
      } finally {
        setIsComputing(false)
      }
    },
    [nodeVectors, state.projectionMethod]
  )

  // Compute clusters when needed
  const computeClusters = useCallback(
    (k: number = state.clusterCount) => {
      if (!projection || projection.points.length === 0) {
        setClusters(null)
        return null
      }

      setIsComputing(true)

      try {
        const result = clusterPoints(projection.points, k)
        setClusters(result)
        return result
      } finally {
        setIsComputing(false)
      }
    },
    [projection, state.clusterCount]
  )

  // Compute similarities when reference node changes
  const computeSimilarities = useCallback(
    (referenceId: string | null) => {
      if (!referenceId || !projection) {
        setSimilarities(null)
        return null
      }

      const refPoint = projection.points.find((p) => p.id === referenceId)
      if (!refPoint?.originalVector) {
        setSimilarities(null)
        return null
      }

      setIsComputing(true)

      try {
        const result = calculateSimilarities(refPoint.originalVector, projection.points)
        setSimilarities(result)
        return result
      } finally {
        setIsComputing(false)
      }
    },
    [projection]
  )

  // Auto-compute projection when vectors change and overlay is enabled
  useEffect(() => {
    if (state.enabled && nodeVectors.length > 0) {
      computeProjection()
    }
  }, [state.enabled, nodeVectors, computeProjection])

  // Auto-compute clusters when mode is clusters and projection exists
  useEffect(() => {
    if (state.enabled && state.mode === 'clusters' && projection) {
      computeClusters()
    }
  }, [state.enabled, state.mode, projection, computeClusters])

  // Auto-compute similarities when mode is similarity and reference changes
  useEffect(() => {
    if (state.enabled && state.mode === 'similarity' && state.referenceNodeId) {
      computeSimilarities(state.referenceNodeId)
    }
  }, [state.enabled, state.mode, state.referenceNodeId, computeSimilarities])

  // Actions
  const setEnabled = useCallback((enabled: boolean) => {
    setState((s) => ({ ...s, enabled }))
  }, [])

  const setMode = useCallback((mode: VectorOverlayMode) => {
    setState((s) => ({ ...s, mode }))
  }, [])

  const setProjectionMethod = useCallback((projectionMethod: ProjectionMethod) => {
    setState((s) => ({ ...s, projectionMethod }))
  }, [])

  const setReferenceNode = useCallback((referenceNodeId: string | null) => {
    setState((s) => ({ ...s, referenceNodeId }))
  }, [])

  const setClusterCount = useCallback((clusterCount: number) => {
    setState((s) => ({ ...s, clusterCount: Math.max(1, Math.min(20, clusterCount)) }))
  }, [])

  const toggleClusterBoundaries = useCallback(() => {
    setState((s) => ({ ...s, showClusterBoundaries: !s.showClusterBoundaries }))
  }, [])

  const recompute = useCallback(() => {
    const proj = computeProjection()
    if (proj && state.mode === 'clusters') {
      computeClusters()
    }
    if (proj && state.mode === 'similarity' && state.referenceNodeId) {
      computeSimilarities(state.referenceNodeId)
    }
  }, [computeProjection, computeClusters, computeSimilarities, state.mode, state.referenceNodeId])

  // Helper to get node color based on current mode
  const getNodeColor = useCallback(
    (nodeId: string, defaultColor: string): string => {
      if (!state.enabled) {
        return defaultColor
      }

      switch (state.mode) {
        case 'similarity':
          if (similarities) {
            const sim = similarities.get(nodeId)
            if (sim !== undefined) {
              return similarityToColor(sim)
            }
          }
          // If no reference node selected or no similarity, use default
          return defaultColor

        case 'clusters':
          if (clusters) {
            const clusterId = clusters.labels.get(nodeId)
            if (clusterId !== undefined) {
              return clusterToColor(clusterId, clusters.clusters.size)
            }
          }
          return defaultColor

        case 'projection':
          // In projection mode, we keep original colors
          return defaultColor

        default:
          return defaultColor
      }
    },
    [state.enabled, state.mode, similarities, clusters]
  )

  // Helper to get node opacity
  const getNodeOpacity = useCallback(
    (nodeId: string): number => {
      if (!state.enabled) {
        return 1
      }

      switch (state.mode) {
        case 'similarity':
          if (similarities) {
            const sim = similarities.get(nodeId)
            if (sim !== undefined) {
              // Higher similarity = higher opacity
              return 0.4 + sim * 0.6
            }
          }
          return 1

        default:
          return 1
      }
    },
    [state.enabled, state.mode, similarities]
  )

  // Memoize the return value
  return useMemo(
    () => ({
      state,
      isComputing,
      projection,
      clusters,
      similarities,
      setEnabled,
      setMode,
      setProjectionMethod,
      setReferenceNode,
      setClusterCount,
      toggleClusterBoundaries,
      recompute,
      getNodeColor,
      getNodeOpacity,
    }),
    [
      state,
      isComputing,
      projection,
      clusters,
      similarities,
      setEnabled,
      setMode,
      setProjectionMethod,
      setReferenceNode,
      setClusterCount,
      toggleClusterBoundaries,
      recompute,
      getNodeColor,
      getNodeOpacity,
    ]
  )
}
