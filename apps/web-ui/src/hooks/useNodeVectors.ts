import { useState, useCallback, useEffect } from 'react'
import { graphqlClient } from '../lib/graphql-client'
import type { VectorData } from './useVectorOverlay'

/**
 * Query to get vector embeddings for nodes
 * This fetches vectors stored in node properties or linked from a vector collection
 */
const NODE_VECTORS_QUERY = `
  query GetNodeVectors($nodeIds: [String!]!) {
    nodeVectors(nodeIds: $nodeIds) {
      nodeId
      vector
    }
  }
`

export interface UseNodeVectorsOptions {
  nodeIds: string[]
  enabled?: boolean
}

export interface UseNodeVectorsResult {
  vectors: VectorData[]
  isLoading: boolean
  error: string | null
  refresh: () => Promise<void>
}

/**
 * Hook to fetch vector embeddings for graph nodes.
 *
 * Note: This requires a GraphQL endpoint that supports fetching vectors for nodes.
 * The actual implementation depends on how vectors are stored in the database.
 *
 * For demo purposes, this can also use mock data or fallback to generating
 * vectors from node properties.
 */
export function useNodeVectors({
  nodeIds,
  enabled = true,
}: UseNodeVectorsOptions): UseNodeVectorsResult {
  const [vectors, setVectors] = useState<VectorData[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    if (!enabled || nodeIds.length === 0) {
      setVectors([])
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      const result = await graphqlClient
        .query(NODE_VECTORS_QUERY, { nodeIds })
        .toPromise()

      if (result.error) {
        // If the query fails (endpoint not implemented), use fallback
        console.warn('NodeVectors query not available, using fallback')
        setVectors(generateFallbackVectors(nodeIds))
        return
      }

      interface NodeVectorResponse {
        nodeId: string
        vector: number[]
      }

      const data = result.data?.nodeVectors as NodeVectorResponse[] | undefined
      if (data) {
        setVectors(data.map((v) => ({ nodeId: v.nodeId, vector: v.vector })))
      } else {
        setVectors(generateFallbackVectors(nodeIds))
      }
    } catch (err) {
      console.warn('Failed to fetch node vectors, using fallback:', err)
      setVectors(generateFallbackVectors(nodeIds))
    } finally {
      setIsLoading(false)
    }
  }, [nodeIds, enabled])

  useEffect(() => {
    refresh()
  }, [refresh])

  return { vectors, isLoading, error, refresh }
}

/**
 * Generate fallback vectors for demonstration.
 * Creates deterministic pseudo-random vectors based on node IDs.
 * This allows the vector overlay to work even without a vector database.
 */
function generateFallbackVectors(nodeIds: string[]): VectorData[] {
  return nodeIds.map((nodeId) => ({
    nodeId,
    vector: generateDeterministicVector(nodeId, 128),
  }))
}

/**
 * Generate a deterministic pseudo-random vector from a string ID.
 * Uses a simple hash-based approach to create reproducible vectors.
 */
function generateDeterministicVector(id: string, dimension: number): number[] {
  const vector: number[] = []

  // Use a simple hash function to seed the random generation
  let hash = 0
  for (let i = 0; i < id.length; i++) {
    hash = ((hash << 5) - hash) + id.charCodeAt(i)
    hash = hash & hash
  }

  // Generate vector components using the hash as a seed
  for (let i = 0; i < dimension; i++) {
    // Simple LCG-based pseudo-random number generator
    hash = (hash * 1103515245 + 12345) & 0x7fffffff
    // Normalize to [-1, 1] range
    vector.push((hash / 0x7fffffff) * 2 - 1)
  }

  // Normalize the vector to unit length
  const magnitude = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0))
  return vector.map((v) => v / magnitude)
}
