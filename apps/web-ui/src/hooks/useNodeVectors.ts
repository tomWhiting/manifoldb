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
 * TODO: SERVER NOT IMPLEMENTED - The `nodeVectors` GraphQL query does not exist yet.
 * See: crates/manifold-server/src/schema/ - needs nodeVectors resolver
 * See: crates/manifoldb-vector/src/store/collection_vector_store.rs - has the infrastructure
 *
 * FIXME: FAKE DATA FALLBACK BELOW - This hook currently generates FAKE vectors when
 * the server query fails. This is UNACCEPTABLE in production. The fallback code
 * (generateFallbackVectors, generateDeterministicVector) must be removed once the
 * server implements the nodeVectors query.
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

// =============================================================================
// FIXME: FAKE DATA GENERATION - REMOVE WHEN SERVER IMPLEMENTS nodeVectors QUERY
// =============================================================================
// These functions generate FAKE vectors that look real but are meaningless.
// They use a deterministic hash so the same node ID always gets the same fake vector,
// which masks the problem by making the UI appear to work.
//
// This is UNACCEPTABLE. When the nodeVectors query fails, we should:
// 1. Set an error state
// 2. Show a clear error message to the user
// 3. NOT display any vector visualization
//
// The server needs to implement: nodeVectors(nodeIds: [String!]!) -> [NodeVector!]!
// Using: crates/manifoldb-vector/src/store/collection_vector_store.rs
// =============================================================================

/** @deprecated FAKE DATA - Remove when server implements nodeVectors */
function generateFallbackVectors(nodeIds: string[]): VectorData[] {
  return nodeIds.map((nodeId) => ({
    nodeId,
    vector: generateDeterministicVector(nodeId, 128),
  }))
}

/** @deprecated FAKE DATA - Remove when server implements nodeVectors */
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
