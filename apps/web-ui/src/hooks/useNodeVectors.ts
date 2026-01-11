import { useState, useCallback, useEffect } from 'react'
import { graphqlClient } from '../lib/graphql-client'
import type { VectorData } from './useVectorOverlay'

/**
 * Query to get vector embeddings for nodes from CollectionVectorStore
 */
const NODE_VECTORS_QUERY = `
  query GetNodeVectors($nodeIds: [ID!]!, $collection: String, $vectorName: String) {
    nodeVectors(nodeIds: $nodeIds, collection: $collection, vectorName: $vectorName) {
      nodeId
      collection
      vectorName
      values
      dimension
    }
  }
`

export interface UseNodeVectorsOptions {
  nodeIds: string[]
  /** Optional collection to filter vectors by */
  collection?: string
  /** Optional vector name to filter by */
  vectorName?: string
  enabled?: boolean
}

export interface UseNodeVectorsResult {
  vectors: VectorData[]
  isLoading: boolean
  error: string | null
  refresh: () => Promise<void>
}

interface NodeVectorResponse {
  nodeId: string
  collection: string | null
  vectorName: string | null
  values: number[]
  dimension: number
}

/**
 * Hook to fetch vector embeddings for graph nodes from CollectionVectorStore.
 *
 * Uses the nodeVectors GraphQL query which fetches vectors from the
 * CollectionVectorStore - the single source of truth for vector data.
 */
export function useNodeVectors({
  nodeIds,
  collection,
  vectorName,
  enabled = true,
}: UseNodeVectorsOptions): UseNodeVectorsResult {
  const [vectors, setVectors] = useState<VectorData[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    if (!enabled || nodeIds.length === 0) {
      setVectors([])
      setError(null)
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      const result = await graphqlClient
        .query(NODE_VECTORS_QUERY, { nodeIds, collection, vectorName })
        .toPromise()

      if (result.error) {
        setError(result.error.message)
        setVectors([])
        return
      }

      const data = result.data?.nodeVectors as NodeVectorResponse[] | undefined
      if (data && data.length > 0) {
        // Map server response to VectorData format
        setVectors(data.map((v) => ({
          nodeId: v.nodeId,
          vector: v.values,
        })))
      } else {
        // No vectors found for these nodes (not an error, just empty)
        setVectors([])
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch node vectors'
      setError(message)
      setVectors([])
    } finally {
      setIsLoading(false)
    }
  }, [nodeIds, collection, vectorName, enabled])

  useEffect(() => {
    refresh()
  }, [refresh])

  return { vectors, isLoading, error, refresh }
}
