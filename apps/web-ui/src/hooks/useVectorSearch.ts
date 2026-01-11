import { useState, useCallback } from 'react'
import { graphqlClient } from '../lib/graphql-client'
import type { VectorSearchResult } from '../types'

const SEARCH_BY_TEXT_QUERY = `
  query SearchByText($collection: String!, $input: TextSearchInput!) {
    searchByText(collection: $collection, input: $input) {
      id
      score
      payload
    }
  }
`

const SEARCH_SIMILAR_QUERY = `
  query SearchSimilar($collection: String!, $input: VectorSearchInput!) {
    searchSimilar(collection: $collection, input: $input) {
      id
      score
      payload
    }
  }
`

export interface TextSearchInput {
  vectorName: string
  queryText: string
  model?: string
  limit?: number
  offset?: number
  withPayload?: boolean
  scoreThreshold?: number
}

export interface VectorSearchInput {
  vectorName: string
  queryVector: number[]
  limit?: number
  offset?: number
  withPayload?: boolean
  scoreThreshold?: number
}

export interface UseVectorSearchResult {
  results: VectorSearchResult[]
  isLoading: boolean
  error: string | null
  searchByText: (collection: string, input: TextSearchInput) => Promise<void>
  searchByVector: (collection: string, input: VectorSearchInput) => Promise<void>
  clear: () => void
}

/**
 * Hook to perform vector similarity search on a collection.
 * Supports both text-based search (auto-embeds the query) and raw vector search.
 */
export function useVectorSearch(): UseVectorSearchResult {
  const [results, setResults] = useState<VectorSearchResult[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const searchByText = useCallback(async (collection: string, input: TextSearchInput) => {
    setIsLoading(true)
    setError(null)

    try {
      const result = await graphqlClient
        .query(SEARCH_BY_TEXT_QUERY, {
          collection,
          input: {
            vectorName: input.vectorName,
            queryText: input.queryText,
            model: input.model,
            limit: input.limit ?? 10,
            offset: input.offset ?? 0,
            withPayload: input.withPayload ?? true,
            scoreThreshold: input.scoreThreshold,
          },
        })
        .toPromise()

      if (result.error) {
        setError(result.error.message)
        setResults([])
        return
      }

      const data = result.data?.searchByText as VectorSearchResult[] | undefined
      setResults(data ?? [])
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Search failed'
      setError(message)
      setResults([])
    } finally {
      setIsLoading(false)
    }
  }, [])

  const searchByVector = useCallback(async (collection: string, input: VectorSearchInput) => {
    setIsLoading(true)
    setError(null)

    try {
      const result = await graphqlClient
        .query(SEARCH_SIMILAR_QUERY, {
          collection,
          input: {
            vectorName: input.vectorName,
            queryVector: input.queryVector,
            limit: input.limit ?? 10,
            offset: input.offset ?? 0,
            withPayload: input.withPayload ?? true,
            scoreThreshold: input.scoreThreshold,
          },
        })
        .toPromise()

      if (result.error) {
        setError(result.error.message)
        setResults([])
        return
      }

      const data = result.data?.searchSimilar as VectorSearchResult[] | undefined
      setResults(data ?? [])
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Search failed'
      setError(message)
      setResults([])
    } finally {
      setIsLoading(false)
    }
  }, [])

  const clear = useCallback(() => {
    setResults([])
    setError(null)
  }, [])

  return { results, isLoading, error, searchByText, searchByVector, clear }
}
