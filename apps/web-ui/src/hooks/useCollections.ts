import { useState, useCallback, useEffect } from 'react'
import { graphqlClient } from '../lib/graphql-client'
import type { CollectionInfo, VectorSearchResult } from '../types'

export interface UseCollectionsResult {
  collections: CollectionInfo[]
  isLoading: boolean
  error: string | null
  refresh: () => Promise<void>
}

export interface UseCollectionBrowserResult {
  collection: CollectionInfo | null
  isLoading: boolean
  error: string | null
  refresh: () => Promise<void>
  searchResults: VectorSearchResult[]
  isSearching: boolean
  searchError: string | null
  search: (vectorName: string, queryVector: number[], options?: SearchOptions) => Promise<void>
  clearSearch: () => void
}

export interface SearchOptions {
  limit?: number
  offset?: number
  scoreThreshold?: number
  withPayload?: boolean
}

const COLLECTIONS_QUERY = `
  query GetCollections {
    collections {
      name
      vectors {
        name
        vectorType
        dimension
        distanceMetric
      }
      pointCount
    }
  }
`

const COLLECTION_QUERY = `
  query GetCollection($name: String!) {
    collection(name: $name) {
      name
      vectors {
        name
        vectorType
        dimension
        distanceMetric
      }
      pointCount
    }
  }
`

const SEARCH_QUERY = `
  query SearchSimilar($collection: String!, $input: VectorSearchInput!) {
    searchSimilar(collection: $collection, input: $input) {
      id
      score
      payload
    }
  }
`

const DELETE_COLLECTION_MUTATION = `
  mutation DeleteCollection($name: String!) {
    deleteCollection(name: $name)
  }
`

interface CollectionResponse {
  name: string
  vectors: {
    name: string
    vectorType: string
    dimension: number | null
    distanceMetric: string
  }[]
  pointCount: number
}

function mapCollection(raw: CollectionResponse): CollectionInfo {
  return {
    name: raw.name,
    vectors: raw.vectors.map((v) => ({
      name: v.name,
      vectorType: v.vectorType as CollectionInfo['vectors'][number]['vectorType'],
      dimension: v.dimension,
      distanceMetric: v.distanceMetric as CollectionInfo['vectors'][number]['distanceMetric'],
    })),
    pointCount: raw.pointCount,
  }
}

export function useCollections(): UseCollectionsResult {
  const [collections, setCollections] = useState<CollectionInfo[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    setIsLoading(true)
    setError(null)

    try {
      const result = await graphqlClient.query(COLLECTIONS_QUERY, {}).toPromise()

      if (result.error) {
        setError(result.error.message)
        return
      }

      const data = result.data?.collections
      if (data) {
        setCollections(data.map(mapCollection))
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch collections')
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  return { collections, isLoading, error, refresh }
}

export function useCollectionBrowser(collectionName: string | null): UseCollectionBrowserResult {
  const [collection, setCollection] = useState<CollectionInfo | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [searchResults, setSearchResults] = useState<VectorSearchResult[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [searchError, setSearchError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    if (!collectionName) {
      setCollection(null)
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      const result = await graphqlClient
        .query(COLLECTION_QUERY, { name: collectionName })
        .toPromise()

      if (result.error) {
        setError(result.error.message)
        return
      }

      const data = result.data?.collection
      if (data) {
        setCollection(mapCollection(data))
      } else {
        setCollection(null)
        setError('Collection not found')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch collection')
    } finally {
      setIsLoading(false)
    }
  }, [collectionName])

  const search = useCallback(
    async (vectorName: string, queryVector: number[], options: SearchOptions = {}) => {
      if (!collectionName) return

      setIsSearching(true)
      setSearchError(null)

      try {
        const input = {
          vectorName,
          queryVector,
          limit: options.limit ?? 10,
          offset: options.offset ?? 0,
          withPayload: options.withPayload ?? true,
          scoreThreshold: options.scoreThreshold,
        }

        const result = await graphqlClient
          .query(SEARCH_QUERY, { collection: collectionName, input })
          .toPromise()

        if (result.error) {
          setSearchError(result.error.message)
          return
        }

        const data = result.data?.searchSimilar
        if (data) {
          setSearchResults(
            data.map((r: { id: string; score: number; payload: unknown }) => ({
              id: r.id,
              score: r.score,
              payload: r.payload as Record<string, unknown> | null,
            }))
          )
        }
      } catch (err) {
        setSearchError(err instanceof Error ? err.message : 'Search failed')
      } finally {
        setIsSearching(false)
      }
    },
    [collectionName]
  )

  const clearSearch = useCallback(() => {
    setSearchResults([])
    setSearchError(null)
  }, [])

  useEffect(() => {
    refresh()
    setSearchResults([])
    setSearchError(null)
  }, [refresh])

  return {
    collection,
    isLoading,
    error,
    refresh,
    searchResults,
    isSearching,
    searchError,
    search,
    clearSearch,
  }
}

export async function deleteCollection(name: string): Promise<boolean> {
  const result = await graphqlClient
    .mutation(DELETE_COLLECTION_MUTATION, { name })
    .toPromise()

  if (result.error) {
    throw new Error(result.error.message)
  }

  return result.data?.deleteCollection ?? false
}
