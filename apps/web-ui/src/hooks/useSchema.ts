import { useState, useCallback, useEffect } from 'react'
import { graphqlClient } from '../lib/graphql-client'

export interface LabelInfo {
  name: string
  count: number
}

export interface EdgeTypeInfo {
  name: string
  count: number
}

export interface GraphSchema {
  nodeCount: number
  edgeCount: number
  labels: LabelInfo[]
  edgeTypes: EdgeTypeInfo[]
  lastUpdated: number
}

interface UseSchemaResult {
  schema: GraphSchema | null
  isLoading: boolean
  error: string | null
  refresh: () => Promise<void>
}

const SCHEMA_QUERY = `
  query GetSchema {
    stats {
      nodeCount
      edgeCount
      labels {
        name
        count
      }
      edgeTypes {
        name
        count
      }
    }
  }
`

export function useSchema(): UseSchemaResult {
  const [schema, setSchema] = useState<GraphSchema | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    setIsLoading(true)
    setError(null)

    try {
      const result = await graphqlClient.query(SCHEMA_QUERY, {}).toPromise()

      if (result.error) {
        setError(result.error.message)
        return
      }

      const stats = result.data?.stats
      if (stats) {
        setSchema({
          nodeCount: stats.nodeCount,
          edgeCount: stats.edgeCount,
          labels: stats.labels.map((l: { name: string; count: number }) => ({
            name: l.name,
            count: l.count,
          })),
          edgeTypes: stats.edgeTypes.map((e: { name: string; count: number }) => ({
            name: e.name,
            count: e.count,
          })),
          lastUpdated: Date.now(),
        })
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch schema')
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  return { schema, isLoading, error, refresh }
}
