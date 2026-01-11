import { useState, useCallback, useEffect, useRef } from 'react'
import { graphqlClient, subscribeToConnectionStatus } from '../lib/graphql-client'
import type { ConnectionStatus } from '../types'

export interface LabelStats {
  name: string
  count: number
}

export interface EdgeTypeStats {
  name: string
  count: number
}

export interface DatabaseStats {
  nodeCount: number
  edgeCount: number
  labels: LabelStats[]
  edgeTypes: EdgeTypeStats[]
  lastUpdated: number
}

interface UseStatsResult {
  stats: DatabaseStats | null
  isLoading: boolean
  error: string | null
  refresh: () => Promise<void>
}

const STATS_QUERY = `
  query GetStats {
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

const AUTO_REFRESH_INTERVAL = 30000 // 30 seconds

export function useStats(): UseStatsResult {
  const [stats, setStats] = useState<DatabaseStats | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected')
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const refresh = useCallback(async () => {
    setIsLoading(true)
    setError(null)

    try {
      const result = await graphqlClient.query(STATS_QUERY, {}).toPromise()

      if (result.error) {
        setError(result.error.message)
        return
      }

      const data = result.data?.stats
      if (data) {
        setStats({
          nodeCount: data.nodeCount,
          edgeCount: data.edgeCount,
          labels: data.labels.map((l: { name: string; count: number }) => ({
            name: l.name,
            count: l.count,
          })),
          edgeTypes: data.edgeTypes.map((e: { name: string; count: number }) => ({
            name: e.name,
            count: e.count,
          })),
          lastUpdated: Date.now(),
        })
        setError(null)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch stats')
    } finally {
      setIsLoading(false)
    }
  }, [])

  // Subscribe to connection status changes
  useEffect(() => {
    const unsubscribe = subscribeToConnectionStatus((status) => {
      setConnectionStatus(status)
    })
    return unsubscribe
  }, [])

  // Fetch stats on mount and when connected
  useEffect(() => {
    if (connectionStatus === 'connected') {
      refresh()
    }
  }, [connectionStatus, refresh])

  // Set up auto-refresh interval
  useEffect(() => {
    if (connectionStatus === 'connected') {
      intervalRef.current = setInterval(() => {
        refresh()
      }, AUTO_REFRESH_INTERVAL)
    }

    return () => {
      if (intervalRef.current !== null) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
    }
  }, [connectionStatus, refresh])

  return { stats, isLoading, error, refresh }
}
