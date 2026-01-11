import { create } from 'zustand'

export type LogLevel = 'info' | 'warn' | 'error' | 'success'
export type LogType = 'query' | 'connection' | 'system'

export interface LogEntry {
  id: string
  timestamp: number
  level: LogLevel
  type: LogType
  message: string
  details?: string
  // Query-specific fields
  query?: string
  queryLanguage?: 'cypher' | 'sql'
  executionTime?: number
  rowCount?: number
  // Error-specific fields
  errorType?: string
  stackTrace?: string
}

// Type filter options
export type LogTypeFilter = LogType | 'all'
export type LogLevelFilter = LogLevel | 'all'

const MAX_LOG_ENTRIES = 500

let logIdCounter = 0
const generateLogId = () => `log-${Date.now()}-${++logIdCounter}`

interface LogsState {
  entries: LogEntry[]
  searchQuery: string
  typeFilter: LogTypeFilter
  levelFilter: LogLevelFilter

  // Actions
  addEntry: (entry: Omit<LogEntry, 'id' | 'timestamp'>) => void
  clearLogs: () => void
  setSearchQuery: (query: string) => void
  setTypeFilter: (filter: LogTypeFilter) => void
  setLevelFilter: (filter: LogLevelFilter) => void
}

export const useLogsStore = create<LogsState>((set) => ({
  entries: [],
  searchQuery: '',
  typeFilter: 'all',
  levelFilter: 'all',

  addEntry: (entry) => {
    const newEntry: LogEntry = {
      ...entry,
      id: generateLogId(),
      timestamp: Date.now(),
    }
    set((state) => ({
      entries: [newEntry, ...state.entries].slice(0, MAX_LOG_ENTRIES),
    }))
  },

  clearLogs: () => {
    set({ entries: [] })
  },

  setSearchQuery: (query) => {
    set({ searchQuery: query })
  },

  setTypeFilter: (filter) => {
    set({ typeFilter: filter })
  },

  setLevelFilter: (filter) => {
    set({ levelFilter: filter })
  },
}))

// Selector for filtered entries
export function useFilteredLogs(): LogEntry[] {
  const entries = useLogsStore((s) => s.entries)
  const searchQuery = useLogsStore((s) => s.searchQuery)
  const typeFilter = useLogsStore((s) => s.typeFilter)
  const levelFilter = useLogsStore((s) => s.levelFilter)

  let filtered = entries

  // Apply type filter
  if (typeFilter !== 'all') {
    filtered = filtered.filter((entry) => entry.type === typeFilter)
  }

  // Apply level filter
  if (levelFilter !== 'all') {
    filtered = filtered.filter((entry) => entry.level === levelFilter)
  }

  // Apply search filter
  if (searchQuery.trim()) {
    const lowerQuery = searchQuery.toLowerCase()
    filtered = filtered.filter((entry) => {
      return (
        entry.message.toLowerCase().includes(lowerQuery) ||
        entry.details?.toLowerCase().includes(lowerQuery) ||
        entry.query?.toLowerCase().includes(lowerQuery)
      )
    })
  }

  return filtered
}

// Helper functions for logging common events
export function logQuery(params: {
  query: string
  language: 'cypher' | 'sql'
  executionTime?: number
  rowCount?: number
  error?: {
    type: string
    message: string
    details?: string
  }
}) {
  const store = useLogsStore.getState()

  if (params.error) {
    store.addEntry({
      level: 'error',
      type: 'query',
      message: `Query failed: ${params.error.message}`,
      details: params.error.details,
      query: params.query,
      queryLanguage: params.language,
      executionTime: params.executionTime,
      errorType: params.error.type,
    })
  } else {
    store.addEntry({
      level: 'success',
      type: 'query',
      message: `Query executed successfully`,
      query: params.query,
      queryLanguage: params.language,
      executionTime: params.executionTime,
      rowCount: params.rowCount,
    })
  }
}

export function logConnection(params: {
  status: 'connected' | 'disconnected' | 'error' | 'connecting'
  serverUrl?: string
  error?: string
}) {
  const store = useLogsStore.getState()

  switch (params.status) {
    case 'connected':
      store.addEntry({
        level: 'success',
        type: 'connection',
        message: `Connected to server`,
        details: params.serverUrl,
      })
      break
    case 'disconnected':
      store.addEntry({
        level: 'info',
        type: 'connection',
        message: 'Disconnected from server',
        details: params.serverUrl,
      })
      break
    case 'connecting':
      store.addEntry({
        level: 'info',
        type: 'connection',
        message: 'Connecting to server...',
        details: params.serverUrl,
      })
      break
    case 'error':
      store.addEntry({
        level: 'error',
        type: 'connection',
        message: `Connection error: ${params.error ?? 'Unknown error'}`,
        details: params.serverUrl,
      })
      break
  }
}

export function logSystem(params: {
  level: LogLevel
  message: string
  details?: string
}) {
  const store = useLogsStore.getState()
  store.addEntry({
    level: params.level,
    type: 'system',
    message: params.message,
    details: params.details,
  })
}

// Export logs as text
export function exportLogsAsText(entries: LogEntry[]): string {
  return entries
    .map((entry) => {
      const timestamp = new Date(entry.timestamp).toISOString()
      const level = entry.level.toUpperCase().padEnd(7)
      const type = `[${entry.type}]`.padEnd(12)
      let line = `${timestamp} ${level} ${type} ${entry.message}`

      if (entry.query) {
        line += `\n  Query: ${entry.query.replace(/\n/g, '\n  ')}`
      }
      if (entry.executionTime !== undefined) {
        line += `\n  Execution time: ${entry.executionTime}ms`
      }
      if (entry.rowCount !== undefined) {
        line += `\n  Row count: ${entry.rowCount}`
      }
      if (entry.details) {
        line += `\n  Details: ${entry.details}`
      }
      if (entry.stackTrace) {
        line += `\n  Stack trace:\n    ${entry.stackTrace.replace(/\n/g, '\n    ')}`
      }

      return line
    })
    .join('\n\n')
}
