export type ViewMode = 'graph' | 'table' | 'json'

export type Theme = 'dark' | 'light' | 'system'

export type WorkspaceMode = 'query' | 'sql-builder' | 'schema' | 'collections'

export type SidebarSection =
  | 'query'
  | 'history'
  | 'overview'
  | 'modules'
  | 'collections'
  | 'schema'
  | 'import-export'
  | 'logs'
  | 'settings'
  | 'assistant'

export type ConnectionStatus = 'connected' | 'disconnected' | 'error' | 'connecting'

export interface QueryResult {
  nodes?: GraphNode[]
  edges?: GraphEdge[]
  rows?: Record<string, unknown>[]
  raw?: unknown
  executionTime?: number
  rowCount?: number
}

export interface GraphNode {
  id: string
  labels: string[]
  properties: Record<string, unknown>
}

export interface GraphEdge {
  id: string
  type: string
  sourceId: string
  targetId: string
  properties: Record<string, unknown>
}

export interface QueryTab {
  id: string
  title: string
  content: string
  language: 'cypher' | 'sql'
  result?: QueryResult
  isExecuting?: boolean
}

export interface ServerStats {
  nodeCount: number
  edgeCount: number
  cpuUsage?: number
  memoryUsage?: number
}

export interface ConnectionError {
  code: string
  message: string
  timestamp: number
}

// Settings types
export interface ConnectionSettings {
  serverUrl: string
  connectionTimeout: number // in milliseconds
}

export interface EditorSettings {
  fontSize: number
  tabSize: 2 | 4
  lineNumbers: boolean
  wordWrap: boolean
  autoComplete: boolean
}

export interface QuerySettings {
  defaultLimit: number
  autoExecuteOnLoad: boolean
  historyLimit: number
}

export interface AppSettings {
  connection: ConnectionSettings
  editor: EditorSettings
  theme: Theme
  query: QuerySettings
}
