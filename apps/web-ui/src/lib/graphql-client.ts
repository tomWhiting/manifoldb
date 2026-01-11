import {
  createClient,
  cacheExchange,
  fetchExchange,
  subscriptionExchange,
} from 'urql'
import type { Client } from 'urql'
import { createClient as createWSClient } from 'graphql-ws'
import type { Client as WSClient } from 'graphql-ws'
import { print } from 'graphql'
import type { ConnectionStatus, ConnectionError, QueryError, QueryResult, GraphNode } from '../types'

const DEFAULT_HTTP_URL = 'http://localhost:6010/graphql'
const DEFAULT_WS_URL = 'ws://localhost:6010/graphql/ws'
const STORAGE_KEY_SERVER_URL = 'manifoldb-server-url'

const MIN_RECONNECT_DELAY = 1000
const MAX_RECONNECT_DELAY = 30000

type ConnectionStatusCallback = (status: ConnectionStatus) => void
type ConnectionErrorCallback = (error: ConnectionError) => void

interface ConnectionState {
  status: ConnectionStatus
  reconnectAttempts: number
  reconnectTimeoutId: ReturnType<typeof setTimeout> | null
}

const state: ConnectionState = {
  status: 'disconnected',
  reconnectAttempts: 0,
  reconnectTimeoutId: null,
}

const statusListeners = new Set<ConnectionStatusCallback>()
const errorListeners = new Set<ConnectionErrorCallback>()

function notifyStatusChange(status: ConnectionStatus): void {
  state.status = status
  statusListeners.forEach((callback) => callback(status))
}

function notifyError(error: ConnectionError): void {
  errorListeners.forEach((callback) => callback(error))
}

export function subscribeToConnectionStatus(callback: ConnectionStatusCallback): () => void {
  statusListeners.add(callback)
  callback(state.status)
  return () => statusListeners.delete(callback)
}

export function subscribeToConnectionErrors(callback: ConnectionErrorCallback): () => void {
  errorListeners.add(callback)
  return () => errorListeners.delete(callback)
}

export function getStoredServerUrl(): string {
  try {
    return localStorage.getItem(STORAGE_KEY_SERVER_URL) ?? DEFAULT_HTTP_URL
  } catch {
    return DEFAULT_HTTP_URL
  }
}

export function setStoredServerUrl(url: string): void {
  try {
    localStorage.setItem(STORAGE_KEY_SERVER_URL, url)
  } catch {
    console.warn('[GraphQL] Failed to persist server URL to localStorage')
  }
}

export function validateServerUrl(url: string): boolean {
  try {
    const parsed = new URL(url)
    return parsed.protocol === 'http:' || parsed.protocol === 'https:'
  } catch {
    return false
  }
}

function httpToWsUrl(httpUrl: string): string {
  try {
    const parsed = new URL(httpUrl)
    parsed.protocol = parsed.protocol === 'https:' ? 'wss:' : 'ws:'
    parsed.pathname = parsed.pathname.replace(/\/?$/, '/ws')
    return parsed.toString()
  } catch {
    return DEFAULT_WS_URL
  }
}

function calculateReconnectDelay(attempts: number): number {
  const delay = MIN_RECONNECT_DELAY * Math.pow(2, attempts)
  return Math.min(delay, MAX_RECONNECT_DELAY)
}

let wsClient: WSClient | null = null
let urqlClient: Client | null = null
let currentHttpUrl: string = getStoredServerUrl()
let currentWsUrl: string = httpToWsUrl(currentHttpUrl)

function cleanupWsClient(): void {
  if (state.reconnectTimeoutId !== null) {
    clearTimeout(state.reconnectTimeoutId)
    state.reconnectTimeoutId = null
  }

  if (wsClient !== null) {
    wsClient.dispose()
    wsClient = null
  }
}

function createWsClientInstance(wsUrl: string): WSClient {
  console.log('[GraphQL WS] Creating client for:', wsUrl)
  return createWSClient({
    url: wsUrl,
    connectionParams: {},
    lazy: false, // Force immediate connection
    shouldRetry: () => true,
    retryAttempts: Infinity,
    retryWait: async (retries: number) => {
      const delay = calculateReconnectDelay(retries)
      state.reconnectAttempts = retries + 1
      notifyStatusChange('connecting')
      console.log(`[GraphQL WS] Reconnecting in ${delay}ms (attempt ${retries + 1})`)
      await new Promise((resolve) => {
        state.reconnectTimeoutId = setTimeout(resolve, delay)
      })
    },
    on: {
      connecting: () => {
        console.log('[GraphQL WS] Connecting...')
        notifyStatusChange('connecting')
      },
      connected: () => {
        console.log('[GraphQL WS] Connected')
        state.reconnectAttempts = 0
        notifyStatusChange('connected')
      },
      error: (error: unknown) => {
        console.error('[GraphQL WS] Error:', error)
        const errorObj: ConnectionError = {
          code: 'WS_ERROR',
          message: error instanceof Error ? error.message : 'WebSocket connection error',
          timestamp: Date.now(),
        }
        notifyError(errorObj)
        notifyStatusChange('error')
      },
      closed: (event: unknown) => {
        const closeEvent = event as CloseEvent | undefined
        console.log('[GraphQL WS] Closed', closeEvent?.code, closeEvent?.reason)
        // Always set to disconnected when closed - the library will handle reconnection
        notifyStatusChange('disconnected')
      },
    },
  })
}

function createUrqlClientInstance(httpUrl: string, ws: WSClient): Client {
  // Custom fetch to force POST method (manifold server only accepts POST for GraphQL)
  // urql puts query params in URL for GET, but we need them in body for POST
  const customFetch: typeof fetch = async (input, init) => {
    const inputUrl = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
    const url = new URL(inputUrl)

    // Extract GraphQL params from URL if present (urql's GET-style encoding)
    const query = url.searchParams.get('query')
    const operationName = url.searchParams.get('operationName')
    const variables = url.searchParams.get('variables')

    // Clear URL params - POST should have clean URL
    url.search = ''

    // Build proper POST body
    const parsedVariables = variables ? JSON.parse(variables) : undefined
    const body = JSON.stringify({
      query,
      operationName,
      variables: parsedVariables,
    })

    return fetch(url.toString(), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...init?.headers,
      },
      body,
    })
  }

  return createClient({
    url: httpUrl,
    fetch: customFetch,
    exchanges: [
      cacheExchange,
      fetchExchange,
      subscriptionExchange({
        forwardSubscription: (request) => ({
          subscribe: (sink) => ({
            unsubscribe: ws.subscribe(
              {
                query: typeof request.query === 'string' ? request.query : (request.query ? print(request.query) : ''),
                variables: request.variables as Record<string, unknown> | undefined,
              },
              sink
            ),
          }),
        }),
      }),
    ],
  })
}

export function initializeClient(httpUrl?: string): Client {
  const url = httpUrl ?? getStoredServerUrl()

  if (!validateServerUrl(url)) {
    const errorObj: ConnectionError = {
      code: 'INVALID_URL',
      message: `Invalid server URL: ${url}`,
      timestamp: Date.now(),
    }
    notifyError(errorObj)
    notifyStatusChange('error')
    throw new Error(errorObj.message)
  }

  cleanupWsClient()

  currentHttpUrl = url
  currentWsUrl = httpToWsUrl(url)
  state.reconnectAttempts = 0

  wsClient = createWsClientInstance(currentWsUrl)
  urqlClient = createUrqlClientInstance(currentHttpUrl, wsClient)

  return urqlClient
}

export function getClient(): Client {
  if (urqlClient === null) {
    return initializeClient()
  }
  return urqlClient
}

export function reconnect(httpUrl?: string): Client {
  const url = httpUrl ?? currentHttpUrl
  setStoredServerUrl(url)
  return initializeClient(url)
}

export function disconnect(): void {
  cleanupWsClient()
  notifyStatusChange('disconnected')
}

export function getConnectionStatus(): ConnectionStatus {
  return state.status
}

export function getReconnectAttempts(): number {
  return state.reconnectAttempts
}

export const graphqlClient = initializeClient()

export const CYPHER_QUERY = `
  query ExecuteCypher($query: String!) {
    cypher(query: $query) {
      nodes {
        id
        labels
        properties
      }
      edges {
        id
        edgeType
        source
        target
        properties
      }
    }
  }
`

export const SQL_QUERY = `
  query ExecuteSQL($query: String!) {
    sql(query: $query) {
      columns
      rows
    }
  }
`

export const STATS_QUERY = `
  query GetStats {
    stats {
      nodeCount
      edgeCount
    }
  }
`

export const GRAPH_CHANGES_SUBSCRIPTION = `
  subscription OnGraphChanges {
    graphChanges {
      type
      node {
        id
        labels
        properties
      }
      edge {
        id
        type
        sourceId
        targetId
        properties
      }
    }
  }
`

// Server response edge type (different field names than UI type)
interface ServerEdge {
  id: string
  edgeType: string
  source: string
  target: string
  properties: Record<string, unknown>
}

interface CypherResponse {
  cypher: {
    nodes: GraphNode[]
    edges: ServerEdge[]
  }
}

function parseGraphQLError(error: unknown): QueryError {
  if (error instanceof Error) {
    const message = error.message.toLowerCase()

    if (error.name === 'AbortError' || message.includes('abort')) {
      return { type: 'cancelled', message: 'Query cancelled' }
    }

    if (message.includes('timeout')) {
      return { type: 'timeout', message: 'Query timed out' }
    }

    if (message.includes('network') || message.includes('fetch') || message.includes('failed to fetch')) {
      return { type: 'network', message: 'Network error: Unable to reach server' }
    }

    if (message.includes('syntax') || message.includes('parse')) {
      const lineMatch = error.message.match(/line\s*(\d+)/i)
      const colMatch = error.message.match(/column\s*(\d+)/i)
      return {
        type: 'syntax',
        message: error.message,
        line: lineMatch ? parseInt(lineMatch[1], 10) : undefined,
        column: colMatch ? parseInt(colMatch[1], 10) : undefined,
      }
    }

    return { type: 'execution', message: error.message }
  }

  if (typeof error === 'string') {
    return { type: 'unknown', message: error }
  }

  return { type: 'unknown', message: 'An unknown error occurred' }
}

function parseUrqlError(error: { message: string; graphQLErrors?: Array<{ message: string; extensions?: Record<string, unknown> }> }): QueryError {
  if (error.graphQLErrors && error.graphQLErrors.length > 0) {
    const gqlError = error.graphQLErrors[0]
    const message = gqlError.message.toLowerCase()

    if (message.includes('syntax') || message.includes('parse')) {
      const lineMatch = gqlError.message.match(/line\s*(\d+)/i)
      const colMatch = gqlError.message.match(/column\s*(\d+)/i)
      return {
        type: 'syntax',
        message: gqlError.message,
        line: lineMatch ? parseInt(lineMatch[1], 10) : undefined,
        column: colMatch ? parseInt(colMatch[1], 10) : undefined,
        details: error.graphQLErrors.slice(1).map(e => e.message).join('\n') || undefined,
      }
    }

    return {
      type: 'execution',
      message: gqlError.message,
      details: error.graphQLErrors.slice(1).map(e => e.message).join('\n') || undefined,
    }
  }

  return parseGraphQLError(new Error(error.message))
}

export interface ExecuteQueryOptions {
  signal?: AbortSignal
}

export async function executeCypherQuery(
  query: string,
  options?: ExecuteQueryOptions
): Promise<QueryResult> {
  const startTime = performance.now()
  const client = getClient()

  try {
    if (options?.signal?.aborted) {
      return {
        executionTime: 0,
        error: { type: 'cancelled', message: 'Query cancelled' },
      }
    }

    const resultPromise = client
      .query<CypherResponse>(CYPHER_QUERY, { query })
      .toPromise()

    const abortPromise = options?.signal
      ? new Promise<never>((_, reject) => {
          options.signal?.addEventListener('abort', () => {
            reject(new DOMException('Query cancelled', 'AbortError'))
          })
        })
      : null

    const result = abortPromise
      ? await Promise.race([resultPromise, abortPromise])
      : await resultPromise

    const executionTime = performance.now() - startTime

    if (result.error) {
      return {
        executionTime,
        error: parseUrqlError(result.error),
        raw: { error: result.error },
      }
    }

    const data = result.data?.cypher
    const nodes = data?.nodes ?? []
    // Map server edge fields to UI expected fields
    const edges = (data?.edges ?? []).map((edge) => ({
      id: edge.id,
      type: edge.edgeType,
      sourceId: edge.source,
      targetId: edge.target,
      properties: edge.properties,
    }))

    return {
      nodes,
      edges,
      raw: data,
      executionTime,
      rowCount: nodes.length + edges.length,
    }
  } catch (err) {
    const executionTime = performance.now() - startTime
    return {
      executionTime,
      error: parseGraphQLError(err),
      raw: { error: String(err) },
    }
  }
}

interface SqlResponse {
  sql: {
    columns: string[]
    rows: unknown[][]
  }
}

export async function executeSqlQuery(
  query: string,
  options?: ExecuteQueryOptions
): Promise<QueryResult> {
  const startTime = performance.now()
  const client = getClient()

  try {
    if (options?.signal?.aborted) {
      return {
        executionTime: 0,
        error: { type: 'cancelled', message: 'Query cancelled' },
      }
    }

    const resultPromise = client
      .query<SqlResponse>(SQL_QUERY, { query })
      .toPromise()

    const abortPromise = options?.signal
      ? new Promise<never>((_, reject) => {
          options.signal?.addEventListener('abort', () => {
            reject(new DOMException('Query cancelled', 'AbortError'))
          })
        })
      : null

    const result = abortPromise
      ? await Promise.race([resultPromise, abortPromise])
      : await resultPromise

    const executionTime = performance.now() - startTime

    if (result.error) {
      return {
        executionTime,
        error: parseUrqlError(result.error),
        raw: { error: result.error },
      }
    }

    const data = result.data?.sql
    const columns: string[] = data?.columns ?? []
    const rawRows: unknown[][] = data?.rows ?? []

    // Convert columnar data to row objects for table display
    const rows: Record<string, unknown>[] = rawRows.map((row) => {
      const rowObj: Record<string, unknown> = {}
      columns.forEach((col, i) => {
        rowObj[col] = row[i] ?? null
      })
      return rowObj
    })

    return {
      rows,
      columns,
      raw: data,
      executionTime,
      rowCount: rows.length,
    }
  } catch (err) {
    const executionTime = performance.now() - startTime
    return {
      executionTime,
      error: parseGraphQLError(err),
      raw: { error: String(err) },
    }
  }
}
