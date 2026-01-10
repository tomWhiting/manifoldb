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
import type { ConnectionStatus, ConnectionError } from '../types'

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
  return createWSClient({
    url: wsUrl,
    connectionParams: {},
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
        if (closeEvent?.code === 1000) {
          notifyStatusChange('disconnected')
        }
      },
    },
  })
}

function createUrqlClientInstance(httpUrl: string, ws: WSClient): Client {
  return createClient({
    url: httpUrl,
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
        type
        sourceId
        targetId
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
