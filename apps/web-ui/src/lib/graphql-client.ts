import { createClient, cacheExchange, fetchExchange, subscriptionExchange } from 'urql'
import { createClient as createWSClient } from 'graphql-ws'

const GRAPHQL_URL = 'http://localhost:6010/graphql'
const WS_URL = 'ws://localhost:6010/graphql/ws'

const wsClient = createWSClient({
  url: WS_URL,
  connectionParams: {},
  on: {
    connected: () => console.log('[GraphQL WS] Connected'),
    error: (err) => console.error('[GraphQL WS] Error:', err),
    closed: () => console.log('[GraphQL WS] Closed'),
  },
})

export const graphqlClient = createClient({
  url: GRAPHQL_URL,
  exchanges: [
    cacheExchange,
    fetchExchange,
    subscriptionExchange({
      forwardSubscription: (operation) => ({
        subscribe: (sink) => ({
          unsubscribe: wsClient.subscribe(
            { query: operation.query!, variables: operation.variables },
            sink
          ),
        }),
      }),
    }),
  ],
})

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
