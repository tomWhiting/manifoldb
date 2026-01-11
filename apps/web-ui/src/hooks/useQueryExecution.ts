import { useCallback } from 'react'
import { useAppStore } from '../stores/app-store'
import { graphqlClient, CYPHER_QUERY, SQL_QUERY } from '../lib/graphql-client'
import type { QueryError } from '../types'

type QueryLanguage = 'cypher' | 'sql'

// SQL keywords that indicate SQL syntax
const SQL_KEYWORDS = [
  'SELECT',
  'INSERT',
  'UPDATE',
  'DELETE',
  'CREATE TABLE',
  'DROP TABLE',
  'ALTER TABLE',
  'FROM',
  'WHERE',
  'JOIN',
  'LEFT JOIN',
  'RIGHT JOIN',
  'INNER JOIN',
  'GROUP BY',
  'ORDER BY',
  'HAVING',
  'LIMIT',
  'OFFSET',
  'UNION',
  'VALUES',
]

// Cypher keywords that indicate Cypher syntax
const CYPHER_KEYWORDS = [
  'MATCH',
  'MERGE',
  'CREATE',
  'RETURN',
  'WHERE',
  'WITH',
  'OPTIONAL MATCH',
  'DETACH DELETE',
  'SET',
  'REMOVE',
  'FOREACH',
  'UNWIND',
  'CALL',
]

/**
 * Detect the query language based on content
 * Returns the detected language or null if ambiguous
 */
export function detectQueryLanguage(query: string): QueryLanguage | null {
  const normalizedQuery = query.toUpperCase().trim()

  // Remove comments
  const withoutComments = normalizedQuery
    .replace(/\/\/.*$/gm, '')
    .replace(/--.*$/gm, '')
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .trim()

  if (!withoutComments) {
    return null
  }

  // Check for Cypher-specific patterns
  const hasCypherPattern = CYPHER_KEYWORDS.some((keyword) => {
    const regex = new RegExp(`\\b${keyword}\\b`, 'i')
    return regex.test(withoutComments)
  })

  // Check for SQL-specific patterns (excluding WHERE which is shared)
  const sqlOnlyKeywords = SQL_KEYWORDS.filter((kw) => !['WHERE'].includes(kw))
  const hasSqlPattern = sqlOnlyKeywords.some((keyword) => {
    const regex = new RegExp(`\\b${keyword.replace(/\s+/g, '\\s+')}\\b`, 'i')
    return regex.test(withoutComments)
  })

  // Node pattern like (n), (n:Label), etc. is strong Cypher indicator
  const hasNodePattern = /\([a-zA-Z_][a-zA-Z0-9_]*(?::[a-zA-Z_][a-zA-Z0-9_]*)?\)/.test(withoutComments)

  // Arrow patterns like -[r]-> or --> are strong Cypher indicators
  const hasArrowPattern = /-\[?[a-zA-Z0-9_:]*\]?->/.test(withoutComments) || /<-\[?[a-zA-Z0-9_:]*\]?-/.test(withoutComments)

  if (hasNodePattern || hasArrowPattern || (hasCypherPattern && !hasSqlPattern)) {
    return 'cypher'
  }

  if (hasSqlPattern && !hasCypherPattern) {
    return 'sql'
  }

  // If both patterns are detected or neither, return null (ambiguous)
  return null
}

/**
 * Parse GraphQL error into a structured QueryError
 */
function parseQueryError(error: unknown): QueryError {
  if (error instanceof Error) {
    // Try to extract line/column info from error message
    const lineMatch = error.message.match(/line\s*(\d+)/i)
    const columnMatch = error.message.match(/column\s*(\d+)/i)

    return {
      message: error.message,
      line: lineMatch ? parseInt(lineMatch[1], 10) : undefined,
      column: columnMatch ? parseInt(columnMatch[1], 10) : undefined,
    }
  }

  return {
    message: String(error),
  }
}

interface ExecuteQueryOptions {
  tabId: string
  query: string
  language: QueryLanguage
}

interface UseQueryExecutionResult {
  executeQuery: (options: ExecuteQueryOptions) => Promise<void>
  executeActiveTab: () => Promise<void>
  detectLanguage: (query: string) => QueryLanguage | null
}

export function useQueryExecution(): UseQueryExecutionResult {
  const tabs = useAppStore((s) => s.tabs)
  const activeTabId = useAppStore((s) => s.activeTabId)
  const setTabResult = useAppStore((s) => s.setTabResult)
  const setTabExecuting = useAppStore((s) => s.setTabExecuting)

  const executeQuery = useCallback(
    async ({ tabId, query, language }: ExecuteQueryOptions): Promise<void> => {
      setTabExecuting(tabId, true)
      const startTime = performance.now()

      try {
        if (language === 'sql') {
          // Execute SQL query
          const result = await graphqlClient
            .query(SQL_QUERY, { query })
            .toPromise()

          const executionTime = performance.now() - startTime

          if (result.error) {
            console.error('SQL query error:', result.error)
            setTabResult(tabId, {
              raw: { error: result.error.message },
              executionTime,
              error: parseQueryError(result.error),
            })
          } else {
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

            setTabResult(tabId, {
              rows,
              columns,
              raw: data,
              executionTime,
              rowCount: rows.length,
            })
          }
        } else {
          // Execute Cypher query
          const result = await graphqlClient
            .query(CYPHER_QUERY, { query })
            .toPromise()

          const executionTime = performance.now() - startTime

          if (result.error) {
            console.error('Cypher query error:', result.error)
            setTabResult(tabId, {
              raw: { error: result.error.message },
              executionTime,
              error: parseQueryError(result.error),
            })
          } else {
            const data = result.data?.cypher
            setTabResult(tabId, {
              nodes: data?.nodes ?? [],
              edges: data?.edges ?? [],
              raw: data,
              executionTime,
              rowCount: (data?.nodes?.length ?? 0) + (data?.edges?.length ?? 0),
            })
          }
        }
      } catch (err) {
        console.error('Query failed:', err)
        setTabResult(tabId, {
          raw: { error: String(err) },
          executionTime: performance.now() - startTime,
          error: parseQueryError(err),
        })
      } finally {
        setTabExecuting(tabId, false)
      }
    },
    [setTabResult, setTabExecuting]
  )

  const executeActiveTab = useCallback(async (): Promise<void> => {
    if (!activeTabId) return

    const activeTab = tabs.find((t) => t.id === activeTabId)
    if (!activeTab) return

    await executeQuery({
      tabId: activeTabId,
      query: activeTab.content,
      language: activeTab.language,
    })
  }, [activeTabId, tabs, executeQuery])

  return {
    executeQuery,
    executeActiveTab,
    detectLanguage: detectQueryLanguage,
  }
}
