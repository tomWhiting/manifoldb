import { useCallback, useRef, useEffect } from 'react'
import { toast } from 'sonner'
import { useAppStore } from '../stores/app-store'
import { useHistoryStore } from '../stores/history-store'
import { executeCypherQuery, executeSqlQuery } from '../lib/graphql-client'
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

function formatError(error: QueryError): string {
  let message = error.message

  if (error.type === 'syntax' && (error.line !== undefined || error.column !== undefined)) {
    const location = [
      error.line !== undefined ? `line ${error.line}` : null,
      error.column !== undefined ? `column ${error.column}` : null,
    ]
      .filter(Boolean)
      .join(', ')

    if (location) {
      message = `${message} (${location})`
    }
  }

  return message
}

interface UseQueryExecutionReturn {
  execute: () => void
  cancel: () => void
  isExecuting: boolean
  detectLanguage: (query: string) => QueryLanguage | null
}

export function useQueryExecution(): UseQueryExecutionReturn {
  const tabs = useAppStore((s) => s.tabs)
  const activeTabId = useAppStore((s) => s.activeTabId)
  const setTabResult = useAppStore((s) => s.setTabResult)
  const setTabExecuting = useAppStore((s) => s.setTabExecuting)
  const addHistoryEntry = useHistoryStore((s) => s.addEntry)

  const abortControllerRef = useRef<AbortController | null>(null)
  const activeTab = tabs.find((t) => t.id === activeTabId)
  const isExecuting = activeTab?.isExecuting ?? false

  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort()
    }
  }, [])

  const execute = useCallback(() => {
    if (!activeTab || !activeTabId) {
      toast.error('No active query tab')
      return
    }

    if (activeTab.isExecuting) {
      return
    }

    const query = activeTab.content.trim()
    if (!query) {
      toast.error('Query is empty')
      return
    }

    abortControllerRef.current?.abort()
    const controller = new AbortController()
    abortControllerRef.current = controller

    setTabExecuting(activeTabId, true)

    // Determine which executor to use based on language
    const language = activeTab.language
    const executeQuery = language === 'sql' ? executeSqlQuery : executeCypherQuery
    const startTime = Date.now()

    executeQuery(query, { signal: controller.signal })
      .then((result) => {
        if (controller.signal.aborted) {
          return
        }

        const executionTime = Date.now() - startTime

        if (result.error) {
          const formattedMessage = formatError(result.error)

          if (result.error.type !== 'cancelled') {
            toast.error('Query failed', {
              description: formattedMessage,
            })

            // Record failed query in history
            addHistoryEntry({
              query,
              language,
              timestamp: Date.now(),
              executionTime,
              status: 'error',
              errorMessage: formattedMessage,
            })
          }
        } else {
          // Record successful query in history
          addHistoryEntry({
            query,
            language,
            timestamp: Date.now(),
            executionTime: result.executionTime ?? executionTime,
            status: 'success',
            rowCount: result.rowCount,
          })
        }

        setTabResult(activeTabId, result)
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setTabExecuting(activeTabId, false)
        }
        if (abortControllerRef.current === controller) {
          abortControllerRef.current = null
        }
      })
  }, [activeTab, activeTabId, setTabResult, setTabExecuting, addHistoryEntry])

  const cancel = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null

      if (activeTabId) {
        setTabExecuting(activeTabId, false)
        setTabResult(activeTabId, {
          executionTime: 0,
          error: { type: 'cancelled', message: 'Query cancelled' },
        })
        toast.info('Query cancelled')
      }
    }
  }, [activeTabId, setTabExecuting, setTabResult])

  return {
    execute,
    cancel,
    isExecuting,
    detectLanguage: detectQueryLanguage,
  }
}
