import { useCallback, useRef, useEffect } from 'react'
import { toast } from 'sonner'
import { useAppStore } from '../stores/app-store'
import { executeCypherQuery } from '../lib/graphql-client'
import type { QueryError } from '../types'

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
}

export function useQueryExecution(): UseQueryExecutionReturn {
  const tabs = useAppStore((s) => s.tabs)
  const activeTabId = useAppStore((s) => s.activeTabId)
  const setTabResult = useAppStore((s) => s.setTabResult)
  const setTabExecuting = useAppStore((s) => s.setTabExecuting)

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

    executeCypherQuery(query, { signal: controller.signal })
      .then((result) => {
        if (controller.signal.aborted) {
          return
        }

        if (result.error) {
          const formattedMessage = formatError(result.error)

          if (result.error.type !== 'cancelled') {
            toast.error('Query failed', {
              description: formattedMessage,
            })
          }
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
  }, [activeTab, activeTabId, setTabResult, setTabExecuting])

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
  }
}
