import { useEffect, useCallback, useRef } from 'react'
import { useMemo } from 'react'
import { Group, Panel, Separator } from 'react-resizable-panels'
import {
  Play,
  Square,
  X,
  Plus,
  Database,
  GitBranch,
  SplitSquareHorizontal,
  SplitSquareVertical,
  XCircle,
} from 'lucide-react'
import CodeMirror from '@uiw/react-codemirror'
import { indentUnit } from '@codemirror/language'
import { EditorState } from '@codemirror/state'
import { EditorView } from '@codemirror/view'
import { toast } from 'sonner'
import { IconButton } from '../shared/IconButton'
import { UnifiedResultView } from '../result-views/UnifiedResultView'
import { useWorkspaceStore } from '../../stores/workspace-store'
import { useHistoryStore } from '../../stores/history-store'
import { useTheme } from '../../hooks/useTheme'
import { useSettingsStore } from '../../stores/settings-store'
import { executeCypherQuery, executeSqlQuery } from '../../lib/graphql-client'
import type { PaneState, QueryError, QueryTab } from '../../types'

interface QueryPaneProps {
  pane: PaneState
  isActive: boolean
}

export function QueryPane({ pane, isActive }: QueryPaneProps) {
  const { isDark } = useTheme()
  const editor = useSettingsStore((s) => s.editor)

  const setActivePane = useWorkspaceStore((s) => s.setActivePane)
  const splitPane = useWorkspaceStore((s) => s.splitPane)
  const closePane = useWorkspaceStore((s) => s.closePane)
  const canSplit = useWorkspaceStore((s) => s.canSplit)
  const layout = useWorkspaceStore((s) => s.layout)
  const paneCount = Object.keys(layout.panes).length

  const addTab = useWorkspaceStore((s) => s.addTab)
  const removeTab = useWorkspaceStore((s) => s.removeTab)
  const setActiveTab = useWorkspaceStore((s) => s.setActiveTab)
  const updateTabContent = useWorkspaceStore((s) => s.updateTabContent)
  const setTabResult = useWorkspaceStore((s) => s.setTabResult)
  const setTabExecuting = useWorkspaceStore((s) => s.setTabExecuting)
  const setTabLanguage = useWorkspaceStore((s) => s.setTabLanguage)

  const addHistoryEntry = useHistoryStore((s) => s.addEntry)

  const activeTab = pane.tabs.find((t) => t.id === pane.activeTabId)
  const isExecuting = activeTab?.isExecuting ?? false
  const abortControllerRef = useRef<AbortController | null>(null)

  // Build extensions based on settings
  const extensions = useMemo(() => {
    const exts = [
      indentUnit.of(' '.repeat(editor.tabSize)),
      EditorState.tabSize.of(editor.tabSize),
    ]

    if (editor.wordWrap) {
      exts.push(EditorView.lineWrapping)
    }

    return exts
  }, [editor.tabSize, editor.wordWrap])

  const formatError = (error: QueryError): string => {
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

  const execute = useCallback(() => {
    if (!activeTab || !pane.activeTabId) {
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

    setTabExecuting(pane.id, pane.activeTabId, true)

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
          addHistoryEntry({
            query,
            language,
            timestamp: Date.now(),
            executionTime: result.executionTime ?? executionTime,
            status: 'success',
            rowCount: result.rowCount,
          })
        }

        if (pane.activeTabId) {
          setTabResult(pane.id, pane.activeTabId, result)
        }
      })
      .finally(() => {
        if (!controller.signal.aborted && pane.activeTabId) {
          setTabExecuting(pane.id, pane.activeTabId, false)
        }
        if (abortControllerRef.current === controller) {
          abortControllerRef.current = null
        }
      })
  }, [activeTab, pane.id, pane.activeTabId, setTabResult, setTabExecuting, addHistoryEntry])

  const cancel = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null

      if (pane.activeTabId) {
        setTabExecuting(pane.id, pane.activeTabId, false)
        setTabResult(pane.id, pane.activeTabId, {
          executionTime: 0,
          error: { type: 'cancelled', message: 'Query cancelled' },
        })
        toast.info('Query cancelled')
      }
    }
  }, [pane.id, pane.activeTabId, setTabExecuting, setTabResult])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort()
    }
  }, [])

  const handleNewTab = () => {
    addTab(pane.id, {
      title: `Query ${pane.tabs.length + 1}`,
      content: '// New query\n',
      language: 'cypher',
    })
  }

  const toggleLanguage = (tabId: string, currentLanguage: 'cypher' | 'sql') => {
    setTabLanguage(pane.id, tabId, currentLanguage === 'cypher' ? 'sql' : 'cypher')
  }

  const handleFocus = () => {
    if (!isActive) {
      setActivePane(pane.id)
    }
  }

  return (
    <div
      className={`flex flex-col h-full w-full ${isActive ? 'ring-1 ring-accent ring-inset' : ''}`}
      onClick={handleFocus}
    >
      {/* Tab bar */}
      <div className="flex items-center border-b border-border bg-bg-secondary">
        <div className="flex items-center gap-1 px-2 py-1 flex-1 overflow-x-auto">
          {pane.tabs.map((tab) => (
            <div
              key={tab.id}
              className={`
                group flex items-center rounded-t text-sm flex-shrink-0
                transition-colors duration-150
                ${
                  pane.activeTabId === tab.id
                    ? 'bg-bg-primary text-text-primary'
                    : 'text-text-muted hover:text-text-secondary hover:bg-bg-tertiary'
                }
              `}
            >
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  toggleLanguage(tab.id, tab.language)
                }}
                title={`Switch to ${tab.language === 'cypher' ? 'SQL' : 'Cypher'}`}
                className={`
                  flex items-center gap-1 px-2 py-1.5 rounded-l
                  hover:bg-bg-tertiary transition-colors
                  ${pane.activeTabId === tab.id ? 'opacity-100' : 'opacity-70 group-hover:opacity-100'}
                `}
              >
                {tab.language === 'sql' ? (
                  <Database size={12} className="text-blue-400" />
                ) : (
                  <GitBranch size={12} className="text-green-400" />
                )}
                <span className="text-[10px] uppercase font-medium">{tab.language}</span>
              </button>

              <button
                onClick={() => setActiveTab(pane.id, tab.id)}
                className="flex items-center gap-2 px-2 py-1.5"
              >
                <span className="truncate max-w-24">{tab.title}</span>
                {pane.tabs.length > 1 && (
                  <span
                    onClick={(e) => {
                      e.stopPropagation()
                      removeTab(pane.id, tab.id)
                    }}
                    className={`
                      p-0.5 rounded hover:bg-bg-tertiary
                      ${pane.activeTabId === tab.id ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'}
                    `}
                  >
                    <X size={12} />
                  </span>
                )}
              </button>
            </div>
          ))}

          <IconButton icon={<Plus size={14} />} onClick={handleNewTab} tooltip="New query" size="sm" />
        </div>

        {/* Action buttons */}
        <div className="flex items-center gap-1 px-2 border-l border-border">
          {canSplit() && (
            <>
              <IconButton
                icon={<SplitSquareVertical size={14} />}
                onClick={() => splitPane(pane.id, 'vertical')}
                tooltip="Split vertical (Cmd+\)"
                size="sm"
              />
              <IconButton
                icon={<SplitSquareHorizontal size={14} />}
                onClick={() => splitPane(pane.id, 'horizontal')}
                tooltip="Split horizontal (Cmd+Shift+\)"
                size="sm"
              />
            </>
          )}

          {paneCount > 1 && (
            <IconButton
              icon={<XCircle size={14} />}
              onClick={() => closePane(pane.id)}
              tooltip="Close pane"
              size="sm"
            />
          )}

          <div className="w-px h-4 bg-border mx-1" />

          {isExecuting ? (
            <IconButton
              icon={<Square size={14} className="fill-current" />}
              onClick={cancel}
              tooltip="Cancel query (Cmd+.)"
              variant="default"
              className="bg-red-600 hover:bg-red-700 text-white"
            />
          ) : (
            <IconButton
              icon={<Play size={16} className="fill-current" />}
              onClick={execute}
              tooltip="Run query (Cmd+Enter)"
              variant="default"
              disabled={!activeTab}
              className="bg-accent hover:bg-accent-hover text-white"
            />
          )}
        </div>
      </div>

      {/* Editor and Results split */}
      <Group orientation="vertical" className="flex-1">
        <Panel id={`${pane.id}-editor`} defaultSize={20} minSize={10}>
          {activeTab ? (
            <CodeMirror
              value={activeTab.content}
              onChange={(value) => updateTabContent(pane.id, activeTab.id, value)}
              height="100%"
              theme={isDark ? 'dark' : 'light'}
              extensions={extensions}
              basicSetup={{
                lineNumbers: editor.lineNumbers,
                highlightActiveLineGutter: true,
                highlightActiveLine: true,
                foldGutter: true,
                dropCursor: true,
                allowMultipleSelections: true,
                indentOnInput: true,
                bracketMatching: true,
                closeBrackets: true,
                autocompletion: editor.autoComplete,
                rectangularSelection: true,
                crosshairCursor: true,
                highlightSelectionMatches: true,
              }}
              className="h-full [&_.cm-editor]:h-full [&_.cm-scroller]:!font-mono"
              style={{ fontSize: `${editor.fontSize}px` }}
            />
          ) : (
            <div className="flex items-center justify-center h-full text-text-muted">
              No query open
            </div>
          )}
        </Panel>
        <Separator className="h-1 bg-border hover:bg-accent transition-colors cursor-row-resize" />
        <Panel id={`${pane.id}-results`} defaultSize={80} minSize={20}>
          <PaneResultView result={activeTab?.result} isExecuting={isExecuting} />
        </Panel>
      </Group>
    </div>
  )
}

interface PaneResultViewProps {
  result: QueryTab['result']
  isExecuting: boolean
}

function PaneResultView({ result, isExecuting }: PaneResultViewProps) {
  // Re-use the unified result view but pass the result from the pane's active tab
  return <UnifiedResultView result={result} isExecuting={isExecuting} />
}
