import { Loader2, GitBranch, Table, Braces, AlertCircle } from 'lucide-react'
import { useAppStore } from '../../stores/app-store'
import { GraphCanvas } from './GraphCanvas'
import { TableView } from './TableView'
import { JSONView } from './JSONView'
import type { ViewMode, QueryError, QueryResult } from '../../types'

const viewModes: { id: ViewMode; icon: typeof GitBranch; label: string }[] = [
  { id: 'graph', icon: GitBranch, label: 'Graph' },
  { id: 'table', icon: Table, label: 'Table' },
  { id: 'json', icon: Braces, label: 'JSON' },
]

function ErrorDisplay({ error }: { error: QueryError }) {
  const typeLabels: Record<QueryError['type'], string> = {
    syntax: 'Syntax Error',
    execution: 'Execution Error',
    timeout: 'Timeout',
    cancelled: 'Cancelled',
    network: 'Network Error',
    unknown: 'Error',
  }

  const location = [
    error.line !== undefined ? `Line ${error.line}` : null,
    error.column !== undefined ? `Column ${error.column}` : null,
  ]
    .filter(Boolean)
    .join(', ')

  return (
    <div className="flex flex-col items-center justify-center h-full p-8 text-center">
      <div className="flex items-center gap-2 text-red-500 mb-3">
        <AlertCircle size={24} />
        <span className="text-lg font-medium">{typeLabels[error.type]}</span>
      </div>
      <p className="text-text-secondary max-w-md mb-2">{error.message}</p>
      {location && (
        <p className="text-text-muted text-sm">{location}</p>
      )}
      {error.details && (
        <p className="text-text-muted text-sm mt-2 max-w-md">{error.details}</p>
      )}
    </div>
  )
}

interface UnifiedResultViewProps {
  result?: QueryResult
  isExecuting?: boolean
}

export function UnifiedResultView({ result: propResult, isExecuting }: UnifiedResultViewProps = {}) {
  const activeViewMode = useAppStore((s) => s.activeViewMode)
  const setViewMode = useAppStore((s) => s.setViewMode)
  const tabs = useAppStore((s) => s.tabs)
  const activeTabId = useAppStore((s) => s.activeTabId)
  const activeTab = tabs.find((t) => t.id === activeTabId)

  // Use prop result if provided, otherwise fall back to app store
  const result = propResult ?? activeTab?.result
  const error = result?.error

  return (
    <div className="flex flex-col h-full">
      {/* View mode switcher */}
      <div className="flex items-center gap-1 px-2 py-1.5 border-b border-border bg-bg-secondary/50">
        {viewModes.map((mode) => {
          const Icon = mode.icon
          const isActive = activeViewMode === mode.id
          return (
            <button
              key={mode.id}
              onClick={() => setViewMode(mode.id)}
              className={`
                flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium
                transition-colors duration-150
                ${
                  isActive
                    ? 'bg-accent-muted text-accent'
                    : 'text-text-muted hover:text-text-secondary hover:bg-bg-tertiary'
                }
              `}
            >
              <Icon size={14} />
              {mode.label}
            </button>
          )
        })}
        {isExecuting && (
          <div className="flex items-center gap-1.5 ml-auto text-xs text-text-muted">
            <Loader2 size={14} className="animate-spin" />
            Executing...
          </div>
        )}
      </div>

      {/* View content */}
      <div className="flex-1 min-h-0">
        {error && error.type !== 'cancelled' ? (
          <ErrorDisplay error={error} />
        ) : (
          <>
            {activeViewMode === 'graph' && <GraphCanvas result={result} />}
            {activeViewMode === 'table' && <TableView result={result} />}
            {activeViewMode === 'json' && <JSONView result={result} />}
          </>
        )}
      </div>
    </div>
  )
}
