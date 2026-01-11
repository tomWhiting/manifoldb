import { useState, useCallback, useRef, useEffect, useMemo } from 'react'
import {
  Search,
  Trash2,
  Download,
  ChevronDown,
  ChevronRight,
  Clock,
  AlertCircle,
  CheckCircle,
  Info,
  AlertTriangle,
  X,
  Filter,
  Terminal,
  Plug,
  Database,
} from 'lucide-react'
import { toast } from 'sonner'
import { IconButton } from '../shared/IconButton'
import {
  useLogsStore,
  useFilteredLogs,
  exportLogsAsText,
  type LogEntry,
  type LogTypeFilter,
  type LogLevelFilter,
} from '../../stores/logs-store'

function formatTime(timestamp: number): string {
  const date = new Date(timestamp)
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

function formatDuration(ms: number | undefined): string {
  if (ms === undefined) return ''
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(2)}s`
}

function getLevelIcon(level: LogEntry['level']) {
  switch (level) {
    case 'error':
      return <AlertCircle size={14} className="text-red-400" />
    case 'warn':
      return <AlertTriangle size={14} className="text-yellow-400" />
    case 'success':
      return <CheckCircle size={14} className="text-green-400" />
    case 'info':
    default:
      return <Info size={14} className="text-blue-400" />
  }
}

function getTypeIcon(type: LogEntry['type']) {
  switch (type) {
    case 'query':
      return <Database size={10} />
    case 'connection':
      return <Plug size={10} />
    case 'system':
    default:
      return <Terminal size={10} />
  }
}

function getLevelClass(level: LogEntry['level']): string {
  switch (level) {
    case 'error':
      return 'border-l-red-500'
    case 'warn':
      return 'border-l-yellow-500'
    case 'success':
      return 'border-l-green-500'
    case 'info':
    default:
      return 'border-l-blue-500'
  }
}

interface LogItemProps {
  entry: LogEntry
  expanded: boolean
  onToggleExpand: () => void
}

function LogItem({ entry, expanded, onToggleExpand }: LogItemProps) {
  return (
    <div
      className={`border-b border-border border-l-2 ${getLevelClass(entry.level)} last:border-b-0`}
    >
      <div
        className="flex items-start gap-2 px-3 py-2 hover:bg-bg-tertiary transition-colors cursor-pointer"
        onClick={onToggleExpand}
      >
        {/* Expand toggle */}
        <button className="mt-0.5 p-0.5 text-text-muted hover:text-text-secondary transition-colors">
          {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </button>

        {/* Level icon */}
        <div className="mt-0.5">{getLevelIcon(entry.level)}</div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="text-sm text-text-secondary">{entry.message}</div>

          {/* Metadata row */}
          <div className="flex items-center gap-3 mt-1 text-xs text-text-muted">
            <span className="flex items-center gap-1">
              <Clock size={10} />
              {formatTime(entry.timestamp)}
            </span>
            <span className="flex items-center gap-1 uppercase text-[10px] font-medium bg-bg-tertiary px-1 py-0.5 rounded">
              {getTypeIcon(entry.type)}
              <span className="ml-0.5">{entry.type}</span>
            </span>
            {entry.queryLanguage && (
              <span className="uppercase text-[10px] font-medium bg-bg-tertiary px-1 py-0.5 rounded">
                {entry.queryLanguage}
              </span>
            )}
            {entry.executionTime !== undefined && (
              <span>{formatDuration(entry.executionTime)}</span>
            )}
            {entry.rowCount !== undefined && <span>{entry.rowCount} rows</span>}
          </div>

          {/* Expanded details */}
          {expanded && (
            <div className="mt-2 space-y-2">
              {entry.query && (
                <div className="bg-bg-tertiary rounded p-2">
                  <pre className="text-xs text-text-secondary font-mono whitespace-pre-wrap overflow-x-auto">
                    {entry.query}
                  </pre>
                </div>
              )}
              {entry.details && (
                <div className="text-xs text-text-muted bg-bg-tertiary rounded p-2">
                  {entry.details}
                </div>
              )}
              {entry.stackTrace && (
                <div className="bg-red-500/10 border border-red-500/20 rounded p-2">
                  <pre className="text-xs text-red-400 font-mono whitespace-pre-wrap overflow-x-auto">
                    {entry.stackTrace}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

interface FilterDropdownProps {
  label: string
  value: string
  options: { value: string; label: string }[]
  onChange: (value: string) => void
}

function FilterDropdown({ label, value, options, onChange }: FilterDropdownProps) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-text-muted">{label}:</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="text-xs bg-bg-tertiary border border-border rounded px-2 py-1
          text-text-secondary focus:outline-none focus:ring-1 focus:ring-accent focus:border-accent"
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  )
}

export function LogsPanel() {
  const searchQuery = useLogsStore((s) => s.searchQuery)
  const setSearchQuery = useLogsStore((s) => s.setSearchQuery)
  const typeFilter = useLogsStore((s) => s.typeFilter)
  const setTypeFilter = useLogsStore((s) => s.setTypeFilter)
  const levelFilter = useLogsStore((s) => s.levelFilter)
  const setLevelFilter = useLogsStore((s) => s.setLevelFilter)
  const clearLogs = useLogsStore((s) => s.clearLogs)

  const filteredLogs = useFilteredLogs()

  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set())
  const [showFilters, setShowFilters] = useState(false)
  const searchInputRef = useRef<HTMLInputElement>(null)
  const searchTimeoutRef = useRef<number | undefined>(undefined)
  const listRef = useRef<HTMLDivElement>(null)

  // Debounced search
  const handleSearchChange = useCallback(
    (value: string) => {
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current)
      }
      searchTimeoutRef.current = window.setTimeout(() => {
        setSearchQuery(value)
      }, 150)
    },
    [setSearchQuery]
  )

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current)
      }
    }
  }, [])

  const handleToggleExpand = useCallback((id: string) => {
    setExpandedIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }, [])

  const handleClearSearch = useCallback(() => {
    setSearchQuery('')
    if (searchInputRef.current) {
      searchInputRef.current.value = ''
    }
  }, [setSearchQuery])

  const handleClearLogs = useCallback(() => {
    if (filteredLogs.length === 0) return
    clearLogs()
    toast.success('Logs cleared')
  }, [clearLogs, filteredLogs.length])

  const handleExport = useCallback(() => {
    if (filteredLogs.length === 0) {
      toast.error('No logs to export')
      return
    }

    const text = exportLogsAsText(filteredLogs)
    const blob = new Blob([text], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `logs-${new Date().toISOString().split('T')[0]}.txt`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)

    toast.success('Logs exported')
  }, [filteredLogs])

  const hasActiveFilters = typeFilter !== 'all' || levelFilter !== 'all'

  const typeOptions = useMemo(
    () => [
      { value: 'all', label: 'All Types' },
      { value: 'query', label: 'Query' },
      { value: 'connection', label: 'Connection' },
      { value: 'system', label: 'System' },
    ],
    []
  )

  const levelOptions = useMemo(
    () => [
      { value: 'all', label: 'All Levels' },
      { value: 'error', label: 'Error' },
      { value: 'warn', label: 'Warning' },
      { value: 'success', label: 'Success' },
      { value: 'info', label: 'Info' },
    ],
    []
  )

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <h2 className="text-lg font-semibold text-text-primary">Logs</h2>
        <div className="flex items-center gap-1">
          <IconButton
            icon={<Download size={16} />}
            onClick={handleExport}
            tooltip="Export logs"
            disabled={filteredLogs.length === 0}
          />
          <IconButton
            icon={<Trash2 size={16} />}
            onClick={handleClearLogs}
            tooltip="Clear logs"
            disabled={filteredLogs.length === 0}
          />
        </div>
      </div>

      {/* Search */}
      <div className="px-3 py-2 border-b border-border space-y-2">
        <div className="flex items-center gap-2">
          <div className="relative flex-1">
            <Search
              size={14}
              className="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-muted"
            />
            <input
              ref={searchInputRef}
              type="text"
              placeholder="Search logs..."
              defaultValue={searchQuery}
              onChange={(e) => handleSearchChange(e.target.value)}
              className="w-full pl-8 pr-8 py-1.5 text-sm bg-bg-tertiary border border-border rounded-md
                text-text-primary placeholder:text-text-muted
                focus:outline-none focus:ring-1 focus:ring-accent focus:border-accent"
            />
            {searchQuery && (
              <button
                onClick={handleClearSearch}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-secondary"
              >
                <X size={14} />
              </button>
            )}
          </div>
          <IconButton
            icon={<Filter size={14} />}
            onClick={() => setShowFilters(!showFilters)}
            tooltip="Toggle filters"
            className={hasActiveFilters ? 'text-accent' : undefined}
          />
        </div>

        {/* Filters row */}
        {showFilters && (
          <div className="flex items-center gap-4 pt-1">
            <FilterDropdown
              label="Type"
              value={typeFilter}
              options={typeOptions}
              onChange={(v) => setTypeFilter(v as LogTypeFilter)}
            />
            <FilterDropdown
              label="Level"
              value={levelFilter}
              options={levelOptions}
              onChange={(v) => setLevelFilter(v as LogLevelFilter)}
            />
          </div>
        )}
      </div>

      {/* Log list */}
      <div ref={listRef} className="flex-1 overflow-y-auto">
        {filteredLogs.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-text-muted px-4">
            <Terminal size={32} className="mb-2 opacity-50" />
            <p className="text-sm text-center">
              {searchQuery || hasActiveFilters ? 'No matching logs found' : 'No logs yet'}
            </p>
            {(searchQuery || hasActiveFilters) && (
              <button
                onClick={() => {
                  handleClearSearch()
                  setTypeFilter('all')
                  setLevelFilter('all')
                }}
                className="mt-2 text-xs text-accent hover:underline"
              >
                Clear filters
              </button>
            )}
          </div>
        ) : (
          filteredLogs.map((entry) => (
            <LogItem
              key={entry.id}
              entry={entry}
              expanded={expandedIds.has(entry.id)}
              onToggleExpand={() => handleToggleExpand(entry.id)}
            />
          ))
        )}
      </div>

      {/* Footer with count */}
      {filteredLogs.length > 0 && (
        <div className="px-4 py-2 border-t border-border bg-bg-secondary">
          <p className="text-xs text-text-muted">
            {filteredLogs.length} {filteredLogs.length === 1 ? 'entry' : 'entries'}
            {(searchQuery || hasActiveFilters) && ' matching'}
          </p>
        </div>
      )}
    </div>
  )
}
