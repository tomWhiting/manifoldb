import { useState, useCallback, useMemo, useRef, useEffect } from 'react'
import {
  Search,
  Trash2,
  Copy,
  Play,
  ChevronDown,
  ChevronRight,
  Clock,
  AlertCircle,
  CheckCircle,
  X,
} from 'lucide-react'
import { toast } from 'sonner'
import { IconButton } from '../shared/IconButton'
import { useAppStore } from '../../stores/app-store'
import {
  useHistoryStore,
  useFilteredHistory,
  groupEntriesByDate,
  DATE_GROUP_LABELS,
  type GroupedHistoryEntry,
} from '../../stores/history-store'
import type { HistoryEntry } from '../../types'

function formatTime(timestamp: number): string {
  const date = new Date(timestamp)
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function formatDuration(ms: number | undefined): string {
  if (ms === undefined) return '-'
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(2)}s`
}

function truncateQuery(query: string, maxLength: number = 80): string {
  const singleLine = query.replace(/\s+/g, ' ').trim()
  if (singleLine.length <= maxLength) return singleLine
  return singleLine.slice(0, maxLength) + '...'
}

function highlightMatch(text: string, query: string): React.ReactNode {
  if (!query.trim()) return text

  const lowerText = text.toLowerCase()
  const lowerQuery = query.toLowerCase()
  const index = lowerText.indexOf(lowerQuery)

  if (index === -1) return text

  return (
    <>
      {text.slice(0, index)}
      <mark className="bg-yellow-500/30 text-inherit">{text.slice(index, index + query.length)}</mark>
      {text.slice(index + query.length)}
    </>
  )
}

interface HistoryItemProps {
  entry: HistoryEntry
  searchQuery: string
  expanded: boolean
  onToggleExpand: () => void
  onCopy: () => void
  onRerun: () => void
  onDelete: () => void
}

function HistoryItem({
  entry,
  searchQuery,
  expanded,
  onToggleExpand,
  onCopy,
  onRerun,
  onDelete,
}: HistoryItemProps) {
  return (
    <div className="border-b border-border last:border-b-0 group">
      <div className="flex items-start gap-2 px-3 py-2 hover:bg-bg-tertiary transition-colors">
        {/* Expand toggle */}
        <button
          onClick={onToggleExpand}
          className="mt-1 p-0.5 text-text-muted hover:text-text-secondary transition-colors"
        >
          {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </button>

        {/* Status icon */}
        <div className="mt-1">
          {entry.status === 'success' ? (
            <CheckCircle size={14} className="text-green-400" />
          ) : (
            <AlertCircle size={14} className="text-red-400" />
          )}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div
            className="text-sm text-text-secondary font-mono cursor-pointer"
            onDoubleClick={onRerun}
            title="Double-click to re-run"
          >
            {expanded ? (
              <pre className="whitespace-pre-wrap text-xs">{highlightMatch(entry.query, searchQuery)}</pre>
            ) : (
              <span className="truncate block">
                {highlightMatch(truncateQuery(entry.query), searchQuery)}
              </span>
            )}
          </div>

          {/* Metadata */}
          <div className="flex items-center gap-3 mt-1 text-xs text-text-muted">
            <span className="flex items-center gap-1">
              <Clock size={10} />
              {formatTime(entry.timestamp)}
            </span>
            <span>{formatDuration(entry.executionTime)}</span>
            {entry.rowCount !== undefined && (
              <span>{entry.rowCount} rows</span>
            )}
            <span className="uppercase text-[10px] font-medium bg-bg-tertiary px-1 py-0.5 rounded">
              {entry.language}
            </span>
          </div>

          {/* Error message */}
          {entry.status === 'error' && entry.errorMessage && expanded && (
            <div className="mt-2 text-xs text-red-400 bg-red-500/10 px-2 py-1 rounded">
              {entry.errorMessage}
            </div>
          )}
        </div>

        {/* Actions (visible on hover) */}
        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <IconButton
            icon={<Play size={12} />}
            onClick={onRerun}
            tooltip="Re-run query"
            size="sm"
          />
          <IconButton
            icon={<Copy size={12} />}
            onClick={onCopy}
            tooltip="Copy to clipboard"
            size="sm"
          />
          <IconButton
            icon={<Trash2 size={12} />}
            onClick={onDelete}
            tooltip="Delete"
            size="sm"
            className="hover:text-red-400"
          />
        </div>
      </div>
    </div>
  )
}

interface DateGroupProps {
  label: string
  entries: GroupedHistoryEntry[]
  searchQuery: string
  expandedIds: Set<string>
  onToggleExpand: (id: string) => void
  onCopy: (entry: HistoryEntry) => void
  onRerun: (entry: HistoryEntry) => void
  onDelete: (id: string) => void
}

function DateGroup({
  label,
  entries,
  searchQuery,
  expandedIds,
  onToggleExpand,
  onCopy,
  onRerun,
  onDelete,
}: DateGroupProps) {
  const [collapsed, setCollapsed] = useState(false)

  if (entries.length === 0) return null

  return (
    <div className="mb-2">
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="flex items-center gap-2 w-full px-3 py-2 text-left hover:bg-bg-tertiary transition-colors"
      >
        {collapsed ? (
          <ChevronRight size={14} className="text-text-muted" />
        ) : (
          <ChevronDown size={14} className="text-text-muted" />
        )}
        <span className="text-xs font-semibold text-text-muted uppercase tracking-wider">
          {label}
        </span>
        <span className="text-xs text-text-muted">({entries.length})</span>
      </button>
      {!collapsed && (
        <div>
          {entries.map((entry) => (
            <HistoryItem
              key={entry.id}
              entry={entry}
              searchQuery={searchQuery}
              expanded={expandedIds.has(entry.id)}
              onToggleExpand={() => onToggleExpand(entry.id)}
              onCopy={() => onCopy(entry)}
              onRerun={() => onRerun(entry)}
              onDelete={() => onDelete(entry.id)}
            />
          ))}
        </div>
      )}
    </div>
  )
}

export function HistoryPanel() {
  const searchQuery = useHistoryStore((s) => s.searchQuery)
  const setSearchQuery = useHistoryStore((s) => s.setSearchQuery)
  const deleteEntry = useHistoryStore((s) => s.deleteEntry)
  const clearAll = useHistoryStore((s) => s.clearAll)

  const filteredEntries = useFilteredHistory()
  const groupedEntries = useMemo(() => groupEntriesByDate(filteredEntries), [filteredEntries])

  const updateTabContent = useAppStore((s) => s.updateTabContent)
  const activeTabId = useAppStore((s) => s.activeTabId)
  const setTabLanguage = useAppStore((s) => s.setTabLanguage)
  const setSidebarSection = useAppStore((s) => s.setSidebarSection)

  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set())
  const searchInputRef = useRef<HTMLInputElement>(null)
  const searchTimeoutRef = useRef<number | undefined>(undefined)

  // Debounced search
  const handleSearchChange = useCallback((value: string) => {
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current)
    }
    searchTimeoutRef.current = window.setTimeout(() => {
      setSearchQuery(value)
    }, 150)
  }, [setSearchQuery])

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

  const handleCopy = useCallback((entry: HistoryEntry) => {
    navigator.clipboard.writeText(entry.query)
    toast.success('Query copied to clipboard')
  }, [])

  const handleRerun = useCallback((entry: HistoryEntry) => {
    if (activeTabId) {
      updateTabContent(activeTabId, entry.query)
      setTabLanguage(activeTabId, entry.language)
      setSidebarSection('query')
      toast.info('Query loaded - press Cmd+Enter to run')
    }
  }, [activeTabId, updateTabContent, setTabLanguage, setSidebarSection])

  const handleDelete = useCallback((id: string) => {
    deleteEntry(id)
    toast.success('Entry deleted')
  }, [deleteEntry])

  const handleClearAll = useCallback(() => {
    if (filteredEntries.length === 0) return
    clearAll()
    toast.success('History cleared')
  }, [clearAll, filteredEntries.length])

  const handleClearSearch = useCallback(() => {
    setSearchQuery('')
    if (searchInputRef.current) {
      searchInputRef.current.value = ''
    }
  }, [setSearchQuery])

  // Group entries by date
  const todayEntries = groupedEntries.filter((e) => e.group === 'today')
  const yesterdayEntries = groupedEntries.filter((e) => e.group === 'yesterday')
  const thisWeekEntries = groupedEntries.filter((e) => e.group === 'thisWeek')
  const olderEntries = groupedEntries.filter((e) => e.group === 'older')

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <h2 className="text-lg font-semibold text-text-primary">History</h2>
        <IconButton
          icon={<Trash2 size={16} />}
          onClick={handleClearAll}
          tooltip="Clear all history"
          disabled={filteredEntries.length === 0}
        />
      </div>

      {/* Search */}
      <div className="px-3 py-2 border-b border-border">
        <div className="relative">
          <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-muted" />
          <input
            ref={searchInputRef}
            type="text"
            placeholder="Search queries..."
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
      </div>

      {/* Entry list */}
      <div className="flex-1 overflow-y-auto">
        {filteredEntries.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-text-muted px-4">
            <Clock size={32} className="mb-2 opacity-50" />
            <p className="text-sm text-center">
              {searchQuery ? 'No matching queries found' : 'No query history yet'}
            </p>
            {searchQuery && (
              <button
                onClick={handleClearSearch}
                className="mt-2 text-xs text-accent hover:underline"
              >
                Clear search
              </button>
            )}
          </div>
        ) : (
          <>
            <DateGroup
              label={DATE_GROUP_LABELS.today}
              entries={todayEntries}
              searchQuery={searchQuery}
              expandedIds={expandedIds}
              onToggleExpand={handleToggleExpand}
              onCopy={handleCopy}
              onRerun={handleRerun}
              onDelete={handleDelete}
            />
            <DateGroup
              label={DATE_GROUP_LABELS.yesterday}
              entries={yesterdayEntries}
              searchQuery={searchQuery}
              expandedIds={expandedIds}
              onToggleExpand={handleToggleExpand}
              onCopy={handleCopy}
              onRerun={handleRerun}
              onDelete={handleDelete}
            />
            <DateGroup
              label={DATE_GROUP_LABELS.thisWeek}
              entries={thisWeekEntries}
              searchQuery={searchQuery}
              expandedIds={expandedIds}
              onToggleExpand={handleToggleExpand}
              onCopy={handleCopy}
              onRerun={handleRerun}
              onDelete={handleDelete}
            />
            <DateGroup
              label={DATE_GROUP_LABELS.older}
              entries={olderEntries}
              searchQuery={searchQuery}
              expandedIds={expandedIds}
              onToggleExpand={handleToggleExpand}
              onCopy={handleCopy}
              onRerun={handleRerun}
              onDelete={handleDelete}
            />
          </>
        )}
      </div>

      {/* Footer with count */}
      {filteredEntries.length > 0 && (
        <div className="px-4 py-2 border-t border-border bg-bg-secondary">
          <p className="text-xs text-text-muted">
            {filteredEntries.length} {filteredEntries.length === 1 ? 'query' : 'queries'}
            {searchQuery && ' matching'}
          </p>
        </div>
      )}
    </div>
  )
}
