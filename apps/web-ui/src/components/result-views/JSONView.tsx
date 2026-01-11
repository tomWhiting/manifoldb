import { useCallback, useRef, useEffect, useState } from 'react'
import {
  Search,
  ChevronUp,
  ChevronDown,
  Copy,
  Check,
  Minimize2,
  Maximize2,
  X,
} from 'lucide-react'
import { useAppStore } from '../../stores/app-store'
import { JSONTree } from './JSONTree'
import { useJSONTree, useJSONSearch, type JsonValue } from './useJSONTree'

import type { QueryResult } from '../../types'

interface JSONViewProps {
  result?: QueryResult
}

export function JSONView({ result: propResult }: JSONViewProps = {}) {
  const tabs = useAppStore((s) => s.tabs)
  const activeTabId = useAppStore((s) => s.activeTabId)
  const activeTab = tabs.find((t) => t.id === activeTabId)
  const result = propResult ?? activeTab?.result

  const [showSearch, setShowSearch] = useState(false)
  const [copied, setCopied] = useState(false)
  const searchInputRef = useRef<HTMLInputElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const data = (result?.raw ?? result ?? null) as JsonValue

  const { expandedPaths, togglePath, expandAll, collapseAll } = useJSONTree(data)
  const {
    searchQuery,
    setSearchQuery,
    matchPaths,
    matchCount,
    currentMatchIndex,
    nextMatch,
    prevMatch,
  } = useJSONSearch(data)

  const copyToClipboard = useCallback(async () => {
    try {
      const json = JSON.stringify(data, null, 2)
      await navigator.clipboard.writeText(json)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      // Clipboard API failed
    }
  }, [data])

  const copyPath = useCallback(async (path: string) => {
    try {
      await navigator.clipboard.writeText(path)
    } catch {
      // Clipboard API failed
    }
  }, [])

  const toggleSearch = useCallback(() => {
    setShowSearch((prev) => !prev)
    if (!showSearch) {
      setTimeout(() => searchInputRef.current?.focus(), 0)
    } else {
      setSearchQuery('')
    }
  }, [showSearch, setSearchQuery])

  const handleSearchKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter') {
        if (e.shiftKey) {
          prevMatch()
        } else {
          nextMatch()
        }
      } else if (e.key === 'Escape') {
        setShowSearch(false)
        setSearchQuery('')
      }
    },
    [nextMatch, prevMatch, setSearchQuery]
  )

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'f') {
        e.preventDefault()
        setShowSearch(true)
        setTimeout(() => searchInputRef.current?.focus(), 0)
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [])

  useEffect(() => {
    if (matchPaths.length > 0 && currentMatchIndex < matchPaths.length) {
      const currentPath = matchPaths[currentMatchIndex]
      const element = containerRef.current?.querySelector(`[data-path="${currentPath}"]`)
      if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'center' })
      }
    }
  }, [currentMatchIndex, matchPaths])

  if (!result) {
    return (
      <div className="flex items-center justify-center h-full text-text-muted">
        Run a query to see results
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center gap-1 px-2 py-1.5 border-b border-border bg-bg-secondary/30">
        {/* Search */}
        {showSearch ? (
          <div className="flex items-center gap-1 flex-1 max-w-md">
            <div className="relative flex-1">
              <Search size={14} className="absolute left-2 top-1/2 -translate-y-1/2 text-text-muted" />
              <input
                ref={searchInputRef}
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={handleSearchKeyDown}
                placeholder="Search..."
                className="w-full pl-7 pr-2 py-1 text-xs bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent"
              />
            </div>
            {matchCount > 0 && (
              <span className="text-xs text-text-muted whitespace-nowrap">
                {currentMatchIndex + 1} / {matchCount}
              </span>
            )}
            <button
              onClick={prevMatch}
              disabled={matchCount === 0}
              className="p-1 rounded hover:bg-bg-tertiary disabled:opacity-50"
              title="Previous match (Shift+Enter)"
            >
              <ChevronUp size={14} className="text-text-muted" />
            </button>
            <button
              onClick={nextMatch}
              disabled={matchCount === 0}
              className="p-1 rounded hover:bg-bg-tertiary disabled:opacity-50"
              title="Next match (Enter)"
            >
              <ChevronDown size={14} className="text-text-muted" />
            </button>
            <button
              onClick={toggleSearch}
              className="p-1 rounded hover:bg-bg-tertiary"
              title="Close search (Esc)"
            >
              <X size={14} className="text-text-muted" />
            </button>
          </div>
        ) : (
          <button
            onClick={toggleSearch}
            className="flex items-center gap-1.5 px-2 py-1 rounded text-xs text-text-muted hover:text-text-secondary hover:bg-bg-tertiary"
            title="Search (Cmd+F)"
          >
            <Search size={14} />
            Search
          </button>
        )}

        <div className="flex-1" />

        {/* Expand/Collapse buttons */}
        <button
          onClick={expandAll}
          className="flex items-center gap-1.5 px-2 py-1 rounded text-xs text-text-muted hover:text-text-secondary hover:bg-bg-tertiary"
          title="Expand all"
        >
          <Maximize2 size={14} />
          Expand
        </button>
        <button
          onClick={collapseAll}
          className="flex items-center gap-1.5 px-2 py-1 rounded text-xs text-text-muted hover:text-text-secondary hover:bg-bg-tertiary"
          title="Collapse all"
        >
          <Minimize2 size={14} />
          Collapse
        </button>

        <div className="w-px h-4 bg-border mx-1" />

        {/* Copy button */}
        <button
          onClick={copyToClipboard}
          className="flex items-center gap-1.5 px-2 py-1 rounded text-xs text-text-muted hover:text-text-secondary hover:bg-bg-tertiary"
          title="Copy JSON"
        >
          {copied ? (
            <>
              <Check size={14} className="text-green-500" />
              Copied
            </>
          ) : (
            <>
              <Copy size={14} />
              Copy
            </>
          )}
        </button>
      </div>

      {/* JSON Content */}
      <div ref={containerRef} className="flex-1 overflow-auto px-4">
        <JSONTree
          data={data}
          searchQuery={searchQuery}
          currentMatchIndex={currentMatchIndex}
          matchPaths={matchPaths}
          onCopyPath={copyPath}
          expandedPaths={expandedPaths}
          onTogglePath={togglePath}
        />
      </div>
    </div>
  )
}
