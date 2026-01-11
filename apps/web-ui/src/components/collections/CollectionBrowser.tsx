import { useState, useCallback } from 'react'
import {
  ArrowLeft,
  RefreshCw,
  Search,
  Database,
  AlertCircle,
  Layers,
  Ruler,
  Hash,
  X,
  ChevronDown,
  ChevronRight,
} from 'lucide-react'
import { useCollectionBrowser } from '../../hooks/useCollections'
import type { VectorSearchResult, VectorConfigInfo } from '../../types'

interface CollectionBrowserProps {
  collectionName: string
  onBack: () => void
}

interface SearchFormProps {
  vectors: VectorConfigInfo[]
  onSearch: (vectorName: string, queryVector: number[], limit: number) => void
  isSearching: boolean
}

function SearchForm({ vectors, onSearch, isSearching }: SearchFormProps) {
  const [selectedVector, setSelectedVector] = useState(vectors[0]?.name ?? '')
  const [queryText, setQueryText] = useState('')
  const [limit, setLimit] = useState(10)
  const [parseError, setParseError] = useState<string | null>(null)

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault()
      setParseError(null)

      if (!queryText.trim()) {
        setParseError('Please enter a query vector')
        return
      }

      try {
        // Try to parse as JSON array first
        let vector: number[]
        const trimmed = queryText.trim()

        if (trimmed.startsWith('[')) {
          vector = JSON.parse(trimmed)
        } else {
          // Try comma-separated or space-separated
          vector = trimmed
            .split(/[,\s]+/)
            .filter((s) => s.length > 0)
            .map((s) => {
              const n = parseFloat(s)
              if (isNaN(n)) throw new Error(`Invalid number: ${s}`)
              return n
            })
        }

        if (!Array.isArray(vector) || vector.length === 0) {
          throw new Error('Vector must be a non-empty array of numbers')
        }

        if (!vector.every((n) => typeof n === 'number' && !isNaN(n))) {
          throw new Error('All elements must be valid numbers')
        }

        onSearch(selectedVector, vector, limit)
      } catch (err) {
        setParseError(err instanceof Error ? err.message : 'Invalid vector format')
      }
    },
    [queryText, selectedVector, limit, onSearch]
  )

  const selectedVectorConfig = vectors.find((v) => v.name === selectedVector)

  return (
    <form onSubmit={handleSubmit} className="p-4 border-b border-border bg-bg-secondary">
      <div className="flex items-center gap-2 mb-3">
        <Search size={16} className="text-text-muted" />
        <span className="text-sm font-medium text-text-primary">Similarity Search</span>
      </div>

      <div className="space-y-3">
        {/* Vector selector */}
        {vectors.length > 1 && (
          <div>
            <label className="block text-xs text-text-muted mb-1">Vector Field</label>
            <select
              value={selectedVector}
              onChange={(e) => setSelectedVector(e.target.value)}
              className="w-full px-3 py-2 bg-bg-primary border border-border rounded text-sm text-text-primary focus:outline-none focus:border-accent"
            >
              {vectors.map((v) => (
                <option key={v.name} value={v.name}>
                  {v.name} ({v.vectorType}, {v.dimension ?? 'sparse'}d)
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Query vector input */}
        <div>
          <label className="block text-xs text-text-muted mb-1">
            Query Vector {selectedVectorConfig?.dimension && `(${selectedVectorConfig.dimension}d)`}
          </label>
          <textarea
            value={queryText}
            onChange={(e) => {
              setQueryText(e.target.value)
              setParseError(null)
            }}
            placeholder="[0.1, 0.2, 0.3, ...] or 0.1 0.2 0.3 ..."
            className="w-full px-3 py-2 bg-bg-primary border border-border rounded text-sm text-text-primary font-mono focus:outline-none focus:border-accent resize-none"
            rows={3}
          />
          {parseError && (
            <p className="text-xs text-red-400 mt-1">{parseError}</p>
          )}
        </div>

        {/* Limit and search button */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <label className="text-xs text-text-muted">Top-K:</label>
            <input
              type="number"
              value={limit}
              onChange={(e) => setLimit(Math.max(1, Math.min(100, parseInt(e.target.value) || 10)))}
              min={1}
              max={100}
              className="w-16 px-2 py-1 bg-bg-primary border border-border rounded text-sm text-text-primary focus:outline-none focus:border-accent"
            />
          </div>
          <button
            type="submit"
            disabled={isSearching}
            className="flex-1 px-4 py-2 bg-accent hover:bg-accent-hover text-white text-sm rounded transition-colors disabled:opacity-50"
          >
            {isSearching ? 'Searching...' : 'Search'}
          </button>
        </div>
      </div>
    </form>
  )
}

interface SearchResultItemProps {
  result: VectorSearchResult
  rank: number
}

function SearchResultItem({ result, rank }: SearchResultItemProps) {
  const [isExpanded, setIsExpanded] = useState(false)
  const hasPayload = result.payload && Object.keys(result.payload).length > 0

  return (
    <div className="border-b border-border last:border-b-0">
      <div
        className="flex items-center gap-2 px-4 py-3 hover:bg-bg-tertiary transition-colors cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        {hasPayload ? (
          isExpanded ? (
            <ChevronDown size={16} className="text-text-muted" />
          ) : (
            <ChevronRight size={16} className="text-text-muted" />
          )
        ) : (
          <div className="w-4" />
        )}
        <span className="text-xs text-text-muted bg-bg-tertiary px-1.5 py-0.5 rounded">
          #{rank}
        </span>
        <span className="text-sm font-mono text-text-secondary flex-1 truncate">
          {result.id}
        </span>
        <span
          className={`text-xs font-medium px-2 py-0.5 rounded ${
            result.score >= 0.9
              ? 'bg-green-500/20 text-green-400'
              : result.score >= 0.7
                ? 'bg-yellow-500/20 text-yellow-400'
                : 'bg-text-muted/20 text-text-muted'
          }`}
        >
          {result.score.toFixed(4)}
        </span>
      </div>

      {isExpanded && hasPayload && (
        <div className="px-6 pb-3">
          <pre className="text-xs bg-bg-tertiary rounded p-3 overflow-x-auto text-text-secondary">
            {JSON.stringify(result.payload, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}

export function CollectionBrowser({ collectionName, onBack }: CollectionBrowserProps) {
  const {
    collection,
    isLoading,
    error,
    refresh,
    searchResults,
    isSearching,
    searchError,
    search,
    clearSearch,
  } = useCollectionBrowser(collectionName)

  const handleSearch = useCallback(
    (vectorName: string, queryVector: number[], limit: number) => {
      search(vectorName, queryVector, { limit })
    },
    [search]
  )

  if (error) {
    return (
      <div className="p-6">
        <div className="flex items-center gap-2 mb-6">
          <button
            onClick={onBack}
            className="p-2 hover:bg-bg-tertiary rounded-md transition-colors"
          >
            <ArrowLeft size={16} className="text-text-muted" />
          </button>
          <h2 className="text-lg font-semibold text-text-primary">{collectionName}</h2>
        </div>
        <div className="flex items-center gap-2 p-4 bg-red-500/10 border border-red-500/20 rounded-md">
          <AlertCircle size={16} className="text-red-400 flex-shrink-0" />
          <p className="text-sm text-red-300">{error}</p>
        </div>
      </div>
    )
  }

  if (isLoading && !collection) {
    return (
      <div className="p-6">
        <div className="flex items-center gap-2 mb-6">
          <button
            onClick={onBack}
            className="p-2 hover:bg-bg-tertiary rounded-md transition-colors"
          >
            <ArrowLeft size={16} className="text-text-muted" />
          </button>
          <h2 className="text-lg font-semibold text-text-primary">{collectionName}</h2>
        </div>
        <div className="flex items-center justify-center py-8">
          <RefreshCw size={24} className="text-text-muted animate-spin" />
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-border">
        <button
          onClick={onBack}
          className="p-2 hover:bg-bg-tertiary rounded-md transition-colors"
          title="Back to collections"
        >
          <ArrowLeft size={16} className="text-text-muted" />
        </button>
        <Database size={16} className="text-text-muted" />
        <h2 className="text-lg font-semibold text-text-primary flex-1 truncate">
          {collectionName}
        </h2>
        <button
          onClick={refresh}
          disabled={isLoading}
          className="p-2 hover:bg-bg-tertiary rounded-md transition-colors disabled:opacity-50"
          title="Refresh collection"
        >
          <RefreshCw size={16} className={`text-text-muted ${isLoading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Collection Stats */}
      {collection && (
        <div className="flex flex-wrap items-center gap-4 px-4 py-3 border-b border-border bg-bg-secondary">
          <div className="flex items-center gap-2">
            <Hash size={14} className="text-text-muted" />
            <span className="text-xs text-text-muted">Points:</span>
            <span className="text-xs font-medium text-text-primary">
              {collection.pointCount.toLocaleString()}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Layers size={14} className="text-text-muted" />
            <span className="text-xs text-text-muted">Vectors:</span>
            <span className="text-xs font-medium text-text-primary">
              {collection.vectors.length}
            </span>
          </div>
          {collection.vectors[0]?.dimension && (
            <div className="flex items-center gap-2">
              <Ruler size={14} className="text-text-muted" />
              <span className="text-xs text-text-muted">Dims:</span>
              <span className="text-xs font-medium text-text-primary">
                {collection.vectors[0].dimension}
              </span>
            </div>
          )}
        </div>
      )}

      {/* Vector configs */}
      {collection && collection.vectors.length > 0 && (
        <div className="px-4 py-2 border-b border-border space-y-1.5">
          {collection.vectors.map((vec) => (
            <div
              key={vec.name}
              className="flex items-center justify-between text-xs bg-bg-tertiary rounded px-2 py-1.5"
            >
              <span className="text-text-secondary font-mono">{vec.name}</span>
              <div className="flex items-center gap-2 text-text-muted">
                <span>{vec.vectorType}</span>
                {vec.dimension && <span>{vec.dimension}d</span>}
                <span className="text-text-muted/60">{vec.distanceMetric}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Search Form */}
      {collection && collection.vectors.length > 0 && (
        <SearchForm
          vectors={collection.vectors}
          onSearch={handleSearch}
          isSearching={isSearching}
        />
      )}

      {/* Search Results or Placeholder */}
      <div className="flex-1 overflow-y-auto">
        {searchError && (
          <div className="m-4 flex items-center gap-2 p-4 bg-red-500/10 border border-red-500/20 rounded-md">
            <AlertCircle size={16} className="text-red-400 flex-shrink-0" />
            <p className="text-sm text-red-300">{searchError}</p>
          </div>
        )}

        {searchResults.length > 0 ? (
          <div>
            <div className="flex items-center justify-between px-4 py-2 bg-bg-secondary border-b border-border">
              <span className="text-xs text-text-muted">
                {searchResults.length} result{searchResults.length !== 1 ? 's' : ''}
              </span>
              <button
                onClick={clearSearch}
                className="flex items-center gap-1 text-xs text-text-muted hover:text-text-primary transition-colors"
              >
                <X size={12} />
                Clear
              </button>
            </div>
            {searchResults.map((result, index) => (
              <SearchResultItem key={result.id} result={result} rank={index + 1} />
            ))}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-12 px-4 text-center">
            <Search size={48} className="text-text-muted/30 mb-4" />
            <p className="text-sm text-text-muted mb-2">Ready to search</p>
            <p className="text-xs text-text-muted/70">
              Enter a query vector above to find similar items
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
