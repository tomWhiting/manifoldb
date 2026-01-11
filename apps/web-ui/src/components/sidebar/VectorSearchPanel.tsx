import { useState, useMemo } from 'react'
import {
  Search,
  RefreshCw,
  ChevronDown,
  AlertCircle,
  Trash2,
  Copy,
  Settings2,
} from 'lucide-react'
import { useCollections } from '../../hooks/useCollections'
import { useVectorSearch } from '../../hooks/useVectorSearch'
import { toast } from 'sonner'
import type { VectorSearchResult } from '../../types'

interface SearchResultItemProps {
  result: VectorSearchResult
  rank: number
}

function SearchResultItem({ result, rank }: SearchResultItemProps) {
  const [isExpanded, setIsExpanded] = useState(false)
  const hasPayload = result.payload && Object.keys(result.payload).length > 0

  const handleCopyId = () => {
    navigator.clipboard.writeText(result.id)
    toast.success('ID copied to clipboard')
  }

  return (
    <div className="border-b border-border last:border-b-0">
      <div
        className="flex items-center gap-2 px-3 py-2 hover:bg-bg-tertiary transition-colors cursor-pointer"
        onClick={() => hasPayload && setIsExpanded(!isExpanded)}
      >
        <span className="text-xs text-text-muted w-6 text-right">#{rank}</span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-mono text-text-primary truncate">{result.id}</span>
            <button
              onClick={(e) => {
                e.stopPropagation()
                handleCopyId()
              }}
              className="p-1 hover:bg-bg-tertiary rounded text-text-muted hover:text-text-primary"
              title="Copy ID"
            >
              <Copy size={12} />
            </button>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-accent bg-accent/10 px-2 py-0.5 rounded">
            {(result.score * 100).toFixed(1)}%
          </span>
          {hasPayload && (
            <ChevronDown
              size={14}
              className={`text-text-muted transition-transform ${isExpanded ? 'rotate-180' : ''}`}
            />
          )}
        </div>
      </div>
      {isExpanded && hasPayload && (
        <div className="px-3 pb-2 pl-10">
          <pre className="text-xs text-text-muted bg-bg-tertiary p-2 rounded overflow-auto max-h-32">
            {JSON.stringify(result.payload, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}

export function VectorSearchPanel() {
  const { collections, isLoading: collectionsLoading, refresh: refreshCollections } = useCollections()
  const { results, isLoading: searchLoading, error: searchError, searchByText, clear } = useVectorSearch()

  const [selectedCollection, setSelectedCollection] = useState<string>('')
  const [selectedVectorName, setSelectedVectorName] = useState<string>('')
  const [queryText, setQueryText] = useState<string>('')
  const [limit, setLimit] = useState<number>(10)
  const [scoreThreshold, setScoreThreshold] = useState<string>('')
  const [showAdvanced, setShowAdvanced] = useState(false)

  // Get vector names for selected collection
  const vectorNames = useMemo(() => {
    const collection = collections.find((c) => c.name === selectedCollection)
    return collection?.vectors.map((v) => v.name) ?? []
  }, [collections, selectedCollection])

  // Auto-select first vector name when collection changes
  const handleCollectionChange = (collectionName: string) => {
    setSelectedCollection(collectionName)
    const collection = collections.find((c) => c.name === collectionName)
    if (collection && collection.vectors.length > 0) {
      setSelectedVectorName(collection.vectors[0].name)
    } else {
      setSelectedVectorName('')
    }
    clear()
  }

  const handleSearch = async () => {
    if (!selectedCollection) {
      toast.error('Please select a collection')
      return
    }
    if (!selectedVectorName) {
      toast.error('Please select a vector field')
      return
    }
    if (!queryText.trim()) {
      toast.error('Please enter search text')
      return
    }

    await searchByText(selectedCollection, {
      vectorName: selectedVectorName,
      queryText: queryText.trim(),
      limit,
      scoreThreshold: scoreThreshold ? parseFloat(scoreThreshold) : undefined,
    })
  }

  const selectedCollectionInfo = collections.find((c) => c.name === selectedCollection)
  const selectedVectorConfig = selectedCollectionInfo?.vectors.find(
    (v) => v.name === selectedVectorName
  )

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <h2 className="text-sm font-semibold text-text-primary">Semantic Search</h2>
        <button
          onClick={refreshCollections}
          disabled={collectionsLoading}
          className="p-1.5 rounded hover:bg-bg-tertiary text-text-muted hover:text-text-primary transition-colors disabled:opacity-50"
          title="Refresh collections"
        >
          <RefreshCw size={14} className={collectionsLoading ? 'animate-spin' : ''} />
        </button>
      </div>

      {/* Search Form */}
      <div className="flex-shrink-0 p-4 space-y-4 border-b border-border">
        {/* Collection Select */}
        <div>
          <label className="block text-xs font-medium text-text-muted mb-1.5">Collection</label>
          <select
            value={selectedCollection}
            onChange={(e) => handleCollectionChange(e.target.value)}
            className="w-full px-3 py-2 text-sm bg-bg-tertiary border border-border rounded-md focus:outline-none focus:border-accent text-text-primary"
          >
            <option value="">Select a collection...</option>
            {collections.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name} ({c.pointCount} vectors)
              </option>
            ))}
          </select>
        </div>

        {/* Vector Name Select */}
        {selectedCollection && vectorNames.length > 0 && (
          <div>
            <label className="block text-xs font-medium text-text-muted mb-1.5">Vector Field</label>
            <select
              value={selectedVectorName}
              onChange={(e) => setSelectedVectorName(e.target.value)}
              className="w-full px-3 py-2 text-sm bg-bg-tertiary border border-border rounded-md focus:outline-none focus:border-accent text-text-primary"
            >
              {vectorNames.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
            {selectedVectorConfig && (
              <div className="mt-1 text-xs text-text-muted">
                {selectedVectorConfig.vectorType} • {selectedVectorConfig.dimension ?? '?'} dims •{' '}
                {selectedVectorConfig.distanceMetric}
                {selectedVectorConfig.embeddingModel && (
                  <> • {selectedVectorConfig.embeddingModel}</>
                )}
              </div>
            )}
          </div>
        )}

        {/* Query Text Input */}
        <div>
          <label className="block text-xs font-medium text-text-muted mb-1.5">
            Search Query
          </label>
          <textarea
            value={queryText}
            onChange={(e) => setQueryText(e.target.value)}
            placeholder="Type your search query here..."
            className="w-full px-3 py-2 text-sm bg-bg-tertiary border border-border rounded-md focus:outline-none focus:border-accent text-text-primary resize-none h-20"
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                handleSearch()
              }
            }}
          />
          <div className="mt-1 text-xs text-text-muted">
            Press Enter to search. The query will be automatically embedded.
          </div>
        </div>

        {/* Advanced Options Toggle */}
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center gap-1 text-xs text-text-muted hover:text-text-primary"
        >
          <Settings2 size={12} />
          {showAdvanced ? 'Hide' : 'Show'} advanced options
          <ChevronDown
            size={12}
            className={`transition-transform ${showAdvanced ? 'rotate-180' : ''}`}
          />
        </button>

        {/* Advanced Options */}
        {showAdvanced && (
          <div className="flex gap-3 pt-2">
            <div className="flex-1">
              <label className="block text-xs font-medium text-text-muted mb-1.5">Limit</label>
              <input
                type="number"
                value={limit}
                onChange={(e) => setLimit(parseInt(e.target.value) || 10)}
                min={1}
                max={100}
                className="w-full px-3 py-2 text-sm bg-bg-tertiary border border-border rounded-md focus:outline-none focus:border-accent text-text-primary"
              />
            </div>
            <div className="flex-1">
              <label className="block text-xs font-medium text-text-muted mb-1.5">
                Min Score (optional)
              </label>
              <input
                type="number"
                value={scoreThreshold}
                onChange={(e) => setScoreThreshold(e.target.value)}
                min={0}
                max={1}
                step={0.01}
                placeholder="0.0 - 1.0"
                className="w-full px-3 py-2 text-sm bg-bg-tertiary border border-border rounded-md focus:outline-none focus:border-accent text-text-primary"
              />
            </div>
          </div>
        )}

        {/* Search Button */}
        <button
          onClick={handleSearch}
          disabled={searchLoading || !selectedCollection || !queryText.trim()}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-accent hover:bg-accent-hover disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-medium rounded-md transition-colors"
        >
          {searchLoading ? (
            <>
              <RefreshCw size={16} className="animate-spin" />
              Searching...
            </>
          ) : (
            <>
              <Search size={16} />
              Search
            </>
          )}
        </button>
      </div>

      {/* Results */}
      <div className="flex-1 overflow-hidden flex flex-col">
        {/* Results Header */}
        {(results.length > 0 || searchError) && (
          <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-bg-secondary">
            <span className="text-xs font-medium text-text-muted">
              {searchError ? 'Error' : `${results.length} result${results.length !== 1 ? 's' : ''}`}
            </span>
            {results.length > 0 && (
              <button
                onClick={clear}
                className="text-xs text-text-muted hover:text-text-primary flex items-center gap-1"
              >
                <Trash2 size={12} />
                Clear
              </button>
            )}
          </div>
        )}

        {/* Error Display */}
        {searchError && (
          <div className="p-4">
            <div className="flex items-start gap-2 text-red-400 bg-red-400/10 p-3 rounded-md">
              <AlertCircle size={16} className="flex-shrink-0 mt-0.5" />
              <div className="text-sm">{searchError}</div>
            </div>
          </div>
        )}

        {/* Results List */}
        {results.length > 0 && (
          <div className="flex-1 overflow-y-auto">
            {results.map((result, index) => (
              <SearchResultItem key={result.id} result={result} rank={index + 1} />
            ))}
          </div>
        )}

        {/* Empty State */}
        {!searchError && results.length === 0 && !searchLoading && (
          <div className="flex-1 flex items-center justify-center p-8">
            <div className="text-center">
              <Search size={48} className="mx-auto text-text-muted/30 mb-4" />
              <p className="text-sm text-text-muted">
                {selectedCollection
                  ? 'Type a query to find similar content'
                  : 'Select a collection to begin'}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
