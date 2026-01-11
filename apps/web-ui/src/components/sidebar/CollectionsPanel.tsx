import { useState } from 'react'
import {
  RefreshCw,
  ChevronDown,
  ChevronRight,
  Database,
  Boxes,
  AlertCircle,
  Trash2,
  Search,
  Layers,
  Ruler,
} from 'lucide-react'
import { useCollections, deleteCollection } from '../../hooks/useCollections'
import { CollectionBrowser } from '../collections/CollectionBrowser'
import type { CollectionInfo } from '../../types'
import { toast } from 'sonner'

interface CollectionItemProps {
  collection: CollectionInfo
  isSelected: boolean
  onSelect: () => void
  onDelete: () => void
}

function CollectionItem({ collection, isSelected, onSelect, onDelete }: CollectionItemProps) {
  const [isExpanded, setIsExpanded] = useState(false)
  const [isDeleting, setIsDeleting] = useState(false)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)

  const handleDelete = async (e: React.MouseEvent) => {
    e.stopPropagation()
    if (!showDeleteConfirm) {
      setShowDeleteConfirm(true)
      return
    }

    setIsDeleting(true)
    try {
      await deleteCollection(collection.name)
      toast.success(`Deleted collection "${collection.name}"`)
      onDelete()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to delete collection')
    } finally {
      setIsDeleting(false)
      setShowDeleteConfirm(false)
    }
  }

  const handleCancelDelete = (e: React.MouseEvent) => {
    e.stopPropagation()
    setShowDeleteConfirm(false)
  }

  const totalDimension = collection.vectors.reduce((sum, v) => sum + (v.dimension ?? 0), 0)
  const vectorTypes = [...new Set(collection.vectors.map((v) => v.vectorType))].join(', ')

  return (
    <div
      className={`border-b border-border last:border-b-0 ${isSelected ? 'bg-accent/10' : ''}`}
    >
      <div
        className="flex items-center gap-2 px-4 py-3 hover:bg-bg-tertiary transition-colors cursor-pointer"
        onClick={onSelect}
      >
        <button
          onClick={(e) => {
            e.stopPropagation()
            setIsExpanded(!isExpanded)
          }}
          className="text-text-muted hover:text-text-primary"
        >
          {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
        </button>
        <Database size={16} className="text-text-muted flex-shrink-0" />
        <span className="text-sm font-medium text-text-primary flex-1 truncate">
          {collection.name}
        </span>
        <span className="text-xs text-text-muted bg-bg-tertiary px-2 py-0.5 rounded-full">
          {collection.pointCount.toLocaleString()} pts
        </span>
        {showDeleteConfirm ? (
          <div className="flex items-center gap-1">
            <button
              onClick={handleDelete}
              disabled={isDeleting}
              className="px-2 py-0.5 text-xs bg-red-600 hover:bg-red-700 text-white rounded disabled:opacity-50"
            >
              {isDeleting ? 'Deleting...' : 'Confirm'}
            </button>
            <button
              onClick={handleCancelDelete}
              className="px-2 py-0.5 text-xs bg-bg-tertiary hover:bg-bg-secondary text-text-muted rounded"
            >
              Cancel
            </button>
          </div>
        ) : (
          <button
            onClick={handleDelete}
            className="p-1 hover:bg-bg-tertiary rounded text-text-muted hover:text-red-400 transition-colors opacity-0 group-hover:opacity-100"
            title="Delete collection"
          >
            <Trash2 size={14} />
          </button>
        )}
      </div>

      {isExpanded && (
        <div className="px-6 pb-3 space-y-2">
          <div className="flex items-center gap-4 text-xs text-text-muted">
            <div className="flex items-center gap-1">
              <Layers size={12} />
              <span>{collection.vectors.length} vector(s)</span>
            </div>
            {totalDimension > 0 && (
              <div className="flex items-center gap-1">
                <Ruler size={12} />
                <span>{totalDimension}d total</span>
              </div>
            )}
          </div>

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

          {vectorTypes && (
            <p className="text-xs text-text-muted">Types: {vectorTypes}</p>
          )}
        </div>
      )}
    </div>
  )
}

export function CollectionsPanel() {
  const { collections, isLoading, error, refresh } = useCollections()
  const [selectedCollection, setSelectedCollection] = useState<string | null>(null)

  const handleCollectionDelete = () => {
    if (selectedCollection) {
      setSelectedCollection(null)
    }
    refresh()
  }

  if (selectedCollection) {
    return (
      <CollectionBrowser
        collectionName={selectedCollection}
        onBack={() => setSelectedCollection(null)}
      />
    )
  }

  if (error) {
    return (
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-text-primary">Collections</h2>
          <button
            onClick={refresh}
            disabled={isLoading}
            className="p-2 hover:bg-bg-tertiary rounded-md transition-colors disabled:opacity-50"
            title="Refresh collections"
          >
            <RefreshCw size={16} className={`text-text-muted ${isLoading ? 'animate-spin' : ''}`} />
          </button>
        </div>
        <div className="flex items-center gap-2 p-4 bg-red-500/10 border border-red-500/20 rounded-md">
          <AlertCircle size={16} className="text-red-400 flex-shrink-0" />
          <p className="text-sm text-red-300">{error}</p>
        </div>
      </div>
    )
  }

  if (isLoading && collections.length === 0) {
    return (
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-text-primary">Collections</h2>
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
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <h2 className="text-lg font-semibold text-text-primary">Collections</h2>
        <button
          onClick={refresh}
          disabled={isLoading}
          className="p-2 hover:bg-bg-tertiary rounded-md transition-colors disabled:opacity-50"
          title="Refresh collections"
        >
          <RefreshCw size={16} className={`text-text-muted ${isLoading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Stats Summary */}
      <div className="flex items-center gap-4 px-4 py-3 border-b border-border bg-bg-secondary">
        <div className="flex items-center gap-2">
          <Boxes size={14} className="text-text-muted" />
          <span className="text-xs text-text-muted">Collections:</span>
          <span className="text-xs font-medium text-text-primary">{collections.length}</span>
        </div>
        <div className="flex items-center gap-2">
          <Search size={14} className="text-text-muted" />
          <span className="text-xs text-text-muted">Total Points:</span>
          <span className="text-xs font-medium text-text-primary">
            {collections.reduce((sum, c) => sum + c.pointCount, 0).toLocaleString()}
          </span>
        </div>
      </div>

      {/* Collection List */}
      <div className="flex-1 overflow-y-auto">
        {collections.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 px-4 text-center">
            <Database size={48} className="text-text-muted/30 mb-4" />
            <p className="text-sm text-text-muted mb-2">No collections found</p>
            <p className="text-xs text-text-muted/70">
              Create a collection using the API or CLI to get started
            </p>
          </div>
        ) : (
          <div className="group">
            {collections.map((collection) => (
              <CollectionItem
                key={collection.name}
                collection={collection}
                isSelected={selectedCollection === collection.name}
                onSelect={() => setSelectedCollection(collection.name)}
                onDelete={handleCollectionDelete}
              />
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      {collections.length > 0 && (
        <div className="px-4 py-2 border-t border-border bg-bg-secondary">
          <p className="text-xs text-text-muted">Click a collection to browse or search</p>
        </div>
      )}
    </div>
  )
}
