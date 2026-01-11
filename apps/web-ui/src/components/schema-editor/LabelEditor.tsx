import { useState } from 'react'
import { Tag, Trash2, Plus, Search } from 'lucide-react'
import type { SchemaLabel } from './SchemaEditor'

interface LabelEditorProps {
  labels: SchemaLabel[]
  selectedLabel: string | null
  onSelectLabel: (name: string | null) => void
  onCreateLabel: (name: string) => void
  onDeleteLabel: (name: string) => void
  isCreating: boolean
  onStartCreate: () => void
  onCancelCreate: () => void
  isExecuting: boolean
}

export function LabelEditor({
  labels,
  selectedLabel,
  onSelectLabel,
  onCreateLabel,
  onDeleteLabel,
  isCreating,
  onStartCreate,
  onCancelCreate,
  isExecuting,
}: LabelEditorProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const [newLabelName, setNewLabelName] = useState('')
  const [labelToDelete, setLabelToDelete] = useState<string | null>(null)

  const filteredLabels = labels.filter(label =>
    label.name.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const handleCreateLabel = () => {
    if (newLabelName.trim() && !labels.some(l => l.name === newLabelName.trim())) {
      onCreateLabel(newLabelName.trim())
      setNewLabelName('')
    }
  }

  const handleDeleteConfirm = () => {
    if (labelToDelete) {
      onDeleteLabel(labelToDelete)
      setLabelToDelete(null)
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-3 border-b border-border">
        <h3 className="text-sm font-medium text-text-primary flex items-center gap-2">
          <Tag size={16} className="text-accent" />
          Labels
          <span className="text-xs text-text-muted">({labels.length})</span>
        </h3>
      </div>

      {/* Search */}
      <div className="px-4 py-2 border-b border-border">
        <div className="relative">
          <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-muted" />
          <input
            type="text"
            placeholder="Search labels..."
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            className="w-full pl-8 pr-3 py-1.5 text-sm bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent"
          />
        </div>
      </div>

      {/* Create new label inline */}
      {isCreating && (
        <div className="px-4 py-3 border-b border-border bg-bg-tertiary">
          <input
            type="text"
            placeholder="New label name"
            value={newLabelName}
            onChange={e => setNewLabelName(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter') handleCreateLabel()
              if (e.key === 'Escape') onCancelCreate()
            }}
            autoFocus
            className="w-full px-3 py-1.5 text-sm bg-bg-primary border border-border rounded focus:outline-none focus:border-accent mb-2"
          />
          <div className="flex gap-2">
            <button
              onClick={handleCreateLabel}
              disabled={isExecuting || !newLabelName.trim() || labels.some(l => l.name === newLabelName.trim())}
              className="flex-1 px-3 py-1.5 text-xs bg-accent hover:bg-accent-hover text-white rounded disabled:opacity-50"
            >
              Create
            </button>
            <button
              onClick={onCancelCreate}
              className="px-3 py-1.5 text-xs text-text-muted hover:text-text-primary"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Labels list */}
      <div className="flex-1 overflow-y-auto">
        {filteredLabels.length === 0 ? (
          <div className="px-4 py-8 text-center">
            <p className="text-sm text-text-muted">
              {searchQuery ? 'No labels match your search' : 'No labels defined'}
            </p>
          </div>
        ) : (
          filteredLabels.map(label => (
            <div
              key={label.name}
              className={`
                group flex items-center gap-2 px-4 py-2 border-b border-border cursor-pointer
                transition-colors
                ${selectedLabel === label.name
                  ? 'bg-accent-muted'
                  : 'hover:bg-bg-tertiary'
                }
              `}
              onClick={() => onSelectLabel(selectedLabel === label.name ? null : label.name)}
            >
              <Tag
                size={14}
                className={selectedLabel === label.name ? 'text-accent' : 'text-text-muted'}
              />
              <span
                className={`flex-1 text-sm truncate ${
                  selectedLabel === label.name ? 'text-accent font-medium' : 'text-text-primary'
                }`}
              >
                {label.name}
              </span>
              <span className="text-xs text-text-muted">
                {label.count.toLocaleString()}
              </span>
              <button
                onClick={e => {
                  e.stopPropagation()
                  setLabelToDelete(label.name)
                }}
                className="opacity-0 group-hover:opacity-100 p-1 text-text-muted hover:text-red-400 transition-all"
                title="Delete label"
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))
        )}
      </div>

      {/* Add button */}
      {!isCreating && (
        <div className="px-4 py-3 border-t border-border">
          <button
            onClick={() => {
              setNewLabelName('')
              onStartCreate()
            }}
            className="flex items-center gap-2 text-sm text-accent hover:underline"
          >
            <Plus size={14} />
            Add Label
          </button>
        </div>
      )}

      {/* Delete confirmation modal */}
      {labelToDelete && (
        <div
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
          onClick={() => setLabelToDelete(null)}
        >
          <div
            className="bg-bg-secondary border border-border rounded-lg shadow-xl p-6 max-w-md"
            onClick={e => e.stopPropagation()}
          >
            <h3 className="text-lg font-medium text-text-primary mb-2">Delete Label?</h3>
            <p className="text-sm text-text-muted mb-4">
              This will permanently delete all nodes with the label "{labelToDelete}" and their
              relationships. This action cannot be undone.
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setLabelToDelete(null)}
                className="px-4 py-2 text-sm text-text-secondary hover:text-text-primary"
              >
                Cancel
              </button>
              <button
                onClick={handleDeleteConfirm}
                disabled={isExecuting}
                className="px-4 py-2 text-sm bg-red-500 hover:bg-red-600 text-white rounded disabled:opacity-50"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
