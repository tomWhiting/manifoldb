import { useState } from 'react'
import { ArrowRightLeft, Trash2, Plus, Search } from 'lucide-react'
import type { SchemaLabel, SchemaRelationship } from './SchemaEditor'

interface RelationshipEditorProps {
  relationships: SchemaRelationship[]
  labels: SchemaLabel[]
  selectedRelationship: string | null
  onSelectRelationship: (name: string | null) => void
  onCreateRelationship: (name: string, sourceLabel: string, targetLabel: string) => void
  onDeleteRelationship: (name: string) => void
  isCreating: boolean
  onCancelCreate: () => void
  isExecuting: boolean
}

export function RelationshipEditor({
  relationships,
  labels,
  selectedRelationship,
  onSelectRelationship,
  onCreateRelationship,
  onDeleteRelationship,
  isCreating,
  onCancelCreate,
  isExecuting,
}: RelationshipEditorProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const [newRelName, setNewRelName] = useState('')
  const [newRelSource, setNewRelSource] = useState(labels[0]?.name ?? '')
  const [newRelTarget, setNewRelTarget] = useState(labels[0]?.name ?? '')
  const [relToDelete, setRelToDelete] = useState<string | null>(null)

  const filteredRelationships = relationships.filter(rel =>
    rel.name.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const handleCreateRelationship = () => {
    if (newRelName.trim() && newRelSource && newRelTarget && !relationships.some(r => r.name === newRelName.trim())) {
      onCreateRelationship(newRelName.trim(), newRelSource, newRelTarget)
      setNewRelName('')
    }
  }

  const handleDeleteConfirm = () => {
    if (relToDelete) {
      onDeleteRelationship(relToDelete)
      setRelToDelete(null)
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-3 border-b border-border">
        <h3 className="text-sm font-medium text-text-primary flex items-center gap-2">
          <ArrowRightLeft size={16} className="text-accent" />
          Relationship Types
          <span className="text-xs text-text-muted">({relationships.length})</span>
        </h3>
      </div>

      {/* Search */}
      <div className="px-4 py-2 border-b border-border">
        <div className="relative">
          <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-muted" />
          <input
            type="text"
            placeholder="Search relationships..."
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            className="w-full pl-8 pr-3 py-1.5 text-sm bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent"
          />
        </div>
      </div>

      {/* Create new relationship inline */}
      {isCreating && labels.length > 0 && (
        <div className="px-4 py-3 border-b border-border bg-bg-tertiary">
          <input
            type="text"
            placeholder="Relationship type name"
            value={newRelName}
            onChange={e => setNewRelName(e.target.value)}
            autoFocus
            className="w-full px-3 py-1.5 text-sm bg-bg-primary border border-border rounded focus:outline-none focus:border-accent mb-2"
          />
          <div className="flex items-center gap-2 mb-2">
            <select
              value={newRelSource}
              onChange={e => setNewRelSource(e.target.value)}
              className="flex-1 px-2 py-1.5 text-sm bg-bg-primary border border-border rounded focus:outline-none focus:border-accent"
            >
              {labels.map(l => (
                <option key={l.name} value={l.name}>
                  {l.name}
                </option>
              ))}
            </select>
            <span className="text-text-muted text-sm">→</span>
            <select
              value={newRelTarget}
              onChange={e => setNewRelTarget(e.target.value)}
              className="flex-1 px-2 py-1.5 text-sm bg-bg-primary border border-border rounded focus:outline-none focus:border-accent"
            >
              {labels.map(l => (
                <option key={l.name} value={l.name}>
                  {l.name}
                </option>
              ))}
            </select>
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleCreateRelationship}
              disabled={
                isExecuting ||
                !newRelName.trim() ||
                !newRelSource ||
                !newRelTarget ||
                relationships.some(r => r.name === newRelName.trim())
              }
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

      {isCreating && labels.length === 0 && (
        <div className="px-4 py-3 border-b border-border bg-bg-tertiary">
          <p className="text-sm text-text-muted text-center">
            Create at least one label before adding relationships
          </p>
          <button
            onClick={onCancelCreate}
            className="w-full mt-2 px-3 py-1.5 text-xs text-text-muted hover:text-text-primary"
          >
            Cancel
          </button>
        </div>
      )}

      {/* Relationships list */}
      <div className="flex-1 overflow-y-auto">
        {filteredRelationships.length === 0 ? (
          <div className="px-4 py-8 text-center">
            <p className="text-sm text-text-muted">
              {searchQuery ? 'No relationship types match your search' : 'No relationship types defined'}
            </p>
          </div>
        ) : (
          filteredRelationships.map(rel => (
            <div
              key={rel.name}
              className={`
                group flex items-center gap-2 px-4 py-2 border-b border-border cursor-pointer
                transition-colors
                ${selectedRelationship === rel.name
                  ? 'bg-accent-muted'
                  : 'hover:bg-bg-tertiary'
                }
              `}
              onClick={() =>
                onSelectRelationship(selectedRelationship === rel.name ? null : rel.name)
              }
            >
              <ArrowRightLeft
                size={14}
                className={selectedRelationship === rel.name ? 'text-accent' : 'text-text-muted'}
              />
              <span
                className={`flex-1 text-sm truncate ${
                  selectedRelationship === rel.name ? 'text-accent font-medium' : 'text-text-primary'
                }`}
              >
                {rel.name}
              </span>
              <span className="text-xs text-text-muted">{rel.count.toLocaleString()}</span>
              <button
                onClick={e => {
                  e.stopPropagation()
                  setRelToDelete(rel.name)
                }}
                className="opacity-0 group-hover:opacity-100 p-1 text-text-muted hover:text-red-400 transition-all"
                title="Delete relationship type"
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
              onCancelCreate()
              setNewRelName('')
              setNewRelSource(labels[0]?.name ?? '')
              setNewRelTarget(labels[0]?.name ?? '')
            }}
            disabled={labels.length === 0}
            className="flex items-center gap-2 text-sm text-accent hover:underline disabled:opacity-50 disabled:no-underline"
          >
            <Plus size={14} />
            Add Relationship Type
          </button>
        </div>
      )}

      {/* Delete confirmation modal */}
      {relToDelete && (
        <div
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
          onClick={() => setRelToDelete(null)}
        >
          <div
            className="bg-bg-secondary border border-border rounded-lg shadow-xl p-6 max-w-md"
            onClick={e => e.stopPropagation()}
          >
            <h3 className="text-lg font-medium text-text-primary mb-2">
              Delete Relationship Type?
            </h3>
            <p className="text-sm text-text-muted mb-4">
              This will permanently delete all edges of type "{relToDelete}". This action cannot be
              undone.
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setRelToDelete(null)}
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
