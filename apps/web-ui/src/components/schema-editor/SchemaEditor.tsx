import { useState, useCallback } from 'react'
import { Group, Panel, Separator } from 'react-resizable-panels'
import {
  RefreshCw,
  Plus,
  Database,
  Tag,
  ArrowRightLeft,
  Key,
  ListTree,
} from 'lucide-react'
import { toast } from 'sonner'
import { SchemaDiagram } from './SchemaDiagram'
import { LabelEditor } from './LabelEditor'
import { RelationshipEditor } from './RelationshipEditor'
import { IconButton } from '../shared/IconButton'
import { CollapsibleSection } from '../shared/CollapsibleSection'
import { useSchema, type LabelInfo, type EdgeTypeInfo } from '../../hooks/useSchema'
import { executeCypherQuery } from '../../lib/graphql-client'

export interface SchemaLabel {
  name: string
  count: number
  properties: SchemaProperty[]
}

export interface SchemaProperty {
  name: string
  type: 'string' | 'number' | 'boolean' | 'date' | 'array' | 'object' | 'unknown'
  nullable: boolean
}

export interface SchemaRelationship {
  name: string
  count: number
  sourceLabels: string[]
  targetLabels: string[]
  properties: SchemaProperty[]
}

export interface SchemaConstraint {
  name: string
  type: 'unique' | 'exists' | 'node_key'
  label: string
  properties: string[]
}

export interface SchemaIndex {
  name: string
  label: string
  properties: string[]
  type: 'btree' | 'fulltext' | 'vector'
}

type EditorMode = 'diagram' | 'labels' | 'relationships' | 'constraints'

function escapeCypherIdentifier(name: string): string {
  if (/[^a-zA-Z0-9_]/.test(name)) {
    return '`' + name.replace(/`/g, '``') + '`'
  }
  return name
}

export function SchemaEditor() {
  const { schema, isLoading, error, refresh } = useSchema()
  const [activeMode, setActiveMode] = useState<EditorMode>('diagram')
  const [selectedLabel, setSelectedLabel] = useState<string | null>(null)
  const [selectedRelationship, setSelectedRelationship] = useState<string | null>(null)
  const [isCreatingLabel, setIsCreatingLabel] = useState(false)
  const [isCreatingRelationship, setIsCreatingRelationship] = useState(false)
  const [constraints, setConstraints] = useState<SchemaConstraint[]>([])
  const [indexes, setIndexes] = useState<SchemaIndex[]>([])
  const [isExecuting, setIsExecuting] = useState(false)

  // Convert schema labels to SchemaLabel format
  const labels: SchemaLabel[] = (schema?.labels ?? []).map((l: LabelInfo) => ({
    name: l.name,
    count: l.count,
    properties: [],
  }))

  // Convert schema edge types to SchemaRelationship format
  const relationships: SchemaRelationship[] = (schema?.edgeTypes ?? []).map((e: EdgeTypeInfo) => ({
    name: e.name,
    count: e.count,
    sourceLabels: [],
    targetLabels: [],
    properties: [],
  }))

  const handleCreateLabel = useCallback(async (name: string) => {
    setIsExecuting(true)
    try {
      const escapedName = escapeCypherIdentifier(name)
      const query = `CREATE (n:${escapedName}) DELETE n`
      const result = await executeCypherQuery(query)

      if (result.error) {
        toast.error(`Failed to create label: ${result.error.message}`)
      } else {
        toast.success(`Label "${name}" created`)
        await refresh()
        setIsCreatingLabel(false)
      }
    } catch {
      toast.error('Failed to create label')
    } finally {
      setIsExecuting(false)
    }
  }, [refresh])

  const handleDeleteLabel = useCallback(async (name: string) => {
    setIsExecuting(true)
    try {
      const escapedName = escapeCypherIdentifier(name)
      const query = `MATCH (n:${escapedName}) DETACH DELETE n`
      const result = await executeCypherQuery(query)

      if (result.error) {
        toast.error(`Failed to delete label: ${result.error.message}`)
      } else {
        toast.success(`Label "${name}" and all its nodes deleted`)
        await refresh()
        setSelectedLabel(null)
      }
    } catch {
      toast.error('Failed to delete label')
    } finally {
      setIsExecuting(false)
    }
  }, [refresh])

  const handleCreateRelationship = useCallback(async (
    name: string,
    sourceLabel: string,
    targetLabel: string
  ) => {
    setIsExecuting(true)
    try {
      const escapedName = escapeCypherIdentifier(name)
      const escapedSource = escapeCypherIdentifier(sourceLabel)
      const escapedTarget = escapeCypherIdentifier(targetLabel)

      // Create a temporary relationship to register the type
      const query = `
        MATCH (a:${escapedSource}), (b:${escapedTarget})
        WITH a, b LIMIT 1
        CREATE (a)-[r:${escapedName}]->(b)
        DELETE r
      `
      const result = await executeCypherQuery(query)

      if (result.error) {
        toast.error(`Failed to create relationship: ${result.error.message}`)
      } else {
        toast.success(`Relationship type "${name}" created`)
        await refresh()
        setIsCreatingRelationship(false)
      }
    } catch {
      toast.error('Failed to create relationship')
    } finally {
      setIsExecuting(false)
    }
  }, [refresh])

  const handleDeleteRelationship = useCallback(async (name: string) => {
    setIsExecuting(true)
    try {
      const escapedName = escapeCypherIdentifier(name)
      const query = `MATCH ()-[r:${escapedName}]->() DELETE r`
      const result = await executeCypherQuery(query)

      if (result.error) {
        toast.error(`Failed to delete relationship type: ${result.error.message}`)
      } else {
        toast.success(`Relationship type "${name}" and all its edges deleted`)
        await refresh()
        setSelectedRelationship(null)
      }
    } catch {
      toast.error('Failed to delete relationship type')
    } finally {
      setIsExecuting(false)
    }
  }, [refresh])

  const handleCreateConstraint = useCallback(async (
    constraintName: string,
    constraintType: 'unique' | 'exists',
    label: string,
    property: string
  ) => {
    setIsExecuting(true)
    try {
      const escapedLabel = escapeCypherIdentifier(label)
      const escapedProp = escapeCypherIdentifier(property)
      const escapedName = escapeCypherIdentifier(constraintName)

      let query: string
      if (constraintType === 'unique') {
        query = `CREATE CONSTRAINT ${escapedName} FOR (n:${escapedLabel}) REQUIRE n.${escapedProp} IS UNIQUE`
      } else {
        query = `CREATE CONSTRAINT ${escapedName} FOR (n:${escapedLabel}) REQUIRE n.${escapedProp} IS NOT NULL`
      }

      const result = await executeCypherQuery(query)

      if (result.error) {
        toast.error(`Failed to create constraint: ${result.error.message}`)
      } else {
        toast.success(`Constraint "${constraintName}" created`)
        setConstraints(prev => [...prev, {
          name: constraintName,
          type: constraintType,
          label,
          properties: [property],
        }])
      }
    } catch {
      toast.error('Failed to create constraint')
    } finally {
      setIsExecuting(false)
    }
  }, [])

  const handleDropConstraint = useCallback(async (constraintName: string) => {
    setIsExecuting(true)
    try {
      const escapedName = escapeCypherIdentifier(constraintName)
      const query = `DROP CONSTRAINT ${escapedName}`
      const result = await executeCypherQuery(query)

      if (result.error) {
        toast.error(`Failed to drop constraint: ${result.error.message}`)
      } else {
        toast.success(`Constraint "${constraintName}" dropped`)
        setConstraints(prev => prev.filter(c => c.name !== constraintName))
      }
    } catch {
      toast.error('Failed to drop constraint')
    } finally {
      setIsExecuting(false)
    }
  }, [])

  const handleCreateIndex = useCallback(async (
    indexName: string,
    label: string,
    properties: string[]
  ) => {
    setIsExecuting(true)
    try {
      const escapedLabel = escapeCypherIdentifier(label)
      const escapedName = escapeCypherIdentifier(indexName)
      const propsStr = properties.map(p => `n.${escapeCypherIdentifier(p)}`).join(', ')

      const query = `CREATE INDEX ${escapedName} FOR (n:${escapedLabel}) ON (${propsStr})`
      const result = await executeCypherQuery(query)

      if (result.error) {
        toast.error(`Failed to create index: ${result.error.message}`)
      } else {
        toast.success(`Index "${indexName}" created`)
        setIndexes(prev => [...prev, {
          name: indexName,
          label,
          properties,
          type: 'btree',
        }])
      }
    } catch {
      toast.error('Failed to create index')
    } finally {
      setIsExecuting(false)
    }
  }, [])

  const handleDropIndex = useCallback(async (indexName: string) => {
    setIsExecuting(true)
    try {
      const escapedName = escapeCypherIdentifier(indexName)
      const query = `DROP INDEX ${escapedName}`
      const result = await executeCypherQuery(query)

      if (result.error) {
        toast.error(`Failed to drop index: ${result.error.message}`)
      } else {
        toast.success(`Index "${indexName}" dropped`)
        setIndexes(prev => prev.filter(i => i.name !== indexName))
      }
    } catch {
      toast.error('Failed to drop index')
    } finally {
      setIsExecuting(false)
    }
  }, [])

  const modeButtons: { mode: EditorMode; icon: React.ReactNode; label: string }[] = [
    { mode: 'diagram', icon: <ListTree size={16} />, label: 'Diagram' },
    { mode: 'labels', icon: <Tag size={16} />, label: 'Labels' },
    { mode: 'relationships', icon: <ArrowRightLeft size={16} />, label: 'Relationships' },
    { mode: 'constraints', icon: <Key size={16} />, label: 'Constraints' },
  ]

  if (error) {
    return (
      <div className="flex flex-col h-full bg-bg-primary">
        <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-bg-secondary">
          <div className="flex items-center gap-2">
            <Database size={18} className="text-accent" />
            <h2 className="text-lg font-semibold text-text-primary">Schema Editor</h2>
          </div>
          <IconButton
            icon={<RefreshCw size={16} />}
            onClick={refresh}
            disabled={isLoading}
            tooltip="Refresh schema"
          />
        </div>
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center p-8">
            <p className="text-red-400 mb-4">{error}</p>
            <button
              onClick={refresh}
              className="px-4 py-2 bg-accent hover:bg-accent-hover text-white rounded transition-colors"
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full bg-bg-primary">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-bg-secondary">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Database size={18} className="text-accent" />
            <h2 className="text-sm font-medium text-text-primary">Schema Editor</h2>
          </div>

          {/* Mode tabs */}
          <div className="flex items-center gap-1 bg-bg-tertiary rounded-md p-1">
            {modeButtons.map(({ mode, icon, label }) => (
              <button
                key={mode}
                onClick={() => setActiveMode(mode)}
                className={`
                  flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded transition-colors
                  ${activeMode === mode
                    ? 'bg-accent text-white'
                    : 'text-text-muted hover:text-text-primary hover:bg-border'
                  }
                `}
              >
                {icon}
                {label}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {(activeMode === 'labels' || activeMode === 'diagram') && (
            <button
              onClick={() => setIsCreatingLabel(true)}
              disabled={isExecuting}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-bg-tertiary hover:bg-border text-text-secondary rounded transition-colors disabled:opacity-50"
            >
              <Plus size={14} />
              New Label
            </button>
          )}
          {(activeMode === 'relationships' || activeMode === 'diagram') && (
            <button
              onClick={() => setIsCreatingRelationship(true)}
              disabled={isExecuting || labels.length < 1}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-bg-tertiary hover:bg-border text-text-secondary rounded transition-colors disabled:opacity-50"
            >
              <Plus size={14} />
              New Relationship
            </button>
          )}
          <IconButton
            icon={<RefreshCw size={16} className={isLoading ? 'animate-spin' : ''} />}
            onClick={refresh}
            disabled={isLoading}
            tooltip="Refresh schema"
          />
        </div>
      </div>

      {/* Stats bar */}
      {schema && (
        <div className="flex items-center gap-6 px-4 py-2 border-b border-border bg-bg-secondary/50 text-xs">
          <div className="flex items-center gap-2">
            <Tag size={14} className="text-text-muted" />
            <span className="text-text-muted">Labels:</span>
            <span className="font-medium text-text-primary">{labels.length}</span>
          </div>
          <div className="flex items-center gap-2">
            <ArrowRightLeft size={14} className="text-text-muted" />
            <span className="text-text-muted">Relationships:</span>
            <span className="font-medium text-text-primary">{relationships.length}</span>
          </div>
          <div className="flex items-center gap-2">
            <Database size={14} className="text-text-muted" />
            <span className="text-text-muted">Nodes:</span>
            <span className="font-medium text-text-primary">{schema.nodeCount.toLocaleString()}</span>
          </div>
          <div className="flex items-center gap-2">
            <ArrowRightLeft size={14} className="text-text-muted" />
            <span className="text-text-muted">Edges:</span>
            <span className="font-medium text-text-primary">{schema.edgeCount.toLocaleString()}</span>
          </div>
        </div>
      )}

      {/* Main content */}
      {isLoading && !schema ? (
        <div className="flex-1 flex items-center justify-center">
          <RefreshCw size={24} className="animate-spin text-text-muted" />
        </div>
      ) : activeMode === 'diagram' ? (
        <SchemaDiagram
          labels={labels}
          relationships={relationships}
          onSelectLabel={setSelectedLabel}
          onSelectRelationship={setSelectedRelationship}
          selectedLabel={selectedLabel}
          selectedRelationship={selectedRelationship}
        />
      ) : activeMode === 'labels' ? (
        <Group orientation="horizontal" className="flex-1">
          <Panel id="labels-list" defaultSize={30} minSize={20} maxSize={50}>
            <div className="h-full overflow-hidden border-r border-border bg-bg-secondary">
              <LabelEditor
                labels={labels}
                selectedLabel={selectedLabel}
                onSelectLabel={setSelectedLabel}
                onCreateLabel={handleCreateLabel}
                onDeleteLabel={handleDeleteLabel}
                isCreating={isCreatingLabel}
                onStartCreate={() => setIsCreatingLabel(true)}
                onCancelCreate={() => setIsCreatingLabel(false)}
                isExecuting={isExecuting}
              />
            </div>
          </Panel>
          <Separator className="w-1 bg-border hover:bg-accent transition-colors cursor-col-resize" />
          <Panel id="label-details" minSize={40}>
            <div className="h-full p-4">
              {selectedLabel ? (
                <LabelDetailView
                  label={labels.find(l => l.name === selectedLabel)}
                  onDelete={() => handleDeleteLabel(selectedLabel)}
                  onCreateConstraint={handleCreateConstraint}
                  onCreateIndex={handleCreateIndex}
                  constraints={constraints.filter(c => c.label === selectedLabel)}
                  indexes={indexes.filter(i => i.label === selectedLabel)}
                  onDropConstraint={handleDropConstraint}
                  onDropIndex={handleDropIndex}
                  isExecuting={isExecuting}
                />
              ) : (
                <div className="flex items-center justify-center h-full text-text-muted text-sm">
                  Select a label to view details
                </div>
              )}
            </div>
          </Panel>
        </Group>
      ) : activeMode === 'relationships' ? (
        <Group orientation="horizontal" className="flex-1">
          <Panel id="relationships-list" defaultSize={30} minSize={20} maxSize={50}>
            <div className="h-full overflow-hidden border-r border-border bg-bg-secondary">
              <RelationshipEditor
                relationships={relationships}
                labels={labels}
                selectedRelationship={selectedRelationship}
                onSelectRelationship={setSelectedRelationship}
                onCreateRelationship={handleCreateRelationship}
                onDeleteRelationship={handleDeleteRelationship}
                isCreating={isCreatingRelationship}
                onStartCreate={() => setIsCreatingRelationship(true)}
                onCancelCreate={() => setIsCreatingRelationship(false)}
                isExecuting={isExecuting}
              />
            </div>
          </Panel>
          <Separator className="w-1 bg-border hover:bg-accent transition-colors cursor-col-resize" />
          <Panel id="relationship-details" minSize={40}>
            <div className="h-full p-4">
              {selectedRelationship ? (
                <RelationshipDetailView
                  relationship={relationships.find(r => r.name === selectedRelationship)}
                  onDelete={() => handleDeleteRelationship(selectedRelationship)}
                  isExecuting={isExecuting}
                />
              ) : (
                <div className="flex items-center justify-center h-full text-text-muted text-sm">
                  Select a relationship type to view details
                </div>
              )}
            </div>
          </Panel>
        </Group>
      ) : activeMode === 'constraints' ? (
        <ConstraintsView
          labels={labels}
          constraints={constraints}
          indexes={indexes}
          onCreateConstraint={handleCreateConstraint}
          onDropConstraint={handleDropConstraint}
          onCreateIndex={handleCreateIndex}
          onDropIndex={handleDropIndex}
          isExecuting={isExecuting}
        />
      ) : null}

      {/* Create Label Modal */}
      {isCreatingLabel && (
        <CreateLabelModal
          onClose={() => setIsCreatingLabel(false)}
          onCreate={handleCreateLabel}
          existingLabels={labels.map(l => l.name)}
          isExecuting={isExecuting}
        />
      )}

      {/* Create Relationship Modal */}
      {isCreatingRelationship && (
        <CreateRelationshipModal
          onClose={() => setIsCreatingRelationship(false)}
          onCreate={handleCreateRelationship}
          labels={labels}
          existingRelationships={relationships.map(r => r.name)}
          isExecuting={isExecuting}
        />
      )}
    </div>
  )
}

// Label Detail View
interface LabelDetailViewProps {
  label: SchemaLabel | undefined
  onDelete: () => void
  onCreateConstraint: (name: string, type: 'unique' | 'exists', label: string, property: string) => void
  onCreateIndex: (name: string, label: string, properties: string[]) => void
  constraints: SchemaConstraint[]
  indexes: SchemaIndex[]
  onDropConstraint: (name: string) => void
  onDropIndex: (name: string) => void
  isExecuting: boolean
}

function LabelDetailView({
  label,
  onDelete,
  onCreateConstraint,
  onCreateIndex,
  constraints,
  indexes,
  onDropConstraint,
  onDropIndex,
  isExecuting,
}: LabelDetailViewProps) {
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [showAddConstraint, setShowAddConstraint] = useState(false)
  const [showAddIndex, setShowAddIndex] = useState(false)
  const [newConstraintName, setNewConstraintName] = useState('')
  const [newConstraintType, setNewConstraintType] = useState<'unique' | 'exists'>('unique')
  const [newConstraintProperty, setNewConstraintProperty] = useState('')
  const [newIndexName, setNewIndexName] = useState('')
  const [newIndexProperty, setNewIndexProperty] = useState('')

  if (!label) return null

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-medium text-text-primary flex items-center gap-2">
            <Tag size={18} className="text-accent" />
            {label.name}
          </h3>
          <p className="text-sm text-text-muted mt-1">{label.count.toLocaleString()} nodes</p>
        </div>
        <button
          onClick={() => setShowDeleteConfirm(true)}
          disabled={isExecuting}
          className="px-3 py-1.5 text-sm text-red-400 hover:bg-red-500/10 rounded transition-colors disabled:opacity-50"
        >
          Delete Label
        </button>
      </div>

      {/* Constraints Section */}
      <CollapsibleSection
        title="Constraints"
        icon={<Key size={16} />}
        count={constraints.length}
        defaultOpen={true}
      >
        <div className="px-4 space-y-2">
          {constraints.length === 0 ? (
            <p className="text-sm text-text-muted italic">No constraints defined</p>
          ) : (
            constraints.map(constraint => (
              <div key={constraint.name} className="flex items-center justify-between p-2 bg-bg-tertiary rounded">
                <div>
                  <p className="text-sm text-text-primary">{constraint.name}</p>
                  <p className="text-xs text-text-muted">
                    {constraint.type.toUpperCase()} on {constraint.properties.join(', ')}
                  </p>
                </div>
                <button
                  onClick={() => onDropConstraint(constraint.name)}
                  disabled={isExecuting}
                  className="text-xs text-red-400 hover:underline disabled:opacity-50"
                >
                  Drop
                </button>
              </div>
            ))
          )}

          {showAddConstraint ? (
            <div className="p-3 bg-bg-tertiary rounded space-y-2">
              <input
                type="text"
                placeholder="Constraint name"
                value={newConstraintName}
                onChange={e => setNewConstraintName(e.target.value)}
                className="w-full px-2 py-1.5 text-sm bg-bg-primary border border-border rounded focus:outline-none focus:border-accent"
              />
              <select
                value={newConstraintType}
                onChange={e => setNewConstraintType(e.target.value as 'unique' | 'exists')}
                className="w-full px-2 py-1.5 text-sm bg-bg-primary border border-border rounded focus:outline-none focus:border-accent"
              >
                <option value="unique">UNIQUE</option>
                <option value="exists">NOT NULL</option>
              </select>
              <input
                type="text"
                placeholder="Property name"
                value={newConstraintProperty}
                onChange={e => setNewConstraintProperty(e.target.value)}
                className="w-full px-2 py-1.5 text-sm bg-bg-primary border border-border rounded focus:outline-none focus:border-accent"
              />
              <div className="flex gap-2">
                <button
                  onClick={() => {
                    if (newConstraintName && newConstraintProperty) {
                      onCreateConstraint(newConstraintName, newConstraintType, label.name, newConstraintProperty)
                      setShowAddConstraint(false)
                      setNewConstraintName('')
                      setNewConstraintProperty('')
                    }
                  }}
                  disabled={isExecuting || !newConstraintName || !newConstraintProperty}
                  className="flex-1 px-3 py-1.5 text-sm bg-accent hover:bg-accent-hover text-white rounded disabled:opacity-50"
                >
                  Create
                </button>
                <button
                  onClick={() => setShowAddConstraint(false)}
                  className="px-3 py-1.5 text-sm text-text-muted hover:text-text-primary"
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <button
              onClick={() => setShowAddConstraint(true)}
              className="flex items-center gap-1 text-sm text-accent hover:underline"
            >
              <Plus size={14} />
              Add Constraint
            </button>
          )}
        </div>
      </CollapsibleSection>

      {/* Indexes Section */}
      <CollapsibleSection
        title="Indexes"
        icon={<ListTree size={16} />}
        count={indexes.length}
        defaultOpen={true}
      >
        <div className="px-4 space-y-2">
          {indexes.length === 0 ? (
            <p className="text-sm text-text-muted italic">No indexes defined</p>
          ) : (
            indexes.map(index => (
              <div key={index.name} className="flex items-center justify-between p-2 bg-bg-tertiary rounded">
                <div>
                  <p className="text-sm text-text-primary">{index.name}</p>
                  <p className="text-xs text-text-muted">
                    {index.type.toUpperCase()} on {index.properties.join(', ')}
                  </p>
                </div>
                <button
                  onClick={() => onDropIndex(index.name)}
                  disabled={isExecuting}
                  className="text-xs text-red-400 hover:underline disabled:opacity-50"
                >
                  Drop
                </button>
              </div>
            ))
          )}

          {showAddIndex ? (
            <div className="p-3 bg-bg-tertiary rounded space-y-2">
              <input
                type="text"
                placeholder="Index name"
                value={newIndexName}
                onChange={e => setNewIndexName(e.target.value)}
                className="w-full px-2 py-1.5 text-sm bg-bg-primary border border-border rounded focus:outline-none focus:border-accent"
              />
              <input
                type="text"
                placeholder="Property name"
                value={newIndexProperty}
                onChange={e => setNewIndexProperty(e.target.value)}
                className="w-full px-2 py-1.5 text-sm bg-bg-primary border border-border rounded focus:outline-none focus:border-accent"
              />
              <div className="flex gap-2">
                <button
                  onClick={() => {
                    if (newIndexName && newIndexProperty) {
                      onCreateIndex(newIndexName, label.name, [newIndexProperty])
                      setShowAddIndex(false)
                      setNewIndexName('')
                      setNewIndexProperty('')
                    }
                  }}
                  disabled={isExecuting || !newIndexName || !newIndexProperty}
                  className="flex-1 px-3 py-1.5 text-sm bg-accent hover:bg-accent-hover text-white rounded disabled:opacity-50"
                >
                  Create
                </button>
                <button
                  onClick={() => setShowAddIndex(false)}
                  className="px-3 py-1.5 text-sm text-text-muted hover:text-text-primary"
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <button
              onClick={() => setShowAddIndex(true)}
              className="flex items-center gap-1 text-sm text-accent hover:underline"
            >
              <Plus size={14} />
              Add Index
            </button>
          )}
        </div>
      </CollapsibleSection>

      {/* Delete Confirmation */}
      {showDeleteConfirm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setShowDeleteConfirm(false)}>
          <div className="bg-bg-secondary border border-border rounded-lg shadow-xl p-6 max-w-md" onClick={e => e.stopPropagation()}>
            <h3 className="text-lg font-medium text-text-primary mb-2">Delete Label?</h3>
            <p className="text-sm text-text-muted mb-4">
              This will permanently delete all nodes with the label "{label.name}" and their relationships.
              This action cannot be undone.
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setShowDeleteConfirm(false)}
                className="px-4 py-2 text-sm text-text-secondary hover:text-text-primary"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  onDelete()
                  setShowDeleteConfirm(false)
                }}
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

// Relationship Detail View
interface RelationshipDetailViewProps {
  relationship: SchemaRelationship | undefined
  onDelete: () => void
  isExecuting: boolean
}

function RelationshipDetailView({ relationship, onDelete, isExecuting }: RelationshipDetailViewProps) {
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)

  if (!relationship) return null

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-medium text-text-primary flex items-center gap-2">
            <ArrowRightLeft size={18} className="text-accent" />
            {relationship.name}
          </h3>
          <p className="text-sm text-text-muted mt-1">{relationship.count.toLocaleString()} edges</p>
        </div>
        <button
          onClick={() => setShowDeleteConfirm(true)}
          disabled={isExecuting}
          className="px-3 py-1.5 text-sm text-red-400 hover:bg-red-500/10 rounded transition-colors disabled:opacity-50"
        >
          Delete Relationship Type
        </button>
      </div>

      <div className="p-4 bg-bg-secondary rounded-lg">
        <h4 className="text-sm font-medium text-text-primary mb-3">Usage Pattern</h4>
        <div className="flex items-center gap-4 text-sm">
          <div className="flex-1 p-2 bg-bg-tertiary rounded text-center">
            <span className="text-text-muted">(source)</span>
          </div>
          <div className="flex items-center gap-2 text-accent">
            <span>-[:{relationship.name}]-&gt;</span>
          </div>
          <div className="flex-1 p-2 bg-bg-tertiary rounded text-center">
            <span className="text-text-muted">(target)</span>
          </div>
        </div>
      </div>

      {/* Delete Confirmation */}
      {showDeleteConfirm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setShowDeleteConfirm(false)}>
          <div className="bg-bg-secondary border border-border rounded-lg shadow-xl p-6 max-w-md" onClick={e => e.stopPropagation()}>
            <h3 className="text-lg font-medium text-text-primary mb-2">Delete Relationship Type?</h3>
            <p className="text-sm text-text-muted mb-4">
              This will permanently delete all edges of type "{relationship.name}".
              This action cannot be undone.
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setShowDeleteConfirm(false)}
                className="px-4 py-2 text-sm text-text-secondary hover:text-text-primary"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  onDelete()
                  setShowDeleteConfirm(false)
                }}
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

// Constraints View
interface ConstraintsViewProps {
  labels: SchemaLabel[]
  constraints: SchemaConstraint[]
  indexes: SchemaIndex[]
  onCreateConstraint: (name: string, type: 'unique' | 'exists', label: string, property: string) => void
  onDropConstraint: (name: string) => void
  onCreateIndex: (name: string, label: string, properties: string[]) => void
  onDropIndex: (name: string) => void
  isExecuting: boolean
}

function ConstraintsView({
  labels,
  constraints,
  indexes,
  onCreateConstraint,
  onDropConstraint,
  onCreateIndex,
  onDropIndex,
  isExecuting,
}: ConstraintsViewProps) {
  const [showAddConstraint, setShowAddConstraint] = useState(false)
  const [showAddIndex, setShowAddIndex] = useState(false)
  const [newConstraintName, setNewConstraintName] = useState('')
  const [newConstraintType, setNewConstraintType] = useState<'unique' | 'exists'>('unique')
  const [newConstraintLabel, setNewConstraintLabel] = useState('')
  const [newConstraintProperty, setNewConstraintProperty] = useState('')
  const [newIndexName, setNewIndexName] = useState('')
  const [newIndexLabel, setNewIndexLabel] = useState('')
  const [newIndexProperty, setNewIndexProperty] = useState('')

  return (
    <div className="flex-1 overflow-y-auto p-4">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Constraints */}
        <div className="bg-bg-secondary rounded-lg border border-border">
          <div className="flex items-center justify-between px-4 py-3 border-b border-border">
            <h3 className="text-sm font-medium text-text-primary flex items-center gap-2">
              <Key size={16} />
              Constraints
            </h3>
            <button
              onClick={() => setShowAddConstraint(true)}
              className="flex items-center gap-1 text-sm text-accent hover:underline"
            >
              <Plus size={14} />
              Add Constraint
            </button>
          </div>

          <div className="p-4">
            {constraints.length === 0 ? (
              <p className="text-sm text-text-muted italic text-center py-4">
                No constraints defined
              </p>
            ) : (
              <div className="space-y-2">
                {constraints.map(constraint => (
                  <div key={constraint.name} className="flex items-center justify-between p-3 bg-bg-tertiary rounded">
                    <div>
                      <p className="text-sm text-text-primary font-medium">{constraint.name}</p>
                      <p className="text-xs text-text-muted">
                        {constraint.type.toUpperCase()} on :{constraint.label} ({constraint.properties.join(', ')})
                      </p>
                    </div>
                    <button
                      onClick={() => onDropConstraint(constraint.name)}
                      disabled={isExecuting}
                      className="text-xs text-red-400 hover:underline disabled:opacity-50"
                    >
                      Drop
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Indexes */}
        <div className="bg-bg-secondary rounded-lg border border-border">
          <div className="flex items-center justify-between px-4 py-3 border-b border-border">
            <h3 className="text-sm font-medium text-text-primary flex items-center gap-2">
              <ListTree size={16} />
              Indexes
            </h3>
            <button
              onClick={() => setShowAddIndex(true)}
              className="flex items-center gap-1 text-sm text-accent hover:underline"
            >
              <Plus size={14} />
              Add Index
            </button>
          </div>

          <div className="p-4">
            {indexes.length === 0 ? (
              <p className="text-sm text-text-muted italic text-center py-4">
                No indexes defined
              </p>
            ) : (
              <div className="space-y-2">
                {indexes.map(index => (
                  <div key={index.name} className="flex items-center justify-between p-3 bg-bg-tertiary rounded">
                    <div>
                      <p className="text-sm text-text-primary font-medium">{index.name}</p>
                      <p className="text-xs text-text-muted">
                        {index.type.toUpperCase()} on :{index.label} ({index.properties.join(', ')})
                      </p>
                    </div>
                    <button
                      onClick={() => onDropIndex(index.name)}
                      disabled={isExecuting}
                      className="text-xs text-red-400 hover:underline disabled:opacity-50"
                    >
                      Drop
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Add Constraint Modal */}
      {showAddConstraint && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setShowAddConstraint(false)}>
          <div className="bg-bg-secondary border border-border rounded-lg shadow-xl p-6 w-96" onClick={e => e.stopPropagation()}>
            <h3 className="text-lg font-medium text-text-primary mb-4">Create Constraint</h3>
            <div className="space-y-3">
              <input
                type="text"
                placeholder="Constraint name"
                value={newConstraintName}
                onChange={e => setNewConstraintName(e.target.value)}
                className="w-full px-3 py-2 text-sm bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent"
              />
              <select
                value={newConstraintType}
                onChange={e => setNewConstraintType(e.target.value as 'unique' | 'exists')}
                className="w-full px-3 py-2 text-sm bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent"
              >
                <option value="unique">UNIQUE</option>
                <option value="exists">NOT NULL</option>
              </select>
              <select
                value={newConstraintLabel}
                onChange={e => setNewConstraintLabel(e.target.value)}
                className="w-full px-3 py-2 text-sm bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent"
              >
                <option value="">Select label...</option>
                {labels.map(l => (
                  <option key={l.name} value={l.name}>{l.name}</option>
                ))}
              </select>
              <input
                type="text"
                placeholder="Property name"
                value={newConstraintProperty}
                onChange={e => setNewConstraintProperty(e.target.value)}
                className="w-full px-3 py-2 text-sm bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent"
              />
            </div>
            <div className="flex gap-3 justify-end mt-6">
              <button
                onClick={() => setShowAddConstraint(false)}
                className="px-4 py-2 text-sm text-text-secondary hover:text-text-primary"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  if (newConstraintName && newConstraintLabel && newConstraintProperty) {
                    onCreateConstraint(newConstraintName, newConstraintType, newConstraintLabel, newConstraintProperty)
                    setShowAddConstraint(false)
                    setNewConstraintName('')
                    setNewConstraintLabel('')
                    setNewConstraintProperty('')
                  }
                }}
                disabled={isExecuting || !newConstraintName || !newConstraintLabel || !newConstraintProperty}
                className="px-4 py-2 text-sm bg-accent hover:bg-accent-hover text-white rounded disabled:opacity-50"
              >
                Create
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Add Index Modal */}
      {showAddIndex && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setShowAddIndex(false)}>
          <div className="bg-bg-secondary border border-border rounded-lg shadow-xl p-6 w-96" onClick={e => e.stopPropagation()}>
            <h3 className="text-lg font-medium text-text-primary mb-4">Create Index</h3>
            <div className="space-y-3">
              <input
                type="text"
                placeholder="Index name"
                value={newIndexName}
                onChange={e => setNewIndexName(e.target.value)}
                className="w-full px-3 py-2 text-sm bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent"
              />
              <select
                value={newIndexLabel}
                onChange={e => setNewIndexLabel(e.target.value)}
                className="w-full px-3 py-2 text-sm bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent"
              >
                <option value="">Select label...</option>
                {labels.map(l => (
                  <option key={l.name} value={l.name}>{l.name}</option>
                ))}
              </select>
              <input
                type="text"
                placeholder="Property name"
                value={newIndexProperty}
                onChange={e => setNewIndexProperty(e.target.value)}
                className="w-full px-3 py-2 text-sm bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent"
              />
            </div>
            <div className="flex gap-3 justify-end mt-6">
              <button
                onClick={() => setShowAddIndex(false)}
                className="px-4 py-2 text-sm text-text-secondary hover:text-text-primary"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  if (newIndexName && newIndexLabel && newIndexProperty) {
                    onCreateIndex(newIndexName, newIndexLabel, [newIndexProperty])
                    setShowAddIndex(false)
                    setNewIndexName('')
                    setNewIndexLabel('')
                    setNewIndexProperty('')
                  }
                }}
                disabled={isExecuting || !newIndexName || !newIndexLabel || !newIndexProperty}
                className="px-4 py-2 text-sm bg-accent hover:bg-accent-hover text-white rounded disabled:opacity-50"
              >
                Create
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// Create Label Modal
interface CreateLabelModalProps {
  onClose: () => void
  onCreate: (name: string) => void
  existingLabels: string[]
  isExecuting: boolean
}

function CreateLabelModal({ onClose, onCreate, existingLabels, isExecuting }: CreateLabelModalProps) {
  const [name, setName] = useState('')
  const isValid = name.trim() && !existingLabels.includes(name.trim())

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={onClose}>
      <div className="bg-bg-secondary border border-border rounded-lg shadow-xl p-6 w-96" onClick={e => e.stopPropagation()}>
        <h3 className="text-lg font-medium text-text-primary mb-4">Create New Label</h3>
        <input
          type="text"
          placeholder="Label name"
          value={name}
          onChange={e => setName(e.target.value)}
          onKeyDown={e => {
            if (e.key === 'Enter' && isValid) {
              onCreate(name.trim())
            }
          }}
          autoFocus
          className="w-full px-3 py-2 text-sm bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent"
        />
        {name && existingLabels.includes(name.trim()) && (
          <p className="text-xs text-red-400 mt-2">A label with this name already exists</p>
        )}
        <div className="flex gap-3 justify-end mt-6">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-text-secondary hover:text-text-primary"
          >
            Cancel
          </button>
          <button
            onClick={() => onCreate(name.trim())}
            disabled={isExecuting || !isValid}
            className="px-4 py-2 text-sm bg-accent hover:bg-accent-hover text-white rounded disabled:opacity-50"
          >
            Create
          </button>
        </div>
      </div>
    </div>
  )
}

// Create Relationship Modal
interface CreateRelationshipModalProps {
  onClose: () => void
  onCreate: (name: string, sourceLabel: string, targetLabel: string) => void
  labels: SchemaLabel[]
  existingRelationships: string[]
  isExecuting: boolean
}

function CreateRelationshipModal({
  onClose,
  onCreate,
  labels,
  existingRelationships,
  isExecuting,
}: CreateRelationshipModalProps) {
  const [name, setName] = useState('')
  const [sourceLabel, setSourceLabel] = useState(labels[0]?.name ?? '')
  const [targetLabel, setTargetLabel] = useState(labels[0]?.name ?? '')

  const isValid = name.trim() && !existingRelationships.includes(name.trim()) && sourceLabel && targetLabel

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={onClose}>
      <div className="bg-bg-secondary border border-border rounded-lg shadow-xl p-6 w-96" onClick={e => e.stopPropagation()}>
        <h3 className="text-lg font-medium text-text-primary mb-4">Create New Relationship Type</h3>
        <div className="space-y-3">
          <input
            type="text"
            placeholder="Relationship type name"
            value={name}
            onChange={e => setName(e.target.value)}
            autoFocus
            className="w-full px-3 py-2 text-sm bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent"
          />
          <div className="flex items-center gap-2">
            <select
              value={sourceLabel}
              onChange={e => setSourceLabel(e.target.value)}
              className="flex-1 px-3 py-2 text-sm bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent"
            >
              {labels.map(l => (
                <option key={l.name} value={l.name}>{l.name}</option>
              ))}
            </select>
            <span className="text-text-muted">→</span>
            <select
              value={targetLabel}
              onChange={e => setTargetLabel(e.target.value)}
              className="flex-1 px-3 py-2 text-sm bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent"
            >
              {labels.map(l => (
                <option key={l.name} value={l.name}>{l.name}</option>
              ))}
            </select>
          </div>
        </div>
        {name && existingRelationships.includes(name.trim()) && (
          <p className="text-xs text-red-400 mt-2">A relationship type with this name already exists</p>
        )}
        <div className="flex gap-3 justify-end mt-6">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-text-secondary hover:text-text-primary"
          >
            Cancel
          </button>
          <button
            onClick={() => onCreate(name.trim(), sourceLabel, targetLabel)}
            disabled={isExecuting || !isValid}
            className="px-4 py-2 text-sm bg-accent hover:bg-accent-hover text-white rounded disabled:opacity-50"
          >
            Create
          </button>
        </div>
      </div>
    </div>
  )
}
