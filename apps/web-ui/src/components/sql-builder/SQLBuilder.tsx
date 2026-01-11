import { useState, useCallback, useMemo } from 'react'
import { Group, Panel, Separator } from 'react-resizable-panels'
import {
  Play,
  Copy,
  RefreshCw,
  Trash2,
  Database,
  Table2,
  ChevronDown,
  ChevronRight,
  Check,
  X,
} from 'lucide-react'
import { toast } from 'sonner'
import { TableCanvas } from './TableCanvas'
import { ColumnPicker } from './ColumnPicker'
import { WhereBuilder } from './WhereBuilder'
import { IconButton } from '../shared/IconButton'
import { CollapsibleSection } from '../shared/CollapsibleSection'
import { UnifiedResultView } from '../result-views/UnifiedResultView'
import { useSchema } from '../../hooks/useSchema'
import { executeSqlQuery } from '../../lib/graphql-client'
import {
  generateSQL,
  createInitialState,
  generateColumnId,
  type SQLBuilderState,
  type SelectedColumn,
  type TableJoin,
  type WhereCondition,
  type JoinType,
} from '../../lib/sql-generator'
import type { QueryResult } from '../../types'

interface TableInfo {
  name: string
  columns: string[]
}

interface JoinEditModalProps {
  join: TableJoin
  onUpdate: (updates: Partial<TableJoin>) => void
  onDelete: () => void
  onClose: () => void
}

function JoinEditModal({ join, onUpdate, onDelete, onClose }: JoinEditModalProps) {
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={onClose}>
      <div
        className="bg-bg-secondary border border-border rounded-lg shadow-xl p-4 min-w-[300px]"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium text-text-primary">Edit Join</h3>
          <button className="p-1 hover:bg-border rounded text-text-muted" onClick={onClose}>
            <X size={16} />
          </button>
        </div>

        <div className="space-y-3">
          <div>
            <label className="block text-xs text-text-muted mb-1">Join Type</label>
            <select
              value={join.joinType}
              onChange={(e) => onUpdate({ joinType: e.target.value as JoinType })}
              className="w-full px-3 py-2 text-sm bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent text-text-primary"
            >
              <option value="INNER">INNER JOIN</option>
              <option value="LEFT">LEFT JOIN</option>
              <option value="RIGHT">RIGHT JOIN</option>
              <option value="FULL">FULL OUTER JOIN</option>
            </select>
          </div>

          <div className="text-xs text-text-muted">
            <p>{join.fromTable}.{join.fromColumn}</p>
            <p className="text-center my-1">↔</p>
            <p>{join.toTable}.{join.toColumn}</p>
          </div>
        </div>

        <div className="flex justify-between mt-4 pt-4 border-t border-border">
          <button
            onClick={onDelete}
            className="px-3 py-1.5 text-sm text-red-400 hover:bg-red-500/10 rounded"
          >
            Delete Join
          </button>
          <button
            onClick={onClose}
            className="px-3 py-1.5 text-sm bg-accent hover:bg-accent-hover text-white rounded"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  )
}

interface TableListItemProps {
  name: string
  columns: string[]
  onAddTable: (name: string) => void
  onAddColumn: (table: string, column: string) => void
  isAdded: boolean
}

function TableListItem({ name, columns, onAddTable, onAddColumn, isAdded }: TableListItemProps) {
  const [isExpanded, setIsExpanded] = useState(false)

  return (
    <div className="border-b border-border last:border-b-0">
      <button
        className={`
          w-full flex items-center gap-2 px-3 py-2 text-left
          hover:bg-bg-tertiary transition-colors
          ${isAdded ? 'text-accent' : 'text-text-secondary'}
        `}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <Table2 size={14} />
        <span className="text-sm flex-1 truncate">{name}</span>
        {isAdded ? (
          <Check size={14} className="text-accent" />
        ) : (
          <button
            className="p-1 hover:bg-border rounded text-text-muted hover:text-accent"
            onClick={(e) => {
              e.stopPropagation()
              onAddTable(name)
            }}
          >
            +
          </button>
        )}
      </button>

      {isExpanded && (
        <div className="bg-bg-tertiary py-1">
          {columns.map(column => (
            <button
              key={column}
              className="w-full flex items-center gap-2 px-6 py-1 text-left text-xs text-text-muted hover:text-text-primary hover:bg-border/50"
              onClick={() => onAddColumn(name, column)}
            >
              <span className="truncate">{column}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

export function SQLBuilder() {
  const { schema, isLoading: schemaLoading, refresh: refreshSchema } = useSchema()
  const [state, setState] = useState<SQLBuilderState>(createInitialState)
  const [result, setResult] = useState<QueryResult | null>(null)
  const [isExecuting, setIsExecuting] = useState(false)
  const [editingJoin, setEditingJoin] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  // Derive available tables from schema (labels = tables in this context)
  // For now, simulate table columns based on common patterns
  const tableInfo = useMemo(() => {
    const map = new Map<string, TableInfo>()

    if (schema?.labels) {
      schema.labels.forEach(label => {
        // Each label becomes a "table" with id and common property columns
        map.set(label.name, {
          name: label.name,
          columns: ['id', 'labels', 'properties', 'created_at', 'updated_at'],
        })
      })
    }

    // Also add edge types as potential tables
    if (schema?.edgeTypes) {
      schema.edgeTypes.forEach(edgeType => {
        map.set(edgeType.name, {
          name: edgeType.name,
          columns: ['id', 'type', 'source_id', 'target_id', 'properties'],
        })
      })
    }

    return map
  }, [schema])

  const availableTables = useMemo(() => Array.from(tableInfo.keys()), [tableInfo])

  // Generate SQL from current state
  const generatedSQL = useMemo(() => generateSQL(state), [state])

  // Add table to canvas
  const handleAddTable = useCallback((name: string, x?: number, y?: number) => {
    const existingCount = state.tables.length

    setState(prev => ({
      ...prev,
      tables: [
        ...prev.tables,
        {
          id: `table-${Date.now()}`,
          name,
          x: x ?? 50 + existingCount * 220,
          y: y ?? 50 + (existingCount % 3) * 100,
        },
      ],
    }))
  }, [state.tables.length])

  // Move table on canvas
  const handleMoveTable = useCallback((id: string, x: number, y: number) => {
    setState(prev => ({
      ...prev,
      tables: prev.tables.map(t => (t.id === id ? { ...t, x, y } : t)),
    }))
  }, [])

  // Remove table from canvas
  const handleRemoveTable = useCallback((id: string) => {
    setState(prev => {
      const table = prev.tables.find(t => t.id === id)
      if (!table) return prev

      // Remove table, its columns, and related joins
      return {
        ...prev,
        tables: prev.tables.filter(t => t.id !== id),
        selectedColumns: prev.selectedColumns.filter(c => c.table !== table.name),
        joins: prev.joins.filter(j => j.fromTable !== table.name && j.toTable !== table.name),
        whereConditions: prev.whereConditions.filter(c => c.table !== table.name),
        orderBy: prev.orderBy.filter(o => o.table !== table.name),
      }
    })
  }, [])

  // Add join
  const handleAddJoin = useCallback((join: Omit<TableJoin, 'id'>) => {
    setState(prev => ({
      ...prev,
      joins: [
        ...prev.joins,
        {
          ...join,
          id: `join-${Date.now()}`,
        },
      ],
    }))
  }, [])

  // Update join
  const handleUpdateJoin = useCallback((id: string, updates: Partial<TableJoin>) => {
    setState(prev => ({
      ...prev,
      joins: prev.joins.map(j => (j.id === id ? { ...j, ...updates } : j)),
    }))
  }, [])

  // Remove join
  const handleRemoveJoin = useCallback((id: string) => {
    setState(prev => ({
      ...prev,
      joins: prev.joins.filter(j => j.id !== id),
    }))
    setEditingJoin(null)
  }, [])

  // Add column to selection
  const handleAddColumn = useCallback((table: string, column: string) => {
    const id = generateColumnId(table, column)

    setState(prev => {
      // Check if already added
      if (prev.selectedColumns.some(c => c.id === id)) {
        return prev
      }

      return {
        ...prev,
        selectedColumns: [
          ...prev.selectedColumns,
          {
            id,
            table,
            column,
            aggregate: 'NONE',
            visible: true,
          },
        ],
      }
    })
  }, [])

  // Update column
  const handleUpdateColumn = useCallback((id: string, updates: Partial<SelectedColumn>) => {
    setState(prev => ({
      ...prev,
      selectedColumns: prev.selectedColumns.map(c => (c.id === id ? { ...c, ...updates } : c)),
    }))
  }, [])

  // Remove column
  const handleRemoveColumn = useCallback((id: string) => {
    setState(prev => ({
      ...prev,
      selectedColumns: prev.selectedColumns.filter(c => c.id !== id),
    }))
  }, [])

  // Reorder columns
  const handleReorderColumns = useCallback((columns: SelectedColumn[]) => {
    setState(prev => ({
      ...prev,
      selectedColumns: columns,
    }))
  }, [])

  // Add WHERE condition
  const handleAddCondition = useCallback(() => {
    const firstColumn = state.selectedColumns[0]
    if (!firstColumn) return

    setState(prev => ({
      ...prev,
      whereConditions: [
        ...prev.whereConditions,
        {
          id: `where-${Date.now()}`,
          table: firstColumn.table,
          column: firstColumn.column,
          operator: '=',
          value: '',
          logic: 'AND',
        },
      ],
    }))
  }, [state.selectedColumns])

  // Update WHERE condition
  const handleUpdateCondition = useCallback((id: string, updates: Partial<WhereCondition>) => {
    setState(prev => ({
      ...prev,
      whereConditions: prev.whereConditions.map(c => (c.id === id ? { ...c, ...updates } : c)),
    }))
  }, [])

  // Remove WHERE condition
  const handleRemoveCondition = useCallback((id: string) => {
    setState(prev => ({
      ...prev,
      whereConditions: prev.whereConditions.filter(c => c.id !== id),
    }))
  }, [])

  // Add to ORDER BY
  const handleAddToOrderBy = useCallback((table: string, column: string) => {
    setState(prev => {
      // Toggle direction if already exists, otherwise add
      const existing = prev.orderBy.find(o => o.table === table && o.column === column)

      if (existing) {
        if (existing.direction === 'ASC') {
          return {
            ...prev,
            orderBy: prev.orderBy.map(o =>
              o.table === table && o.column === column ? { ...o, direction: 'DESC' as const } : o
            ),
          }
        } else {
          // Remove if already DESC
          return {
            ...prev,
            orderBy: prev.orderBy.filter(o => !(o.table === table && o.column === column)),
          }
        }
      }

      return {
        ...prev,
        orderBy: [...prev.orderBy, { table, column, direction: 'ASC' as const }],
      }
    })
  }, [])

  // Execute query
  const handleExecute = useCallback(async () => {
    if (!generatedSQL || generatedSQL.startsWith('--')) {
      toast.error('Please build a valid query first')
      return
    }

    setIsExecuting(true)
    setResult(null)

    try {
      const queryResult = await executeSqlQuery(generatedSQL)
      setResult(queryResult)

      if (queryResult.error) {
        toast.error(queryResult.error.message)
      } else {
        toast.success(`Query executed in ${queryResult.executionTime?.toFixed(0)}ms`)
      }
    } catch {
      toast.error('Query execution failed')
    } finally {
      setIsExecuting(false)
    }
  }, [generatedSQL])

  // Copy SQL to clipboard
  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(generatedSQL)
      setCopied(true)
      toast.success('SQL copied to clipboard')
      setTimeout(() => setCopied(false), 2000)
    } catch {
      toast.error('Failed to copy to clipboard')
    }
  }, [generatedSQL])

  // Reset builder
  const handleReset = useCallback(() => {
    setState(createInitialState())
    setResult(null)
  }, [])

  // Available columns for WHERE builder
  const availableColumns = useMemo(() => {
    return state.selectedColumns.map(c => ({ table: c.table, column: c.column }))
  }, [state.selectedColumns])

  // Find join being edited
  const joinBeingEdited = editingJoin ? state.joins.find(j => j.id === editingJoin) : null

  return (
    <div className="flex flex-col h-full bg-bg-primary">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-bg-secondary">
        <div className="flex items-center gap-2">
          <Database size={18} className="text-accent" />
          <h2 className="text-sm font-medium text-text-primary">SQL Builder</h2>
        </div>

        <div className="flex items-center gap-2">
          <IconButton
            icon={<RefreshCw size={16} />}
            onClick={refreshSchema}
            tooltip="Refresh schema"
            disabled={schemaLoading}
          />
          <IconButton
            icon={<Trash2 size={16} />}
            onClick={handleReset}
            tooltip="Reset builder"
          />
          <div className="w-px h-5 bg-border mx-1" />
          <IconButton
            icon={copied ? <Check size={16} /> : <Copy size={16} />}
            onClick={handleCopy}
            tooltip="Copy SQL"
          />
          <button
            onClick={handleExecute}
            disabled={isExecuting || generatedSQL.startsWith('--')}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-accent hover:bg-accent-hover disabled:opacity-50 text-white text-sm rounded transition-colors"
          >
            <Play size={14} />
            {isExecuting ? 'Running...' : 'Run Query'}
          </button>
        </div>
      </div>

      {/* Main content */}
      <Group orientation="horizontal" className="flex-1">
        {/* Left sidebar - Table list */}
        <Panel id="table-list" defaultSize={20} minSize={15} maxSize={30}>
          <div className="h-full overflow-hidden border-r border-border bg-bg-secondary">
            <div className="flex flex-col h-full">
              <div className="px-3 py-2 border-b border-border">
                <h3 className="text-xs font-medium text-text-muted uppercase tracking-wide">
                  Tables
                </h3>
              </div>

              <div className="flex-1 overflow-y-auto">
                {schemaLoading ? (
                  <div className="flex items-center justify-center py-8">
                    <RefreshCw size={16} className="animate-spin text-text-muted" />
                  </div>
                ) : availableTables.length === 0 ? (
                  <div className="px-3 py-4 text-xs text-text-muted text-center">
                    No tables found in schema
                  </div>
                ) : (
                  availableTables.map(tableName => (
                    <TableListItem
                      key={tableName}
                      name={tableName}
                      columns={tableInfo.get(tableName)?.columns ?? []}
                      onAddTable={handleAddTable}
                      onAddColumn={handleAddColumn}
                      isAdded={state.tables.some(t => t.name === tableName)}
                    />
                  ))
                )}
              </div>
            </div>
          </div>
        </Panel>

        <Separator className="w-1 bg-border hover:bg-accent transition-colors cursor-col-resize" />

        {/* Center - Builder area */}
        <Panel id="builder" minSize={40}>
          <Group orientation="vertical" className="h-full">
            {/* Table canvas */}
            <Panel id="canvas" defaultSize={50} minSize={30}>
              <div className="h-full border-b border-border">
                <TableCanvas
                  tables={state.tables}
                  tableInfo={tableInfo}
                  joins={state.joins}
                  availableTables={availableTables}
                  onAddTable={handleAddTable}
                  onMoveTable={handleMoveTable}
                  onRemoveTable={handleRemoveTable}
                  onAddJoin={handleAddJoin}
                  onRemoveJoin={handleRemoveJoin}
                  onEditJoin={setEditingJoin}
                />
              </div>
            </Panel>

            <Separator className="h-1 bg-border hover:bg-accent transition-colors cursor-row-resize" />

            {/* Query configuration */}
            <Panel id="config" defaultSize={50} minSize={20}>
              <div className="h-full overflow-y-auto bg-bg-secondary">
                <CollapsibleSection
                  title="Columns"
                  icon={<Table2 size={16} />}
                  count={state.selectedColumns.length}
                  defaultOpen={true}
                >
                  <div className="px-2 pb-2">
                    <ColumnPicker
                      columns={state.selectedColumns}
                      onUpdate={handleUpdateColumn}
                      onRemove={handleRemoveColumn}
                      onReorder={handleReorderColumns}
                      onAddToOrderBy={handleAddToOrderBy}
                    />
                  </div>
                </CollapsibleSection>

                <CollapsibleSection
                  title="Filters"
                  icon={<span className="text-xs font-mono">WHERE</span>}
                  count={state.whereConditions.length}
                  defaultOpen={state.whereConditions.length > 0}
                >
                  <div className="px-4 pb-2">
                    <WhereBuilder
                      conditions={state.whereConditions}
                      availableColumns={availableColumns}
                      onAdd={handleAddCondition}
                      onUpdate={handleUpdateCondition}
                      onRemove={handleRemoveCondition}
                    />
                  </div>
                </CollapsibleSection>

                {/* Limit input */}
                <div className="px-4 py-3 border-t border-border">
                  <div className="flex items-center gap-2">
                    <label className="text-sm text-text-secondary">LIMIT:</label>
                    <input
                      type="number"
                      value={state.limit ?? ''}
                      onChange={(e) => setState(prev => ({
                        ...prev,
                        limit: e.target.value ? parseInt(e.target.value, 10) : undefined,
                      }))}
                      placeholder="No limit"
                      min={1}
                      className="w-24 px-2 py-1 text-sm bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent text-text-primary"
                    />
                  </div>
                </div>
              </div>
            </Panel>
          </Group>
        </Panel>

        <Separator className="w-1 bg-border hover:bg-accent transition-colors cursor-col-resize" />

        {/* Right panel - SQL preview & results */}
        <Panel id="sql-results" defaultSize={30} minSize={20}>
          <Group orientation="vertical" className="h-full">
            {/* SQL Preview */}
            <Panel id="sql-preview" defaultSize={40} minSize={20}>
              <div className="h-full flex flex-col border-b border-border">
                <div className="px-3 py-2 border-b border-border bg-bg-secondary flex items-center justify-between">
                  <h3 className="text-xs font-medium text-text-muted uppercase tracking-wide">
                    Generated SQL
                  </h3>
                </div>
                <div className="flex-1 overflow-auto p-3 bg-bg-primary">
                  <pre className="text-sm text-text-primary font-mono whitespace-pre-wrap">
                    {generatedSQL}
                  </pre>
                </div>
              </div>
            </Panel>

            <Separator className="h-1 bg-border hover:bg-accent transition-colors cursor-row-resize" />

            {/* Results */}
            <Panel id="results" defaultSize={60} minSize={20}>
              <div className="h-full flex flex-col">
                <div className="px-3 py-2 border-b border-border bg-bg-secondary">
                  <h3 className="text-xs font-medium text-text-muted uppercase tracking-wide">
                    Results
                  </h3>
                </div>
                <div className="flex-1 overflow-hidden">
                  {result ? (
                    <UnifiedResultView result={result} />
                  ) : (
                    <div className="flex items-center justify-center h-full text-text-muted text-sm">
                      Run a query to see results
                    </div>
                  )}
                </div>
              </div>
            </Panel>
          </Group>
        </Panel>
      </Group>

      {/* Join edit modal */}
      {joinBeingEdited && (
        <JoinEditModal
          join={joinBeingEdited}
          onUpdate={(updates) => handleUpdateJoin(joinBeingEdited.id, updates)}
          onDelete={() => handleRemoveJoin(joinBeingEdited.id)}
          onClose={() => setEditingJoin(null)}
        />
      )}
    </div>
  )
}
