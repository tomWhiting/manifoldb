import { useCallback } from 'react'
import { GripVertical, Eye, EyeOff, ChevronDown, ArrowUpDown } from 'lucide-react'
import type { SelectedColumn, AggregateFunction } from '../../lib/sql-generator'

const AGGREGATE_OPTIONS: { value: AggregateFunction; label: string }[] = [
  { value: 'NONE', label: 'None' },
  { value: 'COUNT', label: 'COUNT' },
  { value: 'COUNT_DISTINCT', label: 'COUNT DISTINCT' },
  { value: 'SUM', label: 'SUM' },
  { value: 'AVG', label: 'AVG' },
  { value: 'MIN', label: 'MIN' },
  { value: 'MAX', label: 'MAX' },
]

interface ColumnRowProps {
  column: SelectedColumn
  onUpdate: (id: string, updates: Partial<SelectedColumn>) => void
  onRemove: (id: string) => void
  onMoveUp: (id: string) => void
  onMoveDown: (id: string) => void
  onAddToOrderBy: (table: string, column: string) => void
  isFirst: boolean
  isLast: boolean
}

function ColumnRow({
  column,
  onUpdate,
  onRemove,
  onMoveUp,
  onMoveDown,
  onAddToOrderBy,
  isFirst,
  isLast,
}: ColumnRowProps) {
  return (
    <tr className="border-b border-border hover:bg-bg-tertiary/50 group">
      {/* Drag handle & visibility */}
      <td className="px-2 py-1.5 w-[60px]">
        <div className="flex items-center gap-1">
          <div className="flex flex-col">
            <button
              className="p-0.5 text-text-muted hover:text-text-primary disabled:opacity-30"
              onClick={() => onMoveUp(column.id)}
              disabled={isFirst}
            >
              <ChevronDown size={12} className="rotate-180" />
            </button>
            <button
              className="p-0.5 text-text-muted hover:text-text-primary disabled:opacity-30"
              onClick={() => onMoveDown(column.id)}
              disabled={isLast}
            >
              <ChevronDown size={12} />
            </button>
          </div>
          <button
            className="p-1 hover:bg-border rounded"
            onClick={() => onUpdate(column.id, { visible: !column.visible })}
            title={column.visible ? 'Hide column' : 'Show column'}
          >
            {column.visible ? (
              <Eye size={14} className="text-accent" />
            ) : (
              <EyeOff size={14} className="text-text-muted" />
            )}
          </button>
        </div>
      </td>

      {/* Table */}
      <td className="px-2 py-1.5 text-sm text-text-muted">
        {column.table}
      </td>

      {/* Column */}
      <td className="px-2 py-1.5 text-sm text-text-primary font-medium">
        {column.column}
      </td>

      {/* Alias */}
      <td className="px-2 py-1.5">
        <input
          type="text"
          value={column.alias ?? ''}
          onChange={(e) => onUpdate(column.id, { alias: e.target.value })}
          placeholder="Alias"
          className="w-full px-2 py-1 text-sm bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent text-text-primary"
        />
      </td>

      {/* Aggregate */}
      <td className="px-2 py-1.5">
        <select
          value={column.aggregate}
          onChange={(e) => onUpdate(column.id, { aggregate: e.target.value as AggregateFunction })}
          className="w-full px-2 py-1 text-sm bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent text-text-primary"
        >
          {AGGREGATE_OPTIONS.map(opt => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </td>

      {/* Actions */}
      <td className="px-2 py-1.5 w-[80px]">
        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            className="p-1 hover:bg-border rounded text-text-muted hover:text-text-primary"
            onClick={() => onAddToOrderBy(column.table, column.column)}
            title="Add to ORDER BY"
          >
            <ArrowUpDown size={14} />
          </button>
          <button
            className="p-1 hover:bg-border rounded text-text-muted hover:text-red-400"
            onClick={() => onRemove(column.id)}
            title="Remove column"
          >
            ×
          </button>
        </div>
      </td>
    </tr>
  )
}

interface ColumnPickerProps {
  columns: SelectedColumn[]
  onUpdate: (id: string, updates: Partial<SelectedColumn>) => void
  onRemove: (id: string) => void
  onReorder: (columns: SelectedColumn[]) => void
  onAddToOrderBy: (table: string, column: string) => void
}

export function ColumnPicker({
  columns,
  onUpdate,
  onRemove,
  onReorder,
  onAddToOrderBy,
}: ColumnPickerProps) {
  const handleMoveUp = useCallback(
    (id: string) => {
      const index = columns.findIndex(c => c.id === id)
      if (index <= 0) return

      const newColumns = [...columns]
      const [item] = newColumns.splice(index, 1)
      newColumns.splice(index - 1, 0, item)
      onReorder(newColumns)
    },
    [columns, onReorder]
  )

  const handleMoveDown = useCallback(
    (id: string) => {
      const index = columns.findIndex(c => c.id === id)
      if (index < 0 || index >= columns.length - 1) return

      const newColumns = [...columns]
      const [item] = newColumns.splice(index, 1)
      newColumns.splice(index + 1, 0, item)
      onReorder(newColumns)
    },
    [columns, onReorder]
  )

  if (columns.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-text-muted text-sm py-8">
        Click on columns in the tables above to add them to your query
      </div>
    )
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left">
        <thead>
          <tr className="border-b border-border bg-bg-tertiary text-xs text-text-muted">
            <th className="px-2 py-2 font-medium">
              <GripVertical size={14} className="inline-block" />
            </th>
            <th className="px-2 py-2 font-medium">Table</th>
            <th className="px-2 py-2 font-medium">Column</th>
            <th className="px-2 py-2 font-medium">Alias</th>
            <th className="px-2 py-2 font-medium">Aggregate</th>
            <th className="px-2 py-2 font-medium"></th>
          </tr>
        </thead>
        <tbody>
          {columns.map((column, index) => (
            <ColumnRow
              key={column.id}
              column={column}
              onUpdate={onUpdate}
              onRemove={onRemove}
              onMoveUp={handleMoveUp}
              onMoveDown={handleMoveDown}
              onAddToOrderBy={onAddToOrderBy}
              isFirst={index === 0}
              isLast={index === columns.length - 1}
            />
          ))}
        </tbody>
      </table>
    </div>
  )
}
