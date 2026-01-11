import { Plus, X } from 'lucide-react'
import type { WhereCondition, WhereOperator, WhereLogic } from '../../lib/sql-generator'

const OPERATOR_OPTIONS: { value: WhereOperator; label: string; needsValue: boolean; needsSecondValue: boolean }[] = [
  { value: '=', label: '= (equals)', needsValue: true, needsSecondValue: false },
  { value: '!=', label: '!= (not equals)', needsValue: true, needsSecondValue: false },
  { value: '<', label: '< (less than)', needsValue: true, needsSecondValue: false },
  { value: '<=', label: '<= (less or equal)', needsValue: true, needsSecondValue: false },
  { value: '>', label: '> (greater than)', needsValue: true, needsSecondValue: false },
  { value: '>=', label: '>= (greater or equal)', needsValue: true, needsSecondValue: false },
  { value: 'LIKE', label: 'LIKE', needsValue: true, needsSecondValue: false },
  { value: 'NOT LIKE', label: 'NOT LIKE', needsValue: true, needsSecondValue: false },
  { value: 'IN', label: 'IN', needsValue: true, needsSecondValue: false },
  { value: 'NOT IN', label: 'NOT IN', needsValue: true, needsSecondValue: false },
  { value: 'IS NULL', label: 'IS NULL', needsValue: false, needsSecondValue: false },
  { value: 'IS NOT NULL', label: 'IS NOT NULL', needsValue: false, needsSecondValue: false },
  { value: 'BETWEEN', label: 'BETWEEN', needsValue: true, needsSecondValue: true },
]

interface ColumnOption {
  table: string
  column: string
}

interface ConditionRowProps {
  condition: WhereCondition
  columns: ColumnOption[]
  onUpdate: (id: string, updates: Partial<WhereCondition>) => void
  onRemove: (id: string) => void
  isFirst: boolean
}

function ConditionRow({
  condition,
  columns,
  onUpdate,
  onRemove,
  isFirst,
}: ConditionRowProps) {
  const operatorInfo = OPERATOR_OPTIONS.find(o => o.value === condition.operator)
  const needsValue = operatorInfo?.needsValue ?? true
  const needsSecondValue = operatorInfo?.needsSecondValue ?? false

  return (
    <div className="flex items-start gap-2 py-2 border-b border-border last:border-b-0">
      {/* Logic (AND/OR) */}
      <div className="w-[60px] flex-shrink-0">
        {isFirst ? (
          <span className="text-sm text-text-muted px-2">WHERE</span>
        ) : (
          <select
            value={condition.logic}
            onChange={(e) => onUpdate(condition.id, { logic: e.target.value as WhereLogic })}
            className="w-full px-2 py-1 text-sm bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent text-text-primary"
          >
            <option value="AND">AND</option>
            <option value="OR">OR</option>
          </select>
        )}
      </div>

      {/* Column selector */}
      <div className="flex-1 min-w-[150px]">
        <select
          value={`${condition.table}.${condition.column}`}
          onChange={(e) => {
            const [table, column] = e.target.value.split('.')
            onUpdate(condition.id, { table, column })
          }}
          className="w-full px-2 py-1 text-sm bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent text-text-primary"
        >
          {columns.map(col => (
            <option key={`${col.table}.${col.column}`} value={`${col.table}.${col.column}`}>
              {col.table}.{col.column}
            </option>
          ))}
        </select>
      </div>

      {/* Operator */}
      <div className="w-[140px] flex-shrink-0">
        <select
          value={condition.operator}
          onChange={(e) => onUpdate(condition.id, { operator: e.target.value as WhereOperator })}
          className="w-full px-2 py-1 text-sm bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent text-text-primary"
        >
          {OPERATOR_OPTIONS.map(opt => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>

      {/* Value(s) */}
      <div className="flex-1 min-w-[120px]">
        {needsValue && (
          <div className="flex items-center gap-2">
            <input
              type="text"
              value={condition.value}
              onChange={(e) => onUpdate(condition.id, { value: e.target.value })}
              placeholder={condition.operator === 'IN' || condition.operator === 'NOT IN' ? 'val1, val2, ...' : 'Value'}
              className="flex-1 px-2 py-1 text-sm bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent text-text-primary"
            />
            {needsSecondValue && (
              <>
                <span className="text-text-muted text-sm">and</span>
                <input
                  type="text"
                  value={condition.value2 ?? ''}
                  onChange={(e) => onUpdate(condition.id, { value2: e.target.value })}
                  placeholder="Value 2"
                  className="flex-1 px-2 py-1 text-sm bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent text-text-primary"
                />
              </>
            )}
          </div>
        )}
        {!needsValue && (
          <span className="text-sm text-text-muted italic px-2">No value needed</span>
        )}
      </div>

      {/* Remove button */}
      <button
        className="p-1 hover:bg-border rounded text-text-muted hover:text-red-400 flex-shrink-0"
        onClick={() => onRemove(condition.id)}
      >
        <X size={16} />
      </button>
    </div>
  )
}

interface WhereBuilderProps {
  conditions: WhereCondition[]
  availableColumns: ColumnOption[]
  onAdd: () => void
  onUpdate: (id: string, updates: Partial<WhereCondition>) => void
  onRemove: (id: string) => void
}

export function WhereBuilder({
  conditions,
  availableColumns,
  onAdd,
  onUpdate,
  onRemove,
}: WhereBuilderProps) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium text-text-primary">WHERE Conditions</h3>
        <button
          onClick={onAdd}
          disabled={availableColumns.length === 0}
          className="flex items-center gap-1 px-2 py-1 text-xs bg-accent hover:bg-accent-hover text-white rounded disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Plus size={12} />
          Add Condition
        </button>
      </div>

      {conditions.length === 0 ? (
        <div className="text-center py-4 text-text-muted text-sm">
          No conditions added. Click "Add Condition" to filter results.
        </div>
      ) : (
        <div className="bg-bg-tertiary rounded-lg p-2">
          {conditions.map((condition, index) => (
            <ConditionRow
              key={condition.id}
              condition={condition}
              columns={availableColumns}
              onUpdate={onUpdate}
              onRemove={onRemove}
              isFirst={index === 0}
            />
          ))}
        </div>
      )}
    </div>
  )
}
