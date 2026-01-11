import { useRef, useState, useCallback, useEffect } from 'react'
import { X, GripVertical, Link2 } from 'lucide-react'
import type { TablePosition, TableJoin, JoinType } from '../../lib/sql-generator'

interface TableInfo {
  name: string
  columns: string[]
}

interface TableCardProps {
  table: TablePosition
  columns: string[]
  onMove: (id: string, x: number, y: number) => void
  onRemove: (id: string) => void
  onColumnClick: (table: string, column: string) => void
  isJoining: boolean
  joinSource: { table: string; column: string } | null
}

function TableCard({
  table,
  columns,
  onMove,
  onRemove,
  onColumnClick,
  isJoining,
  joinSource,
}: TableCardProps) {
  const cardRef = useRef<HTMLDivElement>(null)
  const [isDragging, setIsDragging] = useState(false)
  const dragStartPos = useRef({ x: 0, y: 0 })
  const cardStartPos = useRef({ x: 0, y: 0 })

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if ((e.target as HTMLElement).closest('.no-drag')) return

      setIsDragging(true)
      dragStartPos.current = { x: e.clientX, y: e.clientY }
      cardStartPos.current = { x: table.x, y: table.y }
      e.preventDefault()
    },
    [table.x, table.y]
  )

  useEffect(() => {
    if (!isDragging) return

    const handleMouseMove = (e: MouseEvent) => {
      const dx = e.clientX - dragStartPos.current.x
      const dy = e.clientY - dragStartPos.current.y
      onMove(table.id, cardStartPos.current.x + dx, cardStartPos.current.y + dy)
    }

    const handleMouseUp = () => {
      setIsDragging(false)
    }

    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)

    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
  }, [isDragging, onMove, table.id])

  return (
    <div
      ref={cardRef}
      className={`
        absolute min-w-[180px] bg-bg-secondary border rounded-lg shadow-lg
        ${isDragging ? 'cursor-grabbing shadow-xl z-50' : 'cursor-grab'}
        ${isJoining ? 'ring-2 ring-accent' : 'border-border'}
      `}
      style={{
        left: table.x,
        top: table.y,
        userSelect: 'none',
      }}
      onMouseDown={handleMouseDown}
    >
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-border bg-bg-tertiary rounded-t-lg">
        <GripVertical size={14} className="text-text-muted flex-shrink-0" />
        <span className="text-sm font-medium text-text-primary truncate flex-1">
          {table.name}
        </span>
        <button
          className="no-drag p-0.5 hover:bg-border rounded text-text-muted hover:text-text-primary"
          onClick={() => onRemove(table.id)}
        >
          <X size={14} />
        </button>
      </div>

      {/* Columns */}
      <div className="py-1 max-h-[200px] overflow-y-auto">
        {columns.map(column => {
          const isSource = joinSource?.table === table.name && joinSource?.column === column

          return (
            <button
              key={column}
              className={`
                no-drag w-full flex items-center gap-2 px-3 py-1 text-left text-sm
                hover:bg-bg-tertiary transition-colors
                ${isSource ? 'bg-accent/20 text-accent' : 'text-text-secondary'}
              `}
              onClick={() => onColumnClick(table.name, column)}
            >
              <Link2
                size={12}
                className={isSource ? 'text-accent' : 'text-text-muted opacity-0 group-hover:opacity-100'}
              />
              <span className="truncate">{column}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}

interface JoinLineProps {
  from: { x: number; y: number }
  to: { x: number; y: number }
  joinType: JoinType
  onClick: () => void
}

function JoinLine({ from, to, joinType, onClick }: JoinLineProps) {
  const midX = (from.x + to.x) / 2
  const midY = (from.y + to.y) / 2

  // Calculate the path
  const path = `M ${from.x} ${from.y} C ${midX} ${from.y}, ${midX} ${to.y}, ${to.x} ${to.y}`

  return (
    <g className="cursor-pointer" onClick={onClick}>
      {/* Invisible wider path for easier clicking */}
      <path d={path} stroke="transparent" strokeWidth={20} fill="none" />
      {/* Visible path */}
      <path
        d={path}
        stroke="currentColor"
        strokeWidth={2}
        fill="none"
        className="text-accent"
        strokeDasharray={joinType === 'LEFT' || joinType === 'RIGHT' ? '5,5' : undefined}
      />
      {/* Join type label */}
      <rect
        x={midX - 25}
        y={midY - 10}
        width={50}
        height={20}
        rx={4}
        className="fill-bg-secondary stroke-border"
      />
      <text
        x={midX}
        y={midY + 4}
        textAnchor="middle"
        className="text-xs fill-text-secondary pointer-events-none"
      >
        {joinType}
      </text>
    </g>
  )
}

interface TableCanvasProps {
  tables: TablePosition[]
  tableInfo: Map<string, TableInfo>
  joins: TableJoin[]
  onAddTable: (name: string, x: number, y: number) => void
  onMoveTable: (id: string, x: number, y: number) => void
  onRemoveTable: (id: string) => void
  onAddJoin: (join: Omit<TableJoin, 'id'>) => void
  onEditJoin: (id: string) => void
  availableTables: string[]
}

export function TableCanvas({
  tables,
  tableInfo,
  joins,
  onMoveTable,
  onRemoveTable,
  onAddJoin,
  onEditJoin,
  availableTables,
  onAddTable,
}: TableCanvasProps) {
  const canvasRef = useRef<HTMLDivElement>(null)
  const [joinSource, setJoinSource] = useState<{ table: string; column: string } | null>(null)
  const [showTableMenu, setShowTableMenu] = useState(false)
  const [menuPosition, setMenuPosition] = useState({ x: 0, y: 0 })

  const handleColumnClick = useCallback(
    (table: string, column: string) => {
      if (joinSource === null) {
        // Start join
        setJoinSource({ table, column })
      } else if (joinSource.table === table) {
        // Cancel join (clicked same table)
        setJoinSource(null)
      } else {
        // Complete join
        onAddJoin({
          fromTable: joinSource.table,
          fromColumn: joinSource.column,
          toTable: table,
          toColumn: column,
          joinType: 'INNER',
        })
        setJoinSource(null)
      }
    },
    [joinSource, onAddJoin]
  )

  const handleCanvasDoubleClick = useCallback(
    (e: React.MouseEvent) => {
      if ((e.target as HTMLElement).closest('.table-card')) return

      const rect = canvasRef.current?.getBoundingClientRect()
      if (!rect) return

      setMenuPosition({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      })
      setShowTableMenu(true)
    },
    []
  )

  const handleAddTableFromMenu = useCallback(
    (tableName: string) => {
      onAddTable(tableName, menuPosition.x, menuPosition.y)
      setShowTableMenu(false)
    },
    [menuPosition, onAddTable]
  )

  // Calculate join line positions
  const getJoinPositions = useCallback(
    (join: TableJoin) => {
      const fromTable = tables.find(t => t.name === join.fromTable)
      const toTable = tables.find(t => t.name === join.toTable)

      if (!fromTable || !toTable) return null

      // Get column positions within tables
      const fromInfo = tableInfo.get(join.fromTable)
      const toInfo = tableInfo.get(join.toTable)

      if (!fromInfo || !toInfo) return null

      const fromColIndex = fromInfo.columns.indexOf(join.fromColumn)
      const toColIndex = toInfo.columns.indexOf(join.toColumn)

      // Calculate positions (card width is ~180px, header is ~36px, each row is ~28px)
      const cardWidth = 180
      const headerHeight = 36
      const rowHeight = 28

      const fromY = fromTable.y + headerHeight + (fromColIndex + 0.5) * rowHeight
      const toY = toTable.y + headerHeight + (toColIndex + 0.5) * rowHeight

      // Connect from right edge of left table to left edge of right table
      let fromX: number, toX: number
      if (fromTable.x < toTable.x) {
        fromX = fromTable.x + cardWidth
        toX = toTable.x
      } else {
        fromX = fromTable.x
        toX = toTable.x + cardWidth
      }

      return { from: { x: fromX, y: fromY }, to: { x: toX, y: toY } }
    },
    [tables, tableInfo]
  )

  // Filter available tables to only those not already added
  const addedTableNames = new Set(tables.map(t => t.name))
  const remainingTables = availableTables.filter(t => !addedTableNames.has(t))

  return (
    <div
      ref={canvasRef}
      className="relative w-full h-full bg-bg-primary overflow-hidden"
      onDoubleClick={handleCanvasDoubleClick}
      onClick={() => {
        if (showTableMenu) setShowTableMenu(false)
        if (joinSource) setJoinSource(null)
      }}
    >
      {/* Grid background */}
      <div
        className="absolute inset-0 opacity-10"
        style={{
          backgroundImage:
            'linear-gradient(to right, var(--border) 1px, transparent 1px), linear-gradient(to bottom, var(--border) 1px, transparent 1px)',
          backgroundSize: '20px 20px',
        }}
      />

      {/* SVG for join lines - pointer-events enabled for join line clicks */}
      <svg className="absolute inset-0 w-full h-full" style={{ pointerEvents: 'none' }}>
        <g style={{ pointerEvents: 'auto' }}>
        {joins.map(join => {
          const positions = getJoinPositions(join)
          if (!positions) return null

          return (
            <JoinLine
              key={join.id}
              from={positions.from}
              to={positions.to}
              joinType={join.joinType}
              onClick={() => onEditJoin(join.id)}
            />
          )
        })}
        </g>
      </svg>

      {/* Table cards */}
      {tables.map(table => {
        const info = tableInfo.get(table.name)
        return (
          <TableCard
            key={table.id}
            table={table}
            columns={info?.columns ?? []}
            onMove={onMoveTable}
            onRemove={onRemoveTable}
            onColumnClick={handleColumnClick}
            isJoining={joinSource !== null}
            joinSource={joinSource}
          />
        )
      })}

      {/* Empty state */}
      {tables.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center text-text-muted">
            <p className="text-lg mb-2">No tables selected</p>
            <p className="text-sm">
              Double-click to add a table, or drag from the table list
            </p>
          </div>
        </div>
      )}

      {/* Join mode indicator */}
      {joinSource && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-accent/90 text-white px-4 py-2 rounded-full text-sm">
          Click a column in another table to create a join, or click anywhere to cancel
        </div>
      )}

      {/* Add table menu */}
      {showTableMenu && remainingTables.length > 0 && (
        <div
          className="absolute bg-bg-secondary border border-border rounded-lg shadow-xl z-50 py-1 max-h-[300px] overflow-y-auto min-w-[150px]"
          style={{ left: menuPosition.x, top: menuPosition.y }}
          onClick={e => e.stopPropagation()}
        >
          <div className="px-3 py-1.5 text-xs text-text-muted border-b border-border">
            Add Table
          </div>
          {remainingTables.map(tableName => (
            <button
              key={tableName}
              className="w-full text-left px-3 py-1.5 text-sm text-text-secondary hover:bg-bg-tertiary hover:text-text-primary"
              onClick={() => handleAddTableFromMenu(tableName)}
            >
              {tableName}
            </button>
          ))}
        </div>
      )}

      {showTableMenu && remainingTables.length === 0 && (
        <div
          className="absolute bg-bg-secondary border border-border rounded-lg shadow-xl z-50 py-2 px-3"
          style={{ left: menuPosition.x, top: menuPosition.y }}
          onClick={e => e.stopPropagation()}
        >
          <p className="text-sm text-text-muted">All tables have been added</p>
        </div>
      )}
    </div>
  )
}
