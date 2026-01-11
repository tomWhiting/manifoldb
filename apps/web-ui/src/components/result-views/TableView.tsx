import { useMemo, useState, useRef, useCallback } from 'react'
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
  type VisibilityState,
  type ColumnResizeMode,
  type RowSelectionState,
} from '@tanstack/react-table'
import { useVirtualizer } from '@tanstack/react-virtual'
import {
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  Copy,
  Download,
  Columns,
  ChevronDown,
  ChevronRight,
} from 'lucide-react'
import { useAppStore } from '../../stores/app-store'
import type { GraphNode } from '../../types'
import {
  type RowData,
  type CellValue,
  formatCellValue,
  truncateString,
  isExpandableValue,
  rowsToCSV,
  rowsToClipboardText,
  downloadCSV,
  copyToClipboard,
  mixedSort,
} from '../../lib/table-utils'
import { toast } from 'sonner'

const ROW_HEIGHT = 36
const MAX_CELL_LENGTH = 100

export function TableView() {
  const tabs = useAppStore((s) => s.tabs)
  const activeTabId = useAppStore((s) => s.activeTabId)
  const activeTab = tabs.find((t) => t.id === activeTabId)
  const result = activeTab?.result

  if (!result) {
    return (
      <div className="flex items-center justify-center h-full text-text-muted">
        Run a query to see results
      </div>
    )
  }

  if (result.error) {
    return <ErrorDisplay error={result.error} />
  }

  // SQL results with explicit columns
  if (result.rows && result.columns) {
    return <EnhancedTable rows={result.rows} columns={result.columns} />
  }

  // SQL results without explicit columns (infer from rows)
  if (result.rows && result.rows.length > 0) {
    const columns = Object.keys(result.rows[0])
    return <EnhancedTable rows={result.rows} columns={columns} />
  }

  // Cypher results
  if (result.nodes && result.nodes.length > 0) {
    return <CypherNodesTable nodes={result.nodes} />
  }

  return (
    <div className="flex items-center justify-center h-full text-text-muted">
      No data to display
    </div>
  )
}

interface ErrorDisplayProps {
  error: {
    message: string
    line?: number
    column?: number
  }
}

function ErrorDisplay({ error }: ErrorDisplayProps) {
  return (
    <div className="flex items-center justify-center h-full p-4">
      <div className="max-w-lg w-full bg-red-500/10 border border-red-500/30 rounded-lg p-4">
        <h3 className="text-red-500 font-medium mb-2">Query Error</h3>
        <p className="text-text-secondary text-sm font-mono whitespace-pre-wrap">
          {error.message}
        </p>
        {(error.line !== undefined || error.column !== undefined) && (
          <p className="text-text-muted text-xs mt-2">
            {error.line !== undefined && `Line ${error.line}`}
            {error.line !== undefined && error.column !== undefined && ', '}
            {error.column !== undefined && `Column ${error.column}`}
          </p>
        )}
      </div>
    </div>
  )
}

interface EnhancedTableProps {
  rows: Record<string, unknown>[]
  columns: string[]
}

function EnhancedTable({ rows, columns: columnKeys }: EnhancedTableProps) {
  const [sorting, setSorting] = useState<SortingState>([])
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>({})
  const [rowSelection, setRowSelection] = useState<RowSelectionState>({})
  const [columnSizeMode] = useState<ColumnResizeMode>('onChange')
  const [showColumnMenu, setShowColumnMenu] = useState(false)
  const tableContainerRef = useRef<HTMLDivElement>(null)

  const columns: ColumnDef<RowData>[] = useMemo(
    () => [
      // Selection column
      {
        id: 'select',
        header: ({ table }) => (
          <input
            type="checkbox"
            checked={table.getIsAllRowsSelected()}
            onChange={table.getToggleAllRowsSelectedHandler()}
            className="w-4 h-4 rounded border-border bg-bg-secondary accent-accent cursor-pointer"
          />
        ),
        cell: ({ row }) => (
          <input
            type="checkbox"
            checked={row.getIsSelected()}
            onChange={row.getToggleSelectedHandler()}
            className="w-4 h-4 rounded border-border bg-bg-secondary accent-accent cursor-pointer"
          />
        ),
        size: 40,
        enableResizing: false,
        enableSorting: false,
      },
      // Data columns
      ...columnKeys.map((key) => ({
        accessorKey: key,
        header: key,
        cell: ({ getValue }: { getValue: () => CellValue }) => (
          <CellRenderer value={getValue()} />
        ),
        sortingFn: mixedSort,
        enableResizing: true,
        size: 150,
        minSize: 50,
        maxSize: 500,
      })),
    ],
    [columnKeys]
  )

  const table = useReactTable<RowData>({
    data: rows,
    columns,
    state: {
      sorting,
      columnVisibility,
      rowSelection,
    },
    onSortingChange: setSorting,
    onColumnVisibilityChange: setColumnVisibility,
    onRowSelectionChange: setRowSelection,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    columnResizeMode: columnSizeMode,
    enableRowSelection: true,
  })

  const { rows: tableRows } = table.getRowModel()

  const rowVirtualizer = useVirtualizer({
    count: tableRows.length,
    getScrollElement: () => tableContainerRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 10,
  })

  const virtualRows = rowVirtualizer.getVirtualItems()
  const totalSize = rowVirtualizer.getTotalSize()

  const paddingTop = virtualRows.length > 0 ? virtualRows[0]?.start ?? 0 : 0
  const paddingBottom =
    virtualRows.length > 0
      ? totalSize - (virtualRows[virtualRows.length - 1]?.end ?? 0)
      : 0

  const handleCopySelected = useCallback(async () => {
    const selectedIndices = new Set(
      Object.keys(rowSelection)
        .filter((key) => rowSelection[key])
        .map(Number)
    )
    if (selectedIndices.size === 0) {
      toast.error('No rows selected')
      return
    }
    const text = rowsToClipboardText(rows, columnKeys, selectedIndices)
    const success = await copyToClipboard(text)
    if (success) {
      toast.success(`Copied ${selectedIndices.size} row(s) to clipboard`)
    } else {
      toast.error('Failed to copy to clipboard')
    }
  }, [rows, columnKeys, rowSelection])

  const handleExportCSV = useCallback(() => {
    const selectedIndices = new Set(
      Object.keys(rowSelection)
        .filter((key) => rowSelection[key])
        .map(Number)
    )
    const csv = rowsToCSV(
      rows,
      columnKeys,
      selectedIndices.size > 0 ? selectedIndices : undefined
    )
    const filename = `query-results-${Date.now()}.csv`
    downloadCSV(csv, filename)
    toast.success(
      selectedIndices.size > 0
        ? `Exported ${selectedIndices.size} row(s) to CSV`
        : `Exported ${rows.length} row(s) to CSV`
    )
  }, [rows, columnKeys, rowSelection])

  if (rows.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-text-muted">
        <p>Query executed successfully</p>
        <p className="text-sm">0 rows returned</p>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-border bg-bg-secondary">
        <span className="text-xs text-text-muted">
          {rows.length.toLocaleString()} rows
          {Object.keys(rowSelection).filter((k) => rowSelection[k]).length > 0 &&
            ` • ${Object.keys(rowSelection).filter((k) => rowSelection[k]).length} selected`}
        </span>
        <div className="flex-1" />

        {/* Column visibility dropdown */}
        <div className="relative">
          <button
            onClick={() => setShowColumnMenu(!showColumnMenu)}
            className="flex items-center gap-1 px-2 py-1 text-xs text-text-secondary hover:text-text-primary hover:bg-bg-tertiary rounded transition-colors"
          >
            <Columns className="w-3.5 h-3.5" />
            Columns
          </button>
          {showColumnMenu && (
            <div className="absolute right-0 top-full mt-1 w-48 bg-bg-secondary border border-border rounded-lg shadow-lg z-50 py-1 max-h-64 overflow-auto">
              {table
                .getAllLeafColumns()
                .filter((col) => col.id !== 'select')
                .map((column) => (
                  <label
                    key={column.id}
                    className="flex items-center gap-2 px-3 py-1.5 hover:bg-bg-tertiary cursor-pointer text-sm text-text-secondary"
                  >
                    <input
                      type="checkbox"
                      checked={column.getIsVisible()}
                      onChange={column.getToggleVisibilityHandler()}
                      className="w-4 h-4 rounded border-border bg-bg-secondary accent-accent"
                    />
                    {column.id}
                  </label>
                ))}
            </div>
          )}
        </div>

        <button
          onClick={handleCopySelected}
          className="flex items-center gap-1 px-2 py-1 text-xs text-text-secondary hover:text-text-primary hover:bg-bg-tertiary rounded transition-colors"
          title="Copy selected rows (or all if none selected)"
        >
          <Copy className="w-3.5 h-3.5" />
          Copy
        </button>
        <button
          onClick={handleExportCSV}
          className="flex items-center gap-1 px-2 py-1 text-xs text-text-secondary hover:text-text-primary hover:bg-bg-tertiary rounded transition-colors"
          title="Export to CSV"
        >
          <Download className="w-3.5 h-3.5" />
          CSV
        </button>
      </div>

      {/* Table */}
      <div
        ref={tableContainerRef}
        className="flex-1 overflow-auto"
        onClick={() => setShowColumnMenu(false)}
      >
        <table
          className="w-full text-sm"
          style={{ width: table.getCenterTotalSize() }}
        >
          <thead className="sticky top-0 bg-bg-secondary z-10">
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <th
                    key={header.id}
                    className="relative px-3 py-2 text-left text-xs font-medium text-text-muted uppercase border-b border-border group"
                    style={{ width: header.getSize() }}
                  >
                    {header.isPlaceholder ? null : (
                      <div
                        className={`flex items-center gap-1 ${
                          header.column.getCanSort()
                            ? 'cursor-pointer select-none hover:text-text-primary'
                            : ''
                        }`}
                        onClick={header.column.getToggleSortingHandler()}
                      >
                        {flexRender(
                          header.column.columnDef.header,
                          header.getContext()
                        )}
                        {header.column.getCanSort() && (
                          <span className="ml-1">
                            {header.column.getIsSorted() === 'asc' ? (
                              <ArrowUp className="w-3 h-3" />
                            ) : header.column.getIsSorted() === 'desc' ? (
                              <ArrowDown className="w-3 h-3" />
                            ) : (
                              <ArrowUpDown className="w-3 h-3 opacity-30 group-hover:opacity-70" />
                            )}
                          </span>
                        )}
                      </div>
                    )}
                    {/* Resize handle */}
                    {header.column.getCanResize() && (
                      <div
                        onMouseDown={header.getResizeHandler()}
                        onTouchStart={header.getResizeHandler()}
                        className={`absolute right-0 top-0 h-full w-1 cursor-col-resize select-none touch-none bg-border opacity-0 hover:opacity-100 ${
                          header.column.getIsResizing()
                            ? 'opacity-100 bg-accent'
                            : ''
                        }`}
                      />
                    )}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {paddingTop > 0 && (
              <tr>
                <td style={{ height: `${paddingTop}px` }} />
              </tr>
            )}
            {virtualRows.map((virtualRow) => {
              const row = tableRows[virtualRow.index]
              return (
                <tr
                  key={row.id}
                  className={`
                    ${virtualRow.index % 2 === 0 ? 'bg-bg-primary' : 'bg-bg-secondary/50'}
                    ${row.getIsSelected() ? 'bg-accent-muted' : ''}
                    hover:bg-bg-tertiary transition-colors
                  `}
                  style={{ height: ROW_HEIGHT }}
                >
                  {row.getVisibleCells().map((cell) => (
                    <td
                      key={cell.id}
                      className="px-3 py-2 text-text-secondary"
                      style={{ width: cell.column.getSize() }}
                    >
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext()
                      )}
                    </td>
                  ))}
                </tr>
              )
            })}
            {paddingBottom > 0 && (
              <tr>
                <td style={{ height: `${paddingBottom}px` }} />
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// Cypher nodes table with special handling
function CypherNodesTable({ nodes }: { nodes: GraphNode[] }) {
  // Convert nodes to rows
  const propertyKeys = useMemo(() => {
    const keys = new Set<string>()
    nodes.forEach((node) => {
      Object.keys(node.properties).forEach((key) => keys.add(key))
    })
    return Array.from(keys)
  }, [nodes])

  const rows: RowData[] = useMemo(
    () =>
      nodes.map((node) => ({
        id: node.id,
        labels: node.labels,
        ...node.properties,
      })),
    [nodes]
  )

  const columns = useMemo(
    () => ['id', 'labels', ...propertyKeys],
    [propertyKeys]
  )

  const [sorting, setSorting] = useState<SortingState>([])
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>({})
  const [rowSelection, setRowSelection] = useState<RowSelectionState>({})
  const [showColumnMenu, setShowColumnMenu] = useState(false)
  const tableContainerRef = useRef<HTMLDivElement>(null)

  const columnDefs: ColumnDef<RowData>[] = useMemo(
    () => [
      // Selection column
      {
        id: 'select',
        header: ({ table }) => (
          <input
            type="checkbox"
            checked={table.getIsAllRowsSelected()}
            onChange={table.getToggleAllRowsSelectedHandler()}
            className="w-4 h-4 rounded border-border bg-bg-secondary accent-accent cursor-pointer"
          />
        ),
        cell: ({ row }) => (
          <input
            type="checkbox"
            checked={row.getIsSelected()}
            onChange={row.getToggleSelectedHandler()}
            className="w-4 h-4 rounded border-border bg-bg-secondary accent-accent cursor-pointer"
          />
        ),
        size: 40,
        enableResizing: false,
        enableSorting: false,
      },
      // ID column
      {
        accessorKey: 'id',
        header: 'ID',
        cell: ({ getValue }: { getValue: () => CellValue }) => (
          <span className="font-mono text-xs">{String(getValue())}</span>
        ),
        sortingFn: mixedSort,
        size: 120,
      },
      // Labels column
      {
        accessorKey: 'labels',
        header: 'Labels',
        cell: ({ getValue }: { getValue: () => CellValue }) => {
          const labels = getValue() as string[]
          return (
            <div className="flex flex-wrap gap-1">
              {labels.map((label) => (
                <span
                  key={label}
                  className="inline-block px-1.5 py-0.5 text-xs bg-accent-muted text-accent rounded"
                >
                  {label}
                </span>
              ))}
            </div>
          )
        },
        sortingFn: (rowA, rowB) => {
          const a = (rowA.getValue('labels') as string[]).join(',')
          const b = (rowB.getValue('labels') as string[]).join(',')
          return a.localeCompare(b)
        },
        size: 150,
      },
      // Property columns
      ...propertyKeys.map((key) => ({
        accessorKey: key,
        header: key,
        cell: ({ getValue }: { getValue: () => CellValue }) => (
          <CellRenderer value={getValue()} />
        ),
        sortingFn: mixedSort,
        enableResizing: true,
        size: 150,
        minSize: 50,
        maxSize: 500,
      })),
    ],
    [propertyKeys]
  )

  const table = useReactTable({
    data: rows,
    columns: columnDefs,
    state: {
      sorting,
      columnVisibility,
      rowSelection,
    },
    onSortingChange: setSorting,
    onColumnVisibilityChange: setColumnVisibility,
    onRowSelectionChange: setRowSelection,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    columnResizeMode: 'onChange',
    enableRowSelection: true,
  })

  const { rows: tableRows } = table.getRowModel()

  const rowVirtualizer = useVirtualizer({
    count: tableRows.length,
    getScrollElement: () => tableContainerRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 10,
  })

  const virtualRows = rowVirtualizer.getVirtualItems()
  const totalSize = rowVirtualizer.getTotalSize()

  const paddingTop = virtualRows.length > 0 ? virtualRows[0]?.start ?? 0 : 0
  const paddingBottom =
    virtualRows.length > 0
      ? totalSize - (virtualRows[virtualRows.length - 1]?.end ?? 0)
      : 0

  const handleCopySelected = useCallback(async () => {
    const selectedIndices = new Set(
      Object.keys(rowSelection)
        .filter((key) => rowSelection[key])
        .map(Number)
    )
    if (selectedIndices.size === 0) {
      toast.error('No rows selected')
      return
    }
    const text = rowsToClipboardText(rows, columns, selectedIndices)
    const success = await copyToClipboard(text)
    if (success) {
      toast.success(`Copied ${selectedIndices.size} row(s) to clipboard`)
    } else {
      toast.error('Failed to copy to clipboard')
    }
  }, [rows, columns, rowSelection])

  const handleExportCSV = useCallback(() => {
    const selectedIndices = new Set(
      Object.keys(rowSelection)
        .filter((key) => rowSelection[key])
        .map(Number)
    )
    const csv = rowsToCSV(
      rows,
      columns,
      selectedIndices.size > 0 ? selectedIndices : undefined
    )
    const filename = `query-results-${Date.now()}.csv`
    downloadCSV(csv, filename)
    toast.success(
      selectedIndices.size > 0
        ? `Exported ${selectedIndices.size} row(s) to CSV`
        : `Exported ${rows.length} row(s) to CSV`
    )
  }, [rows, columns, rowSelection])

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-border bg-bg-secondary">
        <span className="text-xs text-text-muted">
          {nodes.length.toLocaleString()} nodes
          {Object.keys(rowSelection).filter((k) => rowSelection[k]).length > 0 &&
            ` • ${Object.keys(rowSelection).filter((k) => rowSelection[k]).length} selected`}
        </span>
        <div className="flex-1" />

        {/* Column visibility dropdown */}
        <div className="relative">
          <button
            onClick={() => setShowColumnMenu(!showColumnMenu)}
            className="flex items-center gap-1 px-2 py-1 text-xs text-text-secondary hover:text-text-primary hover:bg-bg-tertiary rounded transition-colors"
          >
            <Columns className="w-3.5 h-3.5" />
            Columns
          </button>
          {showColumnMenu && (
            <div className="absolute right-0 top-full mt-1 w-48 bg-bg-secondary border border-border rounded-lg shadow-lg z-50 py-1 max-h-64 overflow-auto">
              {table
                .getAllLeafColumns()
                .filter((col) => col.id !== 'select')
                .map((column) => (
                  <label
                    key={column.id}
                    className="flex items-center gap-2 px-3 py-1.5 hover:bg-bg-tertiary cursor-pointer text-sm text-text-secondary"
                  >
                    <input
                      type="checkbox"
                      checked={column.getIsVisible()}
                      onChange={column.getToggleVisibilityHandler()}
                      className="w-4 h-4 rounded border-border bg-bg-secondary accent-accent"
                    />
                    {column.id}
                  </label>
                ))}
            </div>
          )}
        </div>

        <button
          onClick={handleCopySelected}
          className="flex items-center gap-1 px-2 py-1 text-xs text-text-secondary hover:text-text-primary hover:bg-bg-tertiary rounded transition-colors"
          title="Copy selected rows"
        >
          <Copy className="w-3.5 h-3.5" />
          Copy
        </button>
        <button
          onClick={handleExportCSV}
          className="flex items-center gap-1 px-2 py-1 text-xs text-text-secondary hover:text-text-primary hover:bg-bg-tertiary rounded transition-colors"
          title="Export to CSV"
        >
          <Download className="w-3.5 h-3.5" />
          CSV
        </button>
      </div>

      {/* Table */}
      <div
        ref={tableContainerRef}
        className="flex-1 overflow-auto"
        onClick={() => setShowColumnMenu(false)}
      >
        <table
          className="w-full text-sm"
          style={{ width: table.getCenterTotalSize() }}
        >
          <thead className="sticky top-0 bg-bg-secondary z-10">
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <th
                    key={header.id}
                    className="relative px-3 py-2 text-left text-xs font-medium text-text-muted uppercase border-b border-border group"
                    style={{ width: header.getSize() }}
                  >
                    {header.isPlaceholder ? null : (
                      <div
                        className={`flex items-center gap-1 ${
                          header.column.getCanSort()
                            ? 'cursor-pointer select-none hover:text-text-primary'
                            : ''
                        }`}
                        onClick={header.column.getToggleSortingHandler()}
                      >
                        {flexRender(
                          header.column.columnDef.header,
                          header.getContext()
                        )}
                        {header.column.getCanSort() && (
                          <span className="ml-1">
                            {header.column.getIsSorted() === 'asc' ? (
                              <ArrowUp className="w-3 h-3" />
                            ) : header.column.getIsSorted() === 'desc' ? (
                              <ArrowDown className="w-3 h-3" />
                            ) : (
                              <ArrowUpDown className="w-3 h-3 opacity-30 group-hover:opacity-70" />
                            )}
                          </span>
                        )}
                      </div>
                    )}
                    {/* Resize handle */}
                    {header.column.getCanResize() && (
                      <div
                        onMouseDown={header.getResizeHandler()}
                        onTouchStart={header.getResizeHandler()}
                        className={`absolute right-0 top-0 h-full w-1 cursor-col-resize select-none touch-none bg-border opacity-0 hover:opacity-100 ${
                          header.column.getIsResizing()
                            ? 'opacity-100 bg-accent'
                            : ''
                        }`}
                      />
                    )}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {paddingTop > 0 && (
              <tr>
                <td style={{ height: `${paddingTop}px` }} />
              </tr>
            )}
            {virtualRows.map((virtualRow) => {
              const row = tableRows[virtualRow.index]
              return (
                <tr
                  key={row.id}
                  className={`
                    ${virtualRow.index % 2 === 0 ? 'bg-bg-primary' : 'bg-bg-secondary/50'}
                    ${row.getIsSelected() ? 'bg-accent-muted' : ''}
                    hover:bg-bg-tertiary transition-colors
                  `}
                  style={{ height: ROW_HEIGHT }}
                >
                  {row.getVisibleCells().map((cell) => (
                    <td
                      key={cell.id}
                      className="px-3 py-2 text-text-secondary"
                      style={{ width: cell.column.getSize() }}
                    >
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext()
                      )}
                    </td>
                  ))}
                </tr>
              )
            })}
            {paddingBottom > 0 && (
              <tr>
                <td style={{ height: `${paddingBottom}px` }} />
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// Cell renderer with expandable JSON/arrays and truncation
function CellRenderer({ value }: { value: CellValue }) {
  const [expanded, setExpanded] = useState(false)

  if (value === null) {
    return <span className="text-text-muted italic">NULL</span>
  }

  if (value === undefined) {
    return <span className="text-text-muted">{'\u2014'}</span>
  }

  if (typeof value === 'boolean') {
    return (
      <span className={value ? 'text-green-500' : 'text-red-500'}>
        {value ? 'true' : 'false'}
      </span>
    )
  }

  if (isExpandableValue(value)) {
    const jsonStr = JSON.stringify(value, null, expanded ? 2 : undefined)
    const isLong = jsonStr.length > MAX_CELL_LENGTH

    return (
      <div className="group relative">
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-1 text-left"
        >
          {expanded ? (
            <ChevronDown className="w-3 h-3 text-text-muted shrink-0" />
          ) : (
            <ChevronRight className="w-3 h-3 text-text-muted shrink-0" />
          )}
          <span className="font-mono text-xs">
            {expanded
              ? jsonStr
              : isLong
                ? truncateString(jsonStr, MAX_CELL_LENGTH)
                : jsonStr}
          </span>
        </button>
        {!expanded && isLong && (
          <div className="absolute left-0 top-full mt-1 p-2 bg-bg-tertiary border border-border rounded shadow-lg z-50 hidden group-hover:block max-w-md">
            <pre className="text-xs font-mono whitespace-pre-wrap break-all">
              {JSON.stringify(value, null, 2)}
            </pre>
          </div>
        )}
      </div>
    )
  }

  const strValue = formatCellValue(value)
  const isLong = strValue.length > MAX_CELL_LENGTH

  if (isLong) {
    return (
      <div className="group relative">
        <span>{truncateString(strValue, MAX_CELL_LENGTH)}</span>
        <div className="absolute left-0 top-full mt-1 p-2 bg-bg-tertiary border border-border rounded shadow-lg z-50 hidden group-hover:block max-w-md">
          <span className="text-xs whitespace-pre-wrap break-all">
            {strValue}
          </span>
        </div>
      </div>
    )
  }

  return <span>{strValue}</span>
}
