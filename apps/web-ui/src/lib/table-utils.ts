import type { Row } from '@tanstack/react-table'

export type CellValue = unknown

export type RowData = Record<string, unknown>

export type TableSortingFn = (
  rowA: Row<RowData>,
  rowB: Row<RowData>,
  columnId: string
) => number

// Value formatting utilities
export function formatCellValue(value: CellValue): string {
  if (value === null) return 'NULL'
  if (value === undefined) return '\u2014' // em dash
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (typeof value === 'number') return formatNumber(value)
  if (value instanceof Date) return formatDate(value)
  if (typeof value === 'string') {
    // Try parsing as ISO date
    if (isISODateString(value)) {
      return formatDate(new Date(value))
    }
    return value
  }
  if (typeof value === 'object') {
    return JSON.stringify(value)
  }
  return String(value)
}

export function formatNumber(value: number): string {
  if (Number.isInteger(value)) {
    return value.toLocaleString()
  }
  // For floats, show up to 6 decimal places without trailing zeros
  return value.toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: 6,
  })
}

export function formatDate(date: Date): string {
  if (isNaN(date.getTime())) return 'Invalid Date'
  return date.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

export function isISODateString(value: string): boolean {
  // Check for ISO 8601 date formats
  const isoPattern = /^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?)?$/
  return isoPattern.test(value)
}

export function isExpandableValue(value: CellValue): boolean {
  if (value === null || value === undefined) return false
  if (typeof value !== 'object') return false
  if (Array.isArray(value)) return value.length > 0
  return Object.keys(value).length > 0
}

export function truncateString(value: string, maxLength: number): string {
  if (value.length <= maxLength) return value
  return value.slice(0, maxLength - 1) + '\u2026' // ellipsis
}

// Custom sorting function that handles nulls and mixed types
export const mixedSort: TableSortingFn = (rowA, rowB, columnId) => {
  const a = rowA.getValue(columnId) as CellValue
  const b = rowB.getValue(columnId) as CellValue

  // Handle nulls - always sort to bottom
  if (a === null || a === undefined) return 1
  if (b === null || b === undefined) return -1

  // Same types - standard comparison
  if (typeof a === 'number' && typeof b === 'number') {
    return a - b
  }

  if (typeof a === 'string' && typeof b === 'string') {
    return a.localeCompare(b)
  }

  if (typeof a === 'boolean' && typeof b === 'boolean') {
    return a === b ? 0 : a ? -1 : 1
  }

  // Dates
  if (a instanceof Date && b instanceof Date) {
    return a.getTime() - b.getTime()
  }

  // Mixed types - convert to string
  return String(a).localeCompare(String(b))
}

// Export utilities
export function rowsToCSV(
  rows: RowData[],
  columns: string[],
  selectedIndices?: Set<number>
): string {
  const targetRows = selectedIndices
    ? rows.filter((_, index) => selectedIndices.has(index))
    : rows

  const escapeCSV = (value: string): string => {
    if (value.includes(',') || value.includes('"') || value.includes('\n')) {
      return `"${value.replace(/"/g, '""')}"`
    }
    return value
  }

  const header = columns.map(escapeCSV).join(',')
  const dataRows = targetRows.map((row) =>
    columns.map((col) => escapeCSV(formatCellValue(row[col]))).join(',')
  )

  return [header, ...dataRows].join('\n')
}

export function rowsToClipboardText(
  rows: RowData[],
  columns: string[],
  selectedIndices?: Set<number>
): string {
  const targetRows = selectedIndices
    ? rows.filter((_, index) => selectedIndices.has(index))
    : rows

  const header = columns.join('\t')
  const dataRows = targetRows.map((row) =>
    columns.map((col) => formatCellValue(row[col])).join('\t')
  )

  return [header, ...dataRows].join('\n')
}

export function downloadCSV(content: string, filename: string): void {
  const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}

export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    return false
  }
}
