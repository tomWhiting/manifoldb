import type { GraphNode, GraphEdge, QueryResult } from '../types'

export type ExportFormat = 'json' | 'csv'

export interface ExportOptions {
  format: ExportFormat
  prettyPrint?: boolean
  includeNodes?: boolean
  includeEdges?: boolean
  delimiter?: string
  encoding?: 'utf-8' | 'utf-16'
}

export interface ExportProgress {
  phase: 'preparing' | 'exporting' | 'complete' | 'error'
  current: number
  total: number
  message: string
}

export type ExportProgressCallback = (progress: ExportProgress) => void

export interface ExportData {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

function escapeCSVField(value: unknown, delimiter: string): string {
  if (value === null || value === undefined) {
    return ''
  }

  const stringValue = typeof value === 'object' ? JSON.stringify(value) : String(value)
  const needsQuoting = stringValue.includes(delimiter) ||
                       stringValue.includes('"') ||
                       stringValue.includes('\n') ||
                       stringValue.includes('\r')

  if (needsQuoting) {
    return `"${stringValue.replace(/"/g, '""')}"`
  }

  return stringValue
}

export function exportToJson(
  data: ExportData,
  options: ExportOptions,
  onProgress?: ExportProgressCallback
): string {
  const { prettyPrint = true, includeNodes = true, includeEdges = true } = options

  onProgress?.({ phase: 'preparing', current: 0, total: 1, message: 'Preparing JSON export...' })

  const exportObj: { nodes?: GraphNode[]; edges?: GraphEdge[] } = {}

  if (includeNodes) {
    exportObj.nodes = data.nodes
  }
  if (includeEdges) {
    exportObj.edges = data.edges
  }

  onProgress?.({ phase: 'exporting', current: 0, total: 1, message: 'Generating JSON...' })

  const result = prettyPrint
    ? JSON.stringify(exportObj, null, 2)
    : JSON.stringify(exportObj)

  onProgress?.({ phase: 'complete', current: 1, total: 1, message: 'Export complete' })

  return result
}

export function exportNodesToCsv(
  nodes: GraphNode[],
  options: ExportOptions,
  onProgress?: ExportProgressCallback
): string {
  const { delimiter = ',' } = options

  if (nodes.length === 0) {
    return 'id,labels,properties'
  }

  onProgress?.({ phase: 'preparing', current: 0, total: nodes.length, message: 'Preparing node export...' })

  // Collect all unique property keys
  const propertyKeys = new Set<string>()
  for (const node of nodes) {
    Object.keys(node.properties).forEach((key) => propertyKeys.add(key))
  }

  const headers = ['id', 'labels', ...Array.from(propertyKeys)]
  const lines: string[] = [headers.map((h) => escapeCSVField(h, delimiter)).join(delimiter)]

  for (let i = 0; i < nodes.length; i++) {
    const node = nodes[i]

    onProgress?.({
      phase: 'exporting',
      current: i,
      total: nodes.length,
      message: `Exporting node ${i + 1} of ${nodes.length}...`,
    })

    const row = [
      escapeCSVField(node.id, delimiter),
      escapeCSVField(node.labels.join(';'), delimiter),
      ...Array.from(propertyKeys).map((key) => escapeCSVField(node.properties[key], delimiter)),
    ]

    lines.push(row.join(delimiter))
  }

  onProgress?.({ phase: 'complete', current: nodes.length, total: nodes.length, message: 'Export complete' })

  return lines.join('\n')
}

export function exportEdgesToCsv(
  edges: GraphEdge[],
  options: ExportOptions,
  onProgress?: ExportProgressCallback
): string {
  const { delimiter = ',' } = options

  if (edges.length === 0) {
    return 'id,type,sourceId,targetId,properties'
  }

  onProgress?.({ phase: 'preparing', current: 0, total: edges.length, message: 'Preparing edge export...' })

  // Collect all unique property keys
  const propertyKeys = new Set<string>()
  for (const edge of edges) {
    Object.keys(edge.properties).forEach((key) => propertyKeys.add(key))
  }

  const headers = ['id', 'type', 'sourceId', 'targetId', ...Array.from(propertyKeys)]
  const lines: string[] = [headers.map((h) => escapeCSVField(h, delimiter)).join(delimiter)]

  for (let i = 0; i < edges.length; i++) {
    const edge = edges[i]

    onProgress?.({
      phase: 'exporting',
      current: i,
      total: edges.length,
      message: `Exporting edge ${i + 1} of ${edges.length}...`,
    })

    const row = [
      escapeCSVField(edge.id, delimiter),
      escapeCSVField(edge.type, delimiter),
      escapeCSVField(edge.sourceId, delimiter),
      escapeCSVField(edge.targetId, delimiter),
      ...Array.from(propertyKeys).map((key) => escapeCSVField(edge.properties[key], delimiter)),
    ]

    lines.push(row.join(delimiter))
  }

  onProgress?.({ phase: 'complete', current: edges.length, total: edges.length, message: 'Export complete' })

  return lines.join('\n')
}

export function exportQueryResultToCsv(
  result: QueryResult,
  options: ExportOptions,
  onProgress?: ExportProgressCallback
): string {
  const { delimiter = ',' } = options

  // If we have rows/columns (SQL result), export those
  if (result.rows && result.columns) {
    const lines: string[] = [
      result.columns.map((h) => escapeCSVField(h, delimiter)).join(delimiter)
    ]

    const rows = result.rows
    for (let i = 0; i < rows.length; i++) {
      const row = rows[i]
      onProgress?.({
        phase: 'exporting',
        current: i,
        total: rows.length,
        message: `Exporting row ${i + 1} of ${rows.length}...`,
      })

      const values = result.columns.map((col) => escapeCSVField(row[col], delimiter))
      lines.push(values.join(delimiter))
    }

    onProgress?.({ phase: 'complete', current: rows.length, total: rows.length, message: 'Export complete' })
    return lines.join('\n')
  }

  // Otherwise, export nodes and edges separately
  const nodesCsv = result.nodes?.length
    ? exportNodesToCsv(result.nodes, options)
    : ''
  const edgesCsv = result.edges?.length
    ? exportEdgesToCsv(result.edges, options)
    : ''

  if (nodesCsv && edgesCsv) {
    return `# Nodes\n${nodesCsv}\n\n# Edges\n${edgesCsv}`
  }

  return nodesCsv || edgesCsv || ''
}

export function exportQueryResultToJson(
  result: QueryResult,
  options: ExportOptions,
  onProgress?: ExportProgressCallback
): string {
  const { prettyPrint = true } = options

  onProgress?.({ phase: 'preparing', current: 0, total: 1, message: 'Preparing JSON export...' })

  const exportObj: {
    nodes?: GraphNode[]
    edges?: GraphEdge[]
    rows?: Record<string, unknown>[]
    columns?: string[]
  } = {}

  if (result.nodes?.length) {
    exportObj.nodes = result.nodes
  }
  if (result.edges?.length) {
    exportObj.edges = result.edges
  }
  if (result.rows?.length) {
    exportObj.rows = result.rows
    exportObj.columns = result.columns
  }

  onProgress?.({ phase: 'exporting', current: 0, total: 1, message: 'Generating JSON...' })

  const json = prettyPrint
    ? JSON.stringify(exportObj, null, 2)
    : JSON.stringify(exportObj)

  onProgress?.({ phase: 'complete', current: 1, total: 1, message: 'Export complete' })

  return json
}

export function downloadFile(content: string, filename: string, mimeType: string = 'text/plain'): void {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

export function generateExportFilename(baseName: string, format: ExportFormat): string {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
  const extension = format === 'json' ? 'json' : 'csv'
  return `${baseName}-${timestamp}.${extension}`
}

export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`
}
