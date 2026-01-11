import type { GraphNode, GraphEdge } from '../types'
import { graphqlClient } from './graphql-client'

export interface ImportResult {
  nodes: GraphNode[]
  edges: GraphEdge[]
  errors: ImportError[]
  stats: {
    totalRows: number
    successfulNodes: number
    successfulEdges: number
    failedRows: number
  }
}

export interface ImportError {
  line: number
  message: string
  data?: unknown
}

export interface CsvMapping {
  idColumn: string
  labelsColumn?: string
  typeColumn?: string
  sourceColumn?: string
  targetColumn?: string
  propertyColumns: string[]
}

export type ImportFormat = 'json' | 'csv'

export interface ImportProgress {
  phase: 'reading' | 'parsing' | 'validating' | 'complete' | 'error'
  current: number
  total: number
  message: string
}

export type ImportProgressCallback = (progress: ImportProgress) => void

interface JsonImportData {
  nodes?: unknown[]
  edges?: unknown[]
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isValidGraphNode(obj: unknown): obj is GraphNode {
  if (!isRecord(obj)) return false
  if (typeof obj.id !== 'string' && typeof obj.id !== 'number') return false
  if (!Array.isArray(obj.labels)) return false
  if (!isRecord(obj.properties)) return false
  return true
}

function isValidGraphEdge(obj: unknown): obj is GraphEdge {
  if (!isRecord(obj)) return false
  if (typeof obj.id !== 'string' && typeof obj.id !== 'number') return false
  if (typeof obj.type !== 'string') return false
  if (typeof obj.sourceId !== 'string' && typeof obj.sourceId !== 'number') return false
  if (typeof obj.targetId !== 'string' && typeof obj.targetId !== 'number') return false
  if (!isRecord(obj.properties)) return false
  return true
}

function normalizeNode(obj: Record<string, unknown>, index: number): GraphNode | null {
  const id = obj.id ?? obj.ID ?? obj._id ?? `node_${index}`
  const labels = obj.labels ?? obj.label ?? []
  const normalizedLabels = Array.isArray(labels) ? labels : [labels]

  const properties: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(obj)) {
    if (!['id', 'ID', '_id', 'labels', 'label'].includes(key)) {
      properties[key] = value
    }
  }

  return {
    id: String(id),
    labels: normalizedLabels.map(String),
    properties,
  }
}

function normalizeEdge(obj: Record<string, unknown>, index: number): GraphEdge | null {
  const id = obj.id ?? obj.ID ?? obj._id ?? `edge_${index}`
  const type = obj.type ?? obj.TYPE ?? obj.relationType ?? obj.relationship ?? 'RELATED_TO'
  const sourceId = obj.sourceId ?? obj.source ?? obj.from ?? obj.startNode
  const targetId = obj.targetId ?? obj.target ?? obj.to ?? obj.endNode

  if (sourceId === undefined || targetId === undefined) {
    return null
  }

  const properties: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(obj)) {
    if (!['id', 'ID', '_id', 'type', 'TYPE', 'relationType', 'relationship',
          'sourceId', 'source', 'from', 'startNode',
          'targetId', 'target', 'to', 'endNode'].includes(key)) {
      properties[key] = value
    }
  }

  return {
    id: String(id),
    type: String(type),
    sourceId: String(sourceId),
    targetId: String(targetId),
    properties,
  }
}

export async function parseJsonImport(
  content: string,
  onProgress?: ImportProgressCallback
): Promise<ImportResult> {
  const errors: ImportError[] = []
  const nodes: GraphNode[] = []
  const edges: GraphEdge[] = []

  onProgress?.({ phase: 'parsing', current: 0, total: 1, message: 'Parsing JSON...' })

  let data: JsonImportData
  try {
    data = JSON.parse(content) as JsonImportData
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Invalid JSON'
    return {
      nodes: [],
      edges: [],
      errors: [{ line: 1, message: `JSON parse error: ${message}` }],
      stats: { totalRows: 0, successfulNodes: 0, successfulEdges: 0, failedRows: 1 },
    }
  }

  onProgress?.({ phase: 'validating', current: 0, total: 1, message: 'Validating data...' })

  const rawNodes = data.nodes ?? []
  const rawEdges = data.edges ?? []
  const totalItems = rawNodes.length + rawEdges.length

  // Process nodes
  for (let i = 0; i < rawNodes.length; i++) {
    const item = rawNodes[i]
    onProgress?.({
      phase: 'validating',
      current: i,
      total: totalItems,
      message: `Validating node ${i + 1} of ${rawNodes.length}...`,
    })

    if (!isRecord(item)) {
      errors.push({ line: i + 1, message: 'Invalid node format', data: item })
      continue
    }

    if (isValidGraphNode(item)) {
      nodes.push({
        id: String(item.id),
        labels: item.labels.map(String),
        properties: item.properties,
      })
    } else {
      const normalized = normalizeNode(item, i)
      if (normalized) {
        nodes.push(normalized)
      } else {
        errors.push({ line: i + 1, message: 'Could not normalize node', data: item })
      }
    }
  }

  // Process edges
  for (let i = 0; i < rawEdges.length; i++) {
    const item = rawEdges[i]
    onProgress?.({
      phase: 'validating',
      current: rawNodes.length + i,
      total: totalItems,
      message: `Validating edge ${i + 1} of ${rawEdges.length}...`,
    })

    if (!isRecord(item)) {
      errors.push({ line: rawNodes.length + i + 1, message: 'Invalid edge format', data: item })
      continue
    }

    if (isValidGraphEdge(item)) {
      edges.push({
        id: String(item.id),
        type: item.type,
        sourceId: String(item.sourceId),
        targetId: String(item.targetId),
        properties: item.properties,
      })
    } else {
      const normalized = normalizeEdge(item, i)
      if (normalized) {
        edges.push(normalized)
      } else {
        errors.push({
          line: rawNodes.length + i + 1,
          message: 'Edge missing source or target',
          data: item
        })
      }
    }
  }

  onProgress?.({ phase: 'complete', current: totalItems, total: totalItems, message: 'Import complete' })

  return {
    nodes,
    edges,
    errors,
    stats: {
      totalRows: totalItems,
      successfulNodes: nodes.length,
      successfulEdges: edges.length,
      failedRows: errors.length,
    },
  }
}

function parseCsvLine(line: string, delimiter: string = ','): string[] {
  const result: string[] = []
  let current = ''
  let inQuotes = false

  for (let i = 0; i < line.length; i++) {
    const char = line[i]
    const nextChar = line[i + 1]

    if (char === '"') {
      if (inQuotes && nextChar === '"') {
        current += '"'
        i++
      } else {
        inQuotes = !inQuotes
      }
    } else if (char === delimiter && !inQuotes) {
      result.push(current.trim())
      current = ''
    } else {
      current += char
    }
  }

  result.push(current.trim())
  return result
}

export interface CsvImportOptions {
  delimiter?: string
  mapping: CsvMapping
  dataType: 'nodes' | 'edges'
}

export async function parseCsvImport(
  content: string,
  options: CsvImportOptions,
  onProgress?: ImportProgressCallback
): Promise<ImportResult> {
  const { delimiter = ',', mapping, dataType } = options
  const errors: ImportError[] = []
  const nodes: GraphNode[] = []
  const edges: GraphEdge[] = []

  onProgress?.({ phase: 'parsing', current: 0, total: 1, message: 'Parsing CSV...' })

  const lines = content.split(/\r?\n/).filter((line) => line.trim())
  if (lines.length < 2) {
    return {
      nodes: [],
      edges: [],
      errors: [{ line: 1, message: 'CSV must have header and at least one data row' }],
      stats: { totalRows: 0, successfulNodes: 0, successfulEdges: 0, failedRows: 1 },
    }
  }

  const headers = parseCsvLine(lines[0], delimiter)
  const dataRows = lines.slice(1)
  const totalRows = dataRows.length

  onProgress?.({ phase: 'validating', current: 0, total: totalRows, message: 'Processing rows...' })

  for (let i = 0; i < dataRows.length; i++) {
    const lineNumber = i + 2 // +1 for 0-index, +1 for header
    const row = parseCsvLine(dataRows[i], delimiter)

    onProgress?.({
      phase: 'validating',
      current: i,
      total: totalRows,
      message: `Processing row ${i + 1} of ${totalRows}...`,
    })

    const rowData: Record<string, string> = {}
    headers.forEach((header, idx) => {
      rowData[header] = row[idx] ?? ''
    })

    if (dataType === 'nodes') {
      const id = rowData[mapping.idColumn]
      if (!id) {
        errors.push({ line: lineNumber, message: `Missing ID in column '${mapping.idColumn}'`, data: rowData })
        continue
      }

      const labelsValue = mapping.labelsColumn ? rowData[mapping.labelsColumn] : ''
      const labels = labelsValue ? labelsValue.split(';').map((l) => l.trim()).filter(Boolean) : ['Node']

      const properties: Record<string, unknown> = {}
      for (const col of mapping.propertyColumns) {
        if (rowData[col] !== undefined && rowData[col] !== '') {
          const value = rowData[col]
          // Try to parse as number or boolean
          if (value === 'true') properties[col] = true
          else if (value === 'false') properties[col] = false
          else if (!isNaN(Number(value)) && value !== '') properties[col] = Number(value)
          else properties[col] = value
        }
      }

      nodes.push({ id, labels, properties })
    } else {
      // edges
      const id = rowData[mapping.idColumn] ?? `edge_${i}`
      const type = mapping.typeColumn ? rowData[mapping.typeColumn] : 'RELATED_TO'
      const sourceId = mapping.sourceColumn ? rowData[mapping.sourceColumn] : ''
      const targetId = mapping.targetColumn ? rowData[mapping.targetColumn] : ''

      if (!sourceId || !targetId) {
        errors.push({
          line: lineNumber,
          message: 'Missing source or target ID',
          data: rowData
        })
        continue
      }

      const properties: Record<string, unknown> = {}
      for (const col of mapping.propertyColumns) {
        if (rowData[col] !== undefined && rowData[col] !== '') {
          const value = rowData[col]
          if (value === 'true') properties[col] = true
          else if (value === 'false') properties[col] = false
          else if (!isNaN(Number(value)) && value !== '') properties[col] = Number(value)
          else properties[col] = value
        }
      }

      edges.push({ id, type, sourceId, targetId, properties })
    }
  }

  onProgress?.({ phase: 'complete', current: totalRows, total: totalRows, message: 'Import complete' })

  return {
    nodes,
    edges,
    errors,
    stats: {
      totalRows,
      successfulNodes: nodes.length,
      successfulEdges: edges.length,
      failedRows: errors.length,
    },
  }
}

export function detectCsvColumns(content: string, delimiter: string = ','): string[] {
  const lines = content.split(/\r?\n/).filter((line) => line.trim())
  if (lines.length === 0) return []
  return parseCsvLine(lines[0], delimiter)
}

export function previewCsvData(
  content: string,
  delimiter: string = ',',
  maxRows: number = 5
): { headers: string[]; rows: string[][] } {
  const lines = content.split(/\r?\n/).filter((line) => line.trim())
  if (lines.length === 0) return { headers: [], rows: [] }

  const headers = parseCsvLine(lines[0], delimiter)
  const rows = lines.slice(1, maxRows + 1).map((line) => parseCsvLine(line, delimiter))

  return { headers, rows }
}

export async function readFileAsText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = () => reject(new Error('Failed to read file'))
    reader.readAsText(file)
  })
}

export async function readFileInChunks(
  file: File,
  chunkSize: number = 1024 * 1024, // 1MB chunks
  onProgress?: (bytesRead: number, totalBytes: number) => void
): Promise<string> {
  const chunks: string[] = []
  let offset = 0

  while (offset < file.size) {
    const chunk = file.slice(offset, offset + chunkSize)
    const text = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(reader.result as string)
      reader.onerror = () => reject(new Error('Failed to read file chunk'))
      reader.readAsText(chunk)
    })
    chunks.push(text)
    offset += chunkSize
    onProgress?.(Math.min(offset, file.size), file.size)
  }

  return chunks.join('')
}

// =============================================================================
// Backup Import Functions
// =============================================================================

export interface BackupRestoreResult {
  entityCount: number
  edgeCount: number
  metadataCount: number
  totalRecords: number
  success: boolean
  error: string | null
}

const RESTORE_BACKUP_MUTATION = `
  mutation RestoreBackup($content: String!, $verifyReferences: Boolean!, $skipDuplicates: Boolean!) {
    restoreBackup(content: $content, verifyReferences: $verifyReferences, skipDuplicates: $skipDuplicates) {
      entityCount
      edgeCount
      metadataCount
      totalRecords
      success
      error
    }
  }
`

const VERIFY_BACKUP_MUTATION = `
  mutation VerifyBackup($content: String!) {
    verifyBackup(content: $content) {
      entityCount
      edgeCount
      metadataCount
      totalRecords
      success
      error
    }
  }
`

/**
 * Check if content is in ManifoldDB backup format.
 * Backup format starts with a metadata record: {"type":"metadata",...}
 */
export function isBackupFormat(content: string): boolean {
  const firstLine = content.split('\n')[0]?.trim()
  if (!firstLine) return false

  try {
    const parsed = JSON.parse(firstLine)
    return parsed.type === 'metadata' && parsed.data?.format === 'json_lines'
  } catch {
    return false
  }
}

/**
 * Restore a backup to the database.
 */
export async function restoreBackup(
  content: string,
  options: { verifyReferences?: boolean; skipDuplicates?: boolean } = {}
): Promise<BackupRestoreResult> {
  const { verifyReferences = true, skipDuplicates = false } = options

  const result = await graphqlClient
    .mutation(RESTORE_BACKUP_MUTATION, {
      content,
      verifyReferences,
      skipDuplicates,
    })
    .toPromise()

  if (result.error) {
    return {
      entityCount: 0,
      edgeCount: 0,
      metadataCount: 0,
      totalRecords: 0,
      success: false,
      error: result.error.message,
    }
  }

  return result.data?.restoreBackup as BackupRestoreResult
}

/**
 * Verify a backup without importing (dry run).
 */
export async function verifyBackup(content: string): Promise<BackupRestoreResult> {
  const result = await graphqlClient
    .mutation(VERIFY_BACKUP_MUTATION, { content })
    .toPromise()

  if (result.error) {
    return {
      entityCount: 0,
      edgeCount: 0,
      metadataCount: 0,
      totalRecords: 0,
      success: false,
      error: result.error.message,
    }
  }

  return result.data?.verifyBackup as BackupRestoreResult
}

// =============================================================================
// Direct JSONL Import (using Cypher mutations)
// =============================================================================

const EXECUTE_MUTATION = `
  mutation Execute($query: String!) {
    execute(query: $query) {
      table { columns rows }
    }
  }
`

export type JsonlFormat = 'rubicon' | 'generic' | 'unknown'

export interface JsonlAnalysis {
  format: JsonlFormat
  totalLines: number
  entityCount: number
  edgeCount: number
  sampleLabels: string[]
  sampleTypes: string[]
}

/**
 * Analyze JSONL content to determine format and stats
 */
export function analyzeJsonl(content: string): JsonlAnalysis {
  const lines = content.split('\n').filter(l => l.trim())
  let format: JsonlFormat = 'unknown'
  let entityCount = 0
  let edgeCount = 0
  const labels = new Set<string>()
  const types = new Set<string>()

  for (const line of lines) {
    try {
      const record = JSON.parse(line)

      // Detect Rubicon format (Claude session transcripts)
      if (record.sessionId && record.uuid && (record.type === 'user' || record.type === 'assistant')) {
        format = 'rubicon'
        entityCount++
        labels.add(record.type === 'user' ? 'UserMessage' : 'AssistantMessage')
        if (record.parentUuid) {
          edgeCount++ // Will create REPLIES_TO edge
        }
      }
      // Detect backup format entities
      else if (record.type === 'entity' && record.data) {
        format = 'generic'
        entityCount++
        record.data.labels?.forEach((l: string) => labels.add(l))
      }
      // Detect backup format edges
      else if (record.type === 'edge' && record.data) {
        format = 'generic'
        edgeCount++
        if (record.data.type) types.add(record.data.type)
      }
      // Skip metadata records
      else if (record.type === 'metadata') {
        continue
      }
    } catch {
      // Skip malformed lines
    }
  }

  return {
    format,
    totalLines: lines.length,
    entityCount,
    edgeCount,
    sampleLabels: Array.from(labels).slice(0, 5),
    sampleTypes: Array.from(types).slice(0, 5),
  }
}

/**
 * Escape a string for use in Cypher
 */
function escapeForCypher(value: unknown): string {
  if (value === null || value === undefined) return 'null'
  if (typeof value === 'number') return String(value)
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (typeof value === 'string') {
    return `"${value.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\n/g, '\\n')}"`
  }
  if (Array.isArray(value) || typeof value === 'object') {
    return `"${JSON.stringify(value).replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`
  }
  return `"${String(value)}"`
}

/**
 * Convert properties object to Cypher map syntax
 */
function propsToSypher(props: Record<string, unknown>): string {
  const pairs = Object.entries(props)
    .filter(([, v]) => v !== undefined && v !== null)
    .map(([k, v]) => `${k}: ${escapeForCypher(v)}`)
  return pairs.length > 0 ? `{${pairs.join(', ')}}` : ''
}

/**
 * Import JSONL content using direct Cypher mutations.
 * Supports Rubicon (Claude session) and generic entity/edge formats.
 */
export async function importJsonl(
  content: string,
  onProgress?: ImportProgressCallback
): Promise<ImportResult> {
  const lines = content.split('\n').filter(l => l.trim())
  const analysis = analyzeJsonl(content)
  const errors: ImportError[] = []
  const nodes: GraphNode[] = []
  const edges: GraphEdge[] = []

  let successfulNodes = 0
  let successfulEdges = 0
  const nodeIdMap = new Map<string, number>() // Maps original IDs to new IDs

  onProgress?.({
    phase: 'parsing',
    current: 0,
    total: lines.length,
    message: `Detected ${analysis.format} format with ${analysis.entityCount} entities`,
  })

  // First pass: Create all nodes
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    try {
      const record = JSON.parse(line)

      let labels: string[] = []
      let props: Record<string, unknown> = {}
      let originalId: string | null = null

      // Handle Rubicon format
      if (analysis.format === 'rubicon' && record.uuid) {
        if (record.type === 'user' || record.type === 'assistant') {
          labels = [record.type === 'user' ? 'UserMessage' : 'AssistantMessage', 'Message']
          originalId = record.uuid
          props = {
            uuid: record.uuid,
            sessionId: record.sessionId,
            timestamp: record.timestamp,
            role: record.message?.role,
            content: typeof record.message?.content === 'string'
              ? record.message.content
              : JSON.stringify(record.message?.content),
            parentUuid: record.parentUuid,
          }
        } else {
          continue // Skip non-message records
        }
      }
      // Handle generic entity format
      else if (record.type === 'entity' && record.data) {
        labels = record.data.labels || ['Node']
        originalId = String(record.data.id)
        props = record.data.properties || {}
      }
      // Skip other record types
      else {
        continue
      }

      // Create the node
      const labelStr = labels.map(l => `:${l}`).join('')
      const propsStr = propsToSypher(props)
      const query = `CREATE (n${labelStr} ${propsStr}) RETURN id(n) as id`

      const result = await graphqlClient
        .mutation(EXECUTE_MUTATION, { query })
        .toPromise()

      if (result.error) {
        errors.push({ line: i + 1, message: result.error.message })
      } else {
        const newId = result.data?.execute?.table?.rows?.[0]?.[0]
        if (newId !== undefined && originalId) {
          nodeIdMap.set(originalId, newId)
        }
        successfulNodes++
        nodes.push({
          id: String(newId),
          labels,
          properties: props as Record<string, unknown>,
        })
      }

      if (i % 10 === 0) {
        onProgress?.({
          phase: 'validating',
          current: i,
          total: lines.length,
          message: `Created ${successfulNodes} nodes...`,
        })
      }
    } catch (err) {
      errors.push({
        line: i + 1,
        message: err instanceof Error ? err.message : 'Parse error',
      })
    }
  }

  // Second pass: Create edges (for Rubicon, link parent messages)
  if (analysis.format === 'rubicon') {
    onProgress?.({
      phase: 'validating',
      current: lines.length,
      total: lines.length,
      message: 'Creating message relationships...',
    })

    for (let i = 0; i < lines.length; i++) {
      try {
        const record = JSON.parse(lines[i])
        if (record.parentUuid && record.uuid) {
          const sourceId = nodeIdMap.get(record.uuid)
          const targetId = nodeIdMap.get(record.parentUuid)

          if (sourceId !== undefined && targetId !== undefined) {
            const query = `MATCH (a), (b) WHERE id(a) = ${sourceId} AND id(b) = ${targetId} CREATE (a)-[r:REPLIES_TO]->(b) RETURN r`

            const result = await graphqlClient
              .mutation(EXECUTE_MUTATION, { query })
              .toPromise()

            if (result.error) {
              errors.push({ line: i + 1, message: `Edge error: ${result.error.message}` })
            } else {
              successfulEdges++
              edges.push({
                id: `edge_${i}`,
                type: 'REPLIES_TO',
                sourceId: String(sourceId),
                targetId: String(targetId),
                properties: {},
              })
            }
          }
        }
      } catch {
        // Skip edge creation errors silently
      }
    }
  }
  // Handle generic edge format
  else if (analysis.format === 'generic') {
    for (let i = 0; i < lines.length; i++) {
      try {
        const record = JSON.parse(lines[i])
        if (record.type === 'edge' && record.data) {
          const sourceId = nodeIdMap.get(String(record.data.source))
          const targetId = nodeIdMap.get(String(record.data.target))
          const edgeType = record.data.type || 'RELATED_TO'
          const props = record.data.properties || {}

          if (sourceId !== undefined && targetId !== undefined) {
            const propsStr = propsToSypher(props)
            const query = `MATCH (a), (b) WHERE id(a) = ${sourceId} AND id(b) = ${targetId} CREATE (a)-[r:${edgeType} ${propsStr}]->(b) RETURN r`

            const result = await graphqlClient
              .mutation(EXECUTE_MUTATION, { query })
              .toPromise()

            if (!result.error) {
              successfulEdges++
            }
          }
        }
      } catch {
        // Skip edge errors
      }
    }
  }

  onProgress?.({
    phase: 'complete',
    current: lines.length,
    total: lines.length,
    message: `Import complete: ${successfulNodes} nodes, ${successfulEdges} edges`,
  })

  return {
    nodes,
    edges,
    errors,
    stats: {
      totalRows: lines.length,
      successfulNodes,
      successfulEdges,
      failedRows: errors.length,
    },
  }
}
