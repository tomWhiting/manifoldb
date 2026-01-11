import { useAppStore } from '../../stores/app-store'
import type { QueryResult, GraphNode } from '../../types'

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

  // Check if this is an error result
  if (result.error) {
    return <ErrorDisplay error={result.error} />
  }

  // SQL results have rows and columns from the query
  if (result.rows && result.columns) {
    return <SQLTableDisplay result={result} />
  }

  // SQL results without explicit columns (infer from rows)
  if (result.rows && result.rows.length > 0) {
    return <GenericRowsDisplay rows={result.rows} />
  }

  // Cypher results have nodes
  if (result.nodes && result.nodes.length > 0) {
    return <CypherNodesDisplay nodes={result.nodes} />
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

interface SQLTableDisplayProps {
  result: QueryResult
}

function SQLTableDisplay({ result }: SQLTableDisplayProps) {
  const columns = result.columns ?? []
  const rows = result.rows ?? []

  if (rows.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-text-muted">
        <p>Query executed successfully</p>
        <p className="text-sm">0 rows returned</p>
      </div>
    )
  }

  return (
    <div className="h-full overflow-auto">
      <table className="w-full text-sm">
        <thead className="sticky top-0 bg-bg-secondary">
          <tr>
            {columns.map((col) => (
              <th
                key={col}
                className="px-3 py-2 text-left text-xs font-medium text-text-muted uppercase border-b border-border"
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className={i % 2 === 0 ? 'bg-bg-primary' : 'bg-bg-secondary/50'}>
              {columns.map((col) => (
                <td key={col} className="px-3 py-2 text-text-secondary">
                  {formatValue(row[col])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

interface GenericRowsDisplayProps {
  rows: Record<string, unknown>[]
}

function GenericRowsDisplay({ rows }: GenericRowsDisplayProps) {
  // Infer columns from the first row
  const columns = Object.keys(rows[0])

  return (
    <div className="h-full overflow-auto">
      <table className="w-full text-sm">
        <thead className="sticky top-0 bg-bg-secondary">
          <tr>
            {columns.map((col) => (
              <th
                key={col}
                className="px-3 py-2 text-left text-xs font-medium text-text-muted uppercase border-b border-border"
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className={i % 2 === 0 ? 'bg-bg-primary' : 'bg-bg-secondary/50'}>
              {columns.map((col) => (
                <td key={col} className="px-3 py-2 text-text-secondary">
                  {formatValue(row[col])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

interface CypherNodesDisplayProps {
  nodes: GraphNode[]
}

function CypherNodesDisplay({ nodes }: CypherNodesDisplayProps) {
  // Get all property keys from nodes
  const propertyKeys = new Set<string>()
  nodes.forEach((node) => {
    Object.keys(node.properties).forEach((key) => propertyKeys.add(key))
  })
  const columns = ['id', 'labels', ...Array.from(propertyKeys)]

  return (
    <div className="h-full overflow-auto">
      <table className="w-full text-sm">
        <thead className="sticky top-0 bg-bg-secondary">
          <tr>
            {columns.map((col) => (
              <th
                key={col}
                className="px-3 py-2 text-left text-xs font-medium text-text-muted uppercase border-b border-border"
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {nodes.map((node, i) => (
            <tr key={node.id} className={i % 2 === 0 ? 'bg-bg-primary' : 'bg-bg-secondary/50'}>
              <td className="px-3 py-2 text-text-secondary font-mono text-xs">{node.id}</td>
              <td className="px-3 py-2 text-text-secondary">
                {node.labels.map((label) => (
                  <span
                    key={label}
                    className="inline-block px-1.5 py-0.5 mr-1 text-xs bg-accent-muted text-accent rounded"
                  >
                    {label}
                  </span>
                ))}
              </td>
              {Array.from(propertyKeys).map((key) => (
                <td key={key} className="px-3 py-2 text-text-secondary">
                  {formatValue(node.properties[key])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return '\u2014' // em dash
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}
