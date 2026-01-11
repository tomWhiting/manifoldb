import { useAppStore } from '../../stores/app-store'

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

  // Convert nodes to table rows
  const nodes = result.nodes ?? []
  if (nodes.length === 0 && !result.rows) {
    return (
      <div className="flex items-center justify-center h-full text-text-muted">
        No data to display
      </div>
    )
  }

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
  if (value === null || value === undefined) return '—'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}
