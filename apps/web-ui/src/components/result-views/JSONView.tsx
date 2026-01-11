import { useAppStore } from '../../stores/app-store'

export function JSONView() {
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

  const json = JSON.stringify(result.raw ?? result, null, 2)

  return (
    <div className="h-full overflow-auto p-4">
      <pre className="text-sm font-mono text-text-secondary whitespace-pre-wrap">{json}</pre>
    </div>
  )
}
