import { Circle, Cpu, HardDrive, Clock } from 'lucide-react'
import { useAppStore } from '../../stores/app-store'
import type { ConnectionStatus } from '../../types'

function ConnectionIndicator({ status }: { status: ConnectionStatus }) {
  const statusConfig = {
    connected: { color: 'text-green-500', label: 'Connected' },
    disconnected: { color: 'text-neutral-500', label: 'Disconnected' },
    connecting: { color: 'text-yellow-500', label: 'Connecting...' },
    error: { color: 'text-red-500', label: 'Error' },
  }

  const { color, label } = statusConfig[status]

  return (
    <button className="flex items-center gap-2 px-3 py-1.5 rounded hover:bg-bg-tertiary transition-colors">
      <Circle size={8} className={`fill-current ${color}`} />
      <span className="text-xs text-text-muted">{label}</span>
    </button>
  )
}

function ResourceIndicator() {
  const stats = useAppStore((s) => s.stats)

  return (
    <button className="flex items-center gap-3 px-3 py-1.5 rounded hover:bg-bg-tertiary transition-colors">
      <div className="flex items-center gap-1.5">
        <Cpu size={12} className="text-text-muted" />
        <span className="text-xs text-text-muted">{stats?.cpuUsage ?? 0}%</span>
      </div>
      <div className="flex items-center gap-1.5">
        <HardDrive size={12} className="text-text-muted" />
        <span className="text-xs text-text-muted">
          {stats?.nodeCount ?? 0} nodes / {stats?.edgeCount ?? 0} edges
        </span>
      </div>
    </button>
  )
}

function QueryStatus() {
  const tabs = useAppStore((s) => s.tabs)
  const activeTabId = useAppStore((s) => s.activeTabId)
  const activeTab = tabs.find((t) => t.id === activeTabId)

  if (!activeTab) return null

  const isExecuting = activeTab.isExecuting
  const executionTime = activeTab.result?.executionTime
  const rowCount = activeTab.result?.rowCount

  return (
    <div className="flex items-center gap-3 px-3 py-1.5">
      {isExecuting && (
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 border-2 border-accent border-t-transparent rounded-full animate-spin" />
          <span className="text-xs text-text-muted">Executing...</span>
        </div>
      )}
      {!isExecuting && executionTime !== undefined && (
        <>
          <div className="flex items-center gap-1.5">
            <Clock size={12} className="text-text-muted" />
            <span className="text-xs text-text-muted">{executionTime.toFixed(0)}ms</span>
          </div>
          {rowCount !== undefined && (
            <span className="text-xs text-text-muted">{rowCount} rows</span>
          )}
        </>
      )}
    </div>
  )
}

export function Tray() {
  const connectionStatus = useAppStore((s) => s.connectionStatus)

  return (
    <div className="flex items-center justify-between h-7 px-1 text-xs">
      {/* Left: Connection */}
      <div className="flex items-center">
        <ConnectionIndicator status={connectionStatus} />
      </div>

      {/* Center: Resources */}
      <div className="flex items-center">
        <ResourceIndicator />
      </div>

      {/* Right: Query status */}
      <div className="flex items-center">
        <QueryStatus />
      </div>
    </div>
  )
}
