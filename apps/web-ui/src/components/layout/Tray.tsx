import { Circle, Cpu, HardDrive, Clock, RefreshCw } from 'lucide-react'
import { useAppStore } from '../../stores/app-store'
import { getReconnectAttempts } from '../../lib/graphql-client'
import type { ConnectionStatus } from '../../types'

function ConnectionIndicator({ status }: { status: ConnectionStatus }) {
  const reconnectAttempts = getReconnectAttempts()

  const statusConfig = {
    connected: {
      color: 'text-green-500',
      label: 'Connected',
      icon: <Circle size={8} className="fill-current text-green-500" />,
    },
    disconnected: {
      color: 'text-neutral-500',
      label: 'Disconnected',
      icon: <Circle size={8} className="fill-current text-neutral-500" />,
    },
    connecting: {
      color: 'text-yellow-500',
      label: reconnectAttempts > 0 ? `Reconnecting (${reconnectAttempts})...` : 'Connecting...',
      icon: <RefreshCw size={10} className="text-yellow-500 animate-spin" />,
    },
    error: {
      color: 'text-red-500',
      label: 'Error',
      icon: <Circle size={8} className="fill-current text-red-500" />,
    },
  }

  const config = statusConfig[status]

  return (
    <button className="flex items-center gap-2 px-3 py-1.5 rounded hover:bg-white/5 transition-colors">
      {config.icon}
      <span className={`text-xs ${config.color}`}>{config.label}</span>
    </button>
  )
}

function ResourceIndicator() {
  const stats = useAppStore((s) => s.stats)

  return (
    <button className="flex items-center gap-3 px-3 py-1.5 rounded hover:bg-white/5 transition-colors">
      <div className="flex items-center gap-1.5">
        <Cpu size={12} className="text-neutral-500" />
        <span className="text-xs text-neutral-400">{stats?.cpuUsage ?? 0}%</span>
      </div>
      <div className="flex items-center gap-1.5">
        <HardDrive size={12} className="text-neutral-500" />
        <span className="text-xs text-neutral-400">
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
          <div className="w-3 h-3 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-xs text-neutral-400">Executing...</span>
        </div>
      )}
      {!isExecuting && executionTime !== undefined && (
        <>
          <div className="flex items-center gap-1.5">
            <Clock size={12} className="text-neutral-500" />
            <span className="text-xs text-neutral-400">{executionTime.toFixed(0)}ms</span>
          </div>
          {rowCount !== undefined && (
            <span className="text-xs text-neutral-400">{rowCount} rows</span>
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
