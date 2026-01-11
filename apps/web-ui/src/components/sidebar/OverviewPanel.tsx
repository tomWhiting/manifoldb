import {
  RefreshCw,
  Database,
  ArrowRightLeft,
  Wifi,
  WifiOff,
  AlertCircle,
  Tag,
} from 'lucide-react'
import { useStats, type LabelStats, type EdgeTypeStats } from '../../hooks/useStats'
import { useAppStore } from '../../stores/app-store'

interface StatCardProps {
  title: string
  value: number | string
  icon: React.ReactNode
  subtitle?: string
}

function StatCard({ title, value, icon, subtitle }: StatCardProps) {
  return (
    <div className="p-4 bg-bg-secondary rounded-lg border border-border">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-bg-tertiary rounded-md text-text-muted">{icon}</div>
        <div className="flex-1 min-w-0">
          <p className="text-xs text-text-muted uppercase tracking-wide">{title}</p>
          <p className="text-2xl font-semibold text-text-primary">
            {typeof value === 'number' ? value.toLocaleString() : value}
          </p>
          {subtitle && <p className="text-xs text-text-muted mt-0.5">{subtitle}</p>}
        </div>
      </div>
    </div>
  )
}

interface ConnectionCardProps {
  status: 'connected' | 'disconnected' | 'connecting' | 'error'
}

function ConnectionCard({ status }: ConnectionCardProps) {
  const statusConfig = {
    connected: {
      label: 'Connected',
      icon: <Wifi size={18} />,
      color: 'text-green-400',
      bg: 'bg-green-500/10',
    },
    disconnected: {
      label: 'Disconnected',
      icon: <WifiOff size={18} />,
      color: 'text-text-muted',
      bg: 'bg-bg-tertiary',
    },
    connecting: {
      label: 'Connecting...',
      icon: <RefreshCw size={18} className="animate-spin" />,
      color: 'text-yellow-400',
      bg: 'bg-yellow-500/10',
    },
    error: {
      label: 'Error',
      icon: <AlertCircle size={18} />,
      color: 'text-red-400',
      bg: 'bg-red-500/10',
    },
  }

  const config = statusConfig[status]

  return (
    <div className="p-4 bg-bg-secondary rounded-lg border border-border">
      <div className="flex items-center gap-3">
        <div className={`p-2 rounded-md ${config.bg} ${config.color}`}>{config.icon}</div>
        <div className="flex-1 min-w-0">
          <p className="text-xs text-text-muted uppercase tracking-wide">Connection</p>
          <p className={`text-lg font-medium ${config.color}`}>{config.label}</p>
        </div>
      </div>
    </div>
  )
}

interface DistributionBarProps {
  items: Array<{ name: string; count: number }>
  maxCount: number
  type: 'label' | 'edge'
}

function DistributionBar({ items, maxCount, type }: DistributionBarProps) {
  if (items.length === 0) {
    return (
      <p className="text-sm text-text-muted italic px-2">
        No {type === 'label' ? 'labels' : 'relationship types'} found
      </p>
    )
  }

  return (
    <div className="space-y-2">
      {items.map((item) => {
        const percentage = maxCount > 0 ? (item.count / maxCount) * 100 : 0
        return (
          <div key={item.name} className="group">
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm text-text-secondary truncate max-w-[60%]" title={item.name}>
                {item.name}
              </span>
              <span className="text-xs text-text-muted">{item.count.toLocaleString()}</span>
            </div>
            <div className="h-2 bg-bg-tertiary rounded-full overflow-hidden">
              <div
                className="h-full bg-accent rounded-full transition-all duration-300"
                style={{ width: `${Math.max(percentage, 2)}%` }}
              />
            </div>
          </div>
        )
      })}
    </div>
  )
}

interface DistributionSectionProps {
  title: string
  icon: React.ReactNode
  items: LabelStats[] | EdgeTypeStats[]
  type: 'label' | 'edge'
}

function DistributionSection({ title, icon, items, type }: DistributionSectionProps) {
  const sortedItems = [...items].sort((a, b) => b.count - a.count).slice(0, 10)
  const maxCount = sortedItems.length > 0 ? sortedItems[0].count : 0

  return (
    <div className="p-4 bg-bg-secondary rounded-lg border border-border">
      <div className="flex items-center gap-2 mb-4">
        <span className="text-text-muted">{icon}</span>
        <h3 className="text-sm font-medium text-text-primary">{title}</h3>
        <span className="ml-auto text-xs text-text-muted bg-bg-tertiary px-2 py-0.5 rounded-full">
          {items.length}
        </span>
      </div>
      <DistributionBar items={sortedItems} maxCount={maxCount} type={type} />
      {items.length > 10 && (
        <p className="text-xs text-text-muted mt-3 text-center">
          Showing top 10 of {items.length}
        </p>
      )}
    </div>
  )
}

function formatTimestamp(timestamp: number): string {
  const date = new Date(timestamp)
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

export function OverviewPanel() {
  const { stats, isLoading, error, refresh } = useStats()
  const connectionStatus = useAppStore((s) => s.connectionStatus)

  if (error && !stats) {
    return (
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-text-primary">Overview</h2>
          <button
            onClick={refresh}
            disabled={isLoading}
            className="p-2 hover:bg-bg-tertiary rounded-md transition-colors disabled:opacity-50"
            title="Refresh stats"
          >
            <RefreshCw size={16} className={`text-text-muted ${isLoading ? 'animate-spin' : ''}`} />
          </button>
        </div>
        <div className="flex items-center gap-2 p-4 bg-red-500/10 border border-red-500/20 rounded-md">
          <AlertCircle size={16} className="text-red-400 flex-shrink-0" />
          <p className="text-sm text-red-300">{error}</p>
        </div>
      </div>
    )
  }

  if (isLoading && !stats) {
    return (
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-text-primary">Overview</h2>
        </div>
        <div className="flex items-center justify-center py-8">
          <RefreshCw size={24} className="text-text-muted animate-spin" />
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <h2 className="text-lg font-semibold text-text-primary">Overview</h2>
        <button
          onClick={refresh}
          disabled={isLoading}
          className="p-2 hover:bg-bg-tertiary rounded-md transition-colors disabled:opacity-50"
          title="Refresh stats"
        >
          <RefreshCw size={16} className={`text-text-muted ${isLoading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Quick Stats Cards */}
        <div className="grid grid-cols-1 gap-3">
          <ConnectionCard status={connectionStatus} />
          {stats && (
            <>
              <StatCard
                title="Total Nodes"
                value={stats.nodeCount}
                icon={<Database size={18} />}
                subtitle={`${stats.labels.length} label${stats.labels.length !== 1 ? 's' : ''}`}
              />
              <StatCard
                title="Total Edges"
                value={stats.edgeCount}
                icon={<ArrowRightLeft size={18} />}
                subtitle={`${stats.edgeTypes.length} type${stats.edgeTypes.length !== 1 ? 's' : ''}`}
              />
            </>
          )}
        </div>

        {/* Distribution Charts */}
        {stats && (
          <>
            <DistributionSection
              title="Label Distribution"
              icon={<Tag size={16} />}
              items={stats.labels}
              type="label"
            />
            <DistributionSection
              title="Relationship Distribution"
              icon={<ArrowRightLeft size={16} />}
              items={stats.edgeTypes}
              type="edge"
            />
          </>
        )}
      </div>

      {/* Footer with last updated */}
      {stats && (
        <div className="px-4 py-2 border-t border-border bg-bg-secondary flex items-center justify-between">
          <p className="text-xs text-text-muted">
            Last updated: {formatTimestamp(stats.lastUpdated)}
          </p>
          {isLoading && (
            <RefreshCw size={12} className="text-text-muted animate-spin" />
          )}
        </div>
      )}
    </div>
  )
}
