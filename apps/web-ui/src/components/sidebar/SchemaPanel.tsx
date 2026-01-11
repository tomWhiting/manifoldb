import { useState } from 'react'
import {
  RefreshCw,
  ChevronDown,
  ChevronRight,
  Tag,
  ArrowRightLeft,
  Database,
  AlertCircle,
} from 'lucide-react'
import { useSchema, type LabelInfo, type EdgeTypeInfo } from '../../hooks/useSchema'
import { useAppStore } from '../../stores/app-store'

interface CollapsibleSectionProps {
  title: string
  icon: React.ReactNode
  count: number
  defaultOpen?: boolean
  children: React.ReactNode
}

function CollapsibleSection({
  title,
  icon,
  count,
  defaultOpen = true,
  children,
}: CollapsibleSectionProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen)

  return (
    <div className="border-b border-border last:border-b-0">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 w-full px-4 py-3 text-left hover:bg-bg-tertiary transition-colors"
      >
        {isOpen ? (
          <ChevronDown size={16} className="text-text-muted flex-shrink-0" />
        ) : (
          <ChevronRight size={16} className="text-text-muted flex-shrink-0" />
        )}
        <span className="text-text-muted flex-shrink-0">{icon}</span>
        <span className="text-sm font-medium text-text-primary flex-1">{title}</span>
        <span className="text-xs text-text-muted bg-bg-tertiary px-2 py-0.5 rounded-full">
          {count}
        </span>
      </button>
      {isOpen && <div className="pb-2">{children}</div>}
    </div>
  )
}

interface SchemaItemProps {
  name: string
  count: number
  onClick: () => void
}

function SchemaItem({ name, count, onClick }: SchemaItemProps) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-2 w-full px-6 py-1.5 text-left hover:bg-bg-tertiary transition-colors group"
    >
      <span className="text-sm text-text-secondary group-hover:text-accent flex-1 truncate">
        {name}
      </span>
      <span className="text-xs text-text-muted">{count.toLocaleString()}</span>
    </button>
  )
}

function formatTimestamp(timestamp: number): string {
  const date = new Date(timestamp)
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

export function SchemaPanel() {
  const { schema, isLoading, error, refresh } = useSchema()
  const activeTabId = useAppStore((s) => s.activeTabId)
  const updateTabContent = useAppStore((s) => s.updateTabContent)
  const setSidebarSection = useAppStore((s) => s.setSidebarSection)

  const runSampleQuery = (query: string) => {
    if (activeTabId) {
      updateTabContent(activeTabId, query)
      setSidebarSection('query')
    }
  }

  const handleLabelClick = (label: LabelInfo) => {
    const query = `MATCH (n:${label.name}) RETURN n LIMIT 10`
    runSampleQuery(query)
  }

  const handleEdgeTypeClick = (edgeType: EdgeTypeInfo) => {
    const query = `MATCH ()-[r:${edgeType.name}]->() RETURN r LIMIT 10`
    runSampleQuery(query)
  }

  if (error) {
    return (
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-text-primary">Schema</h2>
          <button
            onClick={refresh}
            disabled={isLoading}
            className="p-2 hover:bg-bg-tertiary rounded-md transition-colors disabled:opacity-50"
            title="Refresh schema"
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

  if (isLoading && !schema) {
    return (
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-text-primary">Schema</h2>
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
        <h2 className="text-lg font-semibold text-text-primary">Schema</h2>
        <button
          onClick={refresh}
          disabled={isLoading}
          className="p-2 hover:bg-bg-tertiary rounded-md transition-colors disabled:opacity-50"
          title="Refresh schema"
        >
          <RefreshCw size={16} className={`text-text-muted ${isLoading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Stats Summary */}
      {schema && (
        <div className="flex items-center gap-4 px-4 py-3 border-b border-border bg-bg-secondary">
          <div className="flex items-center gap-2">
            <Database size={14} className="text-text-muted" />
            <span className="text-xs text-text-muted">Nodes:</span>
            <span className="text-xs font-medium text-text-primary">
              {schema.nodeCount.toLocaleString()}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <ArrowRightLeft size={14} className="text-text-muted" />
            <span className="text-xs text-text-muted">Edges:</span>
            <span className="text-xs font-medium text-text-primary">
              {schema.edgeCount.toLocaleString()}
            </span>
          </div>
        </div>
      )}

      {/* Sections */}
      <div className="flex-1 overflow-y-auto">
        {schema && (
          <>
            <CollapsibleSection
              title="Node Labels"
              icon={<Tag size={16} />}
              count={schema.labels.length}
            >
              {schema.labels.length === 0 ? (
                <p className="px-6 py-2 text-sm text-text-muted italic">No labels found</p>
              ) : (
                schema.labels.map((label) => (
                  <SchemaItem
                    key={label.name}
                    name={label.name}
                    count={label.count}
                    onClick={() => handleLabelClick(label)}
                  />
                ))
              )}
            </CollapsibleSection>

            <CollapsibleSection
              title="Relationship Types"
              icon={<ArrowRightLeft size={16} />}
              count={schema.edgeTypes.length}
            >
              {schema.edgeTypes.length === 0 ? (
                <p className="px-6 py-2 text-sm text-text-muted italic">No relationship types found</p>
              ) : (
                schema.edgeTypes.map((edgeType) => (
                  <SchemaItem
                    key={edgeType.name}
                    name={edgeType.name}
                    count={edgeType.count}
                    onClick={() => handleEdgeTypeClick(edgeType)}
                  />
                ))
              )}
            </CollapsibleSection>
          </>
        )}
      </div>

      {/* Footer with last updated */}
      {schema && (
        <div className="px-4 py-2 border-t border-border bg-bg-secondary">
          <p className="text-xs text-text-muted">
            Last updated: {formatTimestamp(schema.lastUpdated)}
          </p>
        </div>
      )}
    </div>
  )
}
