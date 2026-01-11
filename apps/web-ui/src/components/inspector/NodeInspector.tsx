import { Circle, Tag, Hash, ChevronRight } from 'lucide-react'
import { FloatingPanel } from '../shared/FloatingPanel'
import { useInspectorStore, type NodeInspector as NodeInspectorType } from '../../stores/inspector-store'
import { labelToColor } from '../../utils/graph-layout'
import { CollapsibleSection } from '../shared/CollapsibleSection'

interface NodeInspectorProps {
  inspector: NodeInspectorType
  onExpandNeighbors?: (nodeId: string) => void
  onDeleteNode?: (nodeId: string) => void
}

function formatPropertyValue(value: unknown): string {
  if (value === null) return 'null'
  if (value === undefined) return 'undefined'
  if (typeof value === 'string') return `"${value}"`
  if (typeof value === 'object') {
    try {
      return JSON.stringify(value, null, 2)
    } catch {
      return String(value)
    }
  }
  return String(value)
}

function PropertyRow({ name, value }: { name: string; value: unknown }) {
  const formattedValue = formatPropertyValue(value)
  const isMultiline = formattedValue.includes('\n')

  return (
    <div className={`${isMultiline ? 'flex flex-col gap-1' : 'flex items-start justify-between gap-2'} py-1.5 border-b border-border-subtle last:border-0`}>
      <span className="text-xs text-text-muted font-medium shrink-0">{name}</span>
      <span className={`text-xs text-text-secondary font-mono ${isMultiline ? 'whitespace-pre bg-bg-tertiary p-2 rounded overflow-x-auto' : 'text-right truncate'}`}>
        {formattedValue}
      </span>
    </div>
  )
}

export function NodeInspector({
  inspector,
  onExpandNeighbors,
  onDeleteNode,
}: NodeInspectorProps) {
  const { node } = inspector
  const color = labelToColor(node.label)

  const closeInspector = useInspectorStore((s) => s.closeInspector)
  const bringToFront = useInspectorStore((s) => s.bringToFront)
  const updatePosition = useInspectorStore((s) => s.updatePosition)
  const updateSize = useInspectorStore((s) => s.updateSize)
  const toggleMinimized = useInspectorStore((s) => s.toggleMinimized)

  const propertyEntries = Object.entries(node.properties)

  return (
    <FloatingPanel
      title={node.label}
      titleIcon={<Circle size={12} fill={color} stroke={color} />}
      position={inspector.position}
      size={inspector.size}
      minimized={inspector.minimized}
      zIndex={inspector.zIndex}
      onClose={() => closeInspector(inspector.id)}
      onPositionChange={(pos) => updatePosition(inspector.id, pos)}
      onSizeChange={(size) => updateSize(inspector.id, size)}
      onMinimizeToggle={() => toggleMinimized(inspector.id)}
      onFocus={() => bringToFront(inspector.id)}
    >
      <div className="flex flex-col h-full">
        {/* Node ID */}
        <div className="px-3 py-2 border-b border-border bg-bg-tertiary/50">
          <div className="flex items-center gap-2">
            <Hash size={12} className="text-text-muted shrink-0" />
            <span className="text-xs font-mono text-text-muted truncate" title={node.id}>
              {node.id}
            </span>
          </div>
        </div>

        {/* Labels */}
        {node.labels.length > 0 && (
          <CollapsibleSection
            title="Labels"
            icon={<Tag size={12} />}
            defaultOpen
            className="border-b border-border"
          >
            <div className="flex flex-wrap gap-1.5 px-3 py-2">
              {node.labels.map((label, idx) => (
                <span
                  key={idx}
                  className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium"
                  style={{
                    backgroundColor: `${labelToColor(label)}20`,
                    color: labelToColor(label),
                    border: `1px solid ${labelToColor(label)}40`,
                  }}
                >
                  {label}
                </span>
              ))}
            </div>
          </CollapsibleSection>
        )}

        {/* Properties */}
        <CollapsibleSection
          title={`Properties (${propertyEntries.length})`}
          icon={<ChevronRight size={12} />}
          defaultOpen
          className="flex-1 border-b border-border min-h-0"
        >
          <div className="px-3 py-1 overflow-y-auto max-h-48">
            {propertyEntries.length > 0 ? (
              propertyEntries.map(([key, value]) => (
                <PropertyRow key={key} name={key} value={value} />
              ))
            ) : (
              <div className="text-xs text-text-muted py-2 text-center">
                No properties
              </div>
            )}
          </div>
        </CollapsibleSection>

        {/* Actions */}
        <div className="px-3 py-2 border-t border-border mt-auto shrink-0">
          <div className="flex gap-2">
            {onExpandNeighbors && (
              <button
                onClick={() => onExpandNeighbors(node.id)}
                className="flex-1 px-3 py-1.5 text-xs font-medium bg-accent/10 text-accent hover:bg-accent/20 rounded transition-colors"
              >
                Expand Neighbors
              </button>
            )}
            {onDeleteNode && (
              <button
                onClick={() => {
                  if (confirm(`Delete node ${node.id}?`)) {
                    onDeleteNode(node.id)
                  }
                }}
                className="px-3 py-1.5 text-xs font-medium bg-red-500/10 text-red-400 hover:bg-red-500/20 rounded transition-colors"
              >
                Delete
              </button>
            )}
          </div>
        </div>
      </div>
    </FloatingPanel>
  )
}
