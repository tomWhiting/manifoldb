import { ArrowRight, Circle, Hash, ChevronRight } from 'lucide-react'
import { FloatingPanel } from '../shared/FloatingPanel'
import { useInspectorStore, type EdgeInspector as EdgeInspectorType } from '../../stores/inspector-store'
import { labelToColor } from '../../utils/graph-layout'
import { CollapsibleSection } from '../shared/CollapsibleSection'

interface EdgeInspectorProps {
  inspector: EdgeInspectorType
  onNavigateToNode?: (nodeId: string) => void
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

function NodeReference({
  label,
  nodeLabel,
  nodeId,
  onClick,
}: {
  label: string
  nodeLabel: string
  nodeId: string
  onClick?: () => void
}) {
  const color = labelToColor(nodeLabel)

  return (
    <div className="flex items-center justify-between py-2">
      <span className="text-xs text-text-muted">{label}</span>
      <button
        onClick={onClick}
        disabled={!onClick}
        className={`flex items-center gap-2 px-2 py-1 rounded text-xs font-medium transition-colors ${
          onClick
            ? 'bg-bg-tertiary hover:bg-border cursor-pointer'
            : 'bg-bg-tertiary cursor-default'
        }`}
        title={nodeId}
      >
        <Circle size={10} fill={color} stroke={color} />
        <span className="text-text-secondary truncate max-w-[120px]">{nodeLabel}</span>
        {onClick && <ChevronRight size={12} className="text-text-muted" />}
      </button>
    </div>
  )
}

export function EdgeInspector({ inspector, onNavigateToNode }: EdgeInspectorProps) {
  const { edge, sourceNode, targetNode } = inspector

  const closeInspector = useInspectorStore((s) => s.closeInspector)
  const bringToFront = useInspectorStore((s) => s.bringToFront)
  const updatePosition = useInspectorStore((s) => s.updatePosition)
  const updateSize = useInspectorStore((s) => s.updateSize)
  const toggleMinimized = useInspectorStore((s) => s.toggleMinimized)

  const propertyEntries = Object.entries(edge.properties)

  return (
    <FloatingPanel
      title={edge.type}
      titleIcon={<ArrowRight size={12} className="text-text-muted" />}
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
        {/* Edge ID */}
        <div className="px-3 py-2 border-b border-border bg-bg-tertiary/50">
          <div className="flex items-center gap-2">
            <Hash size={12} className="text-text-muted shrink-0" />
            <span className="text-xs font-mono text-text-muted truncate" title={edge.id}>
              {edge.id}
            </span>
          </div>
        </div>

        {/* Relationship */}
        <div className="px-3 py-2 border-b border-border">
          <div className="flex items-center justify-center gap-2">
            <div className="flex items-center gap-1">
              <Circle size={8} fill={labelToColor(sourceNode.label)} stroke={labelToColor(sourceNode.label)} />
              <span className="text-xs text-text-secondary">{sourceNode.label}</span>
            </div>
            <div className="flex items-center gap-1 px-2 py-0.5 bg-bg-tertiary rounded text-xs text-text-muted">
              <ArrowRight size={10} />
              <span className="font-medium">{edge.type}</span>
              <ArrowRight size={10} />
            </div>
            <div className="flex items-center gap-1">
              <Circle size={8} fill={labelToColor(targetNode.label)} stroke={labelToColor(targetNode.label)} />
              <span className="text-xs text-text-secondary">{targetNode.label}</span>
            </div>
          </div>
        </div>

        {/* Source/Target nodes */}
        <CollapsibleSection
          title="Connected Nodes"
          icon={<ChevronRight size={12} />}
          defaultOpen
          className="border-b border-border"
        >
          <div className="px-3">
            <NodeReference
              label="Source"
              nodeLabel={sourceNode.label}
              nodeId={sourceNode.id}
              onClick={onNavigateToNode ? () => onNavigateToNode(sourceNode.id) : undefined}
            />
            <NodeReference
              label="Target"
              nodeLabel={targetNode.label}
              nodeId={targetNode.id}
              onClick={onNavigateToNode ? () => onNavigateToNode(targetNode.id) : undefined}
            />
          </div>
        </CollapsibleSection>

        {/* Properties */}
        <CollapsibleSection
          title={`Properties (${propertyEntries.length})`}
          icon={<ChevronRight size={12} />}
          defaultOpen
          className="flex-1 min-h-0"
        >
          <div className="px-3 py-1 overflow-y-auto max-h-32">
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
      </div>
    </FloatingPanel>
  )
}
