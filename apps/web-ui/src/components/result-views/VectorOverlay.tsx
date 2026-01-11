import { Layers, Target, Grid3x3, X, ChevronDown } from 'lucide-react'
import type { UseVectorOverlayReturn } from '../../hooks/useVectorOverlay'
import type { ProjectionMethod } from '../../utils/vector-projection'

interface VectorOverlayControlsProps {
  overlay: UseVectorOverlayReturn
  hasVectorData: boolean
  onSelectReferenceNode?: () => void
}

export function VectorOverlayControls({
  overlay,
  hasVectorData,
  onSelectReferenceNode,
}: VectorOverlayControlsProps) {
  const { state, isComputing, setEnabled, setMode, setProjectionMethod, setClusterCount } = overlay

  if (!hasVectorData) {
    return null
  }

  return (
    <div className="absolute top-2 left-2 flex flex-col gap-2">
      {/* Toggle button */}
      <button
        onClick={() => setEnabled(!state.enabled)}
        className={`flex items-center gap-2 px-3 py-1.5 rounded-md border transition-colors ${
          state.enabled
            ? 'bg-accent-primary/20 border-accent-primary text-accent-primary'
            : 'bg-bg-secondary/80 backdrop-blur-sm border-border text-text-muted hover:text-text-primary hover:border-border-hover'
        }`}
        title="Toggle vector overlay"
      >
        <Layers size={16} />
        <span className="text-sm">Vectors</span>
        {isComputing && (
          <div className="w-3 h-3 border-2 border-current border-t-transparent rounded-full animate-spin" />
        )}
      </button>

      {/* Expanded controls when enabled */}
      {state.enabled && (
        <div className="bg-bg-secondary/95 backdrop-blur-sm border border-border rounded-md p-3 min-w-[200px] shadow-lg">
          {/* Mode selector */}
          <div className="mb-3">
            <label className="text-xs text-text-muted mb-1 block">Mode</label>
            <div className="flex gap-1">
              <ModeButton
                active={state.mode === 'similarity'}
                onClick={() => setMode('similarity')}
                icon={<Target size={14} />}
                label="Similarity"
                title="Color nodes by similarity to reference"
              />
              <ModeButton
                active={state.mode === 'clusters'}
                onClick={() => setMode('clusters')}
                icon={<Grid3x3 size={14} />}
                label="Clusters"
                title="Group nodes by embedding clusters"
              />
            </div>
          </div>

          {/* Mode-specific controls */}
          {state.mode === 'similarity' && (
            <div className="mb-3">
              <label className="text-xs text-text-muted mb-1 block">Reference Node</label>
              {state.referenceNodeId ? (
                <div className="flex items-center gap-2 bg-bg-tertiary rounded px-2 py-1.5 text-sm">
                  <span className="truncate flex-1">{state.referenceNodeId.slice(0, 12)}...</span>
                  <button
                    onClick={() => overlay.setReferenceNode(null)}
                    className="p-0.5 hover:bg-bg-secondary rounded"
                    title="Clear reference"
                  >
                    <X size={14} />
                  </button>
                </div>
              ) : (
                <button
                  onClick={onSelectReferenceNode}
                  className="w-full text-left text-sm text-text-muted hover:text-text-primary bg-bg-tertiary hover:bg-bg-secondary rounded px-2 py-1.5 transition-colors"
                >
                  Click a node to select...
                </button>
              )}
              <p className="text-xs text-text-muted mt-1">
                Red = similar, Blue = dissimilar
              </p>
            </div>
          )}

          {state.mode === 'clusters' && (
            <div className="mb-3">
              <label className="text-xs text-text-muted mb-1 block">
                Cluster Count: {state.clusterCount}
              </label>
              <input
                type="range"
                min={2}
                max={15}
                value={state.clusterCount}
                onChange={(e) => setClusterCount(parseInt(e.target.value))}
                className="w-full h-1.5 bg-bg-tertiary rounded-lg appearance-none cursor-pointer accent-accent-primary"
              />
            </div>
          )}

          {/* Projection method */}
          <div>
            <label className="text-xs text-text-muted mb-1 block">Projection</label>
            <ProjectionSelect
              value={state.projectionMethod}
              onChange={setProjectionMethod}
            />
          </div>
        </div>
      )}
    </div>
  )
}

interface ModeButtonProps {
  active: boolean
  onClick: () => void
  icon: React.ReactNode
  label: string
  title: string
}

function ModeButton({ active, onClick, icon, label, title }: ModeButtonProps) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 px-2 py-1 rounded text-xs transition-colors ${
        active
          ? 'bg-accent-primary text-white'
          : 'bg-bg-tertiary text-text-muted hover:text-text-primary hover:bg-bg-secondary'
      }`}
      title={title}
    >
      {icon}
      {label}
    </button>
  )
}

interface ProjectionSelectProps {
  value: ProjectionMethod
  onChange: (method: ProjectionMethod) => void
}

function ProjectionSelect({ value, onChange }: ProjectionSelectProps) {
  const options: { value: ProjectionMethod; label: string }[] = [
    { value: 'pca', label: 'PCA (Fast)' },
    { value: 'umap', label: 'UMAP' },
    { value: 'tsne', label: 't-SNE' },
  ]

  return (
    <div className="relative">
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as ProjectionMethod)}
        className="w-full appearance-none bg-bg-tertiary text-text-primary text-sm rounded px-2 py-1.5 pr-7 cursor-pointer hover:bg-bg-secondary transition-colors focus:outline-none focus:ring-1 focus:ring-accent-primary"
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
      <ChevronDown
        size={14}
        className="absolute right-2 top-1/2 -translate-y-1/2 text-text-muted pointer-events-none"
      />
    </div>
  )
}

interface VectorOverlayLegendProps {
  overlay: UseVectorOverlayReturn
}

export function VectorOverlayLegend({ overlay }: VectorOverlayLegendProps) {
  const { state, clusters } = overlay

  if (!state.enabled) {
    return null
  }

  if (state.mode === 'similarity') {
    return (
      <div className="absolute bottom-10 left-2 bg-bg-secondary/95 backdrop-blur-sm border border-border rounded-md p-2">
        <div className="text-xs text-text-muted mb-1">Similarity</div>
        <div className="flex items-center gap-1">
          <div
            className="w-20 h-3 rounded"
            style={{
              background: 'linear-gradient(to right, hsl(240, 70%, 55%), hsl(120, 70%, 55%), hsl(0, 70%, 55%))',
            }}
          />
        </div>
        <div className="flex justify-between text-[10px] text-text-muted mt-0.5">
          <span>Low</span>
          <span>High</span>
        </div>
      </div>
    )
  }

  if (state.mode === 'clusters' && clusters) {
    return (
      <div className="absolute bottom-10 left-2 bg-bg-secondary/95 backdrop-blur-sm border border-border rounded-md p-2">
        <div className="text-xs text-text-muted mb-1">Clusters</div>
        <div className="flex flex-wrap gap-1 max-w-[180px]">
          {Array.from(clusters.clusters.keys()).map((clusterId) => {
            const count = clusters.clusters.get(clusterId)?.length ?? 0
            const hue = (clusterId * 360) / clusters.clusters.size
            return (
              <div
                key={clusterId}
                className="flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px]"
                style={{ backgroundColor: `hsla(${hue}, 65%, 55%, 0.2)` }}
              >
                <div
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: `hsl(${hue}, 65%, 55%)` }}
                />
                <span>{count}</span>
              </div>
            )
          })}
        </div>
      </div>
    )
  }

  return null
}
