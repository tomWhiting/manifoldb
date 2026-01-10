import { GitBranch, Table, Braces } from 'lucide-react'
import { useAppStore } from '../../stores/app-store'
import { GraphCanvas } from './GraphCanvas'
import { TableView } from './TableView'
import { JSONView } from './JSONView'
import type { ViewMode } from '../../types'

const viewModes: { id: ViewMode; icon: typeof GitBranch; label: string }[] = [
  { id: 'graph', icon: GitBranch, label: 'Graph' },
  { id: 'table', icon: Table, label: 'Table' },
  { id: 'json', icon: Braces, label: 'JSON' },
]

export function UnifiedResultView() {
  const activeViewMode = useAppStore((s) => s.activeViewMode)
  const setViewMode = useAppStore((s) => s.setViewMode)

  return (
    <div className="flex flex-col h-full">
      {/* View mode switcher */}
      <div className="flex items-center gap-1 px-2 py-1.5 border-b border-neutral-800 bg-neutral-900/50">
        {viewModes.map((mode) => {
          const Icon = mode.icon
          const isActive = activeViewMode === mode.id
          return (
            <button
              key={mode.id}
              onClick={() => setViewMode(mode.id)}
              className={`
                flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium
                transition-colors duration-150
                ${
                  isActive
                    ? 'bg-blue-600/20 text-blue-400'
                    : 'text-neutral-400 hover:text-neutral-200 hover:bg-white/5'
                }
              `}
            >
              <Icon size={14} />
              {mode.label}
            </button>
          )
        })}
      </div>

      {/* View content */}
      <div className="flex-1 min-h-0">
        {activeViewMode === 'graph' && <GraphCanvas />}
        {activeViewMode === 'table' && <TableView />}
        {activeViewMode === 'json' && <JSONView />}
      </div>
    </div>
  )
}
