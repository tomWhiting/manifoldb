import { X, Plus } from 'lucide-react'
import { useAppStore } from '../../stores/app-store'
import { IconButton } from '../shared/IconButton'

export function QueryTabs() {
  const tabs = useAppStore((s) => s.tabs)
  const activeTabId = useAppStore((s) => s.activeTabId)
  const setActiveTab = useAppStore((s) => s.setActiveTab)
  const removeTab = useAppStore((s) => s.removeTab)
  const addTab = useAppStore((s) => s.addTab)

  const handleNewTab = () => {
    addTab({
      title: `Query ${tabs.length + 1}`,
      content: '// New query\n',
      language: 'cypher',
    })
  }

  return (
    <div className="flex items-center gap-1 px-2 py-1 bg-neutral-900 border-b border-neutral-800">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => setActiveTab(tab.id)}
          className={`
            group flex items-center gap-2 px-3 py-1.5 rounded-t text-sm
            transition-colors duration-150
            ${
              activeTabId === tab.id
                ? 'bg-neutral-950 text-neutral-100'
                : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800/50'
            }
          `}
        >
          <span className="truncate max-w-32">{tab.title}</span>
          {tabs.length > 1 && (
            <span
              onClick={(e) => {
                e.stopPropagation()
                removeTab(tab.id)
              }}
              className={`
                p-0.5 rounded hover:bg-neutral-700
                ${activeTabId === tab.id ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'}
              `}
            >
              <X size={12} />
            </span>
          )}
        </button>
      ))}

      <IconButton icon={<Plus size={14} />} onClick={handleNewTab} tooltip="New query" size="sm" />
    </div>
  )
}
