import { X, Plus, Database, GitBranch } from 'lucide-react'
import { useAppStore } from '../../stores/app-store'
import { IconButton } from '../shared/IconButton'

export function QueryTabs() {
  const tabs = useAppStore((s) => s.tabs)
  const activeTabId = useAppStore((s) => s.activeTabId)
  const setActiveTab = useAppStore((s) => s.setActiveTab)
  const removeTab = useAppStore((s) => s.removeTab)
  const addTab = useAppStore((s) => s.addTab)
  const setTabLanguage = useAppStore((s) => s.setTabLanguage)

  const handleNewTab = () => {
    addTab({
      title: `Query ${tabs.length + 1}`,
      content: '// New query\n',
      language: 'cypher',
    })
  }

  const toggleLanguage = (tabId: string, currentLanguage: 'cypher' | 'sql') => {
    setTabLanguage(tabId, currentLanguage === 'cypher' ? 'sql' : 'cypher')
  }

  return (
    <div className="flex items-center gap-1 px-2 py-1 bg-bg-secondary">
      {tabs.map((tab) => (
        <div
          key={tab.id}
          className={`
            group flex items-center rounded-t text-sm
            transition-colors duration-150
            ${
              activeTabId === tab.id
                ? 'bg-bg-primary text-text-primary'
                : 'text-text-muted hover:text-text-secondary hover:bg-bg-tertiary'
            }
          `}
        >
          {/* Language toggle button */}
          <button
            onClick={(e) => {
              e.stopPropagation()
              toggleLanguage(tab.id, tab.language)
            }}
            title={`Switch to ${tab.language === 'cypher' ? 'SQL' : 'Cypher'}`}
            className={`
              flex items-center gap-1 px-2 py-1.5 rounded-l
              hover:bg-bg-tertiary transition-colors
              ${activeTabId === tab.id ? 'opacity-100' : 'opacity-70 group-hover:opacity-100'}
            `}
          >
            {tab.language === 'sql' ? (
              <Database size={12} className="text-blue-400" />
            ) : (
              <GitBranch size={12} className="text-green-400" />
            )}
            <span className="text-[10px] uppercase font-medium">
              {tab.language}
            </span>
          </button>

          {/* Tab title and close button */}
          <button
            onClick={() => setActiveTab(tab.id)}
            className="flex items-center gap-2 px-2 py-1.5"
          >
            <span className="truncate max-w-32">{tab.title}</span>
            {tabs.length > 1 && (
              <span
                onClick={(e) => {
                  e.stopPropagation()
                  removeTab(tab.id)
                }}
                className={`
                  p-0.5 rounded hover:bg-bg-tertiary
                  ${activeTabId === tab.id ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'}
                `}
              >
                <X size={12} />
              </span>
            )}
          </button>
        </div>
      ))}

      <IconButton icon={<Plus size={14} />} onClick={handleNewTab} tooltip="New query" size="sm" />
    </div>
  )
}
