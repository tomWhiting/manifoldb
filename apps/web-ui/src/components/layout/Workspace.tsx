import { useEffect } from 'react'
import { Group, Panel, Separator } from 'react-resizable-panels'
import { Play, Square } from 'lucide-react'
import { QueryTabs } from '../editor/QueryTabs'
import { QueryEditor } from '../editor/QueryEditor'
import { UnifiedResultView } from '../result-views/UnifiedResultView'
import { SettingsPanel } from '../settings/SettingsPanel'
import { SchemaPanel } from '../sidebar/SchemaPanel'
import { OverviewPanel } from '../sidebar/OverviewPanel'
import { IconButton } from '../shared/IconButton'
import { useAppStore } from '../../stores/app-store'
import { useQueryExecution } from '../../hooks/useQueryExecution'

function QueryWorkspace() {
  const tabs = useAppStore((s) => s.tabs)
  const activeTabId = useAppStore((s) => s.activeTabId)
  const activeTab = tabs.find((t) => t.id === activeTabId)

  const { execute, cancel, isExecuting } = useQueryExecution()

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0
      const modKey = isMac ? e.metaKey : e.ctrlKey

      if (modKey && e.key === 'Enter') {
        e.preventDefault()
        if (!isExecuting) {
          execute()
        }
      }

      if (modKey && e.key === '.') {
        e.preventDefault()
        if (isExecuting) {
          cancel()
        }
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [execute, cancel, isExecuting])

  return (
    <div className="flex flex-col h-full">
      {/* Query tabs with run/cancel button */}
      <div className="flex items-center border-b border-border">
        <QueryTabs />
        <div className="flex-1" />
        <div className="px-2">
          {isExecuting ? (
            <IconButton
              icon={<Square size={14} className="fill-current" />}
              onClick={cancel}
              tooltip="Cancel query (Cmd+.)"
              variant="default"
              className="bg-red-600 hover:bg-red-700 text-white"
            />
          ) : (
            <IconButton
              icon={<Play size={16} className="fill-current" />}
              onClick={execute}
              tooltip="Run query (Cmd+Enter)"
              variant="default"
              disabled={!activeTab}
              className="bg-accent hover:bg-accent-hover text-white"
            />
          )}
        </div>
      </div>

      {/* Split panes: Editor and Results */}
      <Group orientation="vertical" className="flex-1" defaultLayout={{ editor: 40, results: 60 }}>
        <Panel id="editor" minSize={20}>
          <QueryEditor />
        </Panel>
        <Separator className="h-1 bg-border hover:bg-accent transition-colors cursor-row-resize" />
        <Panel id="results" minSize={20}>
          <UnifiedResultView />
        </Panel>
      </Group>
    </div>
  )
}

export function Workspace() {
  const activeSidebarSection = useAppStore((s) => s.activeSidebarSection)

  if (activeSidebarSection === 'settings') {
    return <SettingsPanel />
  }

  if (activeSidebarSection === 'schema') {
    return <SchemaPanel />
  }

  if (activeSidebarSection === 'overview') {
    return <OverviewPanel />
  }

  return <QueryWorkspace />
}
