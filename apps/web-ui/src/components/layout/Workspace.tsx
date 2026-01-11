import { useEffect } from 'react'
import { Group, Panel, Separator } from 'react-resizable-panels'
import { SettingsPanel } from '../settings/SettingsPanel'
import { SchemaPanel } from '../sidebar/SchemaPanel'
import { OverviewPanel } from '../sidebar/OverviewPanel'
import { HistoryPanel } from '../sidebar/HistoryPanel'
import { CollectionsPanel } from '../sidebar/CollectionsPanel'
import { QueryPanel } from '../sidebar/QueryPanel'
import { QueryPane } from './QueryPane'
import { SplitPaneLayout } from './SplitPaneLayout'
import { useAppStore } from '../../stores/app-store'
import { useWorkspaceStore } from '../../stores/workspace-store'

function SplitQueryWorkspace() {
  const layout = useWorkspaceStore((s) => s.layout)
  const updateSizes = useWorkspaceStore((s) => s.updateSizes)
  const splitPane = useWorkspaceStore((s) => s.splitPane)
  const closePane = useWorkspaceStore((s) => s.closePane)

  // Global keyboard shortcuts for split operations
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0
      const modKey = isMac ? e.metaKey : e.ctrlKey

      // Cmd+\ for vertical split
      if (modKey && e.key === '\\' && !e.shiftKey) {
        e.preventDefault()
        splitPane(layout.activePaneId, 'vertical')
      }

      // Cmd+Shift+\ for horizontal split
      if (modKey && e.key === '\\' && e.shiftKey) {
        e.preventDefault()
        splitPane(layout.activePaneId, 'horizontal')
      }

      // Cmd+W to close pane (only if multiple panes)
      if (modKey && e.key === 'w' && Object.keys(layout.panes).length > 1) {
        e.preventDefault()
        closePane(layout.activePaneId)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [layout.activePaneId, layout.panes, splitPane, closePane])

  return (
    <SplitPaneLayout
      node={layout.root}
      panes={layout.panes}
      activePaneId={layout.activePaneId}
      onSizesChange={updateSizes}
      renderPane={(pane, isActive) => (
        <QueryPane pane={pane} isActive={isActive} />
      )}
    />
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

  if (activeSidebarSection === 'history') {
    return <HistoryPanel />
  }

  if (activeSidebarSection === 'collections') {
    return <CollectionsPanel />
  }

  if (activeSidebarSection === 'query') {
    return (
      <Group orientation="horizontal" className="h-full">
        <Panel id="query-panel" defaultSize={25} minSize={15} maxSize={40}>
          <div className="h-full overflow-hidden border-r border-border bg-bg-secondary">
            <QueryPanel />
          </div>
        </Panel>
        <Separator className="w-1 bg-border hover:bg-accent transition-colors cursor-col-resize" />
        <Panel id="query-workspace" minSize={40}>
          <SplitQueryWorkspace />
        </Panel>
      </Group>
    )
  }

  return <SplitQueryWorkspace />
}
