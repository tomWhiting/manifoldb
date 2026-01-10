import { Group, Panel, Separator } from 'react-resizable-panels'
import { Play } from 'lucide-react'
import { QueryTabs } from '../editor/QueryTabs'
import { QueryEditor } from '../editor/QueryEditor'
import { UnifiedResultView } from '../result-views/UnifiedResultView'
import { IconButton } from '../shared/IconButton'
import { useAppStore } from '../../stores/app-store'
import { graphqlClient, CYPHER_QUERY } from '../../lib/graphql-client'

export function Workspace() {
  const tabs = useAppStore((s) => s.tabs)
  const activeTabId = useAppStore((s) => s.activeTabId)
  const setTabResult = useAppStore((s) => s.setTabResult)
  const setTabExecuting = useAppStore((s) => s.setTabExecuting)

  const activeTab = tabs.find((t) => t.id === activeTabId)

  const handleRunQuery = async () => {
    if (!activeTab || !activeTabId) return

    setTabExecuting(activeTabId, true)
    const startTime = performance.now()

    try {
      const result = await graphqlClient
        .query(CYPHER_QUERY, { query: activeTab.content })
        .toPromise()

      const executionTime = performance.now() - startTime

      if (result.error) {
        console.error('Query error:', result.error)
        setTabResult(activeTabId, {
          raw: { error: result.error.message },
          executionTime,
        })
      } else {
        const data = result.data?.cypher
        setTabResult(activeTabId, {
          nodes: data?.nodes ?? [],
          edges: data?.edges ?? [],
          raw: data,
          executionTime,
          rowCount: (data?.nodes?.length ?? 0) + (data?.edges?.length ?? 0),
        })
      }
    } catch (err) {
      console.error('Query failed:', err)
      setTabResult(activeTabId, {
        raw: { error: String(err) },
        executionTime: performance.now() - startTime,
      })
    } finally {
      setTabExecuting(activeTabId, false)
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Query tabs with run button */}
      <div className="flex items-center border-b border-neutral-800">
        <QueryTabs />
        <div className="flex-1" />
        <div className="px-2">
          <IconButton
            icon={<Play size={16} className="fill-current" />}
            onClick={handleRunQuery}
            tooltip="Run query (Cmd+Enter)"
            variant="default"
            disabled={activeTab?.isExecuting}
            className="bg-blue-600 hover:bg-blue-500 text-white"
          />
        </div>
      </div>

      {/* Split panes: Editor and Results */}
      <Group orientation="vertical" className="flex-1" defaultLayout={{ editor: 40, results: 60 }}>
        <Panel id="editor" minSize={20}>
          <QueryEditor />
        </Panel>
        <Separator className="h-1 bg-neutral-800 hover:bg-blue-600 transition-colors cursor-row-resize" />
        <Panel id="results" minSize={20}>
          <UnifiedResultView />
        </Panel>
      </Group>
    </div>
  )
}
