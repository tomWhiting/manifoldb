import CodeMirror from '@uiw/react-codemirror'
import { useAppStore } from '../../stores/app-store'
import { useTheme } from '../../hooks/useTheme'

export function QueryEditor() {
  const tabs = useAppStore((s) => s.tabs)
  const activeTabId = useAppStore((s) => s.activeTabId)
  const updateTabContent = useAppStore((s) => s.updateTabContent)
  const { isDark } = useTheme()

  const activeTab = tabs.find((t) => t.id === activeTabId)

  if (!activeTab) {
    return (
      <div className="flex items-center justify-center h-full text-text-muted">
        No query open
      </div>
    )
  }

  return (
    <CodeMirror
      value={activeTab.content}
      onChange={(value) => updateTabContent(activeTab.id, value)}
      height="100%"
      theme={isDark ? 'dark' : 'light'}
      basicSetup={{
        lineNumbers: true,
        highlightActiveLineGutter: true,
        highlightActiveLine: true,
        foldGutter: true,
        dropCursor: true,
        allowMultipleSelections: true,
        indentOnInput: true,
        bracketMatching: true,
        closeBrackets: true,
        autocompletion: true,
        rectangularSelection: true,
        crosshairCursor: true,
        highlightSelectionMatches: true,
      }}
      className="h-full text-sm [&_.cm-editor]:h-full [&_.cm-scroller]:!font-mono"
    />
  )
}
