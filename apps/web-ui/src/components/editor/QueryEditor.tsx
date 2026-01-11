import { useMemo } from 'react'
import CodeMirror from '@uiw/react-codemirror'
import { indentUnit } from '@codemirror/language'
import { EditorState } from '@codemirror/state'
import { EditorView } from '@codemirror/view'
import { useAppStore } from '../../stores/app-store'
import { useTheme } from '../../hooks/useTheme'
import { useSettingsStore } from '../../stores/settings-store'

export function QueryEditor() {
  const tabs = useAppStore((s) => s.tabs)
  const activeTabId = useAppStore((s) => s.activeTabId)
  const updateTabContent = useAppStore((s) => s.updateTabContent)
  const { isDark } = useTheme()

  // Editor settings
  const editor = useSettingsStore((s) => s.editor)

  const activeTab = tabs.find((t) => t.id === activeTabId)

  // Build extensions based on settings
  const extensions = useMemo(() => {
    const exts = [
      indentUnit.of(' '.repeat(editor.tabSize)),
      EditorState.tabSize.of(editor.tabSize),
    ]

    if (editor.wordWrap) {
      exts.push(EditorView.lineWrapping)
    }

    return exts
  }, [editor.tabSize, editor.wordWrap])

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
      extensions={extensions}
      basicSetup={{
        lineNumbers: editor.lineNumbers,
        highlightActiveLineGutter: true,
        highlightActiveLine: true,
        foldGutter: true,
        dropCursor: true,
        allowMultipleSelections: true,
        indentOnInput: true,
        bracketMatching: true,
        closeBrackets: true,
        autocompletion: editor.autoComplete,
        rectangularSelection: true,
        crosshairCursor: true,
        highlightSelectionMatches: true,
      }}
      className="h-full [&_.cm-editor]:h-full [&_.cm-scroller]:!font-mono"
      style={{ fontSize: `${editor.fontSize}px` }}
    />
  )
}
