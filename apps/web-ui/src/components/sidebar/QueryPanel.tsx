import { useState, useMemo } from 'react'
import {
  Search,
  Play,
  Trash2,
  Wand2,
  ChevronDown,
  ChevronRight,
  Code2,
  Database,
  FileCode,
  Copy,
} from 'lucide-react'
import { IconButton } from '../shared/IconButton'
import { useAppStore } from '../../stores/app-store'
import { useQueryExecution } from '../../hooks/useQueryExecution'
import { CollapsibleSection } from '../shared/CollapsibleSection'
import {
  cypherSnippets,
  sqlSnippets,
  cypherTemplates,
  sqlTemplates,
  applyTemplate,
  type QuerySnippet,
  type QueryTemplate,
  type TemplatePlaceholder,
} from '../../lib/query-snippets'

interface SnippetItemProps {
  snippet: QuerySnippet
  onInsert: (query: string) => void
}

function SnippetItem({ snippet, onInsert }: SnippetItemProps) {
  return (
    <button
      onClick={() => onInsert(snippet.query)}
      className="flex items-start gap-2 w-full px-6 py-2 text-left hover:bg-bg-tertiary transition-colors group"
    >
      <Copy size={14} className="text-text-muted mt-0.5 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0" />
      <div className="flex-1 min-w-0">
        <span className="text-sm text-text-secondary group-hover:text-accent block truncate">
          {snippet.name}
        </span>
        <span className="text-xs text-text-muted block truncate">{snippet.description}</span>
      </div>
    </button>
  )
}

interface TemplateItemProps {
  template: QueryTemplate
  onInsert: (query: string) => void
}

function TemplateItem({ template, onInsert }: TemplateItemProps) {
  const [isExpanded, setIsExpanded] = useState(false)
  const [values, setValues] = useState<Record<string, string>>({})

  const handleInsert = () => {
    const query = applyTemplate(template.template, template.placeholders, values)
    onInsert(query)
    setIsExpanded(false)
    setValues({})
  }

  const handleQuickInsert = () => {
    const query = applyTemplate(template.template, template.placeholders)
    onInsert(query)
  }

  const handleValueChange = (key: string, value: string) => {
    setValues((prev) => ({ ...prev, [key]: value }))
  }

  return (
    <div className="border-b border-border/50 last:border-b-0">
      <div className="flex items-start gap-2 px-6 py-2 hover:bg-bg-tertiary transition-colors group">
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="mt-0.5 p-0.5 hover:bg-bg-tertiary rounded"
        >
          {isExpanded ? (
            <ChevronDown size={14} className="text-text-muted" />
          ) : (
            <ChevronRight size={14} className="text-text-muted" />
          )}
        </button>
        <button onClick={handleQuickInsert} className="flex-1 min-w-0 text-left">
          <span className="text-sm text-text-secondary group-hover:text-accent block truncate">
            {template.name}
          </span>
          <span className="text-xs text-text-muted block truncate">{template.description}</span>
        </button>
      </div>
      {isExpanded && (
        <div className="px-6 pb-3 space-y-2">
          {template.placeholders.map((placeholder) => (
            <PlaceholderInput
              key={placeholder.key}
              placeholder={placeholder}
              value={values[placeholder.key]}
              onChange={(value) => handleValueChange(placeholder.key, value)}
            />
          ))}
          <button
            onClick={handleInsert}
            className="w-full mt-2 px-3 py-1.5 bg-accent hover:bg-accent-hover text-white text-sm rounded transition-colors"
          >
            Insert with values
          </button>
        </div>
      )}
    </div>
  )
}

interface PlaceholderInputProps {
  placeholder: TemplatePlaceholder
  value: string | undefined
  onChange: (value: string) => void
}

function PlaceholderInput({ placeholder, value, onChange }: PlaceholderInputProps) {
  return (
    <div className="flex items-center gap-2">
      <label className="text-xs text-text-muted w-24 flex-shrink-0 truncate" title={placeholder.label}>
        {placeholder.label}
      </label>
      <input
        type="text"
        value={value ?? placeholder.defaultValue}
        onChange={(e) => onChange(e.target.value)}
        className="flex-1 px-2 py-1 text-xs bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent text-text-primary"
        placeholder={placeholder.defaultValue}
      />
    </div>
  )
}

export function QueryPanel() {
  const [searchQuery, setSearchQuery] = useState('')
  const activeTabId = useAppStore((s) => s.activeTabId)
  const tabs = useAppStore((s) => s.tabs)
  const updateTabContent = useAppStore((s) => s.updateTabContent)
  const activeTab = tabs.find((t) => t.id === activeTabId)

  const { execute, isExecuting } = useQueryExecution()

  const insertSnippet = (query: string) => {
    if (activeTabId) {
      updateTabContent(activeTabId, query)
    }
  }

  const handleClearEditor = () => {
    if (activeTabId) {
      updateTabContent(activeTabId, '')
    }
  }

  const handleRunQuery = () => {
    if (!isExecuting) {
      execute()
    }
  }

  // Filter snippets and templates based on search query
  const filteredCypherSnippets = useMemo(() => {
    if (!searchQuery) return cypherSnippets
    const query = searchQuery.toLowerCase()
    return cypherSnippets.filter(
      (s) =>
        s.name.toLowerCase().includes(query) ||
        s.description.toLowerCase().includes(query) ||
        s.query.toLowerCase().includes(query)
    )
  }, [searchQuery])

  const filteredSqlSnippets = useMemo(() => {
    if (!searchQuery) return sqlSnippets
    const query = searchQuery.toLowerCase()
    return sqlSnippets.filter(
      (s) =>
        s.name.toLowerCase().includes(query) ||
        s.description.toLowerCase().includes(query) ||
        s.query.toLowerCase().includes(query)
    )
  }, [searchQuery])

  const filteredCypherTemplates = useMemo(() => {
    if (!searchQuery) return cypherTemplates
    const query = searchQuery.toLowerCase()
    return cypherTemplates.filter(
      (t) =>
        t.name.toLowerCase().includes(query) ||
        t.description.toLowerCase().includes(query) ||
        t.template.toLowerCase().includes(query)
    )
  }, [searchQuery])

  const filteredSqlTemplates = useMemo(() => {
    if (!searchQuery) return sqlTemplates
    const query = searchQuery.toLowerCase()
    return sqlTemplates.filter(
      (t) =>
        t.name.toLowerCase().includes(query) ||
        t.description.toLowerCase().includes(query) ||
        t.template.toLowerCase().includes(query)
    )
  }, [searchQuery])

  const totalResults =
    filteredCypherSnippets.length +
    filteredSqlSnippets.length +
    filteredCypherTemplates.length +
    filteredSqlTemplates.length

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <h2 className="text-lg font-semibold text-text-primary">Query</h2>
      </div>

      {/* Quick Actions */}
      <div className="flex items-center gap-1 px-4 py-2 border-b border-border bg-bg-secondary">
        <IconButton
          icon={<Play size={16} className="fill-current" />}
          onClick={handleRunQuery}
          disabled={!activeTab || isExecuting}
          tooltip="Run query (Cmd+Enter)"
          className="bg-accent hover:bg-accent-hover text-white"
        />
        <IconButton
          icon={<Trash2 size={16} />}
          onClick={handleClearEditor}
          disabled={!activeTabId}
          tooltip="Clear editor"
        />
        <IconButton
          icon={<Wand2 size={16} />}
          disabled
          tooltip="Format query (coming soon)"
        />
      </div>

      {/* Search */}
      <div className="px-4 py-3 border-b border-border">
        <div className="relative">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search snippets..."
            className="w-full pl-9 pr-3 py-2 text-sm bg-bg-tertiary border border-border rounded-md focus:outline-none focus:border-accent text-text-primary placeholder:text-text-muted"
          />
        </div>
        {searchQuery && (
          <p className="mt-2 text-xs text-text-muted">
            {totalResults} result{totalResults !== 1 ? 's' : ''} found
          </p>
        )}
      </div>

      {/* Sections */}
      <div className="flex-1 overflow-y-auto">
        {/* Cypher Snippets */}
        {filteredCypherSnippets.length > 0 && (
          <CollapsibleSection
            title="Cypher Snippets"
            icon={<Code2 size={16} />}
            count={filteredCypherSnippets.length}
          >
            {filteredCypherSnippets.map((snippet) => (
              <SnippetItem key={snippet.id} snippet={snippet} onInsert={insertSnippet} />
            ))}
          </CollapsibleSection>
        )}

        {/* SQL Snippets */}
        {filteredSqlSnippets.length > 0 && (
          <CollapsibleSection
            title="SQL Snippets"
            icon={<Database size={16} />}
            count={filteredSqlSnippets.length}
            defaultOpen={false}
          >
            {filteredSqlSnippets.map((snippet) => (
              <SnippetItem key={snippet.id} snippet={snippet} onInsert={insertSnippet} />
            ))}
          </CollapsibleSection>
        )}

        {/* Cypher Templates */}
        {filteredCypherTemplates.length > 0 && (
          <CollapsibleSection
            title="Cypher Templates"
            icon={<FileCode size={16} />}
            count={filteredCypherTemplates.length}
            defaultOpen={false}
          >
            {filteredCypherTemplates.map((template) => (
              <TemplateItem key={template.id} template={template} onInsert={insertSnippet} />
            ))}
          </CollapsibleSection>
        )}

        {/* SQL Templates */}
        {filteredSqlTemplates.length > 0 && (
          <CollapsibleSection
            title="SQL Templates"
            icon={<Database size={16} />}
            count={filteredSqlTemplates.length}
            defaultOpen={false}
          >
            {filteredSqlTemplates.map((template) => (
              <TemplateItem key={template.id} template={template} onInsert={insertSnippet} />
            ))}
          </CollapsibleSection>
        )}

        {/* No results */}
        {totalResults === 0 && searchQuery && (
          <div className="px-4 py-8 text-center">
            <p className="text-sm text-text-muted">No snippets or templates found</p>
            <p className="text-xs text-text-muted mt-1">Try a different search term</p>
          </div>
        )}
      </div>

      {/* Footer with keyboard shortcuts hint */}
      <div className="px-4 py-2 border-t border-border bg-bg-secondary">
        <p className="text-xs text-text-muted">
          <kbd className="px-1.5 py-0.5 bg-bg-tertiary rounded text-text-secondary">Cmd+Enter</kbd> to run query
        </p>
      </div>
    </div>
  )
}
