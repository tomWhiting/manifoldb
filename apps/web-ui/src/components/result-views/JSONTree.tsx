import { useState, useCallback, useMemo, memo } from 'react'
import { ChevronRight, ChevronDown, Copy, Check } from 'lucide-react'
import type { JsonValue } from './useJSONTree'

interface JSONTreeProps {
  data: JsonValue
  searchQuery?: string
  currentMatchIndex?: number
  matchPaths?: string[]
  onCopyPath?: (path: string) => void
  expandedPaths: Set<string>
  onTogglePath: (path: string) => void
}

interface JSONNodeProps {
  keyName?: string
  value: JsonValue
  path: string
  depth: number
  searchQuery?: string
  currentMatchIndex?: number
  matchPaths?: string[]
  onCopyPath?: (path: string) => void
  expandedPaths: Set<string>
  onTogglePath: (path: string) => void
  isLast: boolean
}

function highlightText(text: string, query: string, path: string, matchPaths?: string[], currentMatchIndex?: number) {
  if (!query || query.length === 0) {
    return <span>{text}</span>
  }

  const lowerText = text.toLowerCase()
  const lowerQuery = query.toLowerCase()
  const parts: React.ReactNode[] = []
  let lastIndex = 0
  let matchIndex = 0

  let index = lowerText.indexOf(lowerQuery)
  while (index !== -1) {
    if (index > lastIndex) {
      parts.push(text.substring(lastIndex, index))
    }

    const isCurrentMatch =
      matchPaths &&
      currentMatchIndex !== undefined &&
      matchPaths[currentMatchIndex] === path &&
      matchPaths.slice(0, currentMatchIndex + 1).filter((p) => p === path).length - 1 === matchIndex

    parts.push(
      <mark
        key={`${index}-${matchIndex}`}
        className={`rounded px-0.5 ${isCurrentMatch ? 'bg-accent text-white' : 'bg-yellow-300 dark:bg-yellow-600'}`}
      >
        {text.substring(index, index + query.length)}
      </mark>
    )

    lastIndex = index + query.length
    matchIndex++
    index = lowerText.indexOf(lowerQuery, lastIndex)
  }

  if (lastIndex < text.length) {
    parts.push(text.substring(lastIndex))
  }

  return <>{parts}</>
}

function CopyButton({ onClick }: { onClick: () => void }) {
  const [copied, setCopied] = useState(false)

  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation()
      onClick()
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    },
    [onClick]
  )

  return (
    <button
      onClick={handleClick}
      className="opacity-0 group-hover:opacity-100 ml-1 p-0.5 rounded hover:bg-bg-tertiary transition-opacity"
      title="Copy path"
    >
      {copied ? <Check size={12} className="text-green-500" /> : <Copy size={12} className="text-text-muted" />}
    </button>
  )
}

const JSONNode = memo(function JSONNode({
  keyName,
  value,
  path,
  depth,
  searchQuery,
  currentMatchIndex,
  matchPaths,
  onCopyPath,
  expandedPaths,
  onTogglePath,
  isLast,
}: JSONNodeProps) {
  const isExpanded = expandedPaths.has(path)
  const isObject = value !== null && typeof value === 'object' && !Array.isArray(value)
  const isArray = Array.isArray(value)
  const isExpandable = isObject || isArray

  const handleToggle = useCallback(() => {
    onTogglePath(path)
  }, [path, onTogglePath])

  const handleCopyPath = useCallback(() => {
    onCopyPath?.(path)
  }, [path, onCopyPath])

  const comma = isLast ? '' : ','
  const indent = depth * 16

  const renderValue = () => {
    if (value === null) {
      return <span className="text-orange-500 dark:text-orange-400">null</span>
    }
    if (typeof value === 'boolean') {
      return <span className="text-purple-600 dark:text-purple-400">{value.toString()}</span>
    }
    if (typeof value === 'number') {
      return <span className="text-blue-600 dark:text-blue-400">{value}</span>
    }
    if (typeof value === 'string') {
      const displayText = `"${value}"`
      return (
        <span className="text-green-600 dark:text-green-400">
          {highlightText(displayText, searchQuery ?? '', path, matchPaths, currentMatchIndex)}
        </span>
      )
    }
    return null
  }

  const renderExpandable = () => {
    const items = isArray ? value : Object.entries(value as Record<string, JsonValue>)
    const itemCount = isArray ? value.length : Object.keys(value as object).length
    const openBracket = isArray ? '[' : '{'
    const closeBracket = isArray ? ']' : '}'

    if (!isExpanded) {
      const preview = isArray ? `${itemCount} items` : `${itemCount} keys`
      return (
        <span className="text-text-primary">
          {openBracket}
          <span className="text-text-muted italic ml-1">{preview}</span>
          {closeBracket}
          {comma}
        </span>
      )
    }

    return (
      <>
        <span className="text-text-primary">{openBracket}</span>
        <div>
          {isArray
            ? (value as JsonValue[]).map((item, index) => (
                <JSONNode
                  key={index}
                  keyName={String(index)}
                  value={item}
                  path={`${path}[${index}]`}
                  depth={depth + 1}
                  searchQuery={searchQuery}
                  currentMatchIndex={currentMatchIndex}
                  matchPaths={matchPaths}
                  onCopyPath={onCopyPath}
                  expandedPaths={expandedPaths}
                  onTogglePath={onTogglePath}
                  isLast={index === (value as JsonValue[]).length - 1}
                />
              ))
            : (items as [string, JsonValue][]).map(([key, val], index, arr) => (
                <JSONNode
                  key={key}
                  keyName={key}
                  value={val}
                  path={path ? `${path}.${key}` : key}
                  depth={depth + 1}
                  searchQuery={searchQuery}
                  currentMatchIndex={currentMatchIndex}
                  matchPaths={matchPaths}
                  onCopyPath={onCopyPath}
                  expandedPaths={expandedPaths}
                  onTogglePath={onTogglePath}
                  isLast={index === arr.length - 1}
                />
              ))}
        </div>
        <div style={{ paddingLeft: indent }} className="text-text-primary">
          {closeBracket}
          {comma}
        </div>
      </>
    )
  }

  return (
    <div
      className="group leading-6 font-mono text-sm"
      style={{ paddingLeft: indent }}
      data-path={path}
    >
      <span className="inline-flex items-center">
        {isExpandable && (
          <button
            onClick={handleToggle}
            className="w-4 h-4 flex items-center justify-center mr-1 hover:bg-bg-tertiary rounded flex-shrink-0"
          >
            {isExpanded ? (
              <ChevronDown size={14} className="text-text-muted" />
            ) : (
              <ChevronRight size={14} className="text-text-muted" />
            )}
          </button>
        )}
        {!isExpandable && <span className="w-5 flex-shrink-0" />}
        {keyName !== undefined && (
          <>
            <span className="text-red-600 dark:text-red-400">
              {highlightText(`"${keyName}"`, searchQuery ?? '', path, matchPaths, currentMatchIndex)}
            </span>
            <span className="text-text-primary">: </span>
          </>
        )}
        {isExpandable ? (
          renderExpandable()
        ) : (
          <>
            {renderValue()}
            <span className="text-text-primary">{comma}</span>
          </>
        )}
        {onCopyPath && <CopyButton onClick={handleCopyPath} />}
      </span>
    </div>
  )
})

export function JSONTree({
  data,
  searchQuery,
  currentMatchIndex,
  matchPaths,
  onCopyPath,
  expandedPaths,
  onTogglePath,
}: JSONTreeProps) {
  const isObject = data !== null && typeof data === 'object' && !Array.isArray(data)
  const isArray = Array.isArray(data)

  const rootContent = useMemo(() => {
    if (!isObject && !isArray) {
      return (
        <JSONNode
          value={data}
          path=""
          depth={0}
          searchQuery={searchQuery}
          currentMatchIndex={currentMatchIndex}
          matchPaths={matchPaths}
          onCopyPath={onCopyPath}
          expandedPaths={expandedPaths}
          onTogglePath={onTogglePath}
          isLast={true}
        />
      )
    }

    return (
      <JSONNode
        value={data}
        path="$"
        depth={0}
        searchQuery={searchQuery}
        currentMatchIndex={currentMatchIndex}
        matchPaths={matchPaths}
        onCopyPath={onCopyPath}
        expandedPaths={expandedPaths}
        onTogglePath={onTogglePath}
        isLast={true}
      />
    )
  }, [data, searchQuery, currentMatchIndex, matchPaths, onCopyPath, expandedPaths, onTogglePath, isObject, isArray])

  return <div className="py-2">{rootContent}</div>
}
