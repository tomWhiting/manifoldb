import { useState, useCallback, useMemo } from 'react'

export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue }

export function useJSONTree(data: JsonValue) {
  const [expandedPaths, setExpandedPaths] = useState<Set<string>>(() => {
    const initial = new Set<string>()
    initial.add('$')
    return initial
  })

  const togglePath = useCallback((path: string) => {
    setExpandedPaths((prev) => {
      const next = new Set(prev)
      if (next.has(path)) {
        next.delete(path)
      } else {
        next.add(path)
      }
      return next
    })
  }, [])

  const expandAll = useCallback(() => {
    const paths = new Set<string>()
    const traverse = (value: JsonValue, path: string) => {
      if (value !== null && typeof value === 'object') {
        paths.add(path)
        if (Array.isArray(value)) {
          value.forEach((item, index) => traverse(item, `${path}[${index}]`))
        } else {
          Object.entries(value).forEach(([key, val]) =>
            traverse(val, path ? `${path}.${key}` : key)
          )
        }
      }
    }
    traverse(data, '$')
    setExpandedPaths(paths)
  }, [data])

  const collapseAll = useCallback(() => {
    setExpandedPaths(new Set<string>())
  }, [])

  return {
    expandedPaths,
    togglePath,
    expandAll,
    collapseAll,
  }
}

export function useJSONSearch(data: JsonValue) {
  const [searchQuery, setSearchQuery] = useState('')
  const [currentMatchIndex, setCurrentMatchIndex] = useState(0)

  const matchPaths = useMemo(() => {
    if (!searchQuery || searchQuery.length === 0) {
      return []
    }

    const matches: string[] = []
    const lowerQuery = searchQuery.toLowerCase()

    const traverse = (value: JsonValue, path: string) => {
      if (typeof value === 'string') {
        if (value.toLowerCase().includes(lowerQuery)) {
          matches.push(path)
        }
      } else if (value !== null && typeof value === 'object') {
        if (Array.isArray(value)) {
          value.forEach((item, index) => traverse(item, `${path}[${index}]`))
        } else {
          Object.entries(value).forEach(([key, val]) => {
            if (key.toLowerCase().includes(lowerQuery)) {
              matches.push(path ? `${path}.${key}` : key)
            }
            traverse(val, path ? `${path}.${key}` : key)
          })
        }
      }
    }

    traverse(data, '$')
    return matches
  }, [data, searchQuery])

  const nextMatch = useCallback(() => {
    if (matchPaths.length > 0) {
      setCurrentMatchIndex((prev) => (prev + 1) % matchPaths.length)
    }
  }, [matchPaths.length])

  const prevMatch = useCallback(() => {
    if (matchPaths.length > 0) {
      setCurrentMatchIndex((prev) => (prev - 1 + matchPaths.length) % matchPaths.length)
    }
  }, [matchPaths.length])

  const updateSearch = useCallback((query: string) => {
    setSearchQuery(query)
    setCurrentMatchIndex(0)
  }, [])

  return {
    searchQuery,
    setSearchQuery: updateSearch,
    matchPaths,
    matchCount: matchPaths.length,
    currentMatchIndex,
    nextMatch,
    prevMatch,
  }
}
