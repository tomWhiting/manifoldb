import type { HistoryEntry } from '../types'

const HISTORY_STORAGE_KEY = 'manifoldb-query-history'
const DEFAULT_MAX_ENTRIES = 100

export function loadHistory(): HistoryEntry[] {
  try {
    const stored = localStorage.getItem(HISTORY_STORAGE_KEY)
    if (!stored) {
      return []
    }
    const parsed = JSON.parse(stored)
    if (!Array.isArray(parsed)) {
      return []
    }
    return parsed as HistoryEntry[]
  } catch {
    console.warn('[History] Failed to load history from localStorage')
    return []
  }
}

export function saveHistory(entries: HistoryEntry[]): void {
  try {
    localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(entries))
  } catch {
    console.warn('[History] Failed to save history to localStorage')
  }
}

export function addHistoryEntry(
  entry: HistoryEntry,
  maxEntries: number = DEFAULT_MAX_ENTRIES
): HistoryEntry[] {
  const entries = loadHistory()
  const updated = [entry, ...entries].slice(0, maxEntries)
  saveHistory(updated)
  return updated
}

export function deleteHistoryEntry(id: string): HistoryEntry[] {
  const entries = loadHistory()
  const updated = entries.filter((e) => e.id !== id)
  saveHistory(updated)
  return updated
}

export function clearHistory(): void {
  try {
    localStorage.removeItem(HISTORY_STORAGE_KEY)
  } catch {
    console.warn('[History] Failed to clear history from localStorage')
  }
}

export function generateHistoryId(): string {
  return `history-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`
}
