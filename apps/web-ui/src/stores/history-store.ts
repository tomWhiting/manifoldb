import { create } from 'zustand'
import type { HistoryEntry } from '../types'
import {
  loadHistory,
  saveHistory,
  clearHistory as clearStoredHistory,
  generateHistoryId,
} from '../lib/history-storage'
import { useSettingsStore } from './settings-store'

interface HistoryState {
  entries: HistoryEntry[]
  searchQuery: string

  // Actions
  addEntry: (entry: Omit<HistoryEntry, 'id'>) => void
  deleteEntry: (id: string) => void
  clearAll: () => void
  setSearchQuery: (query: string) => void
}

export const useHistoryStore = create<HistoryState>((set, get) => ({
  entries: loadHistory(),
  searchQuery: '',

  addEntry: (entry) => {
    const historyLimit = useSettingsStore.getState().query.historyLimit
    const newEntry: HistoryEntry = {
      ...entry,
      id: generateHistoryId(),
    }
    const entries = [newEntry, ...get().entries].slice(0, historyLimit)
    saveHistory(entries)
    set({ entries })
  },

  deleteEntry: (id) => {
    const entries = get().entries.filter((e) => e.id !== id)
    saveHistory(entries)
    set({ entries })
  },

  clearAll: () => {
    clearStoredHistory()
    set({ entries: [] })
  },

  setSearchQuery: (query) => {
    set({ searchQuery: query })
  },
}))

// Selector for filtered entries
export function useFilteredHistory(): HistoryEntry[] {
  const entries = useHistoryStore((s) => s.entries)
  const searchQuery = useHistoryStore((s) => s.searchQuery)

  if (!searchQuery.trim()) {
    return entries
  }

  const lowerQuery = searchQuery.toLowerCase()
  return entries.filter((entry) =>
    entry.query.toLowerCase().includes(lowerQuery)
  )
}

// Group entries by date
type DateGroup = 'today' | 'yesterday' | 'thisWeek' | 'older'

export interface GroupedHistoryEntry extends HistoryEntry {
  group: DateGroup
}

export function groupEntriesByDate(entries: HistoryEntry[]): GroupedHistoryEntry[] {
  const now = new Date()
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
  const yesterdayStart = todayStart - 24 * 60 * 60 * 1000
  const weekStart = todayStart - 7 * 24 * 60 * 60 * 1000

  return entries.map((entry) => {
    let group: DateGroup
    if (entry.timestamp >= todayStart) {
      group = 'today'
    } else if (entry.timestamp >= yesterdayStart) {
      group = 'yesterday'
    } else if (entry.timestamp >= weekStart) {
      group = 'thisWeek'
    } else {
      group = 'older'
    }
    return { ...entry, group }
  })
}

export const DATE_GROUP_LABELS: Record<DateGroup, string> = {
  today: 'Today',
  yesterday: 'Yesterday',
  thisWeek: 'This Week',
  older: 'Older',
}
