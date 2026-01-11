import { create } from 'zustand'
import type {
  ViewMode,
  WorkspaceMode,
  SidebarSection,
  ConnectionStatus,
  QueryTab,
  QueryResult,
  ServerStats,
  Theme,
} from '../types'

interface AppState {
  // Connection
  connectionStatus: ConnectionStatus
  serverUrl: string

  // Sidebar
  sidebarCollapsed: boolean
  activeSidebarSection: SidebarSection

  // Workspace
  workspaceMode: WorkspaceMode
  activeViewMode: ViewMode

  // Queries
  tabs: QueryTab[]
  activeTabId: string | null

  // Server stats
  stats: ServerStats | null

  // Command palette
  commandPaletteOpen: boolean

  // Theme
  theme: Theme

  // Actions
  setConnectionStatus: (status: ConnectionStatus) => void
  setServerUrl: (url: string) => void
  toggleSidebar: () => void
  setSidebarSection: (section: SidebarSection) => void
  setWorkspaceMode: (mode: WorkspaceMode) => void
  setViewMode: (mode: ViewMode) => void
  addTab: (tab: Omit<QueryTab, 'id'>) => string
  removeTab: (id: string) => void
  setActiveTab: (id: string) => void
  updateTabContent: (id: string, content: string) => void
  setTabResult: (id: string, result: QueryResult | undefined) => void
  setTabExecuting: (id: string, isExecuting: boolean) => void
  setTabLanguage: (id: string, language: 'cypher' | 'sql') => void
  setStats: (stats: ServerStats | null) => void
  toggleCommandPalette: () => void
  setTheme: (theme: Theme) => void
  cycleTheme: () => void
}

let tabIdCounter = 0
const generateTabId = () => `tab-${++tabIdCounter}`

const THEME_STORAGE_KEY = 'manifoldb-theme'

const getInitialTheme = (): Theme => {
  if (typeof window === 'undefined') return 'system'
  const stored = localStorage.getItem(THEME_STORAGE_KEY)
  if (stored === 'dark' || stored === 'light' || stored === 'system') {
    return stored
  }
  return 'system'
}

export const useAppStore = create<AppState>((set) => ({
  // Initial state
  connectionStatus: 'disconnected',
  serverUrl: 'http://localhost:6010/graphql',

  sidebarCollapsed: false,
  activeSidebarSection: 'query',

  workspaceMode: 'query',
  activeViewMode: 'graph',

  tabs: [
    {
      id: 'tab-1',
      title: 'Query 1',
      content: '// Write your Cypher query here\nMATCH (n) RETURN n LIMIT 10',
      language: 'cypher',
    },
  ],
  activeTabId: 'tab-1',

  stats: null,
  commandPaletteOpen: false,
  theme: getInitialTheme(),

  // Actions
  setConnectionStatus: (status) => set({ connectionStatus: status }),
  setServerUrl: (url) => set({ serverUrl: url }),

  toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
  setSidebarSection: (section) => set({ activeSidebarSection: section }),

  setWorkspaceMode: (mode) => set({ workspaceMode: mode }),
  setViewMode: (mode) => set({ activeViewMode: mode }),

  addTab: (tab) => {
    const id = generateTabId()
    set((state) => ({
      tabs: [...state.tabs, { ...tab, id }],
      activeTabId: id,
    }))
    return id
  },

  removeTab: (id) =>
    set((state) => {
      const newTabs = state.tabs.filter((t) => t.id !== id)
      const newActiveId =
        state.activeTabId === id
          ? newTabs[newTabs.length - 1]?.id ?? null
          : state.activeTabId
      return { tabs: newTabs, activeTabId: newActiveId }
    }),

  setActiveTab: (id) => set({ activeTabId: id }),

  updateTabContent: (id, content) =>
    set((state) => ({
      tabs: state.tabs.map((t) => (t.id === id ? { ...t, content } : t)),
    })),

  setTabResult: (id, result) =>
    set((state) => ({
      tabs: state.tabs.map((t) => (t.id === id ? { ...t, result } : t)),
    })),

  setTabExecuting: (id, isExecuting) =>
    set((state) => ({
      tabs: state.tabs.map((t) => (t.id === id ? { ...t, isExecuting } : t)),
    })),

  setTabLanguage: (id, language) =>
    set((state) => ({
      tabs: state.tabs.map((t) => (t.id === id ? { ...t, language } : t)),
    })),

  setStats: (stats) => set({ stats }),

  toggleCommandPalette: () => set((state) => ({ commandPaletteOpen: !state.commandPaletteOpen })),

  setTheme: (theme) => {
    localStorage.setItem(THEME_STORAGE_KEY, theme)
    set({ theme })
  },

  cycleTheme: () =>
    set((state) => {
      const nextTheme: Theme = state.theme === 'system' ? 'dark' : state.theme === 'dark' ? 'light' : 'system'
      localStorage.setItem(THEME_STORAGE_KEY, nextTheme)
      return { theme: nextTheme }
    }),
}))
