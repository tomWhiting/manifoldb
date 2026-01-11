import { create } from 'zustand'
import type { AppSettings, EditorSettings, QuerySettings, ServerConnection } from '../types'
import {
  loadSettings,
  saveSettings,
  resetSettings as resetStoredSettings,
  DEFAULT_SETTINGS,
} from '../utils/settings-storage'

interface SettingsState extends AppSettings {
  // Actions
  setEditorSetting: <K extends keyof EditorSettings>(
    key: K,
    value: EditorSettings[K]
  ) => void
  setQuerySetting: <K extends keyof QuerySettings>(
    key: K,
    value: QuerySettings[K]
  ) => void
  setConnectionTimeout: (timeout: number) => void
  resetToDefaults: () => void
  // Server management
  addServer: (server: Omit<ServerConnection, 'id'>) => ServerConnection
  updateServer: (id: string, updates: Partial<Omit<ServerConnection, 'id'>>) => void
  removeServer: (id: string) => void
  setActiveServer: (id: string) => void
  getActiveServer: () => ServerConnection | undefined
}

export const useSettingsStore = create<SettingsState>((set, get) => {
  const initialSettings = loadSettings()

  const saveCurrentSettings = (state: AppSettings) => {
    saveSettings({
      connection: state.connection,
      editor: state.editor,
      query: state.query,
    })
  }

  return {
    ...initialSettings,

    setEditorSetting: (key, value) => {
      set((state) => {
        const updated = {
          ...state,
          editor: { ...state.editor, [key]: value },
        }
        saveCurrentSettings(updated)
        return updated
      })
    },

    setQuerySetting: (key, value) => {
      set((state) => {
        const updated = {
          ...state,
          query: { ...state.query, [key]: value },
        }
        saveCurrentSettings(updated)
        return updated
      })
    },

    setConnectionTimeout: (timeout) => {
      set((state) => {
        const updated = {
          ...state,
          connection: { ...state.connection, connectionTimeout: timeout },
        }
        saveCurrentSettings(updated)
        return updated
      })
    },

    resetToDefaults: () => {
      resetStoredSettings()
      set(DEFAULT_SETTINGS)
    },

    addServer: (server) => {
      const id = `server-${Date.now()}`
      const newServer: ServerConnection = { ...server, id }
      set((state) => {
        const updated = {
          ...state,
          connection: {
            ...state.connection,
            servers: [...state.connection.servers, newServer],
          },
        }
        saveCurrentSettings(updated)
        return updated
      })
      return newServer
    },

    updateServer: (id, updates) => {
      set((state) => {
        const updated = {
          ...state,
          connection: {
            ...state.connection,
            servers: state.connection.servers.map((s) =>
              s.id === id ? { ...s, ...updates } : s
            ),
          },
        }
        saveCurrentSettings(updated)
        return updated
      })
    },

    removeServer: (id) => {
      set((state) => {
        const servers = state.connection.servers.filter((s) => s.id !== id)
        // If we removed the active server, switch to the first available
        const activeServerId =
          state.connection.activeServerId === id
            ? servers[0]?.id ?? null
            : state.connection.activeServerId

        const updated = {
          ...state,
          connection: {
            ...state.connection,
            servers,
            activeServerId,
            // Update serverUrl if active changed
            serverUrl: servers.find((s) => s.id === activeServerId)?.url ?? state.connection.serverUrl,
          },
        }
        saveCurrentSettings(updated)
        return updated
      })
    },

    setActiveServer: (id) => {
      set((state) => {
        const server = state.connection.servers.find((s) => s.id === id)
        if (!server) return state

        const updated = {
          ...state,
          connection: {
            ...state.connection,
            activeServerId: id,
            serverUrl: server.url,
          },
        }
        saveCurrentSettings(updated)
        return updated
      })
    },

    getActiveServer: () => {
      const state = get()
      return state.connection.servers.find(
        (s) => s.id === state.connection.activeServerId
      )
    },
  }
})
