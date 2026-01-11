import { create } from 'zustand'
import type { AppSettings, EditorSettings, QuerySettings } from '../types'
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
}

export const useSettingsStore = create<SettingsState>((set) => {
  const initialSettings = loadSettings()

  return {
    ...initialSettings,

    setEditorSetting: (key, value) => {
      set((state) => {
        const updated = {
          ...state,
          editor: { ...state.editor, [key]: value },
        }
        saveSettings({
          connection: updated.connection,
          editor: updated.editor,
          query: updated.query,
        })
        return updated
      })
    },

    setQuerySetting: (key, value) => {
      set((state) => {
        const updated = {
          ...state,
          query: { ...state.query, [key]: value },
        }
        saveSettings({
          connection: updated.connection,
          editor: updated.editor,
          query: updated.query,
        })
        return updated
      })
    },

    setConnectionTimeout: (timeout) => {
      set((state) => {
        const updated = {
          ...state,
          connection: { ...state.connection, connectionTimeout: timeout },
        }
        saveSettings({
          connection: updated.connection,
          editor: updated.editor,
          query: updated.query,
        })
        return updated
      })
    },

    resetToDefaults: () => {
      resetStoredSettings()
      set(DEFAULT_SETTINGS)
    },
  }
})
