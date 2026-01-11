import type { AppSettings, EditorSettings, QuerySettings, Theme } from '../types'

const SETTINGS_STORAGE_KEY = 'manifoldb-settings'

export const DEFAULT_SETTINGS: AppSettings = {
  connection: {
    serverUrl: 'http://localhost:6010/graphql',
    connectionTimeout: 30000,
  },
  editor: {
    fontSize: 14,
    tabSize: 2,
    lineNumbers: true,
    wordWrap: false,
    autoComplete: true,
  },
  theme: 'system',
  query: {
    defaultLimit: 100,
    autoExecuteOnLoad: false,
    historyLimit: 50,
  },
}

export function loadSettings(): AppSettings {
  try {
    const stored = localStorage.getItem(SETTINGS_STORAGE_KEY)
    if (!stored) {
      return DEFAULT_SETTINGS
    }
    const parsed = JSON.parse(stored) as Partial<AppSettings>
    // Merge with defaults to ensure all fields exist
    return {
      connection: { ...DEFAULT_SETTINGS.connection, ...parsed.connection },
      editor: { ...DEFAULT_SETTINGS.editor, ...parsed.editor },
      theme: parsed.theme ?? DEFAULT_SETTINGS.theme,
      query: { ...DEFAULT_SETTINGS.query, ...parsed.query },
    }
  } catch {
    console.warn('[Settings] Failed to load settings from localStorage')
    return DEFAULT_SETTINGS
  }
}

export function saveSettings(settings: AppSettings): void {
  try {
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings))
  } catch {
    console.warn('[Settings] Failed to save settings to localStorage')
  }
}

export function resetSettings(): AppSettings {
  try {
    localStorage.removeItem(SETTINGS_STORAGE_KEY)
  } catch {
    console.warn('[Settings] Failed to clear settings from localStorage')
  }
  return DEFAULT_SETTINGS
}

// Partial update helpers
export function updateEditorSettings(
  current: AppSettings,
  updates: Partial<EditorSettings>
): AppSettings {
  const updated = {
    ...current,
    editor: { ...current.editor, ...updates },
  }
  saveSettings(updated)
  return updated
}

export function updateQuerySettings(
  current: AppSettings,
  updates: Partial<QuerySettings>
): AppSettings {
  const updated = {
    ...current,
    query: { ...current.query, ...updates },
  }
  saveSettings(updated)
  return updated
}

export function updateTheme(current: AppSettings, theme: Theme): AppSettings {
  const updated = { ...current, theme }
  saveSettings(updated)
  return updated
}

export function updateConnectionTimeout(
  current: AppSettings,
  timeout: number
): AppSettings {
  const updated = {
    ...current,
    connection: { ...current.connection, connectionTimeout: timeout },
  }
  saveSettings(updated)
  return updated
}
