import { useState, useEffect } from 'react'
import {
  Server,
  RefreshCw,
  Check,
  AlertCircle,
  RotateCcw,
  Sun,
  Moon,
  Monitor,
} from 'lucide-react'
import { useConnection } from '../../hooks/useConnection'
import { useSettingsStore } from '../../stores/settings-store'
import { useTheme } from '../../hooks/useTheme'
import type { Theme } from '../../types'

export function SettingsPanel() {
  const { serverUrl, status, connect, isValidUrl, reconnectAttempts } = useConnection()
  const [urlInput, setUrlInput] = useState(serverUrl)
  const [urlError, setUrlError] = useState<string | null>(null)

  // Settings store
  const connectionTimeout = useSettingsStore((s) => s.connection.connectionTimeout)
  const setConnectionTimeout = useSettingsStore((s) => s.setConnectionTimeout)

  const editor = useSettingsStore((s) => s.editor)
  const setEditorSetting = useSettingsStore((s) => s.setEditorSetting)

  const query = useSettingsStore((s) => s.query)
  const setQuerySetting = useSettingsStore((s) => s.setQuerySetting)

  const resetToDefaults = useSettingsStore((s) => s.resetToDefaults)

  // Theme
  const { theme, setTheme } = useTheme()

  useEffect(() => {
    setUrlInput(serverUrl)
  }, [serverUrl])

  const handleUrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value
    setUrlInput(value)

    if (value && !isValidUrl(value)) {
      setUrlError('Please enter a valid HTTP or HTTPS URL')
    } else {
      setUrlError(null)
    }
  }

  const handleConnect = () => {
    if (!urlInput || !isValidUrl(urlInput)) {
      setUrlError('Please enter a valid HTTP or HTTPS URL')
      return
    }

    connect(urlInput)
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleConnect()
    }
  }

  const getStatusIcon = () => {
    switch (status) {
      case 'connected':
        return <Check size={16} className="text-green-500" />
      case 'connecting':
        return <RefreshCw size={16} className="text-yellow-500 animate-spin" />
      case 'error':
        return <AlertCircle size={16} className="text-red-500" />
      default:
        return <AlertCircle size={16} className="text-neutral-500" />
    }
  }

  const getStatusText = () => {
    switch (status) {
      case 'connected':
        return 'Connected'
      case 'connecting':
        return reconnectAttempts > 0 ? `Reconnecting (attempt ${reconnectAttempts})...` : 'Connecting...'
      case 'error':
        return 'Connection error'
      default:
        return 'Disconnected'
    }
  }

  const themeOptions: { value: Theme; label: string; icon: React.ReactNode }[] = [
    { value: 'light', label: 'Light', icon: <Sun size={16} /> },
    { value: 'dark', label: 'Dark', icon: <Moon size={16} /> },
    { value: 'system', label: 'System', icon: <Monitor size={16} /> },
  ]

  const handleReset = () => {
    resetToDefaults()
    setUrlInput('http://localhost:6010/graphql')
    setUrlError(null)
  }

  return (
    <div className="p-6 max-w-2xl overflow-y-auto h-full">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold text-neutral-200">Settings</h2>
        <button
          onClick={handleReset}
          className="flex items-center gap-2 px-3 py-1.5 text-sm text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800 rounded-md transition-colors"
        >
          <RotateCcw size={14} />
          Reset to defaults
        </button>
      </div>

      {/* Connection Settings */}
      <section className="mb-8">
        <h3 className="text-sm font-medium text-neutral-400 uppercase tracking-wider mb-4">
          Server Connection
        </h3>

        <div className="space-y-4">
          <div>
            <label htmlFor="server-url" className="block text-sm text-neutral-300 mb-2">
              Server URL
            </label>
            <div className="flex gap-2">
              <div className="relative flex-1">
                <Server
                  size={16}
                  className="absolute left-3 top-1/2 -translate-y-1/2 text-neutral-500"
                />
                <input
                  id="server-url"
                  type="text"
                  value={urlInput}
                  onChange={handleUrlChange}
                  onKeyDown={handleKeyDown}
                  placeholder="http://localhost:6010/graphql"
                  className={`w-full pl-10 pr-4 py-2 bg-neutral-900 border rounded-md text-sm text-neutral-200 placeholder-neutral-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent ${
                    urlError ? 'border-red-500' : 'border-neutral-700'
                  }`}
                />
              </div>
              <button
                onClick={handleConnect}
                disabled={status === 'connecting' || !!urlError}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-neutral-700 disabled:text-neutral-500 text-white text-sm font-medium rounded-md transition-colors"
              >
                {status === 'connecting' ? 'Connecting...' : 'Connect'}
              </button>
            </div>
            {urlError && <p className="mt-1 text-xs text-red-400">{urlError}</p>}
          </div>

          <div className="flex items-center gap-2 p-3 bg-neutral-900 rounded-md">
            {getStatusIcon()}
            <span className="text-sm text-neutral-300">{getStatusText()}</span>
          </div>

          <div>
            <label htmlFor="connection-timeout" className="block text-sm text-neutral-300 mb-2">
              Connection Timeout
            </label>
            <div className="flex items-center gap-3">
              <input
                id="connection-timeout"
                type="range"
                min={5000}
                max={120000}
                step={5000}
                value={connectionTimeout}
                onChange={(e) => setConnectionTimeout(Number(e.target.value))}
                className="flex-1 h-2 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
              />
              <span className="text-sm text-neutral-400 w-16 text-right">
                {connectionTimeout / 1000}s
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* Theme Settings */}
      <section className="mb-8">
        <h3 className="text-sm font-medium text-neutral-400 uppercase tracking-wider mb-4">
          Theme
        </h3>

        <div className="flex gap-2">
          {themeOptions.map((option) => (
            <button
              key={option.value}
              onClick={() => setTheme(option.value)}
              className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                theme === option.value
                  ? 'bg-blue-600 text-white'
                  : 'bg-neutral-800 text-neutral-300 hover:bg-neutral-700'
              }`}
            >
              {option.icon}
              {option.label}
            </button>
          ))}
        </div>
      </section>

      {/* Editor Settings */}
      <section className="mb-8">
        <h3 className="text-sm font-medium text-neutral-400 uppercase tracking-wider mb-4">
          Editor
        </h3>

        <div className="space-y-4">
          <div>
            <label htmlFor="font-size" className="block text-sm text-neutral-300 mb-2">
              Font Size
            </label>
            <div className="flex items-center gap-3">
              <input
                id="font-size"
                type="range"
                min={10}
                max={24}
                value={editor.fontSize}
                onChange={(e) => setEditorSetting('fontSize', Number(e.target.value))}
                className="flex-1 h-2 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
              />
              <span className="text-sm text-neutral-400 w-12 text-right">
                {editor.fontSize}px
              </span>
            </div>
          </div>

          <div>
            <label className="block text-sm text-neutral-300 mb-2">Tab Size</label>
            <div className="flex gap-2">
              <button
                onClick={() => setEditorSetting('tabSize', 2)}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  editor.tabSize === 2
                    ? 'bg-blue-600 text-white'
                    : 'bg-neutral-800 text-neutral-300 hover:bg-neutral-700'
                }`}
              >
                2 spaces
              </button>
              <button
                onClick={() => setEditorSetting('tabSize', 4)}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  editor.tabSize === 4
                    ? 'bg-blue-600 text-white'
                    : 'bg-neutral-800 text-neutral-300 hover:bg-neutral-700'
                }`}
              >
                4 spaces
              </button>
            </div>
          </div>

          <ToggleSetting
            label="Line Numbers"
            description="Show line numbers in the editor gutter"
            checked={editor.lineNumbers}
            onChange={(checked) => setEditorSetting('lineNumbers', checked)}
          />

          <ToggleSetting
            label="Word Wrap"
            description="Wrap long lines to fit the editor width"
            checked={editor.wordWrap}
            onChange={(checked) => setEditorSetting('wordWrap', checked)}
          />

          <ToggleSetting
            label="Auto-complete"
            description="Show code completion suggestions as you type"
            checked={editor.autoComplete}
            onChange={(checked) => setEditorSetting('autoComplete', checked)}
          />
        </div>
      </section>

      {/* Query Settings */}
      <section className="mb-8">
        <h3 className="text-sm font-medium text-neutral-400 uppercase tracking-wider mb-4">
          Query
        </h3>

        <div className="space-y-4">
          <div>
            <label htmlFor="default-limit" className="block text-sm text-neutral-300 mb-2">
              Default Query Limit
            </label>
            <input
              id="default-limit"
              type="number"
              min={1}
              max={10000}
              value={query.defaultLimit}
              onChange={(e) => setQuerySetting('defaultLimit', Math.max(1, Number(e.target.value)))}
              className="w-32 px-3 py-2 bg-neutral-900 border border-neutral-700 rounded-md text-sm text-neutral-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <p className="mt-1 text-xs text-neutral-500">
              Maximum number of results to return by default
            </p>
          </div>

          <div>
            <label htmlFor="history-limit" className="block text-sm text-neutral-300 mb-2">
              History Limit
            </label>
            <input
              id="history-limit"
              type="number"
              min={10}
              max={500}
              value={query.historyLimit}
              onChange={(e) => setQuerySetting('historyLimit', Math.max(10, Number(e.target.value)))}
              className="w-32 px-3 py-2 bg-neutral-900 border border-neutral-700 rounded-md text-sm text-neutral-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <p className="mt-1 text-xs text-neutral-500">
              Maximum number of queries to store in history
            </p>
          </div>

          <ToggleSetting
            label="Auto-execute on Load"
            description="Automatically run the last query when opening a tab"
            checked={query.autoExecuteOnLoad}
            onChange={(checked) => setQuerySetting('autoExecuteOnLoad', checked)}
          />
        </div>
      </section>

      {/* Connection Info */}
      <section>
        <h3 className="text-sm font-medium text-neutral-400 uppercase tracking-wider mb-4">
          Connection Info
        </h3>
        <div className="text-sm text-neutral-400 space-y-2">
          <p>
            <span className="text-neutral-500">HTTP URL:</span>{' '}
            <code className="text-neutral-300">{serverUrl}</code>
          </p>
          <p className="text-xs text-neutral-500">
            The WebSocket URL is automatically derived from the HTTP URL for subscriptions.
          </p>
        </div>
      </section>
    </div>
  )
}

interface ToggleSettingProps {
  label: string
  description: string
  checked: boolean
  onChange: (checked: boolean) => void
}

function ToggleSetting({ label, description, checked, onChange }: ToggleSettingProps) {
  return (
    <div className="flex items-center justify-between py-2">
      <div>
        <span className="text-sm text-neutral-300">{label}</span>
        <p className="text-xs text-neutral-500">{description}</p>
      </div>
      <button
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
          checked ? 'bg-blue-600' : 'bg-neutral-700'
        }`}
      >
        <span
          className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
            checked ? 'translate-x-6' : 'translate-x-1'
          }`}
        />
      </button>
    </div>
  )
}
