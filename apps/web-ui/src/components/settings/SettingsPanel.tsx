import { useState, useEffect } from 'react'
import { Server, RefreshCw, Check, AlertCircle } from 'lucide-react'
import { useConnection } from '../../hooks/useConnection'

export function SettingsPanel() {
  const { serverUrl, status, connect, isValidUrl, reconnectAttempts } = useConnection()
  const [urlInput, setUrlInput] = useState(serverUrl)
  const [urlError, setUrlError] = useState<string | null>(null)

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

  return (
    <div className="p-6 max-w-2xl">
      <h2 className="text-lg font-semibold text-neutral-200 mb-6">Settings</h2>

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
        </div>
      </section>

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
