import { useState, useRef, useEffect } from 'react'
import { Circle, Cpu, HardDrive, Clock, RefreshCw, ChevronDown, Plus, Trash2, Server } from 'lucide-react'
import { useAppStore } from '../../stores/app-store'
import { useSettingsStore } from '../../stores/settings-store'
import { getReconnectAttempts, reconnect } from '../../lib/graphql-client'
import type { ConnectionStatus } from '../../types'

function ServerSelector({ status }: { status: ConnectionStatus }) {
  const [isOpen, setIsOpen] = useState(false)
  const [isAdding, setIsAdding] = useState(false)
  const [newName, setNewName] = useState('')
  const [newUrl, setNewUrl] = useState('')
  const dropdownRef = useRef<HTMLDivElement>(null)

  const servers = useSettingsStore((s) => s.connection.servers)
  const activeServerId = useSettingsStore((s) => s.connection.activeServerId)
  const setActiveServer = useSettingsStore((s) => s.setActiveServer)
  const addServer = useSettingsStore((s) => s.addServer)
  const removeServer = useSettingsStore((s) => s.removeServer)

  const activeServer = servers.find((s) => s.id === activeServerId)
  const reconnectAttempts = getReconnectAttempts()

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setIsOpen(false)
        setIsAdding(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const statusIcon = {
    connected: <Circle size={8} className="fill-current text-green-500" />,
    disconnected: <Circle size={8} className="fill-current text-text-muted" />,
    connecting: <RefreshCw size={10} className="text-yellow-500 animate-spin" />,
    error: <Circle size={8} className="fill-current text-red-500" />,
  }

  const handleSelectServer = (id: string) => {
    setActiveServer(id)
    setIsOpen(false)
    // Trigger reconnect to the new server
    setTimeout(() => reconnect(), 100)
  }

  const handleAddServer = () => {
    if (newName.trim() && newUrl.trim()) {
      addServer({ name: newName.trim(), url: newUrl.trim() })
      setNewName('')
      setNewUrl('')
      setIsAdding(false)
    }
  }

  const handleRemoveServer = (e: React.MouseEvent, id: string) => {
    e.stopPropagation()
    removeServer(id)
  }

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-2 py-1 rounded hover:bg-bg-tertiary transition-colors"
      >
        {statusIcon[status]}
        <Server size={12} className="text-text-muted" />
        <span className="text-xs text-text-secondary max-w-32 truncate">
          {activeServer?.name ?? 'No server'}
        </span>
        {status === 'connecting' && reconnectAttempts > 0 && (
          <span className="text-xs text-yellow-500">({reconnectAttempts})</span>
        )}
        <ChevronDown size={12} className={`text-text-muted transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <div className="absolute bottom-full left-0 mb-1 w-64 bg-bg-secondary border border-border rounded-md shadow-lg z-50">
          <div className="p-2 border-b border-border">
            <span className="text-xs text-text-muted font-medium">Server Connections</span>
          </div>
          <div className="max-h-48 overflow-y-auto">
            {servers.map((server) => (
              <button
                key={server.id}
                onClick={() => handleSelectServer(server.id)}
                className={`flex items-center justify-between w-full px-3 py-2 text-left hover:bg-bg-tertiary transition-colors ${
                  server.id === activeServerId ? 'bg-accent-muted' : ''
                }`}
              >
                <div className="flex-1 min-w-0">
                  <span className="text-xs text-text-primary block truncate">{server.name}</span>
                  <span className="text-[10px] text-text-muted block truncate">{server.url}</span>
                </div>
                {!server.isDefault && (
                  <button
                    onClick={(e) => handleRemoveServer(e, server.id)}
                    className="p-1 hover:bg-bg-tertiary rounded ml-2"
                    title="Remove server"
                  >
                    <Trash2 size={12} className="text-text-muted hover:text-red-500" />
                  </button>
                )}
              </button>
            ))}
          </div>
          <div className="p-2 border-t border-border">
            {isAdding ? (
              <div className="space-y-2">
                <input
                  type="text"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  placeholder="Server name"
                  className="w-full px-2 py-1 text-xs bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent"
                  autoFocus
                />
                <input
                  type="text"
                  value={newUrl}
                  onChange={(e) => setNewUrl(e.target.value)}
                  placeholder="http://localhost:6010/graphql"
                  className="w-full px-2 py-1 text-xs bg-bg-tertiary border border-border rounded focus:outline-none focus:border-accent"
                />
                <div className="flex gap-2">
                  <button
                    onClick={handleAddServer}
                    disabled={!newName.trim() || !newUrl.trim()}
                    className="flex-1 px-2 py-1 text-xs bg-accent hover:bg-accent-hover text-white rounded disabled:opacity-50"
                  >
                    Add
                  </button>
                  <button
                    onClick={() => setIsAdding(false)}
                    className="flex-1 px-2 py-1 text-xs bg-bg-tertiary hover:bg-border rounded"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <button
                onClick={() => setIsAdding(true)}
                className="flex items-center gap-2 w-full px-2 py-1.5 text-xs text-text-secondary hover:bg-bg-tertiary rounded transition-colors"
              >
                <Plus size={12} />
                Add server
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function ResourceIndicator() {
  const stats = useAppStore((s) => s.stats)

  return (
    <button className="flex items-center gap-3 px-3 py-1.5 rounded hover:bg-bg-tertiary transition-colors">
      <div className="flex items-center gap-1.5">
        <Cpu size={12} className="text-text-muted" />
        <span className="text-xs text-text-muted">{stats?.cpuUsage ?? 0}%</span>
      </div>
      <div className="flex items-center gap-1.5">
        <HardDrive size={12} className="text-text-muted" />
        <span className="text-xs text-text-muted">
          {stats?.nodeCount ?? 0} nodes / {stats?.edgeCount ?? 0} edges
        </span>
      </div>
    </button>
  )
}

function QueryStatus() {
  const tabs = useAppStore((s) => s.tabs)
  const activeTabId = useAppStore((s) => s.activeTabId)
  const activeTab = tabs.find((t) => t.id === activeTabId)

  if (!activeTab) return null

  const isExecuting = activeTab.isExecuting
  const executionTime = activeTab.result?.executionTime
  const rowCount = activeTab.result?.rowCount

  return (
    <div className="flex items-center gap-3 px-3 py-1.5">
      {isExecuting && (
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 border-2 border-accent border-t-transparent rounded-full animate-spin" />
          <span className="text-xs text-text-muted">Executing...</span>
        </div>
      )}
      {!isExecuting && executionTime !== undefined && (
        <>
          <div className="flex items-center gap-1.5">
            <Clock size={12} className="text-text-muted" />
            <span className="text-xs text-text-muted">{executionTime.toFixed(0)}ms</span>
          </div>
          {rowCount !== undefined && (
            <span className="text-xs text-text-muted">{rowCount} rows</span>
          )}
        </>
      )}
    </div>
  )
}

export function Tray() {
  const connectionStatus = useAppStore((s) => s.connectionStatus)

  return (
    <div className="flex items-center justify-between h-7 px-1 text-xs">
      {/* Left: Server connection */}
      <div className="flex items-center">
        <ServerSelector status={connectionStatus} />
      </div>

      {/* Center: Resources */}
      <div className="flex items-center">
        <ResourceIndicator />
      </div>

      {/* Right: Query status */}
      <div className="flex items-center">
        <QueryStatus />
      </div>
    </div>
  )
}
