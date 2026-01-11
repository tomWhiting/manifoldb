import { useEffect, useCallback, useState, useRef } from 'react'
import { toast } from 'sonner'
import { useAppStore } from '../stores/app-store'
import {
  subscribeToConnectionStatus,
  subscribeToConnectionErrors,
  getStoredServerUrl,
  validateServerUrl,
  reconnect,
  getReconnectAttempts,
} from '../lib/graphql-client'
import { logConnection } from '../stores/logs-store'
import type { ConnectionStatus, ConnectionError } from '../types'

interface UseConnectionReturn {
  status: ConnectionStatus
  serverUrl: string
  reconnectAttempts: number
  connect: (url: string) => boolean
  isValidUrl: (url: string) => boolean
}

export function useConnection(): UseConnectionReturn {
  const setConnectionStatus = useAppStore((s) => s.setConnectionStatus)
  const setServerUrl = useAppStore((s) => s.setServerUrl)
  const serverUrl = useAppStore((s) => s.serverUrl)
  const status = useAppStore((s) => s.connectionStatus)

  const [reconnectAttempts, setReconnectAttempts] = useState(0)
  const prevStatusRef = useRef<ConnectionStatus | null>(null)

  useEffect(() => {
    const storedUrl = getStoredServerUrl()
    setServerUrl(storedUrl)
  }, [setServerUrl])

  useEffect(() => {
    const unsubscribeStatus = subscribeToConnectionStatus((newStatus: ConnectionStatus) => {
      setConnectionStatus(newStatus)
      setReconnectAttempts(getReconnectAttempts())

      // Log connection status changes (avoid duplicate logs)
      if (prevStatusRef.current !== newStatus) {
        logConnection({ status: newStatus, serverUrl })
        prevStatusRef.current = newStatus
      }

      if (newStatus === 'connected') {
        toast.success('Connected to server')
      }
    })

    const unsubscribeErrors = subscribeToConnectionErrors((error: ConnectionError) => {
      console.error('[Connection Error]', error)

      let message = error.message
      if (error.code === 'INVALID_URL') {
        message = `Invalid server URL: ${error.message}`
      } else if (error.code === 'WS_ERROR') {
        message = `Connection error: ${error.message}`
      }

      // Log connection error
      logConnection({ status: 'error', serverUrl, error: message })

      toast.error(message, {
        description: reconnectAttempts > 0 ? `Reconnecting (attempt ${reconnectAttempts})...` : undefined,
      })
    })

    return () => {
      unsubscribeStatus()
      unsubscribeErrors()
    }
  }, [setConnectionStatus, reconnectAttempts, serverUrl])

  const connect = useCallback(
    (url: string): boolean => {
      if (!validateServerUrl(url)) {
        toast.error('Invalid server URL', {
          description: 'Please enter a valid HTTP or HTTPS URL',
        })
        return false
      }

      try {
        reconnect(url)
        setServerUrl(url)
        return true
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to connect'
        toast.error('Connection failed', { description: message })
        return false
      }
    },
    [setServerUrl]
  )

  const isValidUrl = useCallback((url: string): boolean => {
    return validateServerUrl(url)
  }, [])

  return {
    status,
    serverUrl,
    reconnectAttempts,
    connect,
    isValidUrl,
  }
}
