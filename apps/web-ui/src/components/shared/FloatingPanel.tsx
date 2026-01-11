import { useRef, useCallback, type ReactNode, type MouseEvent } from 'react'
import { X, Minus, Maximize2 } from 'lucide-react'
import type { InspectorPosition, InspectorSize } from '../../stores/inspector-store'
import { MIN_SIZE, MAX_SIZE } from '../../stores/inspector-store'

interface FloatingPanelProps {
  title: string
  titleIcon?: ReactNode
  position: InspectorPosition
  size: InspectorSize
  minimized: boolean
  zIndex: number
  onClose: () => void
  onPositionChange: (position: InspectorPosition) => void
  onSizeChange: (size: InspectorSize) => void
  onMinimizeToggle: () => void
  onFocus: () => void
  children: ReactNode
}

type ResizeDirection = 'n' | 's' | 'e' | 'w' | 'ne' | 'nw' | 'se' | 'sw'

export function FloatingPanel({
  title,
  titleIcon,
  position,
  size,
  minimized,
  zIndex,
  onClose,
  onPositionChange,
  onSizeChange,
  onMinimizeToggle,
  onFocus,
  children,
}: FloatingPanelProps) {
  const panelRef = useRef<HTMLDivElement>(null)
  const isDraggingRef = useRef(false)
  const isResizingRef = useRef<ResizeDirection | null>(null)
  const startPosRef = useRef({ x: 0, y: 0 })
  const startSizeRef = useRef({ width: 0, height: 0 })
  const startPanelPosRef = useRef({ x: 0, y: 0 })

  const handleMouseDown = useCallback(() => {
    onFocus()
  }, [onFocus])

  const handleTitleBarMouseDown = useCallback(
    (e: MouseEvent<HTMLDivElement>) => {
      // Don't drag if clicking on buttons
      if ((e.target as HTMLElement).closest('button')) return

      e.preventDefault()
      isDraggingRef.current = true
      startPosRef.current = { x: e.clientX, y: e.clientY }
      startPanelPosRef.current = { x: position.x, y: position.y }

      const handleMouseMove = (e: globalThis.MouseEvent) => {
        if (!isDraggingRef.current) return

        const dx = e.clientX - startPosRef.current.x
        const dy = e.clientY - startPosRef.current.y

        const newX = Math.max(0, startPanelPosRef.current.x + dx)
        const newY = Math.max(0, startPanelPosRef.current.y + dy)

        onPositionChange({ x: newX, y: newY })
      }

      const handleMouseUp = () => {
        isDraggingRef.current = false
        document.removeEventListener('mousemove', handleMouseMove)
        document.removeEventListener('mouseup', handleMouseUp)
      }

      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
    },
    [position, onPositionChange]
  )

  const handleResizeStart = useCallback(
    (direction: ResizeDirection) => (e: MouseEvent<HTMLDivElement>) => {
      e.preventDefault()
      e.stopPropagation()

      isResizingRef.current = direction
      startPosRef.current = { x: e.clientX, y: e.clientY }
      startSizeRef.current = { width: size.width, height: size.height }
      startPanelPosRef.current = { x: position.x, y: position.y }

      const handleMouseMove = (e: globalThis.MouseEvent) => {
        if (!isResizingRef.current) return

        const dx = e.clientX - startPosRef.current.x
        const dy = e.clientY - startPosRef.current.y

        let newWidth = startSizeRef.current.width
        let newHeight = startSizeRef.current.height
        let newX = startPanelPosRef.current.x
        let newY = startPanelPosRef.current.y

        // Handle width changes
        if (direction.includes('e')) {
          newWidth = Math.max(MIN_SIZE.width, Math.min(MAX_SIZE.width, startSizeRef.current.width + dx))
        }
        if (direction.includes('w')) {
          newWidth = Math.max(MIN_SIZE.width, Math.min(MAX_SIZE.width, startSizeRef.current.width - dx))
          if (newWidth >= MIN_SIZE.width && newWidth <= MAX_SIZE.width) {
            newX = startPanelPosRef.current.x + (startSizeRef.current.width - newWidth)
          }
        }

        // Handle height changes
        if (direction.includes('s')) {
          newHeight = Math.max(MIN_SIZE.height, Math.min(MAX_SIZE.height, startSizeRef.current.height + dy))
        }
        if (direction.includes('n')) {
          newHeight = Math.max(MIN_SIZE.height, Math.min(MAX_SIZE.height, startSizeRef.current.height - dy))
          if (newHeight >= MIN_SIZE.height && newHeight <= MAX_SIZE.height) {
            newY = startPanelPosRef.current.y + (startSizeRef.current.height - newHeight)
          }
        }

        onSizeChange({ width: newWidth, height: newHeight })
        onPositionChange({ x: newX, y: newY })
      }

      const handleMouseUp = () => {
        isResizingRef.current = null
        document.removeEventListener('mousemove', handleMouseMove)
        document.removeEventListener('mouseup', handleMouseUp)
      }

      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
    },
    [size, position, onSizeChange, onPositionChange]
  )

  const resizeHandleClasses = 'absolute bg-transparent hover:bg-accent/30 transition-colors'

  return (
    <div
      ref={panelRef}
      className="fixed bg-bg-secondary border border-border rounded-lg shadow-xl overflow-hidden flex flex-col"
      style={{
        left: position.x,
        top: position.y,
        width: size.width,
        height: minimized ? 'auto' : size.height,
        zIndex,
        minWidth: MIN_SIZE.width,
        minHeight: minimized ? 'auto' : MIN_SIZE.height,
      }}
      onMouseDown={handleMouseDown}
    >
      {/* Title bar */}
      <div
        className="flex items-center justify-between px-3 py-2 bg-bg-tertiary border-b border-border cursor-move select-none shrink-0"
        onMouseDown={handleTitleBarMouseDown}
      >
        <div className="flex items-center gap-2 min-w-0">
          {titleIcon && <span className="text-text-muted shrink-0">{titleIcon}</span>}
          <span className="text-sm font-medium text-text-primary truncate">{title}</span>
        </div>
        <div className="flex items-center gap-1 shrink-0">
          <button
            onClick={onMinimizeToggle}
            className="p-1 rounded hover:bg-bg-secondary transition-colors"
            title={minimized ? 'Maximize' : 'Minimize'}
          >
            {minimized ? (
              <Maximize2 size={14} className="text-text-muted" />
            ) : (
              <Minus size={14} className="text-text-muted" />
            )}
          </button>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-red-500/20 hover:text-red-400 transition-colors"
            title="Close"
          >
            <X size={14} className="text-text-muted" />
          </button>
        </div>
      </div>

      {/* Content */}
      {!minimized && (
        <div className="flex-1 overflow-auto min-h-0">{children}</div>
      )}

      {/* Resize handles (only when not minimized) */}
      {!minimized && (
        <>
          {/* Edge handles */}
          <div
            className={`${resizeHandleClasses} top-0 left-2 right-2 h-1 cursor-n-resize`}
            onMouseDown={handleResizeStart('n')}
          />
          <div
            className={`${resizeHandleClasses} bottom-0 left-2 right-2 h-1 cursor-s-resize`}
            onMouseDown={handleResizeStart('s')}
          />
          <div
            className={`${resizeHandleClasses} left-0 top-2 bottom-2 w-1 cursor-w-resize`}
            onMouseDown={handleResizeStart('w')}
          />
          <div
            className={`${resizeHandleClasses} right-0 top-2 bottom-2 w-1 cursor-e-resize`}
            onMouseDown={handleResizeStart('e')}
          />

          {/* Corner handles */}
          <div
            className={`${resizeHandleClasses} top-0 left-0 w-2 h-2 cursor-nw-resize`}
            onMouseDown={handleResizeStart('nw')}
          />
          <div
            className={`${resizeHandleClasses} top-0 right-0 w-2 h-2 cursor-ne-resize`}
            onMouseDown={handleResizeStart('ne')}
          />
          <div
            className={`${resizeHandleClasses} bottom-0 left-0 w-2 h-2 cursor-sw-resize`}
            onMouseDown={handleResizeStart('sw')}
          />
          <div
            className={`${resizeHandleClasses} bottom-0 right-0 w-2 h-2 cursor-se-resize`}
            onMouseDown={handleResizeStart('se')}
          />
        </>
      )}
    </div>
  )
}
