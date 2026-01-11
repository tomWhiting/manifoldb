import { useCallback, useRef, useState, useEffect } from 'react'
import type { LayoutNode, PaneState, SplitDirection } from '../../types'

const MIN_PANE_SIZE = 15 // Minimum percentage

interface SplitPaneLayoutProps {
  node: LayoutNode
  panes: Record<string, PaneState>
  activePaneId: string
  onSizesChange: (paneId: string, sizes: [number, number]) => void
  renderPane: (pane: PaneState, isActive: boolean) => React.ReactNode
}

export function SplitPaneLayout({
  node,
  panes,
  activePaneId,
  onSizesChange,
  renderPane,
}: SplitPaneLayoutProps) {
  if (node.type === 'leaf') {
    const pane = panes[node.paneId]
    if (!pane) return null
    return <>{renderPane(pane, pane.id === activePaneId)}</>
  }

  return (
    <SplitContainer
      direction={node.direction}
      sizes={node.sizes}
      onSizesChange={(sizes) => {
        // Get the first pane ID from the left child for identification
        const firstPaneId = getFirstPaneId(node.children[0])
        onSizesChange(firstPaneId, sizes)
      }}
    >
      <SplitPaneLayout
        node={node.children[0]}
        panes={panes}
        activePaneId={activePaneId}
        onSizesChange={onSizesChange}
        renderPane={renderPane}
      />
      <SplitPaneLayout
        node={node.children[1]}
        panes={panes}
        activePaneId={activePaneId}
        onSizesChange={onSizesChange}
        renderPane={renderPane}
      />
    </SplitContainer>
  )
}

function getFirstPaneId(node: LayoutNode): string {
  if (node.type === 'leaf') return node.paneId
  return getFirstPaneId(node.children[0])
}

interface SplitContainerProps {
  direction: SplitDirection
  sizes: [number, number]
  onSizesChange: (sizes: [number, number]) => void
  children: [React.ReactNode, React.ReactNode]
}

function SplitContainer({ direction, sizes, onSizesChange, children }: SplitContainerProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [localSizes, setLocalSizes] = useState(sizes)

  useEffect(() => {
    setLocalSizes(sizes)
  }, [sizes])

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault()
      setIsDragging(true)

      const handleMouseMove = (moveEvent: MouseEvent) => {
        if (!containerRef.current) return

        const rect = containerRef.current.getBoundingClientRect()
        const currentPos = direction === 'horizontal' ? moveEvent.clientX : moveEvent.clientY
        const containerSize = direction === 'horizontal' ? rect.width : rect.height
        const containerStart = direction === 'horizontal' ? rect.left : rect.top

        const relativePos = currentPos - containerStart
        let newFirstSize = (relativePos / containerSize) * 100

        // Enforce minimum sizes
        newFirstSize = Math.max(MIN_PANE_SIZE, Math.min(100 - MIN_PANE_SIZE, newFirstSize))
        const newSecondSize = 100 - newFirstSize

        setLocalSizes([newFirstSize, newSecondSize])
      }

      const handleMouseUp = () => {
        setIsDragging(false)
        document.removeEventListener('mousemove', handleMouseMove)
        document.removeEventListener('mouseup', handleMouseUp)

        // Commit the final sizes
        setLocalSizes((current) => {
          onSizesChange(current)
          return current
        })
      }

      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
    },
    [direction, onSizesChange]
  )

  const isHorizontal = direction === 'horizontal'

  return (
    <div
      ref={containerRef}
      className={`flex h-full w-full ${isHorizontal ? 'flex-row' : 'flex-col'}`}
    >
      <div
        className="overflow-hidden"
        style={{
          [isHorizontal ? 'width' : 'height']: `${localSizes[0]}%`,
          flexShrink: 0,
        }}
      >
        {children[0]}
      </div>

      <div
        onMouseDown={handleMouseDown}
        className={`
          flex-shrink-0 bg-border hover:bg-accent transition-colors
          ${isHorizontal ? 'w-1 cursor-col-resize' : 'h-1 cursor-row-resize'}
          ${isDragging ? 'bg-accent' : ''}
        `}
        style={{ touchAction: 'none' }}
      />

      <div
        className="overflow-hidden"
        style={{
          [isHorizontal ? 'width' : 'height']: `${localSizes[1]}%`,
          flexShrink: 0,
        }}
      >
        {children[1]}
      </div>

      {isDragging && (
        <div
          className="fixed inset-0 z-50"
          style={{ cursor: isHorizontal ? 'col-resize' : 'row-resize' }}
        />
      )}
    </div>
  )
}
