import { useEffect, useRef } from 'react'
import { Info, Expand, Trash2, ArrowRight } from 'lucide-react'
import type { LayoutNode, LayoutEdge } from '../../utils/graph-layout'

export type ContextMenuTarget =
  | { type: 'node'; node: LayoutNode }
  | { type: 'edge'; edge: LayoutEdge; sourceNode: LayoutNode; targetNode: LayoutNode }
  | null

interface ContextMenuProps {
  target: ContextMenuTarget
  position: { x: number; y: number }
  onClose: () => void
  onInspect: () => void
  onExpandNeighbors?: () => void
  onDelete?: () => void
  onNavigateToSource?: () => void
  onNavigateToTarget?: () => void
}

interface MenuItemProps {
  icon: React.ReactNode
  label: string
  onClick: () => void
  variant?: 'default' | 'danger'
}

function MenuItem({ icon, label, onClick, variant = 'default' }: MenuItemProps) {
  return (
    <button
      onClick={onClick}
      className={`w-full flex items-center gap-2 px-3 py-1.5 text-sm text-left transition-colors ${
        variant === 'danger'
          ? 'hover:bg-red-500/10 text-red-400'
          : 'hover:bg-bg-tertiary text-text-secondary hover:text-text-primary'
      }`}
    >
      <span className="w-4 h-4 shrink-0">{icon}</span>
      <span>{label}</span>
    </button>
  )
}

function MenuDivider() {
  return <div className="h-px bg-border my-1" />
}

export function ContextMenu({
  target,
  position,
  onClose,
  onInspect,
  onExpandNeighbors,
  onDelete,
  onNavigateToSource,
  onNavigateToTarget,
}: ContextMenuProps) {
  const menuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        onClose()
      }
    }

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose()
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    document.addEventListener('keydown', handleKeyDown)

    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [onClose])

  // Adjust position to keep menu on screen
  useEffect(() => {
    if (!menuRef.current) return

    const menu = menuRef.current
    const rect = menu.getBoundingClientRect()
    const viewportWidth = window.innerWidth
    const viewportHeight = window.innerHeight

    let adjustedX = position.x
    let adjustedY = position.y

    if (rect.right > viewportWidth) {
      adjustedX = viewportWidth - rect.width - 8
    }

    if (rect.bottom > viewportHeight) {
      adjustedY = viewportHeight - rect.height - 8
    }

    if (adjustedX !== position.x || adjustedY !== position.y) {
      menu.style.left = `${adjustedX}px`
      menu.style.top = `${adjustedY}px`
    }
  }, [position])

  if (!target) return null

  return (
    <div
      ref={menuRef}
      className="fixed bg-bg-secondary border border-border rounded-lg shadow-xl py-1 min-w-[160px] z-[1000]"
      style={{ left: position.x, top: position.y }}
    >
      {target.type === 'node' && (
        <>
          <MenuItem
            icon={<Info size={14} />}
            label="Inspect Node"
            onClick={() => {
              onInspect()
              onClose()
            }}
          />
          {onExpandNeighbors && (
            <MenuItem
              icon={<Expand size={14} />}
              label="Expand Neighbors"
              onClick={() => {
                onExpandNeighbors()
                onClose()
              }}
            />
          )}
          {onDelete && (
            <>
              <MenuDivider />
              <MenuItem
                icon={<Trash2 size={14} />}
                label="Delete Node"
                variant="danger"
                onClick={() => {
                  if (confirm(`Delete node ${target.node.id}?`)) {
                    onDelete()
                    onClose()
                  }
                }}
              />
            </>
          )}
        </>
      )}

      {target.type === 'edge' && (
        <>
          <MenuItem
            icon={<Info size={14} />}
            label="Inspect Edge"
            onClick={() => {
              onInspect()
              onClose()
            }}
          />
          {onNavigateToSource && (
            <MenuItem
              icon={<ArrowRight size={14} className="rotate-180" />}
              label={`Go to ${target.sourceNode.label}`}
              onClick={() => {
                onNavigateToSource()
                onClose()
              }}
            />
          )}
          {onNavigateToTarget && (
            <MenuItem
              icon={<ArrowRight size={14} />}
              label={`Go to ${target.targetNode.label}`}
              onClick={() => {
                onNavigateToTarget()
                onClose()
              }}
            />
          )}
        </>
      )}
    </div>
  )
}
