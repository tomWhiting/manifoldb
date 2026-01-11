import { useEffect, useCallback } from 'react'
import { X } from 'lucide-react'
import { useInspectorStore, type NodeInspector as NodeInspectorType, type EdgeInspector as EdgeInspectorType } from '../../stores/inspector-store'
import { NodeInspector } from './NodeInspector'
import { EdgeInspector } from './EdgeInspector'

interface InspectorContainerProps {
  onExpandNeighbors?: (nodeId: string) => void
  onDeleteNode?: (nodeId: string) => void
  onNavigateToNode?: (nodeId: string) => void
}

export function InspectorContainer({
  onExpandNeighbors,
  onDeleteNode,
  onNavigateToNode,
}: InspectorContainerProps) {
  const inspectors = useInspectorStore((s) => s.inspectors)
  const closeAllInspectors = useInspectorStore((s) => s.closeAllInspectors)

  // Close all inspectors on Escape key
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape' && inspectors.length > 0) {
        closeAllInspectors()
      }
    },
    [inspectors.length, closeAllInspectors]
  )

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

  if (inspectors.length === 0) return null

  return (
    <>
      {/* Close all button */}
      {inspectors.length > 1 && (
        <button
          onClick={closeAllInspectors}
          className="fixed top-2 left-1/2 -translate-x-1/2 z-[1001] flex items-center gap-2 px-3 py-1.5 bg-bg-secondary/90 backdrop-blur-sm border border-border rounded-full text-xs text-text-muted hover:text-text-primary hover:bg-bg-tertiary transition-colors shadow-lg"
          title="Close all inspectors (Escape)"
        >
          <X size={12} />
          Close all ({inspectors.length})
        </button>
      )}

      {/* Render all inspectors */}
      {inspectors.map((inspector) => {
        if (inspector.type === 'node') {
          return (
            <NodeInspector
              key={inspector.id}
              inspector={inspector as NodeInspectorType}
              onExpandNeighbors={onExpandNeighbors}
              onDeleteNode={onDeleteNode}
            />
          )
        } else {
          return (
            <EdgeInspector
              key={inspector.id}
              inspector={inspector as EdgeInspectorType}
              onNavigateToNode={onNavigateToNode}
            />
          )
        }
      })}
    </>
  )
}
