import { useRef, useEffect } from 'react'
import { useAppStore } from '../../stores/app-store'

export function GraphCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const tabs = useAppStore((s) => s.tabs)
  const activeTabId = useAppStore((s) => s.activeTabId)
  const activeTab = tabs.find((t) => t.id === activeTabId)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Handle resize
    const resizeObserver = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect
      canvas.width = width * window.devicePixelRatio
      canvas.height = height * window.devicePixelRatio
      canvas.style.width = `${width}px`
      canvas.style.height = `${height}px`
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio)
      render()
    })
    resizeObserver.observe(canvas)

    function render() {
      if (!ctx || !canvas) return
      const width = canvas.width / window.devicePixelRatio
      const height = canvas.height / window.devicePixelRatio

      // Clear
      ctx.fillStyle = '#0a0a0a'
      ctx.fillRect(0, 0, width, height)

      // Draw grid
      ctx.strokeStyle = '#1a1a1a'
      ctx.lineWidth = 1
      const gridSize = 40
      for (let x = 0; x < width; x += gridSize) {
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, height)
        ctx.stroke()
      }
      for (let y = 0; y < height; y += gridSize) {
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(width, y)
        ctx.stroke()
      }

      // Draw nodes if we have results
      const result = activeTab?.result
      if (result?.nodes && result.nodes.length > 0) {
        const nodes = result.nodes
        const centerX = width / 2
        const centerY = height / 2
        const radius = Math.min(width, height) * 0.3

        nodes.forEach((node, i) => {
          const angle = (i / nodes.length) * Math.PI * 2 - Math.PI / 2
          const x = centerX + Math.cos(angle) * radius
          const y = centerY + Math.sin(angle) * radius

          // Node circle
          ctx.beginPath()
          ctx.arc(x, y, 20, 0, Math.PI * 2)
          ctx.fillStyle = '#3b82f6'
          ctx.fill()
          ctx.strokeStyle = '#60a5fa'
          ctx.lineWidth = 2
          ctx.stroke()

          // Node label
          ctx.fillStyle = '#fff'
          ctx.font = '10px system-ui'
          ctx.textAlign = 'center'
          ctx.textBaseline = 'middle'
          const label = node.labels[0] || node.id.slice(0, 4)
          ctx.fillText(label, x, y)
        })

        // Draw edges
        if (result.edges) {
          const nodePositions = new Map<string, { x: number; y: number }>()
          nodes.forEach((node, i) => {
            const angle = (i / nodes.length) * Math.PI * 2 - Math.PI / 2
            nodePositions.set(node.id, {
              x: centerX + Math.cos(angle) * radius,
              y: centerY + Math.sin(angle) * radius,
            })
          })

          result.edges.forEach((edge) => {
            const source = nodePositions.get(edge.sourceId)
            const target = nodePositions.get(edge.targetId)
            if (source && target) {
              ctx.beginPath()
              ctx.moveTo(source.x, source.y)
              ctx.lineTo(target.x, target.y)
              ctx.strokeStyle = '#525252'
              ctx.lineWidth = 1
              ctx.stroke()
            }
          })
        }
      } else {
        // No data message
        ctx.fillStyle = '#525252'
        ctx.font = '14px system-ui'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText('Run a query to visualize results', width / 2, height / 2)
      }
    }

    render()

    return () => resizeObserver.disconnect()
  }, [activeTab?.result])

  return <canvas ref={canvasRef} className="w-full h-full" />
}
