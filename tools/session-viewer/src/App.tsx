import { useEffect, useState, useCallback, useRef } from 'react'
import { Cosmograph, CosmographProvider, prepareCosmographData, CosmographPointColorStrategy, CosmographPointSizeStrategy } from '@cosmograph/react'
import type { CosmographConfig, CosmographRef } from '@cosmograph/react'
import './App.css'

// Types matching the API
interface Node {
  id: number
  labels: string[]
  properties: Record<string, unknown>
}

interface Edge {
  id: number
  source: number
  target: number
  type: string
  properties: Record<string, unknown>
}

interface GraphData {
  nodes: Node[]
  edges: Edge[]
}

interface Stats {
  node_count: number
  edge_count: number
  labels: Record<string, number>
  edge_types: Record<string, number>
}

// Color mapping for node types
const NODE_COLORS: Record<string, string> = {
  User: '#4CAF50',      // Green
  Assistant: '#2196F3', // Blue
  QueueOperation: '#9E9E9E', // Gray
  Bash: '#FF5722',      // Deep Orange
  Read: '#9C27B0',      // Purple
  Write: '#E91E63',     // Pink
  Grep: '#00BCD4',      // Cyan
  Glob: '#009688',      // Teal
  TodoWrite: '#FFC107', // Amber
  Edit: '#FF9800',      // Orange
  Task: '#3F51B5',      // Indigo
  ToolUse: '#607D8B',   // Blue Gray (default for tools)
  ToolResult: '#795548', // Brown
  Message: '#888888',   // Gray
}

function getNodeColor(labels: string[]): string {
  for (const label of labels) {
    if (NODE_COLORS[label]) {
      return NODE_COLORS[label]
    }
  }
  return '#666666'
}

function getNodeSize(labels: string[]): number {
  if (labels.includes('User') || labels.includes('Assistant')) return 8
  if (labels.includes('ToolUse')) return 5
  if (labels.includes('ToolResult')) return 3
  return 4
}

function getNodeLabel(node: Node): string {
  const labels = node.labels

  if (labels.includes('User')) return 'User'
  if (labels.includes('Assistant')) return 'Assistant'
  if (labels.includes('QueueOperation')) return 'Queue'

  // For tool uses, show the tool name
  for (const label of labels) {
    if (label !== 'ToolUse' && NODE_COLORS[label]) {
      return label
    }
  }

  if (labels.includes('ToolResult')) return 'Result'

  return labels[0] || 'Node'
}

// Transform API data for Cosmograph - only primitive types allowed
interface CosmographNode {
  id: string
  color: string
  size: number
  label: string
  nodeIndex: number // Index back into original array
}

interface CosmographLink {
  source: string
  target: string
}

function transformData(graphData: GraphData): { nodes: CosmographNode[]; links: CosmographLink[] } {
  const nodes: CosmographNode[] = graphData.nodes.map((node, index) => ({
    id: node.id.toString(),
    color: getNodeColor(node.labels),
    size: getNodeSize(node.labels),
    label: getNodeLabel(node),
    nodeIndex: index,
  }))

  const nodeIds = new Set(nodes.map(n => n.id))

  const links: CosmographLink[] = graphData.edges
    .filter(edge => nodeIds.has(edge.source.toString()) && nodeIds.has(edge.target.toString()))
    .map(edge => ({
      source: edge.source.toString(),
      target: edge.target.toString(),
    }))

  return { nodes, links }
}

function App() {
  const [graphData, setGraphData] = useState<GraphData | null>(null)
  const [stats, setStats] = useState<Stats | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [config, setConfig] = useState<CosmographConfig>({})
  const [hoveredNode, setHoveredNode] = useState<Node | null>(null)
  const cosmographRef = useRef<CosmographRef>(null)

  const fetchData = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)

      const [graphRes, statsRes] = await Promise.all([
        fetch('http://localhost:9876/api/graph'),
        fetch('http://localhost:9876/api/stats'),
      ])

      if (!graphRes.ok) {
        throw new Error(`Graph API error: ${graphRes.statusText}`)
      }
      if (!statsRes.ok) {
        throw new Error(`Stats API error: ${statsRes.statusText}`)
      }

      const graph = await graphRes.json()
      const statsData = await statsRes.json()

      setGraphData(graph)
      setStats(statsData)

      // Transform and prepare data for Cosmograph
      const { nodes, links } = transformData(graph)

      const dataConfig = {
        points: {
          pointIdBy: 'id',
          pointColorBy: 'color',
          pointSizeBy: 'size',
        },
        links: {
          linkSourceBy: 'source',
          linkTargetsBy: ['target'],
        },
      }

      const result = await prepareCosmographData(
        dataConfig,
        nodes as unknown as Record<string, unknown>[],
        links as unknown as Record<string, unknown>[]
      )

      if (result) {
        const { points, links: preparedLinks, cosmographConfig } = result
        setConfig({
          points,
          links: preparedLinks,
          ...cosmographConfig,
          // Custom settings for better visualization
          pointColorStrategy: CosmographPointColorStrategy.Direct,
          pointSizeStrategy: CosmographPointSizeStrategy.Direct,
          linkColor: '#444444',
          linkWidth: 0.5,
          linkArrows: true,
          linkArrowsSizeScale: 0.5,
          backgroundColor: '#1a1a2e',
          simulationGravity: 0.1,
          simulationRepulsion: 1,
          simulationLinkSpring: 0.3,
          simulationLinkDistance: 10,
          simulationFriction: 0.85,
          showDynamicLabels: true,
          hoveredPointRingColor: '#ffffff',
        })
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch data')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  const handlePointHover = useCallback((pointIndex: number | undefined) => {
    if (pointIndex !== undefined && graphData) {
      setHoveredNode(graphData.nodes[pointIndex] || null)
    } else {
      setHoveredNode(null)
    }
  }, [graphData])

  if (loading) {
    return (
      <div className="loading">
        <h2>Loading session graph...</h2>
        <p>Make sure the API server is running on port 9876</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="error">
        <h2>Error loading graph</h2>
        <p>{error}</p>
        <button onClick={fetchData}>Retry</button>
        <div className="hint">
          <p>Start the API server with:</p>
          <code>cargo run --package session-viewer-server -- /path/to/db.manifold</code>
        </div>
      </div>
    )
  }

  if (!graphData) {
    return <div className="loading">No data available</div>
  }

  return (
    <div className="app">
      <header className="header">
        <h1>ManifoldDB Session Viewer</h1>
        {stats && (
          <div className="stats">
            <span className="stat">{stats.node_count.toLocaleString()} nodes</span>
            <span className="stat">{stats.edge_count.toLocaleString()} edges</span>
          </div>
        )}
        <div className="controls">
          <button
            className="force-btn"
            onClick={() => cosmographRef.current?.fitView()}
          >
            Fit View
          </button>
          <button className="refresh-btn" onClick={fetchData}>Refresh</button>
        </div>
      </header>

      <div className="graph-container">
        <CosmographProvider>
          <Cosmograph
            ref={cosmographRef}
            {...config}
            onPointMouseOver={handlePointHover}
            onPointMouseOut={() => setHoveredNode(null)}
          />
        </CosmographProvider>
      </div>

      {hoveredNode && (
        <div className="tooltip">
          <div className="tooltip-header">{hoveredNode.labels.join(', ')}</div>
          <div className="tooltip-id">ID: {hoveredNode.id}</div>
          {'uuid' in hoveredNode.properties && hoveredNode.properties.uuid != null && (
            <div className="tooltip-uuid">UUID: {String(hoveredNode.properties.uuid).slice(0, 8)}...</div>
          )}
          {'toolName' in hoveredNode.properties && hoveredNode.properties.toolName != null && (
            <div className="tooltip-tool">Tool: {String(hoveredNode.properties.toolName)}</div>
          )}
          {'command' in hoveredNode.properties && hoveredNode.properties.command != null && (
            <div className="tooltip-command">Command: {String(hoveredNode.properties.command).slice(0, 60)}...</div>
          )}
          {'textContent' in hoveredNode.properties && hoveredNode.properties.textContent != null && (
            <div className="tooltip-content">
              {String(hoveredNode.properties.textContent).slice(0, 150)}...
            </div>
          )}
          {'outputTokens' in hoveredNode.properties && hoveredNode.properties.outputTokens != null && (
            <div className="tooltip-tokens">Tokens: {String(hoveredNode.properties.outputTokens)}</div>
          )}
        </div>
      )}

      <aside className="legend">
        <h3>Legend</h3>
        <div className="legend-section">
          <h4>Nodes</h4>
          <div className="legend-item">
            <span className="legend-color" style={{ background: NODE_COLORS.User }}></span>
            User
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ background: NODE_COLORS.Assistant }}></span>
            Assistant
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ background: NODE_COLORS.Bash }}></span>
            Bash
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ background: NODE_COLORS.Read }}></span>
            Read
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ background: NODE_COLORS.Write }}></span>
            Write
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ background: NODE_COLORS.ToolResult }}></span>
            Tool Result
          </div>
        </div>
        {stats && (
          <div className="legend-section">
            <h4>Node Types</h4>
            {Object.entries(stats.labels).map(([label, count]) => (
              <div key={label} className="legend-item">
                <span className="legend-color" style={{ background: NODE_COLORS[label] || '#666' }}></span>
                {label}: {count.toLocaleString()}
              </div>
            ))}
          </div>
        )}
      </aside>
    </div>
  )
}

export default App
