# ManifoldDB Web UI: Design Document

## Overview

A desktop web application for ManifoldDB, similar in spirit to Memgraph Lab but tailored to ManifoldDB's unified graph-vector-SQL capabilities. The UI provides query editing, result visualization, and database exploration across all three paradigms.

**Target**: Desktop-first. A streamlined mobile-friendly version may come later, but is not an initial priority.

---

## Core Concept: Graph as the Hub

The graph visualization is the central view of the application. All data ultimately connects through the graph—it's the hub that ties together relational, vector, and graph data.

### Unified Result View

Query results can be viewed in multiple formats, with the graph as the primary representation:

| View | Purpose |
|------|---------|
| **Graph** | Primary view. Node-edge visualization. The hub. |
| **Table** | Tabular representation of the same data (nodes as rows, properties as columns) |
| **JSON** | Raw data view for inspection and debugging |

Users can switch between these views for the same result set. The graph doesn't go away—you layer other views on top of it.

### Floating Inspector Windows

Data inspection panels can be "popped out" as floating windows that overlay the graph:
- Drag and position anywhere over the graph canvas
- View node/edge details without leaving the graph context
- Multiple inspectors open simultaneously
- Resize, minimize, or dock back into sidebar

This allows users to inspect tabular data while keeping the graph visible and interactive.

### Vector Overlay

Vector data can be visualized as a layer on the graph:
- UMAP or t-SNE projection as a heatmap overlay
- Color nodes by vector similarity
- Cluster visualization based on embedding space
- Toggle overlay on/off without losing graph state

---

## Workspace Modes

The workspace supports different modes for different tasks:

### 1. Query Execution Mode

The primary mode. Write and run queries, view results.

```
┌─────────────────────────────────────────────────────────────────┐
│ [Sidebar]  │                    Workspace                       │
│            │  ┌─────────────────────────────────────────────┐   │
│            │  │              Query Editor                   │   │
│            │  │         (Iridium / CodeMirror)              │   │
│            │  ├─────────────────────────────────────────────┤   │
│            │  │           Unified Result View               │   │
│            │  │  ┌─────────────────────────────────────┐    │   │
│            │  │  │         Graph Canvas                │    │   │
│            │  │  │    ┌──────────┐                     │    │   │
│            │  │  │    │ Floating │   [View: Graph|     │    │   │
│            │  │  │    │ Table    │    Table|JSON]      │    │   │
│            │  │  │    └──────────┘                     │    │   │
│            │  │  └─────────────────────────────────────┘    │   │
│            │  └─────────────────────────────────────────────┘   │
├────────────┴────────────────────────────────────────────────────┤
│ [Tray] Connection · Resources · Query time                      │
└─────────────────────────────────────────────────────────────────┘
```

### 2. SQL Builder Mode

Dedicated view for building SQL queries visually, inspired by MS Access query designer:
- Visual table selector
- Drag-and-drop join builder
- Column picker with aggregation options
- WHERE clause builder
- Generated SQL preview
- Run and view results

### 3. Schema Editor Mode

View and edit the graph schema:
- Visual node/edge type diagram
- Create new labels and relationship types
- Define property schemas
- Constraint management
- Index management

### 4. Collection Browser Mode

Dedicated view for vector collections:
- Quadrant-style layout
- Browse vectors with payload preview
- Text/content inspection
- Similarity search interface
- Collection statistics

---

## Sidebar

The sidebar is the primary navigation and tool panel. Sections (top to bottom):

### Navigation Sections

| Section | Purpose |
|---------|---------|
| **Query** | Query editor access, new query, run controls |
| **History** | Run history, past queries, re-run or edit |
| **Overview** | Dashboard with database stats, charts, health |
| **Query Modules** | Saved query modules (like Memgraph MAGE procedures) |
| **Collections** | Vector collection browser and inspector |
| **Schema** | Graph schema viewer/editor |
| **Import/Export** | Data import and export tools |
| **Logs** | Query logs, error logs, audit trail |

### Bottom Sections

| Section | Purpose |
|---------|---------|
| **Settings** | App configuration, theme, editor preferences |
| **AI Assistant** | Natural language query help (future) |

### Sidebar Behavior

- Collapsible (icon-only mode)
- Section headers are clickable to expand/collapse
- Active section highlighted
- Keyboard navigable

---

## Bottom Tray

A thin footer bar with essential status information:

| Item | Content |
|------|---------|
| **Connection** | Server connection status indicator (connected/disconnected/error) |
| **Resources** | CPU, memory, query queue status |
| **Query Status** | Current query time, row count, status |

Clicking items expands details (e.g., click connection to see server info, click resources to see detailed metrics).

---

## Query Editor

The query editor uses **Iridium** (GPU-accelerated, Rust/wgpu) for:
- Cypher syntax highlighting and editing
- SQL syntax highlighting and editing
- 120fps rendering, smooth scrolling
- LSP integration for autocomplete/diagnostics (future)

Until Iridium is ready, CodeMirror 6 as fallback.

### Query Helpers

Pre-built query templates insertable via:
- Sidebar snippets section
- Command palette search
- Right-click context menu

Categories:
- Common Cypher patterns (MATCH, CREATE, MERGE, paths)
- Common SQL patterns (SELECT, JOIN, aggregate)
- Vector search templates
- Custom saved queries

---

## Split Panes

Support for splitting the workspace:
- Horizontal split: Two query editors stacked
- Vertical split: Side-by-side views
- Multiple splits: Up to 4 panes (2x2 grid)
- Each pane independent (own query, own results)
- Drag divider to resize

Use cases:
- Compare two query results
- Write query while viewing another's results
- SQL in one pane, Cypher in another

---

## Command Palette

Use **cmdk** for keyboard-driven navigation.

**Trigger**: `Cmd+K` / `Ctrl+K`

**Commands**:
- `New Query` - Open new query tab
- `Run Query` - Execute current query (`Cmd+Enter`)
- `Split Horizontal` / `Split Vertical`
- `Toggle Sidebar`
- `View: Graph` / `View: Table` / `View: JSON`
- `Insert Snippet: [name]`
- `Go to Label: [label]`
- `Go to Collection: [name]`
- `Open Schema Editor`
- `Open SQL Builder`
- `Settings`
- `Toggle Theme`
- `Toggle Vector Overlay`

Fuzzy search across all commands. Recent commands at top.

---

## Result Views

### Graph View (Primary)

Custom WebGPU renderer:
- Force-directed layout
- Pan, zoom, select
- Node/edge hover for details
- Click to expand neighbors
- Style nodes by label, edges by type
- Vector heatmap overlay (UMAP/t-SNE)

### Table View (Overlay or Standalone)

- Nodes as rows, properties as columns
- Nested tables for relationships
- Sortable, resizable columns
- Virtual scrolling for large results
- Can float over graph or dock in panel

### JSON View

- Raw data inspection
- Syntax highlighted
- Collapsible nodes
- Copy to clipboard

### Vector Visualization

When viewing vector data:
- 2D projection (UMAP/t-SNE)
- Can overlay on graph as heatmap
- Similarity clusters visible
- Click point to see payload

---

## Technical Stack

| Component | Choice | Notes |
|-----------|--------|-------|
| Framework | React 18+ | |
| Language | TypeScript | Strict mode |
| Build | Vite | Fast dev, good production builds |
| Styling | Tailwind CSS | Utility-first, minimal custom CSS |
| State | Zustand | Lightweight, simple |
| Data Fetching | GraphQL (urql) | Connects to Manifold server |
| Real-time | GraphQL Subscriptions (WebSocket) | Live updates |
| Command Palette | cmdk | Standard React command menu |
| Table | TanStack Table | Headless, flexible |
| Icons | Lucide React | Clean, consistent icon set |
| Editor (fallback) | CodeMirror 6 | Until Iridium ready |
| Notifications | Sonner | Clean toast notifications |
| Split Panes | react-resizable-panels | Lightweight, flexible |
| Floating Windows | Custom or @floating-ui | For inspector popouts |

### Connection to Manifold Server

- **GraphQL endpoint**: `http://localhost:6010/graphql`
- **WebSocket**: `ws://localhost:6010/graphql/ws` for subscriptions
- Queries: `cypher()`, `sql()`, `node()`, `neighbors()`, `stats()`, etc.
- Mutations: `createNode()`, `createEdge()`, `createNodes()`, `createEdges()`, `updateNode()`, `updateEdge()`, `execute()`, etc.
- Subscriptions: `nodeChanges()`, `edgeChanges()`, `graphChanges()`

---

## UI Components

### Core Layout
- `<AppShell>` - Main layout container
- `<Sidebar>` - Navigation and tools
- `<Workspace>` - Main content area with split support
- `<Tray>` - Bottom status bar

### Editor
- `<QueryEditor>` - Iridium wrapper (or CodeMirror fallback)
- `<QueryTabs>` - Tab bar for multiple queries

### Result Views
- `<UnifiedResultView>` - Container with view switcher
- `<GraphCanvas>` - WebGPU graph renderer
- `<TableOverlay>` - Floating/dockable table view
- `<JSONView>` - Raw data inspector
- `<VectorOverlay>` - UMAP/t-SNE heatmap layer

### Floating Windows
- `<FloatingPanel>` - Draggable, resizable overlay panel
- `<Inspector>` - Node/edge detail inspector

### Sidebar Panels
- `<QueryPanel>` - Query controls and snippets
- `<HistoryPanel>` - Run history browser
- `<OverviewDashboard>` - Stats and charts
- `<ModulesPanel>` - Query module browser
- `<CollectionsPanel>` - Vector collections browser
- `<SchemaPanel>` - Schema viewer/editor access
- `<ImportExportPanel>` - Data I/O tools
- `<LogsPanel>` - Log viewer
- `<SettingsPanel>` - Configuration

### Specialized Views
- `<SQLBuilder>` - Visual SQL query builder
- `<SchemaEditor>` - Visual schema editor
- `<CollectionBrowser>` - Vector collection inspector

### Shared
- `<IconButton>` - Icon-only button with tooltip
- `<SplitPane>` - Resizable split container
- `<CommandPalette>` - cmdk wrapper
- `<Tooltip>` - Consistent tooltip styling

---

## Design Principles

- **Minimal chrome**: Maximize workspace, minimize UI furniture
- **No chunky components**: Avoid thick headers, heavy borders, oversized elements
- **Icon-first**: Buttons use icons (with tooltips), not text labels
- **Clean lines**: Subtle separators, restrained use of color
- **Tray over header**: Status in footer, not fat header bars
- **Contextual controls**: Show controls relevant to current context, hide others
- **Graph as hub**: Everything connects through the graph visualization

---

## Future Additions

Designed to accommodate but not in initial scope:

- **Query history persistence**: Save and search past queries
- **Saved queries/workspaces**: Name and save query sets
- **Collaboration**: Share queries via URL
- **AI assistant**: Natural language → Cypher/SQL
- **Query modules**: MAGE-style stored procedures
- **Plugin system**: User-defined extensions

---

## Next Steps

1. Set up React + Vite + TypeScript project
2. Implement AppShell layout (sidebar, workspace, tray)
3. Integrate CodeMirror 6 as initial editor
4. Connect to Manifold server GraphQL
5. Implement basic query execution flow
6. Integrate graph renderer
7. Build unified result view with view switching
8. Implement floating panels/inspectors
9. Add command palette
10. Build sidebar panels progressively
11. Add SQL Builder mode
12. Add Schema Editor mode
13. Polish and iterate
