import { create } from 'zustand'
import type {
  LayoutNode,
  SplitNode,
  SplitDirection,
  PaneState,
  QueryTab,
  QueryResult,
  WorkspaceLayout,
} from '../types'

const STORAGE_KEY = 'manifoldb-workspace-layout'
const MAX_PANES = 4

let paneIdCounter = 0
let tabIdCounter = 0

const generatePaneId = () => `pane-${++paneIdCounter}`
const generateTabId = () => `tab-${++tabIdCounter}`

function createDefaultTab(): QueryTab {
  return {
    id: generateTabId(),
    title: 'Query 1',
    content: '',
    language: 'cypher',
  }
}

function createDefaultPane(): PaneState {
  const tab = createDefaultTab()
  return {
    id: generatePaneId(),
    tabs: [tab],
    activeTabId: tab.id,
  }
}

function createDefaultLayout(): WorkspaceLayout {
  const pane = createDefaultPane()
  return {
    root: { type: 'leaf', paneId: pane.id },
    panes: { [pane.id]: pane },
    activePaneId: pane.id,
  }
}

function countLeafNodes(node: LayoutNode): number {
  if (node.type === 'leaf') return 1
  return countLeafNodes(node.children[0]) + countLeafNodes(node.children[1])
}

function replaceNode(root: LayoutNode, targetPaneId: string, newNode: LayoutNode): LayoutNode {
  if (root.type === 'leaf') {
    return root.paneId === targetPaneId ? newNode : root
  }

  return {
    ...root,
    children: [
      replaceNode(root.children[0], targetPaneId, newNode),
      replaceNode(root.children[1], targetPaneId, newNode),
    ] as [LayoutNode, LayoutNode],
  }
}

function removePane(root: LayoutNode, paneId: string): LayoutNode | null {
  if (root.type === 'leaf') {
    return root.paneId === paneId ? null : root
  }

  const leftResult = removePane(root.children[0], paneId)
  const rightResult = removePane(root.children[1], paneId)

  if (leftResult === null) return rightResult
  if (rightResult === null) return leftResult

  return {
    ...root,
    children: [leftResult, rightResult] as [LayoutNode, LayoutNode],
  }
}

function getFirstPaneId(node: LayoutNode): string {
  if (node.type === 'leaf') return node.paneId
  return getFirstPaneId(node.children[0])
}

function getAllPaneIds(node: LayoutNode): string[] {
  if (node.type === 'leaf') return [node.paneId]
  return [...getAllPaneIds(node.children[0]), ...getAllPaneIds(node.children[1])]
}

function loadLayout(): WorkspaceLayout | null {
  if (typeof window === 'undefined') return null
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored) {
      const layout = JSON.parse(stored) as WorkspaceLayout
      // Validate structure
      if (layout.root && layout.panes && layout.activePaneId) {
        // Reset counters based on loaded data
        const allPaneIds = getAllPaneIds(layout.root)
        allPaneIds.forEach((id) => {
          const match = id.match(/pane-(\d+)/)
          if (match) {
            const num = parseInt(match[1], 10)
            if (num > paneIdCounter) paneIdCounter = num
          }
          const pane = layout.panes[id]
          if (pane) {
            pane.tabs.forEach((tab) => {
              const tabMatch = tab.id.match(/tab-(\d+)/)
              if (tabMatch) {
                const num = parseInt(tabMatch[1], 10)
                if (num > tabIdCounter) tabIdCounter = num
              }
            })
          }
        })
        return layout
      }
    }
  } catch {
    // Ignore parse errors
  }
  return null
}

function saveLayout(layout: WorkspaceLayout): void {
  if (typeof window === 'undefined') return
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(layout))
  } catch {
    // Ignore storage errors
  }
}

interface WorkspaceState {
  layout: WorkspaceLayout

  // Pane management
  splitPane: (paneId: string, direction: SplitDirection) => void
  closePane: (paneId: string) => void
  setActivePane: (paneId: string) => void
  updateSizes: (parentPaneId: string, sizes: [number, number]) => void

  // Tab management within panes
  addTab: (paneId: string, tab: Omit<QueryTab, 'id'>) => string
  removeTab: (paneId: string, tabId: string) => void
  setActiveTab: (paneId: string, tabId: string) => void
  updateTabContent: (paneId: string, tabId: string, content: string) => void
  setTabResult: (paneId: string, tabId: string, result: QueryResult | undefined) => void
  setTabExecuting: (paneId: string, tabId: string, isExecuting: boolean) => void
  setTabLanguage: (paneId: string, tabId: string, language: 'cypher' | 'sql') => void

  // Convenience getters
  getActivePane: () => PaneState | undefined
  getActivePaneTab: () => QueryTab | undefined
  canSplit: () => boolean

  // Persistence
  persistLayout: () => void
}

export const useWorkspaceStore = create<WorkspaceState>((set, get) => ({
  layout: loadLayout() || createDefaultLayout(),

  splitPane: (paneId, direction) => {
    const { layout } = get()
    if (countLeafNodes(layout.root) >= MAX_PANES) return

    const newPane = createDefaultPane()
    const newSplit: SplitNode = {
      type: 'split',
      direction,
      children: [
        { type: 'leaf', paneId },
        { type: 'leaf', paneId: newPane.id },
      ],
      sizes: [50, 50],
    }

    const newRoot = replaceNode(layout.root, paneId, newSplit)
    const newLayout: WorkspaceLayout = {
      root: newRoot,
      panes: { ...layout.panes, [newPane.id]: newPane },
      activePaneId: newPane.id,
    }

    set({ layout: newLayout })
    saveLayout(newLayout)
  },

  closePane: (paneId) => {
    const { layout } = get()
    if (countLeafNodes(layout.root) <= 1) return

    const newRoot = removePane(layout.root, paneId)
    if (!newRoot) return

    const newPanes = { ...layout.panes }
    delete newPanes[paneId]

    const newActivePaneId =
      layout.activePaneId === paneId ? getFirstPaneId(newRoot) : layout.activePaneId

    const newLayout: WorkspaceLayout = {
      root: newRoot,
      panes: newPanes,
      activePaneId: newActivePaneId,
    }

    set({ layout: newLayout })
    saveLayout(newLayout)
  },

  setActivePane: (paneId) => {
    const { layout } = get()
    if (!layout.panes[paneId]) return

    const newLayout = { ...layout, activePaneId: paneId }
    set({ layout: newLayout })
    saveLayout(newLayout)
  },

  updateSizes: (parentPaneId, sizes) => {
    const { layout } = get()

    function updateSizesInNode(node: LayoutNode): LayoutNode {
      if (node.type === 'leaf') return node

      // Check if this split contains the target pane
      const containsPane = getAllPaneIds(node).includes(parentPaneId)
      if (containsPane && node.type === 'split') {
        // If the left child is the target or contains it as first child
        if (
          (node.children[0].type === 'leaf' && node.children[0].paneId === parentPaneId) ||
          (node.children[0].type === 'split' && getFirstPaneId(node.children[0]) === parentPaneId)
        ) {
          return { ...node, sizes }
        }
      }

      return {
        ...node,
        children: [updateSizesInNode(node.children[0]), updateSizesInNode(node.children[1])] as [
          LayoutNode,
          LayoutNode,
        ],
      }
    }

    const newLayout = { ...layout, root: updateSizesInNode(layout.root) }
    set({ layout: newLayout })
    saveLayout(newLayout)
  },

  addTab: (paneId, tab) => {
    const { layout } = get()
    const pane = layout.panes[paneId]
    if (!pane) return ''

    const id = generateTabId()
    const newTab = { ...tab, id }
    const newPane = {
      ...pane,
      tabs: [...pane.tabs, newTab],
      activeTabId: id,
    }

    const newLayout = {
      ...layout,
      panes: { ...layout.panes, [paneId]: newPane },
    }

    set({ layout: newLayout })
    saveLayout(newLayout)
    return id
  },

  removeTab: (paneId, tabId) => {
    const { layout } = get()
    const pane = layout.panes[paneId]
    if (!pane) return

    const newTabs = pane.tabs.filter((t) => t.id !== tabId)
    const newActiveTabId =
      pane.activeTabId === tabId ? (newTabs[newTabs.length - 1]?.id ?? null) : pane.activeTabId

    const newPane = { ...pane, tabs: newTabs, activeTabId: newActiveTabId }
    const newLayout = {
      ...layout,
      panes: { ...layout.panes, [paneId]: newPane },
    }

    set({ layout: newLayout })
    saveLayout(newLayout)
  },

  setActiveTab: (paneId, tabId) => {
    const { layout } = get()
    const pane = layout.panes[paneId]
    if (!pane) return

    const newPane = { ...pane, activeTabId: tabId }
    const newLayout = {
      ...layout,
      panes: { ...layout.panes, [paneId]: newPane },
      activePaneId: paneId,
    }

    set({ layout: newLayout })
    saveLayout(newLayout)
  },

  updateTabContent: (paneId, tabId, content) => {
    const { layout } = get()
    const pane = layout.panes[paneId]
    if (!pane) return

    const newTabs = pane.tabs.map((t) => (t.id === tabId ? { ...t, content } : t))
    const newPane = { ...pane, tabs: newTabs }
    const newLayout = {
      ...layout,
      panes: { ...layout.panes, [paneId]: newPane },
    }

    set({ layout: newLayout })
    // Don't save on every content change - too frequent
  },

  setTabResult: (paneId, tabId, result) => {
    const { layout } = get()
    const pane = layout.panes[paneId]
    if (!pane) return

    const newTabs = pane.tabs.map((t) => (t.id === tabId ? { ...t, result } : t))
    const newPane = { ...pane, tabs: newTabs }
    const newLayout = {
      ...layout,
      panes: { ...layout.panes, [paneId]: newPane },
    }

    set({ layout: newLayout })
  },

  setTabExecuting: (paneId, tabId, isExecuting) => {
    const { layout } = get()
    const pane = layout.panes[paneId]
    if (!pane) return

    const newTabs = pane.tabs.map((t) => (t.id === tabId ? { ...t, isExecuting } : t))
    const newPane = { ...pane, tabs: newTabs }
    const newLayout = {
      ...layout,
      panes: { ...layout.panes, [paneId]: newPane },
    }

    set({ layout: newLayout })
  },

  setTabLanguage: (paneId, tabId, language) => {
    const { layout } = get()
    const pane = layout.panes[paneId]
    if (!pane) return

    const newTabs = pane.tabs.map((t) => (t.id === tabId ? { ...t, language } : t))
    const newPane = { ...pane, tabs: newTabs }
    const newLayout = {
      ...layout,
      panes: { ...layout.panes, [paneId]: newPane },
    }

    set({ layout: newLayout })
    saveLayout(newLayout)
  },

  getActivePane: () => {
    const { layout } = get()
    return layout.panes[layout.activePaneId]
  },

  getActivePaneTab: () => {
    const { layout } = get()
    const pane = layout.panes[layout.activePaneId]
    if (!pane) return undefined
    return pane.tabs.find((t) => t.id === pane.activeTabId)
  },

  canSplit: () => {
    const { layout } = get()
    return countLeafNodes(layout.root) < MAX_PANES
  },

  persistLayout: () => {
    const { layout } = get()
    saveLayout(layout)
  },
}))
