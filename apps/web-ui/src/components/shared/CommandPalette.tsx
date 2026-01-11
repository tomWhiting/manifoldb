import { useEffect } from 'react'
import { Command } from 'cmdk'
import {
  Plus,
  Play,
  GitBranch,
  Table,
  Braces,
  SplitSquareHorizontal,
  SplitSquareVertical,
  XCircle,
  PanelLeftClose,
  Settings,
  Moon,
  Sun,
  Monitor,
  Layers,
} from 'lucide-react'
import { useAppStore } from '../../stores/app-store'
import { useWorkspaceStore } from '../../stores/workspace-store'
import { useTheme } from '../../hooks/useTheme'

export function CommandPalette() {
  const open = useAppStore((s) => s.commandPaletteOpen)
  const toggleCommandPalette = useAppStore((s) => s.toggleCommandPalette)
  const setViewMode = useAppStore((s) => s.setViewMode)
  const toggleSidebar = useAppStore((s) => s.toggleSidebar)

  const layout = useWorkspaceStore((s) => s.layout)
  const splitPane = useWorkspaceStore((s) => s.splitPane)
  const closePane = useWorkspaceStore((s) => s.closePane)
  const addTab = useWorkspaceStore((s) => s.addTab)
  const canSplit = useWorkspaceStore((s) => s.canSplit)

  const { theme, cycleTheme, setTheme } = useTheme()
  const paneCount = Object.keys(layout.panes).length
  const activePane = layout.panes[layout.activePaneId]

  const ThemeIcon = theme === 'dark' ? Moon : theme === 'light' ? Sun : Monitor
  const themeLabel = theme === 'dark' ? 'Dark' : theme === 'light' ? 'Light' : 'System'

  // Global keyboard shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        toggleCommandPalette()
      }
      if (e.key === 'Escape' && open) {
        toggleCommandPalette()
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [open, toggleCommandPalette])

  if (!open) return null

  const runCommand = (fn: () => void) => {
    fn()
    toggleCommandPalette()
  }

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[20vh]">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/50" onClick={toggleCommandPalette} />

      {/* Command dialog */}
      <Command className="relative w-full max-w-lg bg-bg-secondary rounded-lg shadow-2xl border border-border overflow-hidden">
        <Command.Input
          placeholder="Type a command or search..."
          className="w-full px-4 py-3 bg-transparent text-text-primary placeholder-text-muted outline-none border-b border-border"
          autoFocus
        />

        <Command.List className="max-h-80 overflow-auto p-2">
          <Command.Empty className="py-6 text-center text-sm text-text-muted">
            No results found.
          </Command.Empty>

          <Command.Group heading="Query" className="text-xs text-text-muted px-2 py-1.5">
            <CommandItem
              icon={<Plus size={16} />}
              onSelect={() =>
                runCommand(() =>
                  addTab(layout.activePaneId, {
                    title: `Query ${(activePane?.tabs.length ?? 0) + 1}`,
                    content: '// New query\n',
                    language: 'cypher',
                  })
                )
              }
            >
              New Query
            </CommandItem>
            <CommandItem icon={<Play size={16} />} onSelect={() => runCommand(() => {})}>
              Run Query
            </CommandItem>
          </Command.Group>

          <Command.Group heading="View" className="text-xs text-text-muted px-2 py-1.5">
            <CommandItem
              icon={<GitBranch size={16} />}
              onSelect={() => runCommand(() => setViewMode('graph'))}
            >
              View: Graph
            </CommandItem>
            <CommandItem
              icon={<Table size={16} />}
              onSelect={() => runCommand(() => setViewMode('table'))}
            >
              View: Table
            </CommandItem>
            <CommandItem
              icon={<Braces size={16} />}
              onSelect={() => runCommand(() => setViewMode('json'))}
            >
              View: JSON
            </CommandItem>
          </Command.Group>

          <Command.Group heading="Layout" className="text-xs text-text-muted px-2 py-1.5">
            {canSplit() && (
              <>
                <CommandItem
                  icon={<SplitSquareVertical size={16} />}
                  onSelect={() => runCommand(() => splitPane(layout.activePaneId, 'vertical'))}
                  shortcut="Cmd+\"
                >
                  Split Vertical
                </CommandItem>
                <CommandItem
                  icon={<SplitSquareHorizontal size={16} />}
                  onSelect={() => runCommand(() => splitPane(layout.activePaneId, 'horizontal'))}
                  shortcut="Cmd+Shift+\"
                >
                  Split Horizontal
                </CommandItem>
              </>
            )}
            {paneCount > 1 && (
              <CommandItem
                icon={<XCircle size={16} />}
                onSelect={() => runCommand(() => closePane(layout.activePaneId))}
                shortcut="Cmd+W"
              >
                Close Pane
              </CommandItem>
            )}
            <CommandItem
              icon={<PanelLeftClose size={16} />}
              onSelect={() => runCommand(toggleSidebar)}
            >
              Toggle Sidebar
            </CommandItem>
          </Command.Group>

          <Command.Group heading="Settings" className="text-xs text-text-muted px-2 py-1.5">
            <CommandItem icon={<Settings size={16} />} onSelect={() => runCommand(() => {})}>
              Open Settings
            </CommandItem>
            <CommandItem
              icon={<ThemeIcon size={16} />}
              onSelect={() => runCommand(cycleTheme)}
            >
              Theme: {themeLabel}
            </CommandItem>
            <CommandItem
              icon={<Moon size={16} />}
              onSelect={() => runCommand(() => setTheme('dark'))}
            >
              Theme: Dark
            </CommandItem>
            <CommandItem
              icon={<Sun size={16} />}
              onSelect={() => runCommand(() => setTheme('light'))}
            >
              Theme: Light
            </CommandItem>
            <CommandItem
              icon={<Monitor size={16} />}
              onSelect={() => runCommand(() => setTheme('system'))}
            >
              Theme: System
            </CommandItem>
            <CommandItem icon={<Layers size={16} />} onSelect={() => runCommand(() => {})}>
              Toggle Vector Overlay
            </CommandItem>
          </Command.Group>
        </Command.List>
      </Command>
    </div>
  )
}

function CommandItem({
  children,
  icon,
  onSelect,
  shortcut,
}: {
  children: React.ReactNode
  icon: React.ReactNode
  onSelect: () => void
  shortcut?: string
}) {
  return (
    <Command.Item
      onSelect={onSelect}
      className="flex items-center gap-3 px-2 py-2 rounded text-sm text-text-secondary cursor-pointer data-[selected=true]:bg-bg-tertiary data-[selected=true]:text-text-primary"
    >
      <span className="text-text-muted">{icon}</span>
      <span className="flex-1">{children}</span>
      {shortcut && (
        <span className="text-xs text-text-muted/50">{shortcut}</span>
      )}
    </Command.Item>
  )
}
