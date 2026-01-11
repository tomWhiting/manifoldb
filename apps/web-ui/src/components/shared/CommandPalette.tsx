import { useEffect } from 'react'
import { Command } from 'cmdk'
import {
  Plus,
  Play,
  GitBranch,
  Table,
  Braces,
  SplitSquareHorizontal,
  PanelLeftClose,
  Settings,
  Moon,
  Sun,
  Monitor,
  Layers,
} from 'lucide-react'
import { useAppStore } from '../../stores/app-store'
import { useTheme } from '../../hooks/useTheme'

export function CommandPalette() {
  const open = useAppStore((s) => s.commandPaletteOpen)
  const toggleCommandPalette = useAppStore((s) => s.toggleCommandPalette)
  const addTab = useAppStore((s) => s.addTab)
  const setViewMode = useAppStore((s) => s.setViewMode)
  const toggleSidebar = useAppStore((s) => s.toggleSidebar)
  const tabs = useAppStore((s) => s.tabs)
  const { theme, cycleTheme, setTheme } = useTheme()

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
                  addTab({
                    title: `Query ${tabs.length + 1}`,
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
            <CommandItem
              icon={<SplitSquareHorizontal size={16} />}
              onSelect={() => runCommand(() => {})}
            >
              Split Horizontal
            </CommandItem>
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
}: {
  children: React.ReactNode
  icon: React.ReactNode
  onSelect: () => void
}) {
  return (
    <Command.Item
      onSelect={onSelect}
      className="flex items-center gap-3 px-2 py-2 rounded text-sm text-text-secondary cursor-pointer data-[selected=true]:bg-bg-tertiary data-[selected=true]:text-text-primary"
    >
      <span className="text-text-muted">{icon}</span>
      {children}
    </Command.Item>
  )
}
