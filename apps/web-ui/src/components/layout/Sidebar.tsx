import {
  Play,
  History,
  LayoutDashboard,
  Package,
  Database,
  GitBranch,
  Import,
  FileText,
  Settings,
  Bot,
  PanelLeftClose,
  PanelLeft,
  TableProperties,
  PenTool,
} from 'lucide-react'
import { IconButton } from '../shared/IconButton'
import { useAppStore } from '../../stores/app-store'
import type { SidebarSection } from '../../types'

interface NavItem {
  id: SidebarSection
  icon: typeof Play
  label: string
}

const topNavItems: NavItem[] = [
  { id: 'query', icon: Play, label: 'Query' },
  { id: 'sql-builder', icon: TableProperties, label: 'SQL Builder' },
  { id: 'history', icon: History, label: 'History' },
  { id: 'overview', icon: LayoutDashboard, label: 'Overview' },
  { id: 'modules', icon: Package, label: 'Query Modules' },
  { id: 'collections', icon: Database, label: 'Collections' },
  { id: 'schema', icon: GitBranch, label: 'Schema' },
  { id: 'schema-editor', icon: PenTool, label: 'Schema Editor' },
  { id: 'import-export', icon: Import, label: 'Import/Export' },
  { id: 'logs', icon: FileText, label: 'Logs' },
]

const bottomNavItems: NavItem[] = [
  { id: 'settings', icon: Settings, label: 'Settings' },
  { id: 'assistant', icon: Bot, label: 'AI Assistant' },
]

function SidebarItem({
  item,
  active,
  collapsed,
  onClick,
}: {
  item: NavItem
  active: boolean
  collapsed: boolean
  onClick: () => void
}) {
  const Icon = item.icon

  return (
    <button
      onClick={onClick}
      className={`
        w-full flex items-center gap-3 px-3 py-2 rounded-md
        transition-colors duration-150
        ${active ? 'bg-accent-muted text-accent' : 'text-text-muted hover:bg-bg-tertiary hover:text-text-secondary'}
        ${collapsed ? 'justify-center' : ''}
      `}
      title={collapsed ? item.label : undefined}
    >
      <Icon size={18} className="flex-shrink-0" />
      {!collapsed && <span className="text-sm font-medium truncate">{item.label}</span>}
    </button>
  )
}

export function Sidebar() {
  const collapsed = useAppStore((s) => s.sidebarCollapsed)
  const activeSection = useAppStore((s) => s.activeSidebarSection)
  const toggleSidebar = useAppStore((s) => s.toggleSidebar)
  const setSidebarSection = useAppStore((s) => s.setSidebarSection)

  return (
    <div className="flex flex-col h-full py-2">
      {/* Toggle button */}
      <div className={`px-2 mb-2 ${collapsed ? 'flex justify-center' : ''}`}>
        <IconButton
          icon={collapsed ? <PanelLeft size={18} /> : <PanelLeftClose size={18} />}
          onClick={toggleSidebar}
          tooltip={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        />
      </div>

      {/* Top navigation */}
      <nav className="flex-1 px-2 space-y-1">
        {topNavItems.map((item) => (
          <SidebarItem
            key={item.id}
            item={item}
            active={activeSection === item.id}
            collapsed={collapsed}
            onClick={() => setSidebarSection(item.id)}
          />
        ))}
      </nav>

      {/* Divider */}
      <div className="mx-3 my-2 border-t border-border" />

      {/* Bottom navigation */}
      <nav className="px-2 space-y-1">
        {bottomNavItems.map((item) => (
          <SidebarItem
            key={item.id}
            item={item}
            active={activeSection === item.id}
            collapsed={collapsed}
            onClick={() => setSidebarSection(item.id)}
          />
        ))}
      </nav>
    </div>
  )
}
