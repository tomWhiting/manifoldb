import type { ReactNode } from 'react'
import { useAppStore } from '../../stores/app-store'

interface AppShellProps {
  sidebar: ReactNode
  workspace: ReactNode
  tray: ReactNode
}

export function AppShell({ sidebar, workspace, tray }: AppShellProps) {
  const sidebarCollapsed = useAppStore((s) => s.sidebarCollapsed)

  return (
    <div className="flex h-screen w-screen bg-neutral-950 text-neutral-100 overflow-hidden">
      {/* Sidebar */}
      <aside
        className={`
          flex-shrink-0 border-r border-neutral-800 bg-neutral-900
          transition-[width] duration-200 ease-out
          ${sidebarCollapsed ? 'w-12' : 'w-56'}
        `}
      >
        {sidebar}
      </aside>

      {/* Main content area */}
      <div className="flex flex-col flex-1 min-w-0">
        {/* Workspace */}
        <main className="flex-1 min-h-0 overflow-hidden">{workspace}</main>

        {/* Tray */}
        <footer className="flex-shrink-0 border-t border-neutral-800 bg-neutral-900/50">
          {tray}
        </footer>
      </div>
    </div>
  )
}
