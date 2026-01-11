import { Provider } from 'urql'
import { Toaster } from 'sonner'
import { AppShell } from './components/layout/AppShell'
import { Sidebar } from './components/layout/Sidebar'
import { Workspace } from './components/layout/Workspace'
import { Tray } from './components/layout/Tray'
import { CommandPalette } from './components/shared/CommandPalette'
import { graphqlClient } from './lib/graphql-client'
import { useConnection } from './hooks/useConnection'
import { useTheme } from './hooks/useTheme'

function ConnectionManager() {
  useConnection()
  return null
}

function App() {
  // Initialize theme system
  useTheme()

  return (
    <Provider value={graphqlClient}>
      <ConnectionManager />
      <AppShell sidebar={<Sidebar />} workspace={<Workspace />} tray={<Tray />} />
      <CommandPalette />
      <Toaster
        position="bottom-right"
        theme="dark"
        toastOptions={{
          style: {
            background: 'rgb(23 23 23)',
            border: '1px solid rgb(38 38 38)',
            color: 'rgb(229 229 229)',
          },
        }}
      />
    </Provider>
  )
}

export default App
