import { Provider } from 'urql'
import { AppShell } from './components/layout/AppShell'
import { Sidebar } from './components/layout/Sidebar'
import { Workspace } from './components/layout/Workspace'
import { Tray } from './components/layout/Tray'
import { CommandPalette } from './components/shared/CommandPalette'
import { graphqlClient } from './lib/graphql-client'
import { useTheme } from './hooks/useTheme'

function App() {
  // Initialize theme system
  useTheme()

  return (
    <Provider value={graphqlClient}>
      <AppShell sidebar={<Sidebar />} workspace={<Workspace />} tray={<Tray />} />
      <CommandPalette />
    </Provider>
  )
}

export default App
