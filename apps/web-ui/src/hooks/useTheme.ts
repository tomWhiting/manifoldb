import { useEffect } from 'react'
import { useAppStore } from '../stores/app-store'

/**
 * Hook to manage theme state and apply the appropriate class to the document root.
 * Handles system preference detection and persistence.
 */
export function useTheme() {
  const theme = useAppStore((s) => s.theme)
  const setTheme = useAppStore((s) => s.setTheme)
  const cycleTheme = useAppStore((s) => s.cycleTheme)

  useEffect(() => {
    const root = document.documentElement

    const applyTheme = (resolvedTheme: 'dark' | 'light') => {
      if (resolvedTheme === 'dark') {
        root.classList.add('dark')
      } else {
        root.classList.remove('dark')
      }
    }

    // Resolve the actual theme to apply
    if (theme === 'system') {
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
      applyTheme(mediaQuery.matches ? 'dark' : 'light')

      // Listen for system preference changes
      const handleChange = (e: MediaQueryListEvent) => {
        applyTheme(e.matches ? 'dark' : 'light')
      }

      mediaQuery.addEventListener('change', handleChange)
      return () => mediaQuery.removeEventListener('change', handleChange)
    } else {
      applyTheme(theme)
    }
  }, [theme])

  // Get the resolved theme (what's actually being displayed)
  const resolvedTheme = (): 'dark' | 'light' => {
    if (theme === 'system') {
      return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
    }
    return theme
  }

  return {
    theme,
    setTheme,
    cycleTheme,
    resolvedTheme,
    isDark: resolvedTheme() === 'dark',
  }
}
