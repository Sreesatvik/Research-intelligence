import { useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { login, logout, me, register, type AuthUser } from '../api'
import { AuthContext } from './authContext'

// Asks the server who we are (the session cookie is httpOnly, so scripts cannot read it) and keeps that answer.
export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null | undefined>(undefined)
  const queryClient = useQueryClient()

  useEffect(() => {
    let cancelled = false
    me()
      .then((current) => !cancelled && setUser(current))
      .catch(() => !cancelled && setUser(null))
    // Any request answered with 401 means the session ended.
    const onExpired = () => {
      queryClient.clear() // nothing from the ended session may linger in memory
      setUser(null)
    }
    window.addEventListener('auth:expired', onExpired)
    return () => {
      cancelled = true
      window.removeEventListener('auth:expired', onExpired)
    }
  }, [queryClient])

  const value = useMemo(
    () => ({
      user,
      signIn: async (email: string, password: string) => setUser(await login(email, password)),
      signUp: async (email: string, password: string) => setUser(await register(email, password)),
      signOut: async () => {
        await logout().catch(() => undefined)
        queryClient.clear()
        setUser(null)
      },
    }),
    [user, queryClient],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
