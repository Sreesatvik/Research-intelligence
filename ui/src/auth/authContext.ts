import { createContext, useContext } from 'react'
import type { AuthUser } from '../api'

export interface Auth {
  // undefined while the first check runs, null when signed out
  user: AuthUser | null | undefined
  signIn: (email: string, password: string) => Promise<void>
  signUp: (email: string, password: string) => Promise<void>
  signOut: () => Promise<void>
}

export const AuthContext = createContext<Auth | null>(null)

export function useAuth(): Auth {
  const auth = useContext(AuthContext)
  if (!auth) throw new Error('useAuth must be used inside <AuthProvider>')
  return auth
}
