import { Navigate, Outlet, useLocation } from 'react-router-dom'
import { useAuth } from './authContext'

// Pages inside this route need a signed-in user; everyone else is sent to the sign-in screen and brought back after.
export default function RequireAuth() {
  const { user } = useAuth()
  const { pathname } = useLocation()
  if (user === undefined) return null // still asking the server
  if (user === null) return <Navigate to={`/login?next=${encodeURIComponent(pathname)}`} replace />
  return <Outlet />
}
