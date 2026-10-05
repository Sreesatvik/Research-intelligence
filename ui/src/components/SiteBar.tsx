import { useLocation } from 'react-router-dom'
import { useAuth } from '../auth/authContext'
import TransitionLink from './TransitionLink'

const LINKS = [
  { to: '/chat', label: 'Chat' },
  { to: '/library', label: 'Library' },
  { to: '/map', label: 'Corpus map' },
]

export default function SiteBar() {
  const { pathname } = useLocation()
  const { user, signOut } = useAuth()
  return (
    <header className="sitebar">
      <TransitionLink to="/" className="sitebar__brand">
        Research Intelligence
      </TransitionLink>
      <nav className="sitebar__nav" aria-label="Main">
        {LINKS.map((link) => (
          <TransitionLink
            key={link.to}
            to={link.to}
            className={`sitebar__link${pathname === link.to ? ' sitebar__link--active' : ''}`}
            aria-current={pathname === link.to ? 'page' : undefined}
          >
            {link.label}
          </TransitionLink>
        ))}
        {user ? (
          <>
            <span className="sitebar__user">{user.email}</span>
            <button type="button" className="sitebar__link sitebar__button" onClick={signOut}>
              Sign out
            </button>
          </>
        ) : user === null && pathname !== '/login' ? (
          <TransitionLink to="/login" className="sitebar__link">
            Sign in
          </TransitionLink>
        ) : null}
      </nav>
    </header>
  )
}
