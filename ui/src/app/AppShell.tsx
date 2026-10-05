import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { Link, NavLink, Outlet, useNavigate, useParams } from 'react-router-dom'
import { deleteChat, listChats } from '../api'
import { useAuth } from '../auth/authContext'
import { useWorld } from '../world/worldContext'
import './shell.css'

// The signed-in workspace: a sidebar with the saved chats, and the page (chat or library) beside it.
export default function AppShell() {
  const { user, signOut } = useAuth()
  const { setMode } = useWorld()
  const navigate = useNavigate()
  const { chatId } = useParams()
  const queryClient = useQueryClient()
  const [open, setOpen] = useState(false)

  useEffect(() => setMode('home'), [setMode])

  const chats = useQuery({ queryKey: ['chats'], queryFn: listChats })
  const remove = useMutation({
    mutationFn: (id: number) => deleteChat(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: ['chats'] })
      queryClient.removeQueries({ queryKey: ['chat', String(id)] })
      if (String(id) === chatId) navigate('/chat', { replace: true })
    },
  })

  return (
    <div className="shell">
      <button type="button" className="shell__menu" aria-label="Open menu" onClick={() => setOpen(true)}>
        <span />
        <span />
        <span />
      </button>
      {open && <div className="shell__scrim" onClick={() => setOpen(false)} aria-hidden="true" />}

      <aside
        className={`shell__side${open ? ' shell__side--open' : ''}`}
        aria-label="Navigation"
        onClick={(event) => {
          if ((event.target as HTMLElement).closest('a')) setOpen(false)
        }}
      >
        <Link to="/" className="shell__brand">
          Research Intelligence
        </Link>

        <nav className="shell__nav">
          <NavLink to="/chat" end className={({ isActive }) => `shell__item${isActive && !chatId ? ' shell__item--active' : ''}`}>
            <span className="shell__icon" aria-hidden="true">＋</span>
            New chat
          </NavLink>
          <NavLink to="/library" className={({ isActive }) => `shell__item${isActive ? ' shell__item--active' : ''}`}>
            <span className="shell__icon" aria-hidden="true">▤</span>
            Library
          </NavLink>
          <NavLink to="/map" className="shell__item">
            <span className="shell__icon" aria-hidden="true">◎</span>
            Corpus map
          </NavLink>
        </nav>

        <h2 className="shell__label">Recents</h2>
        <ul className="shell__chats">
          {chats.data?.map((chat) => (
            <li key={chat.id} className="shell__chat">
              <NavLink
                to={`/chat/${chat.id}`}
                className={({ isActive }) => `shell__chat-link${isActive ? ' shell__chat-link--active' : ''}`}
                title={chat.title}
              >
                {chat.title}
              </NavLink>
              <button
                type="button"
                className="shell__chat-delete"
                aria-label={`Delete chat ${chat.title}`}
                onClick={() => remove.mutate(chat.id)}
              >
                ×
              </button>
            </li>
          ))}
          {chats.data?.length === 0 && <li className="shell__empty">No chats yet</li>}
        </ul>

        <div className="shell__account">
          <span className="shell__avatar" aria-hidden="true">{user?.email.slice(0, 2).toUpperCase()}</span>
          <span className="shell__email" title={user?.email}>{user?.email}</span>
          <button type="button" className="shell__signout" onClick={signOut}>
            Sign out
          </button>
        </div>
      </aside>

      <main className="shell__main">
        <Outlet />
      </main>
    </div>
  )
}
