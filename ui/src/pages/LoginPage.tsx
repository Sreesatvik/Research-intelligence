import { useEffect, useState, type FormEvent } from 'react'
import { Navigate, useSearchParams } from 'react-router-dom'
import { providers } from '../api'
import { useAuth } from '../auth/authContext'
import { useWorld } from '../world/worldContext'
import './auth.css'

// Only same-site paths are allowed: ?next=https://evil.example must never redirect off-site.
function safeNext(value: string | null): string {
  return value && value.startsWith('/') && !value.startsWith('//') && !value.startsWith('/login') ? value : '/chat'
}

export default function LoginPage() {
  const { user, signIn, signUp } = useAuth()
  const { setMode } = useWorld()
  const [params] = useSearchParams()
  const next = safeNext(params.get('next'))
  const [creating, setCreating] = useState(false)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(() =>
    params.get('error') === 'google' ? 'Google sign-in did not complete. Please try again.' : null,
  )
  const [googleOn, setGoogleOn] = useState(false)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    setMode('home')
  }, [setMode])

  // The Google button only appears when the server has Google credentials configured.
  useEffect(() => {
    providers()
      .then((available) => setGoogleOn(available.google))
      .catch(() => setGoogleOn(false))
  }, [])

  if (user) return <Navigate to={next} replace />

  async function submit(event: FormEvent) {
    event.preventDefault()
    setBusy(true)
    setError(null)
    try {
      await (creating ? signUp : signIn)(email, password)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Something went wrong')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="auth">
      <form className="panel auth__panel" onSubmit={submit}>
        <h1 className="auth__title">{creating ? 'Create account' : 'Sign in'}</h1>
        <p className="auth__sub">
          {creating ? 'Choose a password of at least 10 characters.' : 'Sign in to ask questions and run research.'}
        </p>

        {googleOn && (
          <>
            <a className="auth__google" href={`/api/auth/google/start?next=${encodeURIComponent(next)}`}>
              Continue with Google
            </a>
            <p className="auth__or">or use email</p>
          </>
        )}

        <label className="auth__field">
          <span>Email</span>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            autoComplete="email"
            required
            autoFocus
          />
        </label>
        <label className="auth__field">
          <span>Password</span>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete={creating ? 'new-password' : 'current-password'}
            minLength={creating ? 10 : undefined}
            required
          />
        </label>

        {error && <p className="notice notice--error" role="alert">{error}</p>}

        <button className="auth__submit" type="submit" disabled={busy || !email || !password}>
          {busy ? 'One moment…' : creating ? 'Create account' : 'Sign in'}
        </button>
        <button
          type="button"
          className="link-button auth__switch"
          onClick={() => {
            setCreating((value) => !value)
            setError(null)
          }}
        >
          {creating ? 'I already have an account' : 'Create an account'}
        </button>
      </form>
    </div>
  )
}
