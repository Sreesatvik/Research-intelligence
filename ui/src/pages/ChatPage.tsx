import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useLayoutEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { ApiError, createChat, getChat, listDocuments, sendChatMessage, streamAnswer, type ChatMessage, type ChatMode } from '../api'
import Answer from '../components/Answer'
import ToolTimeline from '../components/ToolTimeline'
import { useLitDocuments } from '../state/litContext'
import './chat.css'

const POLL_MS = 1500

// While an answer is still arriving, citations may be in the model's own bracket style; show them as [n].
const liveText = (text: string) => text.replace(/\u3010\s*(\d+)[^\u3011]*\u3011/g, '[$1]')
const MODES: { id: ChatMode; label: string }[] = [
  { id: 'ask', label: 'Ask' },
  { id: 'research', label: 'Research' },
]

function Sources({ message }: { message: ChatMessage }) {
  const sources = message.meta.sources ?? []
  const { setIds } = useLitDocuments()
  const navigate = useNavigate()
  if (sources.length === 0) return null
  const documents = [...new Set(sources.map((s) => s.document_id))]
  return (
    <details className="msg__sources">
      <summary>
        Sources ({sources.length})
      </summary>
      <ol>
        {sources.map((source, i) => (
          <li key={source.chunk_id}>
            <span className="msg__source-id">
              [{i + 1}] {source.document_id}
            </span>
            <span className="msg__source-text">{source.text}</span>
          </li>
        ))}
      </ol>
      <button
        type="button"
        className="link-button"
        onClick={() => {
          setIds(documents)
          navigate('/map')
        }}
      >
        Show on the map
      </button>
    </details>
  )
}

function Steps({ message }: { message: ChatMessage }) {
  const calls = message.meta.tool_calls ?? []
  if (calls.length === 0) return null
  if (message.status === 'running') return <ToolTimeline calls={calls} />
  return (
    <details className="msg__sources">
      <summary>Steps ({calls.length})</summary>
      <ToolTimeline calls={calls} />
    </details>
  )
}

function Bubble({ message }: { message: ChatMessage }) {
  const [copied, setCopied] = useState(false)
  if (message.role === 'user') {
    return (
      <div className="msg msg--user">
        <p>{message.content}</p>
      </div>
    )
  }
  if (message.status === 'running') {
    return (
      <div className="msg msg--bot">
        <Steps message={message} />
        <p className="msg__status">
          Researching<span className="dots" aria-hidden="true" />
        </p>
      </div>
    )
  }
  if (message.status === 'failed') {
    return (
      <div className="msg msg--bot">
        <p className="notice notice--error">{message.content}</p>
      </div>
    )
  }
  const { cache_hit: cacheHit, tokens_used: tokens } = message.meta
  return (
    <div className="msg msg--bot">
      <Steps message={message} />
      <Answer text={message.content} />
      {message.meta.standalone && <p className="msg__fact">Searched for: {message.meta.standalone}</p>}
      <Sources message={message} />
      <div className="msg__tools">
        <button
          type="button"
          className="msg__tool"
          onClick={() => {
            navigator.clipboard
              ?.writeText(message.content)
              .then(() => {
                setCopied(true)
                setTimeout(() => setCopied(false), 1500)
              })
              .catch(() => undefined)
          }}
        >
          {copied ? 'Copied' : 'Copy'}
        </button>
        {cacheHit && cacheHit !== 'none' && <span className="msg__fact">cached</span>}
        {!!tokens && <span className="msg__fact">{tokens} tokens</span>}
      </div>
    </div>
  )
}

export default function ChatPage() {
  const { chatId } = useParams()
  const id = chatId ? Number(chatId) : null
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [draft, setDraft] = useState('')
  const [mode, setMode] = useState<ChatMode>('ask')
  const [pending, setPending] = useState<string | null>(null)
  const [live, setLive] = useState<{ text: string; searching: string | null } | null>(null)
  const [sentMode, setSentMode] = useState<ChatMode>('ask')
  const abort = useRef<AbortController | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [shownChat, setShownChat] = useState(chatId)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const endRef = useRef<HTMLDivElement>(null)

  const chat = useQuery({
    queryKey: ['chat', chatId],
    queryFn: () => getChat(id as number),
    enabled: id !== null && Number.isFinite(id),
    refetchInterval: (query) => (query.state.data?.messages.some((m) => m.status === 'running') ? POLL_MS : false),
    retry: false,
  })
  const library = useQuery({ queryKey: ['library'], queryFn: listDocuments })

  if (shownChat !== chatId) {
    setShownChat(chatId) // switching chats clears the previous chat's error
    setError(null)
  }

  const messages = chat.data?.messages ?? []
  const running = messages.some((m) => m.status === 'running')
  const loadedId = chat.data?.id
  const stepCount = messages.at(-1)?.meta.tool_calls?.length
  const busy = pending !== null || running
  const streaming = live !== null
  const liveLength = live?.text.length ?? 0

  // When a research run finishes, the sidebar title and token counts may have changed.
  useEffect(() => {
    if (!running && loadedId !== undefined) queryClient.invalidateQueries({ queryKey: ['chats'] })
  }, [running, loadedId, queryClient])

  const send = useMutation({
    mutationFn: async ({ text, sendMode }: { text: string; sendMode: ChatMode }) => {
      const target = id ?? (await createChat()).id
      if (sendMode === 'research') return sendChatMessage(target, text, sendMode)
      const controller = new AbortController()
      abort.current = controller
      try {
        return await streamAnswer(
          target,
          text,
          {
            onSearching: (question) => setLive((current) => current && { ...current, searching: question }),
            onDelta: (piece) => setLive((current) => current && { ...current, text: current.text + piece }),
          },
          controller.signal,
        )
      } catch (e) {
        if (controller.signal.aborted) return getChat(target) // stopped on purpose: show what was saved
        throw e
      }
    },
    onSuccess: (detail) => {
      queryClient.setQueryData(['chat', String(detail.id)], detail)
      queryClient.invalidateQueries({ queryKey: ['chats'] })
      if (id === null) navigate(`/chat/${detail.id}`, { replace: true })
    },
    onError: (e, { text }) => {
      setError(e instanceof ApiError || e instanceof Error ? e.message : 'Something went wrong')
      setDraft(text)
    },
    onSettled: () => {
      abort.current = null
      setPending(null)
      setLive(null)
    },
  })

  function submit(event?: FormEvent) {
    event?.preventDefault()
    const text = draft.trim()
    if (!text || busy) return
    setError(null)
    setPending(text)
    setSentMode(mode)
    setLive(mode === 'ask' ? { text: '', searching: null } : null)
    setDraft('')
    send.mutate({ text, sendMode: mode })
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault()
      submit()
    }
  }

  useLayoutEffect(() => {
    const input = inputRef.current
    if (!input) return
    input.style.height = 'auto'
    input.style.height = `${Math.min(input.scrollHeight, 200)}px`
  }, [draft])

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'end', behavior: liveLength > 0 ? 'auto' : 'smooth' })
  }, [messages.length, pending, running, stepCount, liveLength])

  useEffect(() => {
    inputRef.current?.focus()
  }, [chatId])

  const empty = id === null && pending === null
  const noDocuments = library.data?.length === 0
  const missing = id !== null && chat.error instanceof ApiError && chat.error.status === 404

  const composer = (
    <form className="composer" onSubmit={submit}>
      <textarea
        ref={inputRef}
        className="composer__input"
        value={draft}
        rows={1}
        placeholder={mode === 'ask' ? 'Ask about your documents' : 'Describe what to research'}
        aria-label="Message"
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={onKeyDown}
        maxLength={4000}
      />
      <div className="composer__bar">
        <div className="composer__modes" role="radiogroup" aria-label="Mode">
          {MODES.map((option) => (
            <button
              key={option.id}
              type="button"
              role="radio"
              aria-checked={mode === option.id}
              className={`composer__mode${mode === option.id ? ' composer__mode--on' : ''}`}
              onClick={() => setMode(option.id)}
            >
              {option.label}
            </button>
          ))}
        </div>
        {streaming ? (
          <button type="button" className="composer__send" aria-label="Stop" onClick={() => abort.current?.abort()}>
            <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
              <rect x="5" y="5" width="10" height="10" rx="2" fill="currentColor" />
            </svg>
          </button>
        ) : (
          <button type="submit" className="composer__send" disabled={!draft.trim() || busy} aria-label="Send">
            <svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true">
              <path d="M10 16V4m0 0-5 5m5-5 5 5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        )}
      </div>
    </form>
  )

  if (missing) {
    return (
      <div className="chat chat--empty">
        <h1 className="chat__heading">That chat does not exist</h1>
        <Link to="/chat" className="link-button">
          Start a new chat
        </Link>
      </div>
    )
  }

  if (empty) {
    return (
      <div className="chat chat--empty">
        <h1 className="chat__heading">{noDocuments ? 'Add documents to begin' : 'What do you want to know?'}</h1>
        {noDocuments ? (
          <Link to="/library" className="chat__cta">
            Open Library
          </Link>
        ) : (
          composer
        )}
        {error && <p className="notice notice--error" role="alert">{error}</p>}
      </div>
    )
  }

  return (
    <div className="chat">
      <div className="chat__scroll">
        <div className="chat__thread">
          {messages.map((message) => (
            <Bubble key={message.id} message={message} />
          ))}
          {pending !== null && (
            <>
              <div className="msg msg--user">
                <p>{pending}</p>
              </div>
              <div className="msg msg--bot">
                {live && live.text ? (
                  <div className="msg__live">
                    <Answer text={liveText(live.text)} />
                    <span className="caret" aria-hidden="true" />
                  </div>
                ) : (
                  <p className="msg__status">
                    {sentMode === 'research' ? 'Starting research' : live?.searching ? `Searching for: ${live.searching}` : 'Thinking'}
                    <span className="dots" aria-hidden="true" />
                  </p>
                )}
              </div>
            </>
          )}
          {error && <p className="notice notice--error" role="alert">{error}</p>}
          <div ref={endRef} />
        </div>
      </div>
      <div className="chat__dock">{composer}</div>
    </div>
  )
}
