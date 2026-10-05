// All backend calls live here; components import these functions instead of calling fetch.

export interface SourceItem {
  chunk_id: string
  document_id: string
  score: number
  text: string
}

export type Route = 'retrieval' | 'rag' | 'agent' | 'overview'
export type CacheHit = 'none' | 'exact' | 'semantic'

export interface AskResponse {
  answer: string
  cited_chunk_ids: string[]
  refused_by_retrieval: boolean
  sources: SourceItem[]
  tokens_used: number
  cache_hit: CacheHit
  route: Route // "agent" means the question needs POST /research
}

export interface SearchResponse {
  results: SourceItem[]
}

export interface ResearchStarted {
  job_id: string
  status: string
}

export interface ToolCallInfo {
  step: number
  tool: string
  arguments: string // raw JSON text
  is_error: boolean
}

export type JobStatus = 'running' | 'done' | 'failed'

export interface ResearchStatus {
  job_id: string
  status: JobStatus
  answer: string | null
  stopped_because: string | null
  tool_calls: ToolCallInfo[]
  tokens_used: number
  error: string | null
}

// Carries the HTTP status so callers can tell e.g. a 429 (quota) from a 503.
// Fields are assigned by hand because tsconfig enables erasableSyntaxOnly.
export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

// FastAPI sends {"detail": "..."} or, for validation errors, {"detail": [{"msg": "..."}]}.
function detailToMessage(detail: unknown): string {
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail.map((d) => (d && typeof d.msg === 'string' ? d.msg : JSON.stringify(d))).join('; ')
  }
  return 'Unexpected error'
}

// The dev server proxies /api to the backend (see vite.config.ts).
const BASE_URL = '/api'

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      ...init,
      credentials: 'same-origin', // the httpOnly session cookie travels with every request
      headers: { 'Content-Type': 'application/json', ...init.headers },
    })
  } catch {
    throw new ApiError(0, 'Cannot reach the server. Is the backend running on port 8000?')
  }

  if (!response.ok) {
    if (response.status === 401) window.dispatchEvent(new Event('auth:expired')) // the session ended or never existed
    let message = `Request failed (${response.status})`
    try {
      const body = await response.json()
      message = detailToMessage(body.detail)
    } catch {
      // body was not JSON; keep the generic message
    }
    throw new ApiError(response.status, message)
  }
  return (await response.json()) as T
}

function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, { method: 'POST', body: JSON.stringify(body) })
}

export function health(): Promise<{ status: string }> {
  return request('/health')
}

export function search(query: string, k = 5): Promise<SearchResponse> {
  return post('/search', { query, k })
}

export function ask(question: string, k = 3): Promise<AskResponse> {
  return post('/ask', { question, k })
}

export function startResearch(question: string): Promise<ResearchStarted> {
  return post('/research', { question })
}

export function getResearch(jobId: string): Promise<ResearchStatus> {
  return request(`/research/${encodeURIComponent(jobId)}`)
}

export interface AuthUser {
  email: string
}

export function register(email: string, password: string): Promise<AuthUser> {
  return post('/auth/register', { email, password })
}

export function login(email: string, password: string): Promise<AuthUser> {
  return post('/auth/login', { email, password })
}

export function logout(): Promise<{ ok: boolean }> {
  return post('/auth/logout', {})
}

export function me(): Promise<AuthUser> {
  return request('/auth/me')
}

export function providers(): Promise<{ google: boolean }> {
  return request('/auth/providers')
}

export type ChatMode = 'ask' | 'research'
export type MessageStatus = 'done' | 'running' | 'failed'

export interface ChatSummary {
  id: number
  title: string
  updated_at: string
}

export interface MessageMeta {
  mode?: ChatMode
  route?: Route
  standalone?: string | null
  cache_hit?: CacheHit
  tokens_used?: number
  refused?: boolean
  sources?: SourceItem[]
  cited_chunk_ids?: string[]
  tool_calls?: ToolCallInfo[]
  stopped_because?: string
}

export interface ChatMessage {
  id: number
  role: 'user' | 'assistant'
  content: string
  status: MessageStatus
  meta: MessageMeta
  created_at: string
}

export interface ChatDetail extends ChatSummary {
  messages: ChatMessage[]
}

export interface StreamHandlers {
  onSearching: (question: string) => void
  onDelta: (text: string) => void
}

// Ask mode streams: the server sends events as the answer is written, and the last event is the saved chat.
export async function streamAnswer(
  id: number,
  content: string,
  handlers: StreamHandlers,
  signal: AbortSignal,
): Promise<ChatDetail> {
  let response: Response
  try {
    response = await fetch(`${BASE_URL}/chats/${id}/stream`, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content, mode: 'ask' }),
      signal,
    })
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') throw e
    throw new ApiError(0, 'Cannot reach the server. Is the backend running on port 8000?')
  }
  if (!response.ok || !response.body) {
    if (response.status === 401) window.dispatchEvent(new Event('auth:expired'))
    let message = `Request failed (${response.status})`
    try {
      message = detailToMessage((await response.json()).detail)
    } catch {
      // not JSON: keep the generic message
    }
    throw new ApiError(response.status, message)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let finished: ChatDetail | null = null
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let end: number
    while ((end = buffer.indexOf('\n\n')) >= 0) {
      const block = buffer.slice(0, end)
      buffer = buffer.slice(end + 2)
      const [eventLine, dataLine] = block.split('\n')
      const data = JSON.parse(dataLine.slice('data: '.length))
      const event = eventLine.slice('event: '.length)
      if (event === 'delta') handlers.onDelta(data.text)
      else if (event === 'searching') handlers.onSearching(data.question)
      else if (event === 'done') finished = data as ChatDetail
    }
  }
  if (!finished) throw new ApiError(0, 'The connection ended before the answer finished')
  return finished
}

export function listChats(): Promise<ChatSummary[]> {
  return request('/chats')
}

export function createChat(): Promise<ChatSummary> {
  return post('/chats', {})
}

export function getChat(id: number): Promise<ChatDetail> {
  return request(`/chats/${id}`)
}

export function deleteChat(id: number): Promise<{ ok: boolean }> {
  return request(`/chats/${id}`, { method: 'DELETE' })
}

export function sendChatMessage(id: number, content: string, mode: ChatMode): Promise<ChatDetail> {
  return post(`/chats/${id}/messages`, { content, mode })
}

export interface DocumentInfo {
  id: string
  name: string
  kind: string
  chunks: number
  created_at: string
}

export interface CorpusPoint {
  id: string
  title: string
  topic: string
  x: number
  y: number
  z: number
  cluster: number
}

export interface CorpusMap {
  points: CorpusPoint[]
  clusters: { id: number; keywords: string[]; size: number }[]
}

export function listDocuments(): Promise<DocumentInfo[]> {
  return request('/library/documents')
}

// The file travels as the raw request body, so the server needs no multipart parser.
export function uploadDocument(file: File): Promise<DocumentInfo> {
  return request(`/library/documents/${encodeURIComponent(file.name)}`, {
    method: 'PUT',
    body: file,
    headers: { 'Content-Type': 'application/octet-stream' },
  })
}

export function deleteDocument(id: string): Promise<{ ok: boolean }> {
  return request(`/library/documents/${encodeURIComponent(id)}`, { method: 'DELETE' })
}

export function getCorpusMap(): Promise<CorpusMap> {
  return request('/library/map')
}

export function checkDuplicates(hashes: string[]): Promise<{ duplicates: Record<string, { id: string; name: string }> }> {
  return post('/library/check', { hashes })
}
