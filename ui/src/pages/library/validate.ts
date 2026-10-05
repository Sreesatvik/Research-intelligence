export const MAX_BYTES = 15 * 1024 * 1024
export const MAX_BATCH = 20
const EXTENSIONS = ['pdf', 'txt', 'md']

export const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))
export const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches

export function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

// Returns why the file is unusable, or null when it is fine. A PDF must really start like a PDF.
export async function checkFormat(file: File): Promise<string | null> {
  const extension = file.name.split('.').pop()?.toLowerCase() ?? ''
  if (!EXTENSIONS.includes(extension)) return 'Only PDF, TXT and Markdown files are supported'
  if (file.size === 0) return 'The file is empty'
  if (extension === 'pdf') {
    const head = new TextDecoder().decode(await file.slice(0, 5).arrayBuffer())
    if (head !== '%PDF-') return 'This is not a real PDF'
  }
  return null
}

export function checkSize(file: File): string | null {
  return file.size > MAX_BYTES ? `Larger than ${MAX_BYTES / (1024 * 1024)} MB` : null
}

// SHA-256 of the file bytes, the same fingerprint the server keeps. Null where the browser offers no crypto.
export async function fingerprint(file: File): Promise<string | null> {
  if (!crypto?.subtle) return null
  const digest = await crypto.subtle.digest('SHA-256', await file.arrayBuffer())
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, '0')).join('')
}

// Picking the same file twice in one selection is dropped before anything else happens.
export function dropRepeats(files: File[]): File[] {
  const seen = new Set<string>()
  return files.filter((file) => {
    const id = `${file.name}|${file.size}|${file.lastModified}`
    if (seen.has(id)) return false
    seen.add(id)
    return true
  })
}
