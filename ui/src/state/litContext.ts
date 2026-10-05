import { createContext, useContext } from 'react'

// Papers to highlight on the Corpus map; written by the Ask page, read by the map.
export interface LitDocuments {
  ids: string[]
  setIds: (ids: string[]) => void
}

export const LitContext = createContext<LitDocuments | null>(null)

export function useLitDocuments(): LitDocuments {
  const context = useContext(LitContext)
  if (!context) throw new Error('useLitDocuments must be used inside <LitDocumentsProvider>')
  return context
}
