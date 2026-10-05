import { useMemo, useState, type ReactNode } from 'react'
import { LitContext } from './litContext'

export function LitDocumentsProvider({ children }: { children: ReactNode }) {
  const [ids, setIds] = useState<string[]>([])
  const value = useMemo(() => ({ ids, setIds }), [ids])
  return <LitContext.Provider value={value}>{children}</LitContext.Provider>
}
