import { createContext, useContext } from 'react'
import type { CorpusMap } from '../api'
import type { WorldMode } from './scene'

export type CorpusPoint = CorpusMap['points'][number]
export type CorpusData = CorpusMap

// The one persistent 3D world. Pages describe what state it should be in; they never own a canvas.
export interface World {
  data: CorpusData | null
  setMode: (mode: WorldMode) => void
  setScroll: (progress: number) => void
  setMatches: (ids: string[]) => void
  setSlotTop: (px: number) => void
  setHighlight: (ids: string[]) => void
  setClusterFilter: (cluster: number | null) => void
  subscribeHover: (listener: (id: string | null) => void) => () => void
  subscribeSelect: (listener: (cluster: number) => void) => () => void // a paper was clicked in the map
  transitionTo: (path: string) => void // animates the current page out, then navigates
}

export const WorldContext = createContext<World | null>(null)

export function useWorld(): World {
  const world = useContext(WorldContext)
  if (!world) throw new Error('useWorld must be used inside <WorldLayout>')
  return world
}
