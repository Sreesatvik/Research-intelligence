import { useQuery } from '@tanstack/react-query'
import { gsap } from 'gsap'
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Outlet, useLocation, useNavigate } from 'react-router-dom'
import { getCorpusMap } from '../api'
import { useAuth } from '../auth/authContext'
import SiteBar from '../components/SiteBar'
import { DECOR } from './decor'
import type { WorldMode, WorldScene } from './scene'
import { WorldContext, type CorpusData, type World } from './worldContext'
import './world.css'

const WORKSPACE = /^\/(chat|library)(\/|$)/ // these pages bring their own sidebar

// Owns the single full-viewport 3D canvas for every page, and the animated hand-off between pages.
export default function WorldLayout() {
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  const navigate = useNavigate()
  const { pathname } = useLocation()
  const containerRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<WorldScene | null>(null)
  const hoverListeners = useRef(new Set<(id: string | null) => void>())
  const selectListeners = useRef(new Set<(cluster: number) => void>())
  const leaving = useRef(false)
  // What the pages asked for, so it can be applied the moment the scene finishes loading.
  const wanted = useRef({ mode: 'home' as WorldMode, matches: [] as string[], slotTop: 0, highlight: [] as string[], filter: null as number | null })
  const { user } = useAuth()
  const [mode, setModeState] = useState<WorldMode>('home')
  const map = useQuery({ queryKey: ['corpus-map'], queryFn: getCorpusMap, enabled: !!user })
  const data: CorpusData | null = map.data ?? null
  // The world shows the signed-in user's own documents; anonymous sheets stand in until there are some.
  const points = useMemo(() => (map.data && map.data.points.length > 0 ? map.data.points : DECOR), [map.data])

  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    let cancelled = false
    let scene: WorldScene | null = null

    // Three.js loads as its own chunk, so the first paint never waits for it.
    import('./scene').then(({ createWorldScene }) => {
      if (cancelled) return
      scene = createWorldScene(container, points, reducedMotion)
      sceneRef.current = scene
      scene.setHoverListener((id) => hoverListeners.current.forEach((listener) => listener(id)))
      scene.setSelectListener((cluster) => selectListeners.current.forEach((listener) => listener(cluster)))
      const want = wanted.current
      scene.setMode(want.mode)
      scene.setHighlight(want.highlight)
      scene.setClusterFilter(want.filter)
      scene.setSlotTop(want.slotTop)
      scene.setMatches(want.matches)
      if (!reducedMotion) {
        // Explicit end value: the scene is rebuilt when documents load, and a plain `from` would treat a half-faded opacity as its target.
        gsap.fromTo(container, { opacity: 0 }, { opacity: 1, duration: 1.4, ease: 'power2.out', overwrite: true, clearProps: 'opacity' })
      }
    })

    return () => {
      cancelled = true
      gsap.killTweensOf(container)
      container.style.opacity = ''
      scene?.dispose()
      sceneRef.current = null
    }
  }, [reducedMotion, points])

  const setMode = useCallback((next: WorldMode) => {
    wanted.current.mode = next
    setModeState(next)
    sceneRef.current?.setMode(next)
  }, [])
  const setScroll = useCallback((progress: number) => sceneRef.current?.setScroll(progress), [])
  const setMatches = useCallback((ids: string[]) => {
    wanted.current.matches = ids
    sceneRef.current?.setMatches(ids)
  }, [])
  const setSlotTop = useCallback((px: number) => {
    wanted.current.slotTop = px
    sceneRef.current?.setSlotTop(px)
  }, [])
  const setHighlight = useCallback((ids: string[]) => {
    wanted.current.highlight = ids
    sceneRef.current?.setHighlight(ids)
  }, [])
  const setClusterFilter = useCallback((cluster: number | null) => {
    wanted.current.filter = cluster
    sceneRef.current?.setClusterFilter(cluster)
  }, [])
  const subscribeHover = useCallback((listener: (id: string | null) => void) => {
    hoverListeners.current.add(listener)
    return () => {
      hoverListeners.current.delete(listener)
    }
  }, [])

  const subscribeSelect = useCallback((listener: (cluster: number) => void) => {
    selectListeners.current.add(listener)
    return () => {
      selectListeners.current.delete(listener)
    }
  }, [])

  // Everything marked data-leave on the current page animates out, then the route changes.
  const transitionTo = useCallback(
    (path: string) => {
      if (leaving.current || path === pathname) return // already here: nothing to animate out of
      const items = document.querySelectorAll('[data-leave]')
      if (reducedMotion || items.length === 0) {
        navigate(path)
        return
      }
      leaving.current = true
      gsap.to(items, {
        opacity: 0,
        y: -36,
        duration: 0.45,
        stagger: 0.05,
        ease: 'power2.in',
        onComplete: () => {
          leaving.current = false
          navigate(path)
        },
      })
    },
    [navigate, pathname, reducedMotion],
  )

  const world = useMemo<World>(
    () => ({ data, setMode, setScroll, setMatches, setSlotTop, setHighlight, setClusterFilter, subscribeHover, subscribeSelect, transitionTo }),
    [data, setMode, setScroll, setMatches, setSlotTop, setHighlight, setClusterFilter, subscribeHover, subscribeSelect, transitionTo],
  )

  return (
    <WorldContext.Provider value={world}>
      <div className={`world__canvas${mode === 'map' ? ' world__canvas--interactive' : ''}`} ref={containerRef} aria-hidden="true" />
      {pathname !== '/' && !WORKSPACE.test(pathname) && <SiteBar />}
      <Suspense fallback={null}>
        <Outlet />
      </Suspense>
    </WorldContext.Provider>
  )
}
