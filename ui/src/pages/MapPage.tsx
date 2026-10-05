import { useGSAP } from '@gsap/react'
import { gsap } from 'gsap'
import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { useLitDocuments } from '../state/litContext'
import { clusterColor } from '../theme/palette'
import { useWorld } from '../world/worldContext'
import './map.css'

// The 3D world fills the whole page; this page only adds frosted panels floating over it.
export default function MapPage() {
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  const { data, setMode, setHighlight, setClusterFilter, subscribeHover, subscribeSelect } = useWorld()
  const { ids: litIds, setIds: setLitIds } = useLitDocuments()
  const rootRef = useRef<HTMLDivElement>(null)
  const [hovered, setHovered] = useState<string | null>(null)
  const [filter, setFilter] = useState<number | null>(null)

  const empty = data !== null && data.points.length === 0

  useEffect(() => {
    setMode(empty ? 'home' : 'map') // the map is only interactive when there is something real on it
    return () => {
      setHighlight([])
      setClusterFilter(null)
    }
  }, [empty, setMode, setHighlight, setClusterFilter])

  useEffect(() => setHighlight(litIds), [litIds, setHighlight])
  useEffect(() => setClusterFilter(filter), [filter, setClusterFilter])
  useEffect(() => subscribeHover(setHovered), [subscribeHover])
  useEffect(() => subscribeSelect((cluster) => setFilter((current) => (current === cluster ? null : cluster))), [subscribeSelect])

  useGSAP(
    () => {
      if (reducedMotion) return
      gsap.from('[data-enter]', { y: 30, opacity: 0, duration: 0.9, stagger: 0.1, ease: 'power3.out', delay: 0.3 })
    },
    { scope: rootRef },
  )

  const paper = hovered ? data?.points.find((p) => p.id === hovered) : undefined
  const selected = filter !== null ? data?.clusters.find((c) => c.id === filter) : undefined
  const matched = data ? data.points.filter((p) => litIds.includes(p.id)).length : 0

  return (
    <div className="map" ref={rootRef}>
      {empty && (
        <div className="panel map__lit" data-enter data-leave>
          <span>Your library is empty, so there is nothing to map yet.</span>
          <Link to="/library" className="link-button">
            Add documents
          </Link>
        </div>
      )}

      <aside className="panel map__legend" aria-label="Clusters" data-enter data-leave>
        <h2 className="map__heading">Clusters</h2>
        <ul className="map__list">
          {data?.clusters.map((cluster) => (
            <li key={cluster.id}>
              <button
                type="button"
                className={`map__cluster${filter === cluster.id ? ' map__cluster--active' : ''}`}
                aria-pressed={filter === cluster.id}
                onClick={() => setFilter(filter === cluster.id ? null : cluster.id)}
              >
                <span className="map__swatch" style={{ background: clusterColor(cluster.id) }} />
                <span className="map__words">{cluster.keywords.slice(0, 3).join(', ')}</span>
                <span className="map__count">{cluster.size}</span>
              </button>
            </li>
          ))}
        </ul>
        {filter !== null && (
          <button type="button" className="link-button" onClick={() => setFilter(null)}>
            Show all
          </button>
        )}
      </aside>

      <div className="panel map__info" aria-live="polite" data-enter data-leave>
        {paper ? (
          <>
            <strong className="map__title">{paper.title}</strong>
            <span className="map__meta">
              <code>{paper.id}</code> · {paper.topic}
            </span>
          </>
        ) : selected ? (
          <>
            <strong className="map__title">{selected.keywords.slice(0, 3).join(', ')}</strong>
            <span className="map__meta">{selected.size} documents · click a document or the cluster again to release</span>
          </>
        ) : (
          <span className="map__meta">Drag to rotate · scroll to zoom · click a document to gather its cluster</span>
        )}
      </div>

      {matched > 0 && (
        <div className="panel map__lit" data-enter data-leave>
          <span>
            {matched} document{matched === 1 ? '' : 's'} cited in your last answer {matched === 1 ? 'stands' : 'stand'} out.
          </span>
          <button type="button" className="link-button" onClick={() => setLitIds([])}>
            Clear
          </button>
        </div>
      )}
    </div>
  )
}
