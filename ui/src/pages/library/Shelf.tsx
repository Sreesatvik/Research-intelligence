import { useGSAP } from '@gsap/react'
import { gsap } from 'gsap'
import { useRef } from 'react'
import type { DocumentInfo } from '../../api'
import { CountUp, Heading, Sheet } from './parts'
import { useStageIn } from './useStageIn'
import { reducedMotion } from './validate'

function Tile({ doc, index, onRemove }: { doc: DocumentInfo; index: number; onRemove: (id: string) => void }) {
  const tile = useRef<HTMLLIElement>(null)

  useGSAP(
    () => {
      if (!tile.current || reducedMotion()) return
      gsap.from(tile.current, {
        y: 40,
        rotate: index % 2 ? 2 : -2,
        opacity: 0,
        duration: 0.7,
        delay: 0.55 + Math.min(index, 10) * 0.05,
        ease: 'power3.out',
        clearProps: 'transform,opacity',
      })
    },
    { scope: tile },
  )

  function remove() {
    const el = tile.current
    if (!el || reducedMotion()) return onRemove(doc.id)
    gsap
      .timeline({ onComplete: () => onRemove(doc.id) })
      .to(el, { y: -24, rotate: 5, opacity: 0, scale: 0.9, duration: 0.4, ease: 'power3.in' })
  }

  return (
    <li ref={tile} className="tile">
      <Sheet />
      <strong className="tile__name" title={doc.name}>
        {doc.name}
      </strong>
      <span className="tile__meta">
        {doc.kind.toUpperCase()} · {doc.chunks} passages
      </span>
      <button type="button" className="tile__remove" aria-label={`Remove ${doc.name}`} onClick={remove}>
        Remove
      </button>
    </li>
  )
}

interface Props {
  docs: DocumentInfo[]
  loaded: boolean
  onAdd: () => void
  onRemove: (id: string) => void
}

export default function Shelf({ docs, loaded, onAdd, onRemove }: Props) {
  const root = useRef<HTMLDivElement>(null)
  useStageIn(root)
  const passages = docs.reduce((sum, d) => sum + d.chunks, 0)

  return (
    <div className="stage" ref={root}>
      <header className="stage__head">
        <div>
          <Heading text="Library" />
          {docs.length > 0 && (
            <p className="stage__sub" data-in>
              <CountUp to={docs.length} /> document{docs.length === 1 ? '' : 's'} · <CountUp to={passages} /> passages
            </p>
          )}
        </div>
        {docs.length > 0 && (
          <button type="button" className="btn btn--primary" onClick={onAdd} data-in>
            Add documents
          </button>
        )}
      </header>

      {loaded && docs.length === 0 && (
        <div className="empty" data-in>
          <div className="stack" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          <p>Your shelf is empty.</p>
          <button type="button" className="btn btn--primary" onClick={onAdd}>
            Add your first documents
          </button>
        </div>
      )}

      <ul className="shelf">
        {docs.map((doc, i) => (
          <Tile key={doc.id} doc={doc} index={i} onRemove={onRemove} />
        ))}
      </ul>
    </div>
  )
}
