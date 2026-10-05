import { useRef } from 'react'
import { CountUp, Heading } from './parts'
import { useStageIn } from './useStageIn'
import type { Outcome } from './Process'

interface Props {
  outcomes: Outcome[]
  onDone: () => void
  onMore: () => void
}

export default function Summary({ outcomes, onDone, onMore }: Props) {
  const root = useRef<HTMLDivElement>(null)
  useStageIn(root)
  const added = outcomes.filter((o) => o.state === 'added').length
  const duplicates = outcomes.filter((o) => o.state === 'duplicate').length
  const failed = outcomes.filter((o) => o.state === 'failed').length
  const problems = outcomes.filter((o) => o.state === 'duplicate' || o.state === 'failed')

  return (
    <div className="stage" ref={root}>
      <header className="stage__head">
        <Heading text="All done" />
      </header>

      <div className="numbers">
        <div className="number" data-in>
          <CountUp to={added} className="number__value number__value--ok" />
          <span>added</span>
        </div>
        <div className="number" data-in>
          <CountUp to={duplicates} className="number__value" />
          <span>duplicates skipped</span>
        </div>
        <div className="number" data-in>
          <CountUp to={failed} className="number__value" />
          <span>failed</span>
        </div>
      </div>

      {problems.length > 0 && (
        <ul className="notes">
          {problems.map((outcome, i) => (
            <li key={`${outcome.name}-${i}`} data-in>
              <strong>{outcome.name}</strong>
              <span>{outcome.message}</span>
            </li>
          ))}
        </ul>
      )}

      <div className="actions actions--on">
        <button type="button" className="btn btn--primary" onClick={onDone} data-in>
          Back to library
        </button>
        <button type="button" className="btn btn--ghost" onClick={onMore} data-in>
          Add more
        </button>
      </div>
    </div>
  )
}
