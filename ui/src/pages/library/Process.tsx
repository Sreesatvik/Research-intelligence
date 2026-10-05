import { useQueryClient } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import { ApiError, uploadDocument } from '../../api'
import { CountUp, Heading, Sheet } from './parts'
import { useStageIn } from './useStageIn'
import { formatSize, sleep } from './validate'

export interface Outcome {
  name: string
  state: 'wait' | 'reading' | 'added' | 'duplicate' | 'failed'
  message?: string
  passages?: number
}

interface Props {
  files: File[]
  onFinished: (outcomes: Outcome[]) => void
}

const TONE = { added: 'ok', duplicate: 'dup', failed: 'bad' } as const

export default function Process({ files, onFinished }: Props) {
  const root = useRef<HTMLDivElement>(null)
  const queryClient = useQueryClient()
  const started = useRef(false)
  const finished = useRef(onFinished)
  const [outcomes, setOutcomes] = useState<Outcome[]>(() => files.map((file) => ({ name: file.name, state: 'wait' })))
  useStageIn(root)

  useEffect(() => {
    finished.current = onFinished
  }, [onFinished])

  useEffect(() => {
    if (started.current) return // React runs effects twice in development: never upload twice
    started.current = true
    const results: Outcome[] = files.map((file) => ({ name: file.name, state: 'wait' }))
    const set = (index: number, patch: Partial<Outcome>) => {
      results[index] = { ...results[index], ...patch }
      setOutcomes([...results])
    }

    void (async () => {
      await sleep(900)
      for (const [index, file] of files.entries()) {
        set(index, { state: 'reading' })
        const started = Date.now()
        try {
          const doc = await uploadDocument(file)
          await sleep(Math.max(0, 900 - (Date.now() - started))) // long enough for the reading to be seen
          set(index, { state: 'added', passages: doc.chunks })
        } catch (e) {
          await sleep(Math.max(0, 900 - (Date.now() - started)))
          const duplicate = e instanceof ApiError && e.status === 409
          set(index, { state: duplicate ? 'duplicate' : 'failed', message: e instanceof Error ? e.message : 'Upload failed' })
        }
        queryClient.invalidateQueries({ queryKey: ['library'] })
        queryClient.invalidateQueries({ queryKey: ['corpus-map'] })
        await sleep(450)
      }
      await sleep(500)
      finished.current(results)
    })()
  }, [files, queryClient])

  const total = outcomes.length
  const complete = outcomes.filter((o) => o.state !== 'wait' && o.state !== 'reading').length

  return (
    <div className="stage" ref={root}>
      <header className="stage__head">
        <div>
          <Heading text="Reading" />
          <p className="stage__sub" data-in>
            {complete} of {total} finished
          </p>
        </div>
      </header>

      <div className="meter" data-in aria-hidden="true">
        <span style={{ transform: `scaleX(${total ? complete / total : 0})` }} />
      </div>

      <ul className="reads">
        {outcomes.map((outcome, i) => (
          <li key={`${outcome.name}-${i}`} className={`read read--${outcome.state}`} data-in>
            <Sheet scanning={outcome.state === 'reading'} tone={outcome.state in TONE ? TONE[outcome.state as keyof typeof TONE] : undefined} />
            <div className="read__body">
              <strong title={outcome.name}>{outcome.name}</strong>
              <span>
                {outcome.state === 'wait' && `Waiting · ${formatSize(files[i].size)}`}
                {outcome.state === 'reading' && 'Reading, splitting and indexing'}
                {outcome.state === 'added' && (
                  <>
                    Added · <CountUp to={outcome.passages ?? 0} /> passages
                  </>
                )}
                {(outcome.state === 'duplicate' || outcome.state === 'failed') && outcome.message}
              </span>
            </div>
          </li>
        ))}
      </ul>
    </div>
  )
}
