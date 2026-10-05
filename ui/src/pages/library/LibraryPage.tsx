import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { gsap } from 'gsap'
import { useRef, useState } from 'react'
import { deleteDocument, listDocuments } from '../../api'
import Process, { type Outcome } from './Process'
import Review from './Review'
import Select from './Select'
import Shelf from './Shelf'
import Summary from './Summary'
import { dropRepeats, MAX_BATCH, reducedMotion } from './validate'
import './library.css'

type Stage = 'shelf' | 'select' | 'review' | 'process' | 'summary'

const CURTAIN_WORD: Record<Stage, string> = {
  shelf: 'Library',
  select: 'Add',
  review: 'Check',
  process: 'Read',
  summary: 'Done',
}

// One page, five stages. A curtain sweeps across the whole page between stages, and the new stage is swapped in
// while the page is fully covered.
export default function LibraryPage() {
  const queryClient = useQueryClient()
  const scroller = useRef<HTMLDivElement>(null)
  const curtain = useRef<HTMLDivElement>(null)
  const word = useRef<HTMLSpanElement>(null)
  const moving = useRef(false)
  const [stage, setStage] = useState<Stage>('shelf')
  const [picked, setPicked] = useState<File[]>([])
  const [accepted, setAccepted] = useState<File[]>([])
  const [outcomes, setOutcomes] = useState<Outcome[]>([])

  const documents = useQuery({ queryKey: ['library'], queryFn: listDocuments })
  const remove = useMutation({
    mutationFn: (id: string) => deleteDocument(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['library'] })
      queryClient.invalidateQueries({ queryKey: ['corpus-map'] })
    },
  })

  function go(next: Stage, apply?: () => void) {
    if (moving.current) return
    const swap = () => {
      apply?.()
      setStage(next)
      scroller.current?.scrollTo({ top: 0 })
    }
    if (reducedMotion() || !curtain.current || !word.current) return swap()
    moving.current = true
    word.current.textContent = CURTAIN_WORD[next]
    gsap
      .timeline({ onComplete: () => (moving.current = false) })
      .set(curtain.current, { pointerEvents: 'auto' })
      .fromTo(curtain.current, { clipPath: 'inset(0 100% 0 0)' }, { clipPath: 'inset(0 0% 0 0)', duration: 0.55, ease: 'power3.inOut' })
      .fromTo(word.current, { yPercent: 110, opacity: 0 }, { yPercent: 0, opacity: 1, duration: 0.5, ease: 'power4.out' }, '-=0.15')
      .add(swap)
      .to(word.current, { yPercent: -110, opacity: 0, duration: 0.4, ease: 'power3.in' }, '+=0.25')
      .to(curtain.current, { clipPath: 'inset(0 0 0 100%)', duration: 0.6, ease: 'power3.inOut' }, '-=0.15')
      .set(curtain.current, { pointerEvents: 'none' })
  }

  const docs = documents.data ?? []

  return (
    <div className="library">
      <div className="library__scroll" ref={scroller}>
        <div className="library__inner">
          {stage === 'shelf' && (
            <Shelf docs={docs} loaded={documents.isSuccess} onAdd={() => go('select')} onRemove={(id) => remove.mutate(id)} />
          )}
          {stage === 'select' && (
            <Select
              onBack={() => go('shelf')}
              onPick={(files) => {
                const chosen = dropRepeats(files).slice(0, MAX_BATCH)
                go('review', () => setPicked(chosen))
              }}
            />
          )}
          {stage === 'review' && (
            <Review
              files={picked}
              onCancel={() => go('shelf')}
              onConfirm={(valid) => go('process', () => setAccepted(valid))}
            />
          )}
          {stage === 'process' && (
            <Process files={accepted} onFinished={(results) => go('summary', () => setOutcomes(results))} />
          )}
          {stage === 'summary' && (
            <Summary outcomes={outcomes} onDone={() => go('shelf')} onMore={() => go('select')} />
          )}
        </div>
      </div>

      <div className="curtain" ref={curtain} aria-hidden="true">
        <span className="curtain__mask">
          <span className="curtain__word" ref={word} />
        </span>
      </div>
    </div>
  )
}
