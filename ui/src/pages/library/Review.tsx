import { useEffect, useRef, useState } from 'react'
import { checkDuplicates } from '../../api'
import { Heading, Sheet, Tick, type CheckState } from './parts'
import { useStageIn } from './useStageIn'
import { checkFormat, checkSize, fingerprint, formatSize, sleep } from './validate'

type Check = 'format' | 'size' | 'unique'
const LABELS: Record<Check, string> = { format: 'Format and content', size: 'Size', unique: 'Not already in your library' }
const ORDER: Check[] = ['format', 'size', 'unique']

interface Item {
  file: File
  checks: Record<Check, CheckState>
  reason?: string
}

interface Props {
  files: File[]
  onConfirm: (valid: File[]) => void
  onCancel: () => void
}

const fresh = (file: File): Item => ({ file, checks: { format: 'wait', size: 'wait', unique: 'wait' } })
const failed = (item: Item) => ORDER.some((check) => item.checks[check] === 'bad')
const settled = (item: Item) => ORDER.every((check) => item.checks[check] !== 'wait' && item.checks[check] !== 'run')

export default function Review({ files, onConfirm, onCancel }: Props) {
  const root = useRef<HTMLDivElement>(null)
  const [items, setItems] = useState<Item[]>(() => files.map(fresh))
  useStageIn(root)

  useEffect(() => {
    let cancelled = false
    const set = (index: number, check: Check, state: CheckState, reason?: string) => {
      if (cancelled) return
      setItems((current) =>
        current.map((item, i) => (i === index ? { ...item, checks: { ...item.checks, [check]: state }, reason: reason ?? item.reason } : item)),
      )
    }
    const seen = new Set<string>()

    async function inspect(file: File, index: number) {
      await sleep(900 + index * 150) // let the stage settle, then start the files one after another
      let reason = null as string | null
      set(index, 'format', 'run')
      await sleep(260)
      reason = await checkFormat(file)
      set(index, 'format', reason ? 'bad' : 'ok', reason ?? undefined)
      if (reason) return set(index, 'size', 'skip'), set(index, 'unique', 'skip')

      set(index, 'size', 'run')
      await sleep(220)
      reason = checkSize(file)
      set(index, 'size', reason ? 'bad' : 'ok', reason ?? undefined)
      if (reason) return set(index, 'unique', 'skip')

      set(index, 'unique', 'run')
      await sleep(260)
      const hash = await fingerprint(file)
      if (hash && seen.has(hash)) return set(index, 'unique', 'bad', 'Selected more than once')
      if (hash) seen.add(hash)
      try {
        const known = hash ? (await checkDuplicates([hash])).duplicates[hash] : undefined
        if (known) return set(index, 'unique', 'bad', `Already in your library as "${known.name}"`)
      } catch {
        // The server checks again when the file is uploaded, so a failed pre-check is not fatal.
      }
      set(index, 'unique', 'ok')
    }

    files.forEach((file, index) => void inspect(file, index))
    return () => {
      cancelled = true
    }
  }, [files])

  const done = items.every(settled)
  const valid = items.filter((item) => settled(item) && !failed(item))
  const rejected = items.filter(failed).length

  return (
    <div className="stage" ref={root}>
      <header className="stage__head">
        <div>
          <Heading text="Checking" />
          <p className="stage__sub" data-in>
            {done ? `${valid.length} ready${rejected ? ` · ${rejected} rejected` : ''}` : 'Looking at each file before anything is uploaded'}
          </p>
        </div>
      </header>

      <ul className="checks">
        {items.map((item, i) => {
          const working = ORDER.some((check) => item.checks[check] === 'run')
          const bad = failed(item)
          return (
            <li key={`${item.file.name}-${i}`} className={`check${bad ? ' check--bad' : ''}`} data-in>
              <Sheet scanning={working} tone={settled(item) ? (bad ? 'bad' : 'ok') : undefined} />
              <div className="check__body">
                <strong title={item.file.name}>{item.file.name}</strong>
                <span className="check__size">{formatSize(item.file.size)}</span>
                <ul className="check__steps">
                  {ORDER.map((check) => (
                    <li key={check} className={`check__step check__step--${item.checks[check]}`}>
                      <Tick state={item.checks[check]} />
                      {LABELS[check]}
                    </li>
                  ))}
                </ul>
                {bad && <span className="check__reason">{item.reason}</span>}
              </div>
            </li>
          )
        })}
      </ul>

      <div className={`actions${done ? ' actions--on' : ''}`}>
        <button type="button" className="btn btn--primary" disabled={!done || valid.length === 0} onClick={() => onConfirm(valid.map((v) => v.file))}>
          {valid.length === 1 ? 'Add 1 document' : `Add ${valid.length} documents`}
        </button>
        <button type="button" className="btn btn--ghost" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </div>
  )
}
