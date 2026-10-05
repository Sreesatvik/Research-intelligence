import { useRef, useState, type DragEvent } from 'react'
import { Heading } from './parts'
import { useStageIn } from './useStageIn'

interface Props {
  onPick: (files: File[]) => void
  onBack: () => void
}

export default function Select({ onPick, onBack }: Props) {
  const root = useRef<HTMLDivElement>(null)
  const input = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)
  useStageIn(root)

  function drop(event: DragEvent) {
    event.preventDefault()
    setOver(false)
    if (event.dataTransfer.files.length) onPick(Array.from(event.dataTransfer.files))
  }

  return (
    <div className="stage" ref={root}>
      <header className="stage__head">
        <Heading text="Add documents" />
        <button type="button" className="btn btn--ghost" onClick={onBack} data-in>
          Back to library
        </button>
      </header>

      <div
        className={`drop${over ? ' drop--over' : ''}`}
        data-in
        onDragOver={(e) => {
          e.preventDefault()
          setOver(true)
        }}
        onDragLeave={() => setOver(false)}
        onDrop={drop}
      >
        <svg className="drop__frame" aria-hidden="true">
          <rect />
        </svg>
        <div className="stack" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
        <p className="drop__lead">{over ? 'Release to continue' : 'Drop files anywhere in this box'}</p>
        <p className="drop__hint">PDF, TXT or Markdown · up to 15 MB each</p>
        <button type="button" className="btn btn--primary" onClick={() => input.current?.click()}>
          Choose files
        </button>
        <input
          ref={input}
          type="file"
          accept=".pdf,.txt,.md"
          multiple
          hidden
          onChange={(e) => {
            if (e.target.files?.length) onPick(Array.from(e.target.files))
            e.target.value = ''
          }}
        />
      </div>
    </div>
  )
}
