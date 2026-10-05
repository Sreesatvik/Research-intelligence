import { useGSAP } from '@gsap/react'
import { gsap } from 'gsap'
import { useRef } from 'react'
import { reducedMotion } from './validate'

const LINES = [100, 88, 100, 76, 94, 52]

// A paper sheet. `scanning` runs the reading bar; `tone` stamps the verdict on it.
export function Sheet({ scanning = false, tone }: { scanning?: boolean; tone?: 'ok' | 'bad' | 'dup' }) {
  return (
    <div className={`sheet${scanning ? ' sheet--scan' : ''}${tone ? ` sheet--${tone}` : ''}`} aria-hidden="true">
      {LINES.map((width, i) => (
        <span key={i} className="sheet__line" style={{ width: `${width}%`, animationDelay: `${i * 0.12}s` }} />
      ))}
      <span className="sheet__scan" />
      <span className="sheet__stamp">{tone === 'ok' ? '✓' : tone === 'dup' ? '=' : tone === 'bad' ? '✕' : ''}</span>
    </div>
  )
}

export type CheckState = 'wait' | 'run' | 'ok' | 'bad' | 'skip'

export function Tick({ state }: { state: CheckState }) {
  return (
    <span className={`tick tick--${state}`} aria-hidden="true">
      <svg viewBox="0 0 16 16" width="16" height="16">
        {state === 'ok' && <path d="M3 8.5 6.6 12 13 4.5" />}
        {state === 'bad' && <path d="M4 4l8 8M12 4l-8 8" />}
        {state === 'skip' && <path d="M4 8h8" />}
      </svg>
    </span>
  )
}

export function CountUp({ to, className }: { to: number; className?: string }) {
  const ref = useRef<HTMLSpanElement>(null)
  useGSAP(
    () => {
      const el = ref.current
      if (!el) return
      if (reducedMotion()) {
        el.textContent = String(to)
        return
      }
      const counter = { n: Number(el.textContent) || 0 }
      gsap.to(counter, {
        n: to,
        duration: 0.9,
        ease: 'power2.out',
        onUpdate: () => {
          el.textContent = String(Math.round(counter.n))
        },
      })
    },
    { dependencies: [to] },
  )
  return (
    <span ref={ref} className={className}>
      0
    </span>
  )
}

// Big heading whose letters rise out of a mask when the stage appears.
export function Heading({ text }: { text: string }) {
  const ref = useRef<HTMLHeadingElement>(null)
  useGSAP(
    () => {
      if (reducedMotion()) return
      gsap.from('.stage__char', { yPercent: 115, duration: 0.8, stagger: 0.035, delay: 0.4, ease: 'power4.out' })
    },
    { scope: ref },
  )
  return (
    <h1 className="stage__title" aria-label={text} ref={ref}>
      {[...text].map((char, i) => (
        <span className="stage__mask" key={i} aria-hidden="true">
          <span className="stage__char">{char === ' ' ? ' ' : char}</span>
        </span>
      ))}
    </h1>
  )
}
