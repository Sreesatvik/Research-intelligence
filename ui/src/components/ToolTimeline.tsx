import { useGSAP } from '@gsap/react'
import gsap from 'gsap'
import { useRef } from 'react'
import type { ToolCallInfo } from '../api'

function describeArguments(raw: string): string {
  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>
    return Object.entries(parsed)
      .map(([key, value]) => `${key}: ${typeof value === 'string' ? value : JSON.stringify(value)}`)
      .join('  ·  ')
  } catch {
    return raw
  }
}

// Each row animates itself on mount, so only rows that arrive during polling animate in.
function TimelineItem({ call, index }: { call: ToolCallInfo; index: number }) {
  const ref = useRef<HTMLLIElement>(null)

  useGSAP(
    () => {
      if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
      gsap.from(ref.current, {
        x: -24,
        opacity: 0,
        duration: 0.45,
        ease: 'power2.out',
        delay: Math.min(index, 5) * 0.08,
        clearProps: 'transform,opacity',
      })
    },
    { scope: ref },
  )

  return (
    <li ref={ref} className={`timeline-item${call.is_error ? ' timeline-item--error' : ''}`}>
      <span className="timeline-item__step">{call.step}</span>
      <div className="timeline-item__body">
        <strong>{call.tool}</strong>
        {call.is_error && <span className="badge badge--error">error</span>}
        <div className="timeline-item__args">{describeArguments(call.arguments)}</div>
      </div>
    </li>
  )
}

export default function ToolTimeline({ calls }: { calls: ToolCallInfo[] }) {
  if (calls.length === 0) return null
  return (
    <ol className="timeline" aria-label="Agent tool calls">
      {calls.map((call, i) => (
        <TimelineItem key={`${call.step}-${call.tool}-${i}`} call={call} index={i} />
      ))}
    </ol>
  )
}
