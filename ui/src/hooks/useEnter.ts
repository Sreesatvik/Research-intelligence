import { useGSAP } from '@gsap/react'
import { gsap } from 'gsap'
import type { RefObject } from 'react'

function enter(ref: RefObject<HTMLElement | null>, from: gsap.TweenVars, index: number) {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
  gsap.from(ref.current, {
    ...from,
    opacity: 0,
    duration: 0.7,
    ease: 'power3.out',
    delay: Math.min(index, 8) * 0.07,
    clearProps: 'transform,opacity',
  })
}

// Rise into place like a sheet of paper when mounted.
export function useRiseIn(ref: RefObject<HTMLElement | null>, index: number, distance = 56) {
  useGSAP(() => enter(ref, { y: distance, rotate: 1.2 }, index), { scope: ref })
}

// Slide in from the right, i.e. from the 3D node the element belongs to.
export function useSlideIn(ref: RefObject<HTMLElement | null>, index: number, distance = 72) {
  useGSAP(() => enter(ref, { x: distance }, index), { scope: ref })
}
