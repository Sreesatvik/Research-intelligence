import { useGSAP } from '@gsap/react'
import { gsap } from 'gsap'
import type { RefObject } from 'react'
import { reducedMotion } from './validate'

// Everything marked data-in inside the stage settles into place right after the curtain lifts.
export function useStageIn(scope: RefObject<HTMLElement | null>, dependencies: unknown[] = []) {
  useGSAP(
    () => {
      if (reducedMotion()) return
      gsap.from('[data-in]', { y: 34, opacity: 0, duration: 0.75, stagger: 0.07, delay: 0.5, ease: 'power3.out', clearProps: 'transform,opacity' })
    },
    { scope, dependencies },
  )
}
