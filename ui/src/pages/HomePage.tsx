import { useGSAP } from '@gsap/react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import Lenis from 'lenis'
import { useEffect, useLayoutEffect, useRef } from 'react'
import { useAuth } from '../auth/authContext'
import TransitionLink from '../components/TransitionLink'
import { useWorld } from '../world/worldContext'
import './home.css'

gsap.registerPlugin(useGSAP, ScrollTrigger)

const TITLE_LINES = ['Research', 'Intelligence']

const SECTIONS = [
  {
    index: '01',
    word: 'Chat',
    to: '/chat',
    copy: 'Ask your documents anything, or hand a bigger question to a research agent. Every conversation is saved.',
    points: ['Answers with numbered citations', 'Ask for quick answers or deep research', 'Every source shown'],
  },
  {
    index: '02',
    word: 'Library',
    to: '/library',
    copy: 'Your own documents, kept private to your account. Only these are ever searched.',
    points: ['Upload PDF, text and Markdown', 'Searched only for you', 'Remove a file and it is gone'],
  },
  {
    index: '03',
    word: 'Map',
    to: '/map',
    copy: 'Your library as a space you can fly through.',
    points: ['Every document in 3D, grouped by topic', 'Click a document to gather its cluster', 'Sources from your chats stand out'],
  },
]

export default function HomePage() {
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  const { setMode, setScroll } = useWorld()
  const { user, signOut } = useAuth()
  const rootRef = useRef<HTMLDivElement>(null)
  const titleRef = useRef<HTMLHeadingElement>(null)

  useEffect(() => {
    setMode('home')
    return () => setScroll(0)
  }, [setMode, setScroll])

  // Fit the title to the window by measuring: render at 100px, find the widest line, scale to the space.
  useLayoutEffect(() => {
    const title = titleRef.current
    if (!title) return
    const lines = Array.from(title.querySelectorAll<HTMLElement>('.home__line'))

    function fit() {
      if (!title || !title.isConnected) return
      const probe = 100
      title.style.fontSize = `${probe}px`
      const available = title.clientWidth
      let size = Infinity
      lines.forEach((line) => {
        const first = line.firstElementChild as HTMLElement | null
        const last = line.lastElementChild as HTMLElement | null
        if (!first || !last) return
        const textWidth = last.offsetLeft + last.offsetWidth - first.offsetLeft
        const indent = parseFloat(getComputedStyle(line).marginLeft) || 0
        size = Math.min(size, (probe * (available - indent)) / textWidth)
      })
      title.style.fontSize = `${Math.floor(Math.min(size * 0.99, window.innerHeight * 0.22))}px`
    }

    fit()
    window.addEventListener('resize', fit)
    document.fonts?.ready.then(fit)
    return () => window.removeEventListener('resize', fit)
  }, [])

  // Smooth scrolling on GSAP's ticker, so Lenis and ScrollTrigger share one clock.
  useEffect(() => {
    window.scrollTo(0, 0)
    if (reducedMotion) return
    const lenis = new Lenis()
    lenis.on('scroll', ScrollTrigger.update)
    const tick = (time: number) => lenis.raf(time * 1000)
    gsap.ticker.add(tick)
    gsap.ticker.lagSmoothing(0)
    return () => {
      gsap.ticker.remove(tick)
      gsap.ticker.lagSmoothing(500, 33)
      lenis.destroy()
    }
  }, [reducedMotion])

  useGSAP(
    () => {
      if (reducedMotion) return

      gsap
        .timeline({ defaults: { ease: 'power4.out' } })
        .from('.char', { yPercent: 115, rotate: 6, duration: 1.1, stagger: 0.035 }, 0.2)
        .from('.home__sub', { y: 28, opacity: 0, duration: 0.9 }, '-=0.5')
        .from('.home__link', { x: 60, opacity: 0, duration: 0.9, stagger: 0.1 }, '-=0.7')
        .from('.home__scroll', { opacity: 0, duration: 0.8 }, '-=0.3')

      // Lenis already smooths the scroll, so scrub is plain: extra easing would make the text trail the scene.
      const hero = { trigger: '.home__hero', start: 'top top', end: 'bottom top', scrub: true }
      gsap.to('.home__title', { yPercent: -12, ease: 'none', scrollTrigger: hero })
      gsap.to('.home__sub', { yPercent: -22, ease: 'none', scrollTrigger: hero })
      gsap.to('.home__nav', { yPercent: -10, ease: 'none', scrollTrigger: hero })

      ScrollTrigger.create({
        trigger: rootRef.current,
        start: 'top top',
        end: 'bottom bottom',
        onUpdate: (self) => setScroll(self.progress),
      })

      // A matte veil dims the 3D scene as the sections arrive, so the text reads clearly.
      gsap.fromTo(
        '.home__veil',
        { opacity: 0 },
        { opacity: 0.72, ease: 'none', scrollTrigger: { trigger: '.home__section', start: 'top bottom', end: 'top 25%', scrub: true } },
      )

      gsap.utils.toArray<HTMLElement>('.home__section').forEach((section) => {
        gsap.from(section.querySelectorAll('.home__reveal'), {
          y: 40,
          opacity: 0,
          duration: 0.9,
          stagger: 0.12,
          ease: 'power3.out',
          scrollTrigger: { trigger: section, start: 'top 70%', toggleActions: 'play none none reverse' },
        })
      })
    },
    { scope: rootRef },
  )

  return (
    <div className="home" ref={rootRef}>
      <div className="home__veil" aria-hidden="true" />
      <section className="home__hero">
        <div className="home__auth" data-leave>
          {user ? (
            <>
              <span>{user.email}</span>
              <button type="button" onClick={signOut}>Sign out</button>
            </>
          ) : user === null ? (
            <TransitionLink to="/login">Sign in</TransitionLink>
          ) : null}
        </div>
        <h1 className="home__title" aria-label="Research Intelligence" ref={titleRef} data-leave>
          {TITLE_LINES.map((line) => (
            <span className="home__line" key={line} aria-hidden="true">
              {[...line].map((char, i) => (
                <span className="char" key={i}>
                  {char}
                </span>
              ))}
            </span>
          ))}
        </h1>

        <div className="home__bottom">
          <p className="home__sub" data-leave>
            Ask questions about your own documents. Every answer shows its sources.
          </p>
          <nav className="home__nav" aria-label="Main" data-leave>
            <ul>
              {SECTIONS.map((section) => (
                <li key={section.to}>
                  <TransitionLink to={section.to} className="home__link">
                    <span className="home__link-index">{section.index}</span>
                    <span>{section.word === 'Map' ? 'Corpus map' : section.word}</span>
                    <span className="home__link-arrow" aria-hidden="true">→</span>
                  </TransitionLink>
                </li>
              ))}
            </ul>
          </nav>
        </div>

        <span className="home__scroll" aria-hidden="true">Scroll</span>
      </section>

      {SECTIONS.map((section) => (
        <section className="home__section" key={section.index}>
          <div className="home__meta home__reveal">
            <span className="home__index">{section.index}</span>
            <span className="home__of">/ {String(SECTIONS.length).padStart(2, '0')}</span>
          </div>
          <div className="home__grid">
            <h2 className="home__word">{section.word}</h2>
            <div className="home__detail">
              <p className="home__copy home__reveal">{section.copy}</p>
              <ul className="home__points home__reveal">
                {section.points.map((point) => (
                  <li key={point}>{point}</li>
                ))}
              </ul>
              <TransitionLink to={section.to} className="home__open home__reveal">
                <span>Open {section.word.toLowerCase()}</span>
                <span aria-hidden="true">→</span>
              </TransitionLink>
            </div>
          </div>
        </section>
      ))}
    </div>
  )
}
