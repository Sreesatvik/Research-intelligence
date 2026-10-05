// Where the matching papers sit on screen in search mode. Both the 3D scene (to place nodes)
// and the page (to place cards beside them) use this one function, so they always line up.

export interface Slot {
  x: number // CSS pixels from the left of the viewport
  y: number // CSS pixels from the top
}

export const isWide = (width: number, height: number) => width / height > 1.15

export const maxSlots = (height: number) => (height < 760 ? 4 : 5)

const CARD_PITCH = 92 // vertical space one card needs

// How many cards fit below the typing area without overlapping each other.
export function slotCapacity(height: number, minTop = 0): number {
  const bottom = 0.92 * height
  const top = Math.max(0.44 * height, Math.min(minTop, bottom - CARD_PITCH))
  return Math.max(1, Math.min(maxSlots(height), Math.floor((bottom - top) / CARD_PITCH) + 1))
}

// minTop keeps the slots below the typing area, however tall it grows.
export function slotsFor(count: number, width: number, height: number, minTop = 0): Slot[] {
  const n = Math.min(count, slotCapacity(height, minTop))
  const bottom = 0.92 * height
  const top = Math.max(0.44 * height, Math.min(minTop, bottom - CARD_PITCH * Math.max(n - 1, 0)))
  const centreX = isWide(width, height) ? 0.7 * width : 0.84 * width
  return Array.from({ length: n }, (_, k) => {
    const t = n === 1 ? 0.5 : k / (n - 1)
    return {
      x: centreX + Math.sin(Math.PI * t) * 0.035 * width, // a gentle arc
      y: n === 1 ? (top + bottom) / 2 : top + (bottom - top) * t,
    }
  })
}
