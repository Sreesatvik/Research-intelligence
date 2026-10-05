import type { WorldNode } from './scene'

// Anonymous sheets that fill the world before a library has any documents. They carry no titles and are
// never shown on the map page, which stays empty until there is something real to plot.
const COUNT = 42
const CLUSTERS = 5

export const DECOR: WorldNode[] = Array.from({ length: COUNT }, (_, i) => {
  const y = 1 - (i / (COUNT - 1)) * 2
  const radius = Math.sqrt(1 - y * y)
  const angle = i * Math.PI * (3 - Math.sqrt(5))
  return {
    id: `decor-${i}`,
    x: Math.cos(angle) * radius * 0.9,
    y: y * 0.9,
    z: Math.sin(angle) * radius * 0.9,
    cluster: i % CLUSTERS,
  }
})
