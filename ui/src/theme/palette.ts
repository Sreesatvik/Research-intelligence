// One palette for the whole app, built on deep blues (coolors: 3E92CC, 2A628F, 13293D, 16324F, 18435A).
// Cream text and the pale paper tints are derived neutrals: the blues alone offer no light text colour.
export const BG = '#13293D'
export const LINE = '#7FA6C6' // guide rings and the constellation chain
export const ACCENT = '#5AA6DB' // text and rules (a tint of 3E92CC that clears 4.5:1 contrast on navy)
export const ACCENT_NODE = '#3E92CC' // active papers in the 3D scene

// Paper tints for the clusters: pale, matte stock that multiplies the printed texture.
export const CLUSTER_COLORS = ['#8EC1E6', '#D9CDB0', '#A5C3B2', '#B4C0CC', '#B7AECF', '#9CCFCB', '#E2BFA6', '#E9EEF2']

export function clusterColor(cluster: number): string {
  return CLUSTER_COLORS[cluster % CLUSTER_COLORS.length]
}
