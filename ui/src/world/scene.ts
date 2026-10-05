import * as THREE from 'three'
import { ACCENT_NODE, BG, LINE, clusterColor } from '../theme/palette'
import { slotsFor } from './layout'

export type WorldMode = 'home' | 'map' | 'search'

export interface WorldNode {
  id: string
  x: number
  y: number
  z: number
  cluster: number
}

export interface WorldScene {
  setMode: (mode: WorldMode) => void
  setScroll: (progress: number) => void // 0..1 across the home page
  setMatches: (ids: string[]) => void // search: papers that pull together into a constellation
  setSlotTop: (px: number) => void // search: the constellation starts below this screen y
  setHighlight: (ids: string[]) => void // map: retrieved papers shown in the accent colour
  setClusterFilter: (cluster: number | null) => void // map: the selected cluster gathers into a ring
  setHoverListener: (listener: (id: string | null) => void) => void
  setSelectListener: (listener: (cluster: number) => void) => void // a paper was clicked in the map
  dispose: () => void
}

const SHELL_RADIUS = 1.5
const PAPER_SIZE = 0.2 // height of one sheet at rest, in world units
const MAX_MATCHES = 5
const MAX_CHAIN = 16 // longest chain of hairlines: a selected cluster plus its closing link
const REST_Z = 3.4
const SEARCH_Z = 3.0
const TURN = Math.PI * 2
const RING_SEGMENTS = 160

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))

// The printed face of a sheet: a title block, author line, two text columns and a figure.
// It is drawn in greys so each cluster's tint colours the paper and the print together.
function makePaperTexture(maxAnisotropy: number): THREE.CanvasTexture {
  const width = 256
  const height = 362 // A4 proportions
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')!
  let seed = 7
  const random = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) // fixed seed: the same print every load

  ctx.fillStyle = '#f7f8f9'
  ctx.fillRect(0, 0, width, height)
  ctx.fillStyle = '#2b3138'
  ctx.fillRect(24, 30, width - 96, 11)
  ctx.fillRect(24, 48, width - 150, 11)
  ctx.fillStyle = '#8a939b'
  ctx.fillRect(24, 72, 96, 5)

  const columnWidth = (width - 48 - 12) / 2
  for (let column = 0; column < 2; column++) {
    const x = 24 + column * (columnWidth + 12)
    ctx.fillStyle = '#aab2b9'
    for (let y = 98; y < height - 28; y += 9) {
      ctx.fillRect(x, y, columnWidth * (0.7 + random() * 0.3), 3)
    }
  }
  ctx.fillStyle = '#d3dae0' // a figure block in the left column
  ctx.fillRect(24, 178, columnWidth, 62)
  ctx.fillStyle = '#8a939b'
  ;[18, 34, 26, 44].forEach((bar, i) => ctx.fillRect(34 + i * 18, 232 - bar, 11, bar))

  const texture = new THREE.CanvasTexture(canvas)
  texture.colorSpace = THREE.SRGBColorSpace
  texture.anisotropy = maxAnisotropy
  return texture
}

// A thin circle of line, used for the sphere's guide rings.
function makeRing(radius: number, material: THREE.Material, rotX: number, rotZ: number): THREE.LineLoop {
  const points = Array.from({ length: RING_SEGMENTS }, (_, i) => {
    const angle = (i / RING_SEGMENTS) * TURN
    return new THREE.Vector3(Math.cos(angle) * radius, Math.sin(angle) * radius, 0)
  })
  const ring = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(points), material)
  ring.rotation.set(rotX, 0, rotZ)
  return ring
}

export function createWorldScene(container: HTMLElement, nodes: WorldNode[], reducedMotion: boolean): WorldScene {
  const scene = new THREE.Scene()
  const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100)
  camera.position.z = REST_Z

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.domElement.style.display = 'block'
  container.appendChild(renderer.domElement)

  // Soft, shadowless light. The matte material has no emission, so nothing can glow.
  scene.add(new THREE.HemisphereLight(0xffffff, 0x2a4560, 1.5))
  const key = new THREE.DirectionalLight(0xffffff, 1.3)
  key.position.set(-2, 3, 4)
  scene.add(key)

  const group = new THREE.Group()
  scene.add(group)

  // ---- Papers: matte sheets of paper, one InstancedMesh ----
  // Each paper keeps the direction it has in the embedding map, projected onto a clean sphere shell.
  const count = nodes.length
  const base = nodes.map((n) => {
    const direction = new THREE.Vector3(n.x, n.y, n.z)
    if (direction.lengthSq() < 1e-8) direction.set(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5)
    return direction.normalize().multiplyScalar(SHELL_RADIUS)
  })
  const baseColor = nodes.map((n) => new THREE.Color(clusterColor(n.cluster)))
  const background = new THREE.Color(BG)
  const accent = new THREE.Color(ACCENT_NODE)

  const pos = base.map((v) => v.clone())
  const tPos = base.map((v) => v.clone())
  const col = baseColor.map((c) => c.clone())
  const tCol = baseColor.map((c) => c.clone())
  const scl = new Float32Array(count).fill(1)
  const tScl = new Float32Array(count).fill(1)
  const dissolve = new Float32Array(count)
  const tDissolve = new Float32Array(count)
  const phase = Float32Array.from({ length: count }, () => Math.random())

  // Sheets face outward from the sphere (like pages on a globe); matches turn to face the camera.
  const look = new THREE.Object3D()
  function facing(position: THREE.Vector3): THREE.Quaternion {
    look.position.copy(position)
    look.lookAt(position.x * 2, position.y * 2, position.z * 2)
    return look.quaternion.clone()
  }
  const quat = base.map((v) => facing(v))
  const tQuat = quat.map((q) => q.clone())

  const paperTexture = makePaperTexture(renderer.capabilities.getMaxAnisotropy())
  const mesh = new THREE.InstancedMesh(
    new THREE.BoxGeometry(0.707, 1, 0.025), // A4 proportions, paper-thin
    new THREE.MeshStandardMaterial({ map: paperTexture, roughness: 0.9, metalness: 0 }),
    count,
  )
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
  col.forEach((c, i) => mesh.setColorAt(i, c))
  group.add(mesh)

  // ---- Guide rings: two thin circles that outline the shell ----
  const ringMaterial = new THREE.LineBasicMaterial({ color: LINE, transparent: true, opacity: 0.14, depthWrite: false })
  group.add(makeRing(SHELL_RADIUS, ringMaterial, Math.PI / 2, 0), makeRing(SHELL_RADIUS, ringMaterial, Math.PI / 3, Math.PI / 5))

  // The constellation chain that joins the matching papers in search.
  const chainAttribute = new THREE.BufferAttribute(new Float32Array(MAX_CHAIN * 6), 3)
  chainAttribute.setUsage(THREE.DynamicDrawUsage)
  const chainGeometry = new THREE.BufferGeometry()
  chainGeometry.setAttribute('position', chainAttribute)
  chainGeometry.setDrawRange(0, 0)
  const chainMaterial = new THREE.LineBasicMaterial({ color: LINE, transparent: true, opacity: 0, depthWrite: false })
  const chain = new THREE.LineSegments(chainGeometry, chainMaterial)
  chain.frustumCulled = false
  group.add(chain)

  // ---- State ----
  let mode: WorldMode = 'home'
  let matches: number[] = [] // node indices, strongest first
  let highlight = new Set<string>()
  let clusterFilter: number | null = null
  let hoverIndex = -1
  let hoverListener: ((id: string | null) => void) | null = null
  let selectListener: ((cluster: number) => void) | null = null
  let slotTop = 0 // screen y (px) above which the constellation must not reach
  let selectionActive = false // a cluster is gathered into a ring in the map
  let chainList: number[] = [] // node indices joined by the hairline chain
  let chainClosed = false
  const indexById = new Map(nodes.map((n, i) => [n.id, i]))

  const pointer = { x: 0, y: 0, tx: 0, ty: 0 }
  const scroll = { value: 0, target: 0 }
  const view = { yaw: 0, pitch: 0, vYaw: 0, vPitch: 0, distance: 3 }
  const drag = { active: false, x: 0, y: 0, startX: 0, startY: 0 }
  const ease = (rate: number) => (reducedMotion ? 1 : rate)

  // Slots are computed from a camera parked at its search position, so they stay put while the camera moves.
  const slotCamera = camera.clone()
  function slotPositions(total: number): THREE.Vector3[] {
    const width = container.clientWidth
    const height = container.clientHeight
    slotCamera.aspect = width / height
    slotCamera.position.set(0, 0, SEARCH_Z)
    slotCamera.updateProjectionMatrix()
    slotCamera.updateMatrixWorld(true)
    return slotsFor(total, width, height, slotTop).map((slot) => {
      const direction = new THREE.Vector3((slot.x / width) * 2 - 1, -(slot.y / height) * 2 + 1, 0.5)
        .unproject(slotCamera)
        .sub(slotCamera.position)
        .normalize()
      return slotCamera.position.clone().addScaledVector(direction, -slotCamera.position.z / direction.z)
    })
  }

  // The selected cluster sits on a ring facing the camera, right of centre on wide screens (the legend is on the left).
  function ringSlots(total: number): THREE.Vector3[] {
    const wide = container.clientWidth / container.clientHeight > 1.15
    const centreX = wide ? 0.55 : 0
    const radius = total <= 1 ? 0 : clamp(0.28 + total * 0.06, 0.5, 1.05)
    return Array.from({ length: total }, (_, k) => {
      const angle = Math.PI / 2 + (k / total) * TURN
      return new THREE.Vector3(centreX + Math.cos(angle) * radius, Math.sin(angle) * radius, 0)
    })
  }

  // Work out where every node, its size and its colour should be heading for the current state.
  function retarget() {
    const searching = mode === 'search'
    const members =
      mode === 'map' && clusterFilter !== null
        ? nodes.map((n, i) => (n.cluster === clusterFilter ? i : -1)).filter((i) => i >= 0).slice(0, MAX_CHAIN - 1)
        : []
    selectionActive = members.length > 0
    const gathered = searching ? matches : members // the papers that pull together
    const gathering = gathered.length > 0
    const slots = !gathering ? [] : searching ? slotPositions(gathered.length) : ringSlots(gathered.length)
    const slotOf = new Map(gathered.map((nodeIndex, k) => [nodeIndex, k]))
    chainList = gathering ? gathered : []
    chainClosed = selectionActive && gathered.length > 2

    nodes.forEach((node, i) => {
      tPos[i].copy(base[i])
      tCol[i].copy(baseColor[i])
      tScl[i] = 1
      tDissolve[i] = 0
      tQuat[i].copy(facing(base[i]))

      if (gathering) {
        const slot = slotOf.get(i)
        if (slot !== undefined && slots[slot]) {
          tPos[i].copy(slots[slot])
          tQuat[i].identity()
          if (searching) {
            tCol[i].copy(accent)
            tScl[i] = 1.4
          } else {
            tScl[i] = 1.3
            if (highlight.has(node.id)) tCol[i].copy(accent)
          }
        } else {
          // Everything else drifts outward and thins into quiet noise.
          tPos[i].multiplyScalar(1.45)
          tCol[i].lerp(background, 0.72)
          tScl[i] = 0.6
          tDissolve[i] = 1
        }
      } else if (mode === 'map' && highlight.has(node.id)) {
        tCol[i].copy(accent)
        tScl[i] = 1.6
      }
      if (mode === 'map' && i === hoverIndex) tScl[i] *= 1.35
    })
    if (reducedMotion) settle()
  }

  function settle() {
    nodes.forEach((_, i) => {
      pos[i].copy(tPos[i])
      col[i].copy(tCol[i])
      scl[i] = tScl[i]
      dissolve[i] = tDissolve[i]
      quat[i].copy(tQuat[i])
    })
  }

  const dummy = new THREE.Object3D()
  function step(time: number, dt: number) {
    pointer.x += (pointer.tx - pointer.x) * ease(0.05)
    pointer.y += (pointer.ty - pointer.y) * ease(0.05)
    scroll.value += (scroll.target - scroll.value) * ease(0.1)

    // View: slow turn at rest, drag inertia in the map, faces the camera in search.
    const faceCamera = mode === 'search' || selectionActive
    if (!faceCamera) {
      if (!reducedMotion && !drag.active) view.yaw += dt * (mode === 'map' ? 0.03 : 0.05)
      if (!drag.active) {
        view.yaw += view.vYaw
        view.pitch = clamp(view.pitch + view.vPitch, -1.2, 1.2)
        view.vYaw *= 0.94
        view.vPitch *= 0.94
      }
    }
    const home = mode === 'home'
    const targetYaw =
      faceCamera
        ? TURN * Math.round(group.rotation.y / TURN)
        : view.yaw + (home ? pointer.x * 0.35 + scroll.value * Math.PI * 0.35 : 0)
    const targetPitch = faceCamera ? 0 : view.pitch + (home ? pointer.y * 0.2 + scroll.value * 0.15 : 0)
    group.rotation.y += (targetYaw - group.rotation.y) * ease(0.08)
    group.rotation.x += (targetPitch - group.rotation.x) * ease(0.08)
    group.position.y += ((home ? scroll.value * 0.5 : 0) - group.position.y) * ease(0.08)
    const targetZ = mode === 'map' ? view.distance : mode === 'search' ? SEARCH_Z : REST_Z - scroll.value * 0.4
    camera.position.z += (targetZ - camera.position.z) * ease(0.06)

    // Nodes.
    for (let i = 0; i < count; i++) {
      pos[i].lerp(tPos[i], ease(0.08))
      col[i].lerp(tCol[i], ease(0.1))
      scl[i] += (tScl[i] - scl[i]) * ease(0.1)
      dissolve[i] += (tDissolve[i] - dissolve[i]) * ease(0.06)
      quat[i].slerp(tQuat[i], ease(0.08))
      const drift = dissolve[i] * 0.1
      dummy.position.set(
        pos[i].x + drift * Math.sin(time * 0.7 + phase[i] * TURN),
        pos[i].y + drift * Math.cos(time * 0.6 + phase[i] * 5.1),
        pos[i].z + drift * Math.sin(time * 0.5 + phase[i] * 3.7),
      )
      dummy.quaternion.copy(quat[i])
      dummy.scale.setScalar(PAPER_SIZE * scl[i])
      dummy.updateMatrix()
      mesh.setMatrixAt(i, dummy.matrix)
      mesh.setColorAt(i, col[i])
    }
    mesh.instanceMatrix.needsUpdate = true
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true

    // Hairlines: the chain follows its nodes while they fly into place; the guide rings step back.
    const total = chainList.length
    const links = total < 2 ? 0 : chainClosed ? total : total - 1
    for (let k = 0; k < links; k++) {
      pos[chainList[k]].toArray(chainAttribute.array as Float32Array, k * 6)
      pos[chainList[(k + 1) % total]].toArray(chainAttribute.array as Float32Array, k * 6 + 3)
    }
    chainAttribute.needsUpdate = true
    chainGeometry.setDrawRange(0, links * 2)
    chainMaterial.opacity += ((links > 0 ? 0.55 : 0) - chainMaterial.opacity) * ease(0.08)
    ringMaterial.opacity += ((faceCamera ? 0 : 0.14) - ringMaterial.opacity) * ease(0.08)
  }

  function resize() {
    const width = container.clientWidth
    const height = container.clientHeight
    renderer.setSize(width, height)
    camera.aspect = width / height
    camera.updateProjectionMatrix()
    if (mode === 'search') retarget()
  }
  resize()
  const observer = new ResizeObserver(resize)
  observer.observe(container)

  // ---- Input ----
  const raycaster = new THREE.Raycaster()
  const ndc = new THREE.Vector2()
  function onWindowMove(event: PointerEvent) {
    pointer.tx = (event.clientX / window.innerWidth) * 2 - 1
    pointer.ty = (event.clientY / window.innerHeight) * 2 - 1
    if (drag.active) {
      const dx = (event.clientX - drag.x) * 0.006
      const dy = (event.clientY - drag.y) * 0.006
      view.yaw += dx
      view.pitch = clamp(view.pitch + dy, -1.2, 1.2)
      view.vYaw = dx * 0.5
      view.vPitch = dy * 0.5
      drag.x = event.clientX
      drag.y = event.clientY
    }
  }
  function pickAt(clientX: number, clientY: number): number {
    const rect = container.getBoundingClientRect()
    ndc.set(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1)
    raycaster.setFromCamera(ndc, camera)
    return raycaster.intersectObject(mesh)[0]?.instanceId ?? -1
  }
  function onContainerMove(event: PointerEvent) {
    if (mode !== 'map' || drag.active || count === 0) return
    const index = pickAt(event.clientX, event.clientY)
    if (index !== hoverIndex) {
      hoverIndex = index
      retarget()
      hoverListener?.(index >= 0 ? nodes[index].id : null)
    }
  }
  function onContainerLeave() {
    if (hoverIndex !== -1) {
      hoverIndex = -1
      retarget()
      hoverListener?.(null)
    }
  }
  function onDown(event: PointerEvent) {
    if (mode !== 'map') return
    drag.active = true
    drag.x = drag.startX = event.clientX
    drag.y = drag.startY = event.clientY
    container.style.cursor = 'grabbing'
  }
  function onUp(event: PointerEvent) {
    if (!drag.active) return
    drag.active = false
    container.style.cursor = ''
    // Barely moved: that was a click, not a drag. Select the cluster of the paper under the pointer.
    if (Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) < 5) {
      const index = pickAt(event.clientX, event.clientY)
      if (index >= 0) selectListener?.(nodes[index].cluster)
    }
  }
  function onWheel(event: WheelEvent) {
    if (mode !== 'map') return
    event.preventDefault()
    view.distance = clamp(view.distance + event.deltaY * 0.0015, 1.4, 5)
  }
  window.addEventListener('pointermove', onWindowMove)
  window.addEventListener('pointerup', onUp)
  container.addEventListener('pointermove', onContainerMove)
  container.addEventListener('pointerleave', onContainerLeave)
  container.addEventListener('pointerdown', onDown)
  container.addEventListener('wheel', onWheel, { passive: false })

  const timer = new THREE.Timer()
  timer.connect(document)
  let frame = 0
  function animate() {
    timer.update()
    step(reducedMotion ? 0 : timer.getElapsed(), reducedMotion ? 0 : timer.getDelta())
    renderer.render(scene, camera)
    frame = requestAnimationFrame(animate)
  }
  animate()

  return {
    setMode(next) {
      mode = next
      hoverIndex = -1
      if (next === 'map') {
        view.distance = 3
        if (!reducedMotion) camera.position.z = Math.max(camera.position.z, 6.5) // fly in from afar
      }
      retarget()
    },
    setScroll(progress) {
      scroll.target = progress
    },
    setMatches(ids) {
      matches = ids.map((id) => indexById.get(id)).filter((i): i is number => i !== undefined).slice(0, MAX_MATCHES)
      retarget()
    },
    setSlotTop(px) {
      if (px === slotTop) return
      slotTop = px
      if (mode === 'search') retarget()
    },
    setHighlight(ids) {
      highlight = new Set(ids)
      retarget()
    },
    setClusterFilter(cluster) {
      clusterFilter = cluster
      retarget()
    },
    setHoverListener(listener) {
      hoverListener = listener
    },
    setSelectListener(listener) {
      selectListener = listener
    },
    dispose() {
      cancelAnimationFrame(frame)
      timer.dispose()
      observer.disconnect()
      window.removeEventListener('pointermove', onWindowMove)
      window.removeEventListener('pointerup', onUp)
      container.removeEventListener('pointermove', onContainerMove)
      container.removeEventListener('pointerleave', onContainerLeave)
      container.removeEventListener('pointerdown', onDown)
      container.removeEventListener('wheel', onWheel)
      mesh.geometry.dispose()
      paperTexture.dispose()
      ;(mesh.material as THREE.Material).dispose()
      mesh.dispose()
      group.children.forEach((child) => {
        if (child instanceof THREE.LineLoop) child.geometry.dispose()
      })
      ringMaterial.dispose()
      chainGeometry.dispose()
      chainMaterial.dispose()
      renderer.dispose()
      renderer.domElement.remove()
    },
  }
}
