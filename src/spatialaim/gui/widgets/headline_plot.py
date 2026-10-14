"""
Headline plot: a Plotly figure drawn client-side (CCv2 + Plotly.js) with state
selection handled in the browser, without reruns.

* single click on a legend entry or any point -> toggle that state
* double click -> isolate that state (or, if already isolated, restore all)
* right click (with ``context_menu``) -> report the state to Python, which opens
  the cell-type modal

Inactive states are recoloured to one light grey rather than hidden, so
overlapping points never stack into odd shades. State traces are matched by
their ``legendgroup`` ("state<n>"); other traces (grey below-threshold,
continuous confidence) are left untouched. Marker sizes are rescaled on zoom, so
the dots can stay small for the dense spatial panel without vanishing when a
region is zoomed into. Plotly.js is loaded from a CDN (reused if the page already
exposes it); an offline page shows an inline error instead of the plot.
"""

from __future__ import annotations

import streamlit as st

from spatialaim.gui.widgets.plotly_fit import FIT_JS

_HEADLINE_JS = """
const CDN = "https://esm.sh/plotly.js-dist-min@2.35.2"

const stateOf = (tr) => {
  const lg = (tr && tr.legendgroup) || ""
  return lg.indexOf("state") === 0 ? parseInt(lg.slice(5), 10) : null
}

export default async function (component) {
  const { parentElement, data, key, setTriggerValue } = component

  let gd = parentElement.querySelector(".spatialaim-gd")
  if (!gd) {
    gd = document.createElement("div")
    gd.className = "spatialaim-gd"
    gd.style.width = "100%"
    parentElement.appendChild(gd)
  }

  let Plotly = window.Plotly || window.__aimPlotly
  if (!Plotly) {
    try {
      Plotly = (await import(CDN)).default
      window.__aimPlotly = Plotly
    } catch (e) {
      gd.textContent = "Could not load Plotly.js (needs internet): " + e
      return
    }
  }

  const fig = JSON.parse(data.figureJson)
  const offColor = data.offColor || "#dcdcdc"

  const baseColors = fig.data.map((tr) => (tr.marker && tr.marker.color) || null)
  const states = Array.from(
    new Set(fig.data.map(stateOf).filter((s) => s !== null))
  ).sort((a, b) => a - b)
  const sig = states.join(",")

  // Active-state set, kept on window per component key. Reset when the set of
  // states changes (e.g. K changed), preserved across same-K reruns.
  if (!window.__aimActive) window.__aimActive = {}
  const store = window.__aimActive
  if (!store[key] || store[key].sig !== sig) {
    store[key] = { sig, active: new Set(states) }
  }

  // Plotly.react merges the new layout over the previous one and leaves STALE
  // subplot domains when the panel count changes (e.g. toggling the shared-gene
  // UMAP flips 2<->3 panels), which overlaps the panels. Purge first whenever the
  // axis structure changes so react rebuilds from a clean layout; an unchanged
  // structure (e.g. a K drag or a threshold change) still updates smoothly in
  // place. Purge drops Plotly's event listeners, so re-wire them after.
  const axisSig = Object.keys(fig.layout || {})
    .filter((k) => /^xaxis/.test(k))
    .sort()
    .join(",")
  if (gd.__aimAxisSig && gd.__aimAxisSig !== axisSig) {
    Plotly.purge(gd)
    gd.__aimWired = false
  }
  gd.__aimAxisSig = axisSig

  await drawFitted(Plotly, gd, fig.data, fig.layout || {})

  const applyColors = () => {
    const active = store[key].active
    const idx = [], cols = []
    fig.data.forEach((tr, i) => {
      const s = stateOf(tr)
      if (s !== null && baseColors[i]) {
        idx.push(i)
        cols.push(active.has(s) ? baseColors[i] : offColor)
      }
    })
    if (idx.length) Plotly.restyle(gd, { "marker.color": cols }, idx)
  }

  // Marker sizes are in pixels, so zooming only spreads the points apart: a dot
  // small enough that the dense spatial panel is not one solid blob at full
  // extent stays a speck once zoomed in. Rescale each panel from its own zoom
  // factor instead -- sublinear and capped, so zooming in makes the dots a
  // little bigger, never blobs, and zooming out keeps the un-zoomed size.
  // Panels are equal-aspect (y scaleanchored to x), so the x span alone is a
  // faithful zoom measure.
  const ZOOM_EXP = 0.6   // 4x zoom -> ~2.3x dots
  const ZOOM_MIN = 1.0   // zooming out never shrinks below the base size
  const ZOOM_MAX = 4.0

  const axisOf = (tr) => (tr.xaxis || "x").replace("x", "xaxis")
  const baseSizes = fig.data.map((tr) =>
    tr.marker && typeof tr.marker.size === "number" ? tr.marker.size : null
  )
  // Un-zoomed span per axis, from the explicit ranges render/scatter.py pins.
  const baseSpans = {}
  fig.data.forEach((tr) => {
    const ax = axisOf(tr)
    const r = ((fig.layout || {})[ax] || {}).range
    if (r && baseSpans[ax] === undefined) baseSpans[ax] = Math.abs(r[1] - r[0])
  })

  // Current ranges come from Plotly's resolved layout: the relayout payload only
  // carries the keys that changed, and carries none on an autorange reset.
  let sizing = false, lastSig = ""
  const applySizes = () => {
    if (sizing) return
    const full = gd._fullLayout || {}
    const idx = [], sizes = []
    fig.data.forEach((tr, i) => {
      const base = baseSizes[i]
      const ax = axisOf(tr)
      const span0 = baseSpans[ax]
      const r = ((full[ax] || {}).range) || null
      if (base === null || !span0 || !r) return
      const span = Math.abs(r[1] - r[0])
      const f = span > 0
        ? Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, Math.pow(span0 / span, ZOOM_EXP)))
        : 1
      idx.push(i)
      sizes.push(base * f)
    })
    // Panning fires relayout without changing any span; skip the restyle then,
    // so dragging a large Scattergl panel stays smooth.
    const sig = sizes.join(",")
    if (!idx.length || sig === lastSig) return
    lastSig = sig
    sizing = true
    try {
      Plotly.restyle(gd, { "marker.size": sizes }, idx)
    } finally {
      sizing = false
    }
  }

  const toggle = (s) => {
    const a = store[key].active
    if (a.has(s)) a.delete(s)
    else a.add(s)
    applyColors()
  }
  const isolate = (s) => {
    const a = store[key].active
    if (a.size === 1 && a.has(s)) store[key].active = new Set(states)
    else store[key].active = new Set([s])
    applyColors()
  }

  applyColors()
  applySizes()

  // Refresh the live context read by the (once-attached) event handlers.
  gd.__aimCtx = {
    toggle,
    isolate,
    applySizes,
    fig,
    contextMenu: !!data.contextMenu,
    openMenu: (s) => setTriggerValue("menu_state", s),
  }
  if (!gd.__aimWired) {
    gd.__aimWired = true

    gd.on("plotly_relayout", () => gd.__aimCtx.applySizes())

    // A right click on a legend entry only opens the modal, it never toggles.
    // Plotly reports it as an ordinary legend click, either before the
    // contextmenu event (Windows) or after it (macOS), and a touchpad's
    // two-finger tap may not even flag the right button. So a right-button
    // click is ignored outright, a legend click just after a contextmenu is
    // skipped, and a toggle just before one is undone (see ``onMenu``).
    let legendToggle = null, skipLegendUntil = 0
    gd.on("plotly_legendclick", (ev) => {
      if (ev.event && ev.event.button === 2) return false
      if (Date.now() < skipLegendUntil) return false
      const s = stateOf(gd.__aimCtx.fig.data[ev.curveNumber])
      if (s !== null) {
        gd.__aimCtx.toggle(s)
        legendToggle = { s, t: Date.now() }
      }
      return false // prevent the native hide
    })
    const undoLegendToggle = () => {
      const now = Date.now()
      if (legendToggle && now - legendToggle.t < 400) gd.__aimCtx.toggle(legendToggle.s)
      legendToggle = null
      skipLegendUntil = now + 400
    }
    gd.on("plotly_legenddoubleclick", (ev) => {
      const s = stateOf(gd.__aimCtx.fig.data[ev.curveNumber])
      if (s !== null) gd.__aimCtx.isolate(s)
      return false
    })

    // Right click on a state's point or legend entry opens that cell type's
    // modal in Python (trigger "menu_state"); the browser's own context menu is
    // suppressed only then. Plotly has no right-click event, so the point under
    // the cursor is the last hovered one, and a legend entry is found from the
    // elements under the cursor (each legend item's datum carries its trace).
    // A touchpad's two-finger tap can send its mousedown with no pressed button
    // (`buttons === 0`), which Plotly takes for a left drag: it covers the whole
    // page with a ".dragcover" div (the contextmenu then targets that cover, not
    // this plot) and unhovers on the slightest finger jitter. So listen on the
    // document, locate the plot with elementsFromPoint (which sees through the
    // cover), and fall back to the state hovered when the button went down.
    let hovered = null, downHovered = null, lastMenu = 0
    gd.on("plotly_hover", (ev) => {
      const pt = ev.points && ev.points[0]
      hovered = pt ? stateOf(pt.data) : null
    })
    gd.on("plotly_unhover", () => { hovered = null })
    const onDown = () => { downHovered = hovered }
    const onMenu = (ev) => {
      if (!gd.isConnected) {
        document.removeEventListener("contextmenu", onMenu, true)
        return
      }
      const under = document
        .elementsFromPoint(ev.clientX, ev.clientY)
        .filter((el) => gd.contains(el))
      if (!under.length) return
      const item = under.map((el) => el.closest(".legend .traces")).find(Boolean)
      if (item) undoLegendToggle()
      if (!gd.__aimCtx.contextMenu) return
      const datum = item && item.__data__ && item.__data__[0]
      const s = datum && datum.trace
        ? stateOf(datum.trace)
        : (hovered !== null ? hovered : downHovered)
      if (s === null || s === undefined) return
      ev.preventDefault()
      // Plotly can re-dispatch the event; open the modal once per click.
      const now = Date.now()
      if (now - lastMenu < 400) return
      lastMenu = now
      gd.__aimCtx.openMenu(s)
    }
    // A purge re-wires; drop the previous wiring's native listeners first.
    if (gd.__aimMenuWiring) {
      gd.removeEventListener("pointerdown", gd.__aimMenuWiring.onDown, true)
      document.removeEventListener("contextmenu", gd.__aimMenuWiring.onMenu, true)
    }
    gd.__aimMenuWiring = { onDown, onMenu }
    gd.addEventListener("pointerdown", onDown, true)
    document.addEventListener("contextmenu", onMenu, true)

    // Points have no native double-click event, so time it ourselves.
    let timer = null, pending = null
    gd.on("plotly_click", (ev) => {
      if (ev.event && ev.event.button === 2) return // right click: the modal
      const pt = ev.points && ev.points[0]
      if (!pt) return
      const s = stateOf(pt.data)
      if (s === null) return
      if (timer && pending === s) {
        clearTimeout(timer); timer = null; pending = null
        gd.__aimCtx.isolate(s)
      } else {
        if (timer) { clearTimeout(timer); gd.__aimCtx.toggle(pending) }
        pending = s
        timer = setTimeout(() => {
          timer = null; pending = null
          gd.__aimCtx.toggle(s)
        }, 300)
      }
    })
  }
}
""" + FIT_JS

# Plotly renders more reliably in the light DOM (it injects global styles and
# hover nodes), so opt out of the shadow-root isolation for this component.
_HEADLINE_PLOT = st.components.v2.component(
    "aim_headline_plot", js=_HEADLINE_JS, isolate_styles=False
)


def headline_plot(
    fig, key: str, *, off_color: str = "#dcdcdc", context_menu: bool = False
) -> int | None:
    """Render ``fig`` (a Plotly figure) with client-side state toggle/isolate.

    ``off_color`` is the single light-gray colour used for every deactivated
    state. With ``context_menu`` a right click on a state's point or legend entry
    reruns the script and returns that state (else ``None``) -- the caller opens
    the cell-type modal for it.
    """
    result = _HEADLINE_PLOT(
        key=key,
        data={
            "figureJson": fig.to_json(),
            "offColor": off_color,
            "contextMenu": context_menu,
        },
        on_menu_state_change=lambda: None,
    )
    value = getattr(result, "menu_state", None)
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None
