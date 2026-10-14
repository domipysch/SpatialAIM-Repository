"""
Linked plot: a Plotly figure drawn client-side and linked by K to every other
instance in the same ``group``, so the K-sweep cards behave like one figure.

* hovering a point highlights the same K in every linked figure (browser only,
  no rerun)
* clicking a point emits that K to Python (one rerun), so the shared K slider can
  follow

The link is a per-group registry on ``window`` (CCv2 components share the page,
they are not iframes), which each instance joins on render and leaves on
unmount.

Contract with the figures (see ``render/ksweep.py``): every K-indexed trace
carries ``customdata`` whose first column is K; a trace's ``meta`` may be
"nullV" / "nullH" to mark the two null-crosshair lines, whose per-K coordinates
and axis span come from ``layout.meta.spatialaim`` = {nulls: {K: [x, y]},
ranges: {x, y}}.
"""

from __future__ import annotations

import streamlit as st

from spatialaim.gui.widgets.plotly_fit import FIT_JS

_LINKED_JS = """
const CDN = "https://esm.sh/plotly.js-dist-min@2.35.2"

const kOf = (cd) => {
  if (cd == null) return null
  const v = Array.isArray(cd) ? cd[0] : cd
  return v == null || isNaN(v) ? null : Number(v)
}

export default async function (component) {
  const { parentElement, data, key, setTriggerValue } = component

  let gd = parentElement.querySelector(".spatialaim-linked-gd")
  if (!gd) {
    gd = document.createElement("div")
    gd.className = "spatialaim-linked-gd"
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
  const group = String(data.group || "default")
  const spatialaim = (fig.layout && fig.layout.meta && fig.layout.meta.spatialaim) || {}

  await drawFitted(Plotly, gd, fig.data, fig.layout || {})

  // Per-trace K of every point, base marker sizes, and the null-line traces.
  const kPerTrace = fig.data.map((tr) =>
    (tr.customdata || []).map((cd) => kOf(cd))
  )
  const baseSize = fig.data.map((tr) => Number((tr.marker && tr.marker.size) || 7))
  const nullTrace = { nullV: -1, nullH: -1 }
  fig.data.forEach((tr, i) => {
    if (tr.meta === "nullV" || tr.meta === "nullH") nullTrace[tr.meta] = i
  })

  // Highlight the given K here (null clears): grow that point and give it a ring
  // in every K-indexed trace, and move the null crosshair to that K's null.
  const highlight = (k) => {
    const idx = [], sizes = [], widths = []
    kPerTrace.forEach((ks, i) => {
      if (!ks.length || fig.data[i].meta === "nullV" || fig.data[i].meta === "nullH")
        return
      idx.push(i)
      sizes.push(ks.map((v) => (k !== null && v === k ? baseSize[i] * 1.8 : baseSize[i])))
      widths.push(ks.map((v) => (k !== null && v === k ? 2 : 0)))
    })
    if (idx.length) {
      Plotly.restyle(gd, { "marker.size": sizes, "marker.line.width": widths }, idx)
    }

    const nulls = spatialaim.nulls || {}
    const ranges = spatialaim.ranges || {}
    const pair = k !== null ? nulls[String(k)] : null
    const place = (tag, axis, value) => {
      const i = nullTrace[tag]
      if (i < 0) return
      const span = ranges[axis === "x" ? "y" : "x"]
      if (value == null || !span) {
        Plotly.restyle(gd, { visible: false }, [i])
        return
      }
      const along = [value, value]
      const across = [span[0], span[1]]
      Plotly.restyle(
        gd,
        axis === "x"
          ? { x: [along], y: [across], visible: true }
          : { x: [across], y: [along], visible: true },
        [i]
      )
    }
    place("nullV", "x", pair ? pair[0] : null)
    place("nullH", "y", pair ? pair[1] : null)
  }

  // Join the group registry (shared across every instance on the page).
  if (!window.__aimLinked) window.__aimLinked = {}
  const registry = window.__aimLinked
  if (!registry[group]) registry[group] = new Map()
  const peers = registry[group]
  peers.set(key, highlight)

  const broadcast = (k) => peers.forEach((fn) => fn(k))

  // Re-point the (once-attached) handlers at this render's closures.
  gd.__aimLinkCtx = { broadcast, setTriggerValue }
  if (!gd.__aimLinkWired) {
    gd.__aimLinkWired = true
    gd.on("plotly_hover", (ev) => {
      const pt = ev.points && ev.points[0]
      const k = pt ? kOf(pt.customdata) : null
      if (k !== null) gd.__aimLinkCtx.broadcast(k)
    })
    gd.on("plotly_unhover", () => gd.__aimLinkCtx.broadcast(null))
    gd.on("plotly_click", (ev) => {
      const pt = ev.points && ev.points[0]
      const k = pt ? kOf(pt.customdata) : null
      if (k !== null) gd.__aimLinkCtx.setTriggerValue("picked_k", k)
    })
  }

  // Leave the group on unmount -- but only if this render's entry is still the
  // registered one, so a late cleanup cannot unregister a fresh re-render.
  return () => {
    if (peers.get(key) === highlight) peers.delete(key)
  }
}
""" + FIT_JS

_LINKED_PLOT = st.components.v2.component(
    "aim_linked_plot", js=_LINKED_JS, isolate_styles=False
)


def linked_plot(fig, key: str, *, group: str, on_pick=None) -> None:
    """Render ``fig`` linked to every other ``linked_plot`` in the same ``group``.

    Hovering a point highlights the same K in all of them without a rerun;
    clicking one emits that K as the ``picked_k`` trigger, which ``on_pick`` (a
    no-argument callback, run before the script body) can act on -- read the K
    from ``st.session_state[key]``.
    """
    _LINKED_PLOT(
        key=key,
        data={"figureJson": fig.to_json(), "group": group},
        on_picked_k_change=on_pick if on_pick is not None else lambda: None,
    )


def picked_k(key: str) -> int | None:
    """The K last clicked in the ``linked_plot`` mounted under ``key``, if any.

    Reads the component's ``picked_k`` trigger out of session state (which is a
    dict on some Streamlit versions and an attribute bag on others), so it works
    from inside the ``on_pick`` callback.
    """
    state = st.session_state.get(key)
    if state is None:
        return None
    value = getattr(state, "picked_k", None)
    if value is None and hasattr(state, "get"):
        value = state.get("picked_k")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
