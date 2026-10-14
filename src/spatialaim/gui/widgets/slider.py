"""
Live sliders: drop-in replacements for ``st.slider`` / ``st.select_slider`` that
emit their value continuously while dragging, not only on release.

Both are one inline Streamlit Custom Component v2: an ``<input type="range">``
whose ``input`` event (every drag step) calls ``setStateValue`` and so triggers a
rerun, which makes the plots follow the handle live.

Trade-off: every drag step is a rerun. The confidence-threshold slider lives in a
per-mapper ``st.fragment``, so only its tab reruns, and the UMAP panels come from
``render.scatter.umap_panel_data``'s cache, so a drag rebuilds just that tab's
spatial panel. The K slider is global (it drives every tab), so a K drag is a full
rerun that reloads that K's mapping -- heavier on large datasets.
"""

from __future__ import annotations

import streamlit as st

# Theme-aware styling that mimics Streamlit's native (BaseWeb) slider: a thin
# rounded track with a primary-colour fill up to a circular thumb, the current
# value floating above the thumb, and min/max labels at the ends. ``--st-*``
# variables are injected into the shadow root, so it follows the active theme.
_CSS = """
.wrap {
  width: 100%;
  box-sizing: border-box;
  color: var(--st-text-color, inherit);
  font-family: var(--st-font, sans-serif);
  /* Track/thumb geometry + theme colours in one place. */
  --track: var(--st-border-color, rgba(151, 166, 195, 0.35));
  --fill: var(--st-primary-color, #ff4b4b);
  --bar: 6px;     /* track thickness */
  --thumb: 16px;  /* thumb diameter  */
}
.label {
  font-size: 0.8rem;
  opacity: 0.6;
  margin-bottom: 0.25rem;
}
.track-area {
  position: relative;
  /* Room for the floating value bubble above the track, with breathing space. */
  padding-top: 1.7rem;
}
.bubble {
  position: absolute;
  top: 0;
  transform: translateX(-50%);
  font-size: 0.8rem;
  font-weight: 600;
  white-space: nowrap;
  color: var(--st-text-color, #31333f);
  font-variant-numeric: tabular-nums;
  pointer-events: none;
}
/* The input is as tall as the thumb so the thumb never clips; the visible thin
   track is drawn by the ::track pseudo-element, which the browser centres inside
   that full-height box. The fill percentage is driven by the ``--pct`` variable
   (0..100) that the JS updates on every drag step. */
input.range {
  --pct: 0;
  -webkit-appearance: none;
  appearance: none;
  display: block;
  width: 100%;
  height: var(--thumb);
  background: transparent;
  outline: none;
  margin: 0;
  padding: 0;
  cursor: pointer;
}
input.range::-webkit-slider-runnable-track {
  height: var(--bar);
  border-radius: 9999px;
  background: linear-gradient(
    90deg,
    var(--fill) calc(var(--pct) * 1%),
    var(--track) calc(var(--pct) * 1%)
  );
}
input.range::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: var(--thumb);
  height: var(--thumb);
  /* Centre the thumb on the (thinner) track. */
  margin-top: calc((var(--bar) - var(--thumb)) / 2);
  border: none;
  border-radius: 50%;
  background: var(--fill);
  cursor: pointer;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
  transition: box-shadow 0.15s ease;
}
input.range::-webkit-slider-thumb:hover {
  box-shadow: 0 0 0 6px color-mix(in srgb, var(--fill) 25%, transparent);
}
input.range:focus-visible::-webkit-slider-thumb {
  box-shadow: 0 0 0 6px color-mix(in srgb, var(--fill) 25%, transparent);
}
/* Firefox: the track + native progress fill are value-driven, no JS needed. */
input.range::-moz-range-track {
  height: var(--bar);
  border-radius: 9999px;
  background: var(--track);
}
input.range::-moz-range-progress {
  height: var(--bar);
  border-radius: 9999px;
  background: var(--fill);
}
input.range::-moz-range-thumb {
  width: var(--thumb);
  height: var(--thumb);
  border: none;
  border-radius: 50%;
  background: var(--fill);
  cursor: pointer;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
}
.ends {
  display: flex;
  justify-content: space-between;
  margin-top: 0.4rem;
  font-size: 0.72rem;
  opacity: 0.5;
  font-variant-numeric: tabular-nums;
}
"""

_HTML = """
<div class="wrap">
  <div class="label"></div>
  <div class="track-area">
    <div class="bubble"></div>
    <input class="range" type="range" />
    <div class="ends"><span class="end-min"></span><span class="end-max"></span></div>
  </div>
</div>
"""

_JS = """
const THUMB = 16
export default function (component) {
  const { parentElement, data, setStateValue } = component
  const label = parentElement.querySelector(".label")
  const bubble = parentElement.querySelector(".bubble")
  const input = parentElement.querySelector(".range")
  const endMin = parentElement.querySelector(".end-min")
  const endMax = parentElement.querySelector(".end-max")
  if (!input) return

  const d = data || {}
  const options = d.options || null
  const decimals = d.decimals ?? 2
  const prefix = d.prefix || ""
  const min = Number(d.min)
  const max = Number(d.max)

  const fmt = (v) => {
    if (options) {
      const i = Math.round(v)
      return options[i] != null ? String(options[i]) : String(v)
    }
    return prefix + Number(v).toFixed(decimals)
  }

  label.textContent = d.label || ""
  input.min = d.min
  input.max = d.max
  input.step = d.step
  endMin.textContent = fmt(min)
  endMax.textContent = fmt(max)

  // Drive the WebKit fill via --pct (the ::track gradient reads it) + position the
  // value bubble over the thumb. Firefox fills natively via ::-moz-range-progress.
  const paint = (v) => {
    const pct = max > min ? ((v - min) / (max - min)) * 100 : 0
    input.style.setProperty("--pct", String(pct))
    // Correct for thumb width so the bubble stays centred over the handle.
    const offset = (0.5 - pct / 100) * THUMB
    bubble.style.left = `calc(${pct}% + ${offset}px)`
    bubble.textContent = fmt(v)
  }

  // True sync: reflect Python's value, but don't fight an active drag.
  const nextValue = d.value
  if (nextValue != null && Number(input.value) !== Number(nextValue)) {
    input.value = nextValue
  }
  paint(Number(input.value))

  input.oninput = (e) => {
    const v = Number(e.target.value)
    paint(v)                    // instant local feedback
    setStateValue("value", v)   // -> rerun (live update)
  }
}
"""

_LIVE_SLIDER = st.components.v2.component(
    "aim_live_slider", html=_HTML, css=_CSS, js=_JS
)


def live_slider(
    label: str,
    min_value: float,
    max_value: float,
    step: float,
    value: float,
    key: str,
    *,
    decimals: int = 2,
    prefix: str = "",
) -> float:
    """A numeric slider that updates live while dragging. Returns the value."""
    state = st.session_state.get(key, {})
    try:
        cur = float(state.get("value", value))
    except (TypeError, ValueError):
        cur = float(value)
    cur = min(max(cur, min_value), max_value)

    result = _LIVE_SLIDER(
        key=key,
        data={
            "label": label,
            "min": min_value,
            "max": max_value,
            "step": step,
            "value": cur,
            "decimals": decimals,
            "prefix": prefix,
            "options": None,
        },
        default={"value": cur},
        on_value_change=lambda: None,
    )
    v = cur if result.value is None else float(result.value)
    return min(max(v, min_value), max_value)


def live_select_slider(label: str, options, value, key: str):
    """A discrete slider over ``options`` that updates live while dragging.

    Emits the selected option's index and maps it back to the option value, so
    non-contiguous option sets (e.g. a K range with a step) work correctly.
    """
    options = list(options)
    if not options:
        return None
    default_idx = options.index(value) if value in options else 0

    state = st.session_state.get(key, {})
    try:
        idx = int(state.get("value", default_idx))
    except (TypeError, ValueError):
        idx = default_idx
    idx = min(max(idx, 0), len(options) - 1)

    result = _LIVE_SLIDER(
        key=key,
        data={
            "label": label,
            "min": 0,
            "max": len(options) - 1,
            "step": 1,
            "value": idx,
            "decimals": 0,
            "prefix": "",
            "options": [str(o) for o in options],
        },
        default={"value": idx},
        on_value_change=lambda: None,
    )
    sel = idx if result.value is None else int(result.value)
    sel = min(max(sel, 0), len(options) - 1)
    return options[sel]
