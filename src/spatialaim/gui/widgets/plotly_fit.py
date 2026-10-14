"""
Explicit plot sizing shared by the client-side Plotly components.

Plotly's own autosizing is unreliable inside Streamlit: a plot (re)drawn while
its tab is hidden (zero width) falls back to Plotly's 700 px default and stays
squeezed into the left when the tab is shown, and ``Plots.resize`` drops the
figure height and re-derives it from the div, so after a sidebar toggle the
page-flow height can disagree with the drawn plot and overlap what follows.
So the components size the plot themselves: the div reserves the figure's pixel
height in CSS, the layout width is always the container's width, and a
ResizeObserver re-fits whenever the container width differs from the width
Plotly actually drew (not from the last *observed* width, which misses a redraw
that happened while hidden).
"""

FIT_JS = """
// Prepare ``layout`` for drawing into ``gd``: pin the div height to the figure
// height and the layout width to the container (or, while hidden, the last
// drawn width, so a hidden redraw does not collapse to Plotly's default).
const sizeLayout = (gd, layout) => {
  if (layout.height) gd.style.height = layout.height + "px"
  const w = Math.floor(gd.getBoundingClientRect().width)
  const prev = gd._fullLayout && gd._fullLayout.width
  if (w > 0) layout.width = w
  else if (prev) layout.width = prev
  layout.autosize = false
  return layout
}

// Re-fit the drawn width to the container; no-op while hidden, mid-draw, or
// already matching.
const refit = (Plotly, gd) => {
  if (gd.__aimDrawing) return
  const w = Math.floor(gd.getBoundingClientRect().width)
  const fl = gd._fullLayout
  if (w > 0 && fl && Math.abs((fl.width || 0) - w) > 1) {
    Plotly.relayout(gd, { width: w })
  }
}

// Draw (or update) the figure with explicit sizing, then keep it fitted.
const drawFitted = async (Plotly, gd, data, layout) => {
  gd.__aimDrawing = true
  try {
    await Plotly.react(gd, data, sizeLayout(gd, layout), {
      responsive: false,
      displaylogo: false,
    })
  } finally {
    gd.__aimDrawing = false
  }
  if (!gd.__aimFitWired && typeof ResizeObserver !== "undefined") {
    gd.__aimFitWired = true
    let frame = 0
    new ResizeObserver(() => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(() => refit(Plotly, gd))
    }).observe(gd)
  }
  // The container may have changed size while the draw was in flight.
  refit(Plotly, gd)
}
"""
