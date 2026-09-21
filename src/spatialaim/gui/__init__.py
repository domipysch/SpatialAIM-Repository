"""Interactive Streamlit GUI for browsing SpatialAIM sweep results.

Launched via ``spatialaim gui`` (see ``gui/__main__.py``), which starts a Streamlit
server running ``gui/app.py``. The GUI drives the SpatialAIM sweep in-process (one run
root per mapper under the given output dir) and renders per-K views (Plotly,
built in ``gui/render.py``) from the machine-readable metrics the analysis
writes -- it does not compute any metric itself.
"""
