"""
Interactive Streamlit GUI for running SpatialAIM and browsing its results.

``spatialaim gui`` (``__main__.py``) starts a Streamlit server on ``app.py``. The
GUI runs sweeps in-process (one run root per mapper under the output dir) and
draws every plot (Plotly) from the metrics the analysis writes; it computes no
metric itself.

- ``app.py``          entry point: sidebar, run queue, result tabs, polling
- ``sidebar.py``      data paths, run settings, method selection, footer
- ``validation.py``   pair validation status + dialogs, input requirements
- ``views/``          the result tabs, the cell-type modal, cards and controls
- ``render/``         every Plotly figure
- ``widgets/``        custom components (live sliders, client-side plots)
- ``loaders.py``      Streamlit-cached reads of run outputs and the scaffold
- ``data_access.py``  plain readers of the sweep outputs on disk
- ``scaffold.py``     the reference scaffold (cached at the output dir)
- ``compute.py``      a mapper sweep on a background thread
- ``state_names.py``  user-set cell-type names (the GUI's only write)
- ``texts.py``        mapper labels and the report cards' help texts
"""
