"""
Figure production for the GUI: every figure is an interactive Plotly object
built from the metrics on disk (each K's ``analysis/data``) and, for the UMAP,
marker and fraction views, the reference scaffold.

- ``common``     palettes, shared layout, labels, axes, static export
- ``scatter``    the per-cell-type UMAP / spatial / start-cluster traces
- ``headline``   the multi-panel figures sharing one cell-type legend
- ``markers``    marker-gene rankings, the dotplot, gene expression lookups
- ``report``     the report cards' figures (per mapper and in Compare)
- ``shape``      cell shape features (segmentation mask)
- ``cell_type``  the cell-type modal's figures
- ``ksweep``     the "Choose K" line and scatter figures
"""
