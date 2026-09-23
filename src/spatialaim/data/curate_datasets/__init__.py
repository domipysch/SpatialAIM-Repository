"""
Reproducible curation of the SpatialAIM dataset collection.

One module per scRNA reference (``curate_<nn>_<name>.py``). Each downloads its
reference and every ST slice paired with it from the *original* public source
and rebuilds the h5ad files that ``Data/01_Datasets_Used`` ships, byte-for-byte
where the source allows it.

Each script takes a single argument, ``--out-dir``; it downloads its sources
into a temporary folder below it and deletes them again, so only the final
h5ad files remain. Shared download/write helpers live in ``_common.py``.
"""
