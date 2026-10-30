"""
The Overview page: every starting point × mapper of the pair output in one
table, each at its own best overall K, so the best configuration stands out.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import streamlit as st

from spatialaim.analysis import compare_ks as scores
from spatialaim.gui import data_access

# The overall score, then each criterion's score, with their column titles.
_SCORES: dict[str, str] = {"overall": "Overall"} | {
    c.key: c.short for c in scores.CRITERIA
}

HELP_OVERVIEW = """
Every combination of starting clusters and mapping method, each at the K its own
sweep proposes (its best overall K), with the overall score and each criterion's
score at that K. Sorted by the overall score; click a column header to sort by
another one.

All scores share one scale across starts and methods: reconstruction is a cosine
similarity, spatial organisation a chance-corrected assortativity, coherence a
modularity. One caveat: the reference half of coherence (the modularity of the
states on the reference's shared-gene graph) favours the Leiden start, whose
clusters are themselves Leiden clusters of a closely related graph.
"""


def overview_table(starts: list[data_access.Start]) -> pd.DataFrame:
    """One row per start × mapper with a K comparison: ``start`` (label),
    ``folder`` (start folder name), ``mapper``, ``best_k`` (best overall K) and the
    overall and criterion scores at that K (NaN without a scored K)."""
    rows = []
    for start in starts:
        for mapper in data_access.list_mappers(start.folder):
            df = data_access.ksweep_table(data_access.run_root(start.folder, mapper))
            if df is None or df.empty:
                continue
            table = scores.score_table(df)
            table["overall"] = scores.overall(table)
            hit = scores.best_ks(table)["overall"]
            row = {"start": start.label, "folder": start.folder.name, "mapper": mapper}
            at_best = table[table["k"] == hit[0]].iloc[0] if hit else None
            row["best_k"] = hit[0] if hit else np.nan
            for key in _SCORES:
                row[key] = float(at_best[key]) if at_best is not None else np.nan
            rows.append(row)
    columns = ["start", "folder", "mapper", "best_k", *_SCORES]
    return pd.DataFrame(rows, columns=columns)


def overview_view(
    starts: list[data_access.Start],
    on_open: Callable[[data_access.Start, str], None],
) -> None:
    """The Overview page. ``on_open(start, mapper)`` is called when the user
    opens a row: it shows that start and mapper at its best K."""
    table = overview_table(starts)
    if table.empty:
        st.info("No finished runs yet.")
        return

    st.subheader("Overview: all starting clusters × methods", help=HELP_OVERVIEW)
    start_labels = list(dict.fromkeys(table["start"]))
    mapper_names = list(dict.fromkeys(table["mapper"]))
    col_starts, col_mappers = st.columns(2)
    with col_starts:
        shown_starts = st.multiselect(
            "Starting clusters",
            start_labels,
            default=start_labels,
            key="ov_starts",
            persist_state="session",
        )
    with col_mappers:
        shown_mappers = st.multiselect(
            "Methods",
            mapper_names,
            default=mapper_names,
            key="ov_mappers",
            persist_state="session",
        )

    shown = table[
        table["start"].isin(shown_starts) & table["mapper"].isin(shown_mappers)
    ].sort_values("overall", ascending=False, na_position="last")
    picked = st.dataframe(
        shown.drop(columns="folder")
        .rename(columns={"start": "starting clusters", "best_k": "best K", **_SCORES})
        .round(3),
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key="ov_table",
    )
    rows = picked.selection.rows if picked is not None else []
    if rows:
        row = shown.iloc[rows[0]]
        if st.button(
            f"Open {row['start']} · {row['mapper']}"
            + (f" at K = {int(row['best_k'])}" if np.isfinite(row["best_k"]) else ""),
            key="ov_open",
            type="primary",
        ):
            start = next(s for s in starts if s.folder.name == row["folder"])
            on_open(start, str(row["mapper"]))
    else:
        st.caption("Select a row to explore that combination at its best K.")
