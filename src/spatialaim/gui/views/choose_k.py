"""
The "Choose K" card of a mapper tab: one linked line plot per criterion, the
best K per criterion and overall (each a button moving the K slider), and the
criterion-vs-criterion scatters in a dialog.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from spatialaim.analysis import compare_ks as scores
from spatialaim.gui import data_access
from spatialaim.gui.render.ksweep import (
    BEST_K_COLOR,
    render_ksweep_criterion_figure,
    render_ksweep_scatter_figure,
)
from spatialaim.gui.views.cards import plot_card
from spatialaim.gui.views.controls import set_ctrl_k
from spatialaim.gui.widgets.linked_plot import picked_k


def _pick_k_callback(state_key: str, ks_all: list[int]):
    """``on_pick`` for a linked K-plot card: move the shared K slider to the K
    that was clicked. Runs as a widget callback, i.e. before the script body
    re-instantiates the slider, so writing its session-state key is allowed."""

    def _on_pick() -> None:
        k = picked_k(state_key)
        if k is not None and k in ks_all:
            set_ctrl_k(ks_all.index(k))
            # The K slider and sibling tabs live outside this tab's fragment; ask
            # the fragment body to escalate to a full app rerun so they follow.
            st.session_state["_ctrl_k_pick_pending"] = True

    return _on_pick


@st.dialog("Criterion trade-offs", width="large")
def _ksweep_scatter_dialog(
    mapper: str, score_table: pd.DataFrame, ks_all: list[int]
) -> None:
    """The three criterion-vs-criterion scatters, kept out of the card body so the
    card itself stays short."""
    pareto_only = st.checkbox(
        "Show only pareto-optimal points",
        value=True,
        key=f"ks_pareto_{mapper}",
        help="Keep only the K that no other K beats on all three criteria at once. "
        "A criterion that could not be computed for a K counts as its worst "
        "possible value.",
    )
    mask = scores.pareto_mask(score_table) if pareto_only else None
    group = f"ksweep_scatter_{mapper}"

    for col, (x_key, y_key) in zip(st.columns(3), scores.SCATTER_PAIRS):
        with col:
            key = f"ks_scatter_{x_key}_{y_key}_{mapper}"
            plot_card(
                render_ksweep_scatter_figure(
                    score_table,
                    scores.CRITERION[x_key],
                    scores.CRITERION[y_key],
                    mask=mask,
                ),
                key=key,
                stem=f"{mapper}_ksweep_scatter_{x_key}_{y_key}",
                link_group=group,
                on_pick=_pick_k_callback(key, ks_all),
            )

    shown = (
        f"Showing the {int(mask.sum())} pareto-optimal of {len(score_table)} K. "
        if mask is not None
        else ""
    )
    st.caption(
        f"One dot per K, each criterion summarised by the harmonic mean of its two "
        f"curves. {shown}Hover a dot to highlight that K in every panel and to show "
        "the grey label-shuffle null crosshair; click a dot to set the K slider. The "
        "axes span the nulls as well, so drag to zoom in on the dots."
    )


def _select_k_callback(k: int, ks_all: list[int]):
    """``on_click`` for a "Best K" row: move the shared K slider onto that K."""

    def _on_click() -> None:
        if k in ks_all:
            set_ctrl_k(ks_all.index(k))
            # The slider and sibling tabs live outside this tab's fragment; ask
            # the fragment body to escalate to a full app rerun so they follow.
            st.session_state["_ctrl_k_pick_pending"] = True

    return _on_click


def _best_k_rows(score_table: pd.DataFrame) -> list[tuple[str, int, float, str]]:
    """``(description, k, score, slug)`` for the best overall K and the best K per
    criterion.

    Presentation wrapper around :func:`spatialaim.analysis.compare_ks.best_ks`, which the
    sweep also writes to ``k_selection.json`` — the card and the file therefore
    always agree. A criterion with no finite score anywhere is skipped rather than
    reported as a winner.
    """
    best = scores.best_ks(score_table)
    labels = {"overall": "**Best overall** (harmonic mean across the criteria):"} | {
        c.key: f"**Best {c.label.lower()}** (harmonic mean):" for c in scores.CRITERIA
    }
    return [
        (labels[slug], hit[0], hit[1], slug)
        for slug, hit in best.items()
        if hit is not None
    ]


def best_overall_k(root: Path) -> int | None:
    """One mapper's best overall K, or ``None`` while its sweep table is missing
    or nothing in it could be scored yet. Same source as the "Choose K" card."""
    df = data_access.ksweep_table(root)
    if df is None or df.empty:
        return None
    hit = scores.best_ks(scores.score_table(df)).get("overall")
    return None if hit is None else int(hit[0])


def render_choose_k(
    mapper: str, root: Path, ks_all: list[int], current_k: int | None = None
) -> None:
    """The "Choose K" card: one line plot per criterion (its two curves plus their
    harmonic mean), each criterion's winning K below it, the best overall K below
    them, and the criterion-vs-criterion scatters behind a button.

    The line plots are linked: hovering a dot highlights the same K in all three,
    clicking one moves the K slider. Every number here — the nulls included — is
    read from the sweep's ``k_comparison.csv``.

    Collapsed, the card is a dead end unless its header says where things stand, so
    the label carries the K on show and the best overall K.
    """
    df = data_access.ksweep_table(root)
    if df is None or df.empty:
        return

    score_table = scores.score_table(df)
    best = {
        slug: (text, k, value) for text, k, value, slug in _best_k_rows(score_table)
    }
    # The best overall K is marked in all three cards, so the plots and the rows
    # below agree on one K.
    best_overall_k = best["overall"][1] if "overall" in best else None

    label = "Choose K (number of cell types)"
    state = []
    if current_k is not None:
        state.append(f"showing K = {current_k}")
    if best_overall_k is not None:
        state.append(f"best overall K = {best_overall_k}")
    if state:
        label += " — " + ", ".join(state)

    with st.expander(label, expanded=False):
        group = f"ksweep_{mapper}"

        # Tint the starred rows' material star to the same amber the plots use.
        # Streamlit renders the button's `icon` as an icon-font span, which it
        # already aligns with the label — unlike a "★" text glyph, whose font
        # bearings sat off-centre.
        st.markdown(
            "<style>div[class*='st-key-ks_bestkstar_'] button span"
            "[data-testid='stIconMaterial']"
            f"{{color:{BEST_K_COLOR};font-size:1.3rem;}}</style>",
            unsafe_allow_html=True,
        )

        def _best_k_row(slug: str) -> None:
            """One line: the label, then its K as the button. Skipped if unscored."""
            entry = best.get(slug)
            if entry is None:
                return
            text, k, value = entry
            row = st.container(
                horizontal=True, vertical_alignment="center", gap="small"
            )
            row.markdown(text)
            # Star the K the plots mark — on every row that lands on it, not just
            # the 'best overall' one. Starred rows get their own key prefix so the
            # CSS above can tint just that icon.
            starred = k == best_overall_k
            prefix = "ks_bestkstar" if starred else "ks_bestk"
            row.button(
                f"K = {k}",
                icon=":material/star:" if starred else None,
                key=f"{prefix}_{slug}_{mapper}",
                on_click=_select_k_callback(k, ks_all),
                disabled=k not in ks_all,
                help=f"Score {value:.2f}. Moves the K slider to K = {k}.",
            )

        # Each criterion's best K sits under its own plot; the overall winner —
        # which spans all three — gets its own row below them.
        for col, (index, criterion) in zip(st.columns(3), enumerate(scores.CRITERIA)):
            with col:
                key = f"ks_{criterion.key}_{mapper}"
                plot_card(
                    render_ksweep_criterion_figure(
                        df, criterion, index=index, best_k=best_overall_k
                    ),
                    key=key,
                    stem=f"{mapper}_ksweep_{criterion.key}",
                    link_group=group,
                    on_pick=_pick_k_callback(key, ks_all),
                )
                _best_k_row(criterion.key)

        st.divider()
        if not best:
            st.caption("No criterion could be scored for this run yet.")
        _best_k_row("overall")

        if st.button(
            "Compare criteria against each other",
            key=f"ks_scatter_btn_{mapper}",
            width="stretch",
            help="Opens the three criterion-vs-criterion scatter plots (one dot "
            "per K) with the pareto filter.",
        ):
            _ksweep_scatter_dialog(mapper, score_table, ks_all)
