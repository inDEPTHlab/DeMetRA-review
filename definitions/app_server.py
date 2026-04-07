from pathlib import Path

from shiny import reactive, render, ui
from shinywidgets import render_plotly

import plotly.io as pio
import json

from definitions.submission_module import mps_block_ui, mps_block_server
from definitions.submission_validator import validate_and_structure

from definitions.table_reactivity import _filter_litreview_table
from definitions.plotting_funcs import _single_sankey

# ── Load cache once at startup ────────────────────────────────

_cache = Path(__file__).parent.parent / "dynamic_cache"

with open(_cache / "metadata" / "stats.json") as f:
    STATS = json.load(f)

FIGS = {
    name: pio.read_json(str(_cache / "figures" / f"{name}.json"))
    for name in [
        "multilevel_piechart",
        "phenotype_pub_count",
        "mps_count_histogram",
        "category_over_years",
        "publication_histogram",
        "publication_network",
        "sample_size_over_time",
    ]
}


def app_server(input, output, session):

    # ==============================================================
    # Overview page 
    # ==============================================================
    
    @reactive.Calc
    def filter_overview_page_table():
        return _filter_litreview_table(
            selected_category   = input.overview_page_selected_category(),
            selected_phenotype  = input.overview_page_selected_phenotype(),
            selected_period     = input.overview_page_selected_developmentalperiod(),
            selected_year_range = input.overview_page_selected_year(),
            based_on_filter     = input.overview_page_basedon(),
            which_table         = input.overview_page_which_table(),
        )

    @output
    @render.text
    def paper_count():
        return str(STATS["publication_count"])

    @output
    @render.text
    def mpss_count():
        return str(STATS["mps_count"])

    @output
    @render.text
    def phenotype_count():
        return str(STATS["phenotype_count"])

    @output
    @render.text
    def last_updated():
        return str(STATS["last_updated"])

    @output
    @render.data_frame
    def overview_page_table():
        filtered_data, table_style = filter_overview_page_table()
        return render.DataTable(
            data = filtered_data, selection_mode = "rows", 
            height = "450px", width = "100%", styles = table_style,
        )

    # ==============================================================
    # Literature exploration page (all figures served from cache)
    # ==============================================================

    for _name in FIGS:
        # create a closure to capture the name
        def _make_renderer(name):
            def _renderer():
                return FIGS[name]
            _renderer.__name__ = name
            output(render_plotly(_renderer))
        _make_renderer(_name)

    # ── Target vs. Base (depends on input & review data) ──────────
    
    @render.plot(width=1000, height=1000)
    def sankey_target_base():
        var = input.comparison_selected_variable()
        filter_base_type = input.comparison_selected_base_type()
        # ... same label order logic as original app.py ...
        return _single_sankey(var=var, left_label_order=[], right_label_order=[],
                              filter=filter_base_type)

    # ==============================================================
    # Submit your MPS page
    # ==============================================================
    page_id  = "submit_page"

    mps_ids     = reactive.Value(["mps_1"])
    mps_getters = {"mps_1": mps_block_server("mps_1")}

    # mps_ids  = reactive.Value([])   # list of namespace IDs inserted so far
    # mps_getters = {}  # namespace_id → reactive.Calc (values)

    # # Register static first block
    # mps_getters["mps_1"] = mps_block_server("mps_1")
    # mps_ids.set(["mps_1"])

    def _insert_block():
        ns_id = f"mps_{len(mps_ids.get()) + 1}"
        idx   = len(mps_ids.get()) + 1
        ui.insert_ui(
            ui = mps_block_ui(ns_id, idx),
            selector=f"#{page_id}_mps_container",
            where="beforeEnd")

        mps_getters[ns_id] = mps_block_server(ns_id)
        mps_ids.set(mps_ids.get() + [ns_id])

    @reactive.Effect
    @reactive.event(input[f"{page_id}_add_mps"], ignore_init=True)
    def _on_add_mps():
        _insert_block()

    # ── Submit PR ─────────────────────────────────────────────────
    @reactive.Effect
    @reactive.event(input[f"{page_id}_submit"])

    def _handle_submission():
        # Collect publication fields 
        pub = {
        "Title":       input[f"{page_id}_title"](),
        "Author_list": input[f"{page_id}_author"](),
        "Contact":     input[f"{page_id}_contact"](),
        "DOI":         input[f"{page_id}_doi"](),
        "Journal":     input[f"{page_id}_journal"](),
        "Date":        input[f"{page_id}_year"]()
        }

        # Collect MPS blocks
        mps_list = [mps_getters[ns_id]() for ns_id in mps_ids.get()]

        # Validate input and structure data for submission
        validation, df = validate_and_structure(pub, mps_list)

        if not validation.valid:
            ui.notification_show(
                "⚠️ Please fix the errors below before submitting.",
                type="warning", duration=5)

            @output
            @render.ui
            def submit_page_result():
                return ui.div(
                    ui.tags.strong("⚠️ Please fix the following:"),
                    ui.tags.ul(
                        *[ui.tags.li(ui.markdown(e)) for e in validation.flat_errors()]),
                    class_="submit-result-error")
            return

        # All input is valid: submit PR 
        try:
            pr_url = _open_pr(df, contact = "") # TODO: collect contact info

            ui.notification_show(
                ui.HTML(f"✅ Submitted! <a href='{pr_url}' target='_blank'>View PR →</a>"),
                type="message", duration=10)

            @output
            @render.ui
            def submit_page_result():
                return ui.div(
                    ui.markdown(
                        f"✅ **Thank you!** {len(df)} new MPS(s) submitted.  \n"
                        f"[View the pull request ↗]({pr_url})"),
                    class_="submit-result-success")

        except Exception as e:
            ui.notification_show(
                "❌ Submission failed — see details below.", 
                type="error", duration=8)

            @output
            @render.ui
            def submit_page_result():
                return ui.div(
                    ui.tags.strong("❌ Submission failed:"),
                    ui.tags.pre(str(e), style="font-size:0.85em; margin-top:6px;"),
                    class_="submit-result-error")