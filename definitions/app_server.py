import os
import io
from datetime import datetime

import pandas as pd
from shiny import reactive, render, ui, req
from github import Github

from definitions.submission_module import mps_block_ui, mps_block_server

from definitions.backend_funcs import (
    _count_papers, _count_mpss, _count_phenotypes,
    _filter_litreview_table,
    _multilevel_piechart, _mps_count_histogram, _category_over_years,
    _publication_histogram, _publication_network,
    _sample_size_over_time, _single_sankey,
)
from shinywidgets import render_plotly


def app_server(input, output, session):

    # =========================================================
    # Overview page  (unchanged from original)
    # =========================================================
    @reactive.Calc
    def filter_overview_page_table():
        return _filter_litreview_table(
            selected_category=input.overview_page_selected_category(),
            selected_phenotype=input.overview_page_selected_phenotype(),
            selected_period=input.overview_page_selected_developmentalperiod(),
            selected_year_range=input.overview_page_selected_year(),
            based_on_filter=input.overview_page_basedon(),
            which_table=input.overview_page_which_table(),
        )

    @output
    @render.text
    def paper_count():
        return str(_count_papers())

    @output
    @render.text
    def mpss_count():
        return str(_count_mpss())

    @output
    @render.text
    def phenotype_count():
        return str(_count_phenotypes())

    @output
    @render.data_frame
    def overview_page_table():
        filtered_data, table_style = filter_overview_page_table()
        return render.DataTable(
            data=filtered_data,
            selection_mode="rows",
            height="450px",
            width="100%",
            styles=table_style,
        )

    # =========================================================
    # Phenotypes page  (unchanged)
    # =========================================================
    @render_plotly
    def multilevel_piechart():
        return _multilevel_piechart()

    @render_plotly
    def mps_count_histogram():
        return _mps_count_histogram()

    @render_plotly
    def category_over_years():
        return _category_over_years()

    # =========================================================
    # Publications page  (unchanged)
    # =========================================================
    @render_plotly
    def publication_histogram():
        return _publication_histogram()

    @render_plotly
    def publication_network():
        return _publication_network()

    # =========================================================
    # Target vs. Base page  (unchanged)
    # =========================================================
    @render.plot(width=1000, height=1000)
    def sankey_target_base():
        var = input.comparison_selected_variable()
        filter_base_type = input.comparison_selected_base_type()
        # ... same label order logic as original app.py ...
        return _single_sankey(var=var, left_label_order=[], right_label_order=[],
                              filter=filter_base_type)

    # =========================================================
    # Sample size page  (unchanged)
    # =========================================================
    @render_plotly
    def sample_size_over_time():
        return _sample_size_over_time()

    # =========================================================
    # Submit page  — modules + insert_ui pattern
    # =========================================================
    page_id  = "submit_page"
    mps_ids  = reactive.Value([])   # list of namespace IDs inserted so far
    mps_getters = {}  # namespace_id → reactive.Calc (values)

    # Register static first block
    mps_getters["mps_1"] = mps_block_server("mps_1")
    mps_ids.set(["mps_1"])

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

    # ── Submit ─────────────────────────────────────────────────
    @reactive.Effect
    @reactive.event(input[f"{page_id}_submit"])
    def _handle_submission():
        
        # ── Publication validation ────────────────────────────────────
        pub_required = {
            "Author": input[f"{page_id}_author"](),
            "Year":   input[f"{page_id}_year"](),
            "Title":  input[f"{page_id}_title"](),
            "DOI":    input[f"{page_id}_doi"](),
        }
        pub_errors = [
            label for label, val in pub_required.items()
            if val is None or (isinstance(val, str) and val.strip() == "")
        ]

        # ── MPS validation ────────────────────────────────────────────
        REQUIRED_MPS = ["Phenotype", "Category", "Tissue",
                        "Developmental period", "Sample size", "n CpGs", "Method"]
        all_mps_errors = {}
        for i, ns_id in enumerate(mps_ids.get(), start=1):
            row = mps_getters[ns_id]()
            missing = [f for f in REQUIRED_MPS
                    if row.get(f) is None or str(row.get(f, "")).strip() == ""]
            if missing:
                all_mps_errors[f"MPS #{i}"] = missing

        # ── Block if invalid ─────────────────────────────────────────
        if pub_errors or all_mps_errors:
            error_lines = []
            if pub_errors:
                error_lines.append("**Publication info:** " + ", ".join(pub_errors))
            for block_label, missing in all_mps_errors.items():
                error_lines.append(f"**{block_label}:** " + ", ".join(missing))

            ui.notification_show(
                "⚠️ Some required fields are missing — see details below.",
                type="warning",
                duration=5,
            )

            @output
            @render.ui
            def submit_page_result():
                return ui.div(
                    ui.tags.strong("⚠️ Please complete these required fields:"),
                    ui.tags.ul(*[ui.tags.li(ui.markdown(line)) for line in error_lines]),
                    class_="submit-result-error",
                )
            return

        # ── All valid — submit ────────────────────────────────────────
        pub_info = {
            "Author":   input[f"{page_id}_author"](),
            "Year":     input[f"{page_id}_year"](),
            "Title":    input[f"{page_id}_title"](),
            "DOI":      input[f"{page_id}_doi"](),
            "Date":     datetime.today().strftime("%Y-%m-%d"),
            "Based on": "Submitted",
        }
        mps_rows = [
            {**pub_info, **mps_getters[ns_id]()}
            for ns_id in mps_ids.get()
        ]

        try:
            pr_url = _open_pr(mps_rows, pub_info)
            ui.notification_show(
                ui.HTML(f"✅ Submission received! "
                        f"<a href='{pr_url}' target='_blank'>View pull request →</a>"),
                type="message", duration=10,
            )

            @output
            @render.ui
            def submit_page_result():
                return ui.div(
                    ui.markdown(
                        f"✅ **Thank you!** {len(mps_rows)} MPS row(s) submitted.  \n"
                        f"[View the pull request ↗]({pr_url})"
                    ),
                    class_="submit-result-success",
                )

        except Exception as e:
            ui.notification_show("❌ Submission failed — see details below.",
                                type="error", duration=8)

            @output
            @render.ui
            def submit_page_result():
                return ui.div(
                    ui.tags.strong("❌ Submission failed:"),
                    ui.tags.pre(str(e), style="font-size:0.85em; margin-top:6px;"),
                    class_="submit-result-error",
                )


def _open_pr(mps_rows: list[dict], pub_info: dict):
    """Push updated CSV to a new branch and open a PR."""

    g = Github(os.environ["GITHUB_PAT"])
    repo = g.get_repo("inDEPTHlab/DeMetRA-review")
    csv_path = "assets/MPS_table_cleaned.csv"

    file_obj = repo.get_contents(csv_path, ref="main")
    current_csv = pd.read_csv(io.StringIO(file_obj.decoded_content.decode()))
    updated_csv = pd.concat([current_csv, pd.DataFrame(mps_rows)], ignore_index=True)

    branch_name = (
        f"submission/"
        f"{pub_info['Author'].split()[0].lower()}-"
        f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    main_sha = repo.get_branch("main").commit.sha
    repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=main_sha)
    repo.update_file(
        csv_path,
        f"Submission: {pub_info['Title'][:60]}",
        updated_csv.to_csv(index=False),
        file_obj.sha,
        branch=branch_name,
    )

    n  = len(mps_rows)
    pr_body = (
        f"**Submitted via app** — {n} MPS(s)\n\n"
        f"**Author:** {pub_info['Author']}  \n"
        f"**Year:** {pub_info['Year']}  \n"
        f"**DOI:** https://doi.org/{pub_info['DOI']}  \n\n"
        + "\n\n".join(
            f"### MPS #{i + 1}\n"
            + "\n".join(f"- **{k}**: {v}" for k, v in row.items() if k not in pub_info)
            for i, row in enumerate(mps_rows)
        )
    )

    pr = repo.create_pull(
        title=f"Submission: {pub_info['Author']} ({pub_info['Year']}) — {n} MPS(s)",
        body=pr_body,
        head=branch_name,
        base="main"
    )

    return pr.html_url
