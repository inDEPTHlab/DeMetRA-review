# from shiny import App, render, ui
from shiny import App, Inputs, Outputs, Session, module, reactive, render, ui
import faicons as fa

from shinywidgets import output_widget, render_plotly

import definitions.layout_styles as styles
from definitions.backend_funcs import _count_papers, _count_mpss, _count_phenotypes, data_subset, \
    _multilevel_piechart, _sample_size_over_time, _category_over_years

app_ui = ui.page_fluid(
    ui.card(
        ui.page_navbar(
            ui.nav_panel("Overview",
                ui.layout_column_wrap(
                    ui.value_box('', ui.output_text('paper_count'), 'papers',
                                 showcase=fa.icon_svg('file-circle-check')),
                    ui.value_box('', ui.output_text("mpss_count"), 'unique MPSs',
                                 showcase=fa.icon_svg("dna")),
                    ui.value_box('', ui.output_text("phenotype_count"), 'unique Phenotypes',
                                 showcase=fa.icon_svg("stethoscope")),
                    # col_widths=[-3, 3, 3, 3],            
                    fill=False),
                ui.input_action_button("reset_filter_df", "Reset filters"),
                ui.output_data_frame("litreview_df"),

                ),

            ui.nav_panel("MPSs by phenotype",
                ui.card(output_widget('multilevel_piechart'))),

            ui.nav_panel("Sample size over time",
                ui.card(output_widget('sample_size_over_time'))),

            ui.nav_panel("Phenotype categories over time",
                ui.card(output_widget('category_over_years'))),

            id="navbar", title="DeMetRA: literature review"
        )
    )
)


def server(input, output, session):

    @output
    @render.text
    def paper_count():
        c = _count_papers()
        return f'{c}'

    @render.text
    def mpss_count():
        c = _count_mpss()
        return f'{c}'

    @render.text
    def phenotype_count():
        c = _count_phenotypes()
        return f'{c}'

    @render.data_frame
    def litreview_df():
        return render.DataTable(data_subset, filters=True)

    @reactive.effect
    @reactive.event(input.reset_filter_df)
    async def _():
        await litreview_df.update_filter(None)

        # PLOTS TABS ============================================

    @render_plotly
    def multilevel_piechart():
        p = _multilevel_piechart()
        return p

    @render_plotly
    def sample_size_over_time():
        p = _sample_size_over_time()
        return p

    @render_plotly
    def category_over_years():
        p = _category_over_years()
        return p


app = App(app_ui, server)

