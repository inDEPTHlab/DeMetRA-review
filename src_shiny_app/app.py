# from shiny import App, render, ui
from shiny import App, Inputs, Outputs, Session, module, reactive, render, ui
import faicons as fa

from shinywidgets import output_widget, render_plotly

import definitions.layout_styles as styles
from definitions.backend_funcs import _count_papers, _multilevel_piechart, _sample_size_over_time, _category_over_years

app_ui = ui.page_fluid(
    ui.card(
        ui.card_header("DeMetRA: Developmental Methylation Risk Atlas - literature review"),
        ui.layout_sidebar(
            ui.sidebar("Sidebar", bg="#f8f8f8"),

            ui.layout_column_wrap(
                ui.value_box("Papers",
                    ui.output_text('paper_count'), showcase=fa.icon_svg('file-circle-check')),

                    # ui.value_box(
                    #     "unique MPS",
                    #     ui.output_text("bill_length"),
                    #     showcase=fa.icon_svg("ruler-horizontal")),
                fill=False),
            ui.card(output_widget('multilevel_piechart')),
            ui.card(output_widget('sample_size_over_time')),
            ui.card(output_widget('category_over_years')),
        ),
    )
)


def server(input, output, session):

    @output
    @render.ui
    def paper_count():
        c = _count_papers()
        return f'{c} papers'

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

