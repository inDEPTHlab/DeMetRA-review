# from shiny import App, render, ui
from shiny import App, Inputs, Outputs, Session, module, reactive, render, ui

from shinywidgets import output_widget, render_plotly

import definitions.layout_styles as styles
from definitions.backend_funcs import _sample_size_over_time

app_ui = ui.page_fillable(
    ui.card(
        ui.card_header("DeMetRA: Developmental Methylation Risk Atlas - literature review"),
        ui.layout_sidebar(
            ui.sidebar("Sidebar",
                       bg="#f8f8f8"),
            output_widget('sample_size_over_time') ,
        ),
    )
)


def server(input, output, session):

    @output
    @render_plotly
    def sample_size_over_time():
        p = _sample_size_over_time()
        return p


app = App(app_ui, server)

