# from shiny import App, render, ui
from shiny import App, reactive, render, ui

from shinywidgets import output_widget, render_plotly

import definitions.layout_styles as styles
from definitions.backend_funcs import _count_papers, _count_mpss, _count_phenotypes, data_subset, _target_base_sankey, \
    _multilevel_piechart, _sample_size_over_time, _category_over_years, _publication_histogram, _publication_network

from definitions.page_uis import overview_page, phenotypes_page, publications_page, target_base_comparison_page, \
    sample_size_page

app_ui = ui.page_fluid(
    ui.page_navbar(
        overview_page(),
        phenotypes_page(),
        publications_page(),
        target_base_comparison_page(),
        sample_size_page(),

        id="navbar", 
        title="DeMetRA : Developmental Methylation Risk Atlas"
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
        return render.DataTable(data_subset, filters=True, 
                                styles=styles.DATATABLE_STYLE)


    @reactive.effect
    @reactive.event(input.reset_filter_df)
    async def _():
        await litreview_df.update_filter(None)

    # PLOTS TABS ============================================
    @render_plotly
    def publication_histogram():
        p = _publication_histogram()
        return p
    
    @render_plotly
    def publication_network():
        p = _publication_network()
        return p
    
    @render_plotly
    def multilevel_piechart():
        p = _multilevel_piechart()
        return p

    @render_plotly
    def sample_size_over_time():
        p = _sample_size_over_time()
        return p
    
    @render.plot(width=1300, height=1400)
    def sankey_target_base():
        p = _target_base_sankey()
        return p
    
    @render_plotly
    def category_over_years():
        p = _category_over_years()
        return p


app = App(app_ui, server)

