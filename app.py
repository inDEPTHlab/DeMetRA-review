from pathlib import Path
from shiny import App, reactive, render, ui, Inputs

from shinywidgets import render_plotly

import definitions.layout_styles as styles
from definitions.backend_funcs import _count_papers, _count_mpss, _count_phenotypes, \
    _filter_litreview_table, _single_sankey, \
    _multilevel_piechart, _mps_count_histogram, _category_over_years, \
    _publication_histogram, _publication_network, \
    _sample_size_over_time

from definitions.page_uis import overview_page, phenotypes_page, publications_page, target_base_comparison_page, \
    sample_size_page

here = Path(__file__).parent

css_file = here / 'css' / 'custom_styles.css'

app_ui = ui.page_fluid(
    ui.include_css(css_file),
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

    # ====================== Overview page ======================
    @reactive.Calc
    def filter_overview_page_table():
       
        return _filter_litreview_table(selected_category = input.overview_page_selected_category(),
                                       selected_phenotype = input.overview_page_selected_phenotype(),
                                       selected_period = input.overview_page_selected_developmentalperiod(),
                                       selected_year_range = input.overview_page_selected_year(),
                                       based_on_filter = input.overview_page_basedon(),
                                       which_table = input.overview_page_which_table())
        
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
    
    @output
    @render.data_frame
    def overview_page_table():
        filtered_data, table_style = filter_overview_page_table()
        return render.DataTable(data=filtered_data, 
                                selection_mode='rows',
                                height='450px',
                                width='100%',
                                styles=table_style)
    
    # ================= Categories piechart page ==================

    @render_plotly
    def multilevel_piechart():
        p = _multilevel_piechart()
        return p
    
    @render_plotly
    def mps_count_histogram():
        p = _mps_count_histogram()
        return p
    
    @render_plotly
    def category_over_years():
        p = _category_over_years()
        return p
    
    # ================= Publication network page ==================
    @render_plotly
    def publication_histogram():
        p = _publication_histogram()
        return p
    
    @render_plotly
    def publication_network():
        p = _publication_network()
        return p
    
    # =================== Target vs. base page ====================
    
    @render.plot(width=1300, height=1400)
    def sankey_target_base():
        note = None

        var = input.comparison_selected_variable()

        if var == "Array":
            right_order = ['450K', 'EPICv1', 'EPICv2', 'Nanopore sequencing',
                           'Multiple (450K, EPICv1)',
                           'Multiple (450K, GMEL (~3000 CpGs from EPICv1))',
                           'Multiple (450K, EPICv2)']
            left_order = ['450K', 'EPICv1',
                          'Multiple (450K, EPICv1)', 
                          'Multiple (450K, EPICv1, PCR)', 
                          'Multiple (450K, PCR)']
            note = '* ~3000 CpGs from EPICv1'
        elif var == "Tissue":
            right_order = ['Peripheral blood', 'Whole blood', 'Dried bloodspot', 'Blood-clots',
                           'Cord blood', 'Saliva', 'Buccal cells', 'Tumour cells', 'Not reported'] 
            left_order = ['Peripheral blood', 'Whole blood',
                          'Cord blood',
                          'Multiple (Cord blood, Dried bloodspot)',
                          'Multiple (Cord blood, Whole blood)',
                          'Multiple (Whole blood, HPCs)',
                          'Buccal cells',
                          'Leukocytes',
                          'Tumour cells']
        elif var == "Ancestry":
            right_order = ['White', 'European', 'Mixed', 'Hispanic', 'African', 'Not reported']
            left_order = ['White', 'European', 'Mixed', 'Hispanic', 'Not reported']
        else: # Developmental period
            right_order = ['Birth', 'Very early childhood', 'Early childhood', 'Mid childhood',
                          'Late childhood', 'Childhood and adolescence', 'Adolescence', 'Not reported']
            left_order = ['Birth', 'Mid childhood', 'Late childhood', 'Childhood',
                          'Childhood and adolescence', 'Birth, Childhood and adolescence',
                          'Adolescence', 'Adults', 'Not reported']

        p = _single_sankey(var = var, left_order = left_order, right_order = right_order, 
                           note = note)  

        # p = _target_base_sankey()
        return p
    
    # ================ Sample size over time page =================
    
    @render_plotly
    def sample_size_over_time():
        p = _sample_size_over_time()
        return p
    
    


app = App(app_ui, server)

