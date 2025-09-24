from shiny import ui
import faicons as fa

from shinywidgets import output_widget

from definitions.backend_funcs import mps_table
import definitions.layout_styles as style


def add_value_box(value, label, icon):
    _label = ui.markdown(label+"<br>")
    _width = '0.8em' if icon != 'dna' else '0.5em'
    _height = '0.7em' if icon != 'dna' else '0.7em'

    return ui.value_box('', ui.output_text(value), _label,
                        showcase=fa.icon_svg(icon, width=_width, height=_height,
                                             margin_right='0.1em'),
                        showcase_layout="top right",
                        style='display: flex; padding-bottom: 2px; padding-top: 5px; margin-right: 2px;',
                        max_height='120px')

def var_selector(page_id, variable, title=None):
    _options = [f for f in mps_table[variable].unique()]
    title = variable if not title else title
    variable_id = variable.lower().replace(' ', '')

    return ui.input_selectize(id=f'{page_id}_selected_{variable_id}',
                              label=ui.h6(title, style='font-weight: bold;'),
                              choices=_options,
                              selected=[],
                              multiple=True,
                              width='95%')

def var_slider(page_id, variable, title=None):
    _min = int(mps_table[variable].min())
    _max = int(mps_table[variable].max())

    title = variable if not title else title

    variable_id = variable.lower().replace(' ', '')
    return ui.input_slider(id=f"{page_id}_selected_{variable_id}",
                           label=ui.h6(title, style='font-weight: bold;'), 
                           min=_min, max=_max, value=[_min, _max],
                           width='90%',
                           sep="")

def var_checkbox(page_id, variable, title=None):
    
    _options = sorted([f for f in mps_table[variable].unique()])

    title = variable if not title else title
    variable_id = variable.lower().replace(' ', '')

    return ui.input_checkbox_group(id=f'{page_id}_{variable_id}',
                                   label=ui.h6(title, style='font-weight: bold;'),
                                   choices=_options,
                                   selected=_options)


def overview_page(page_id='overview_page'):
    return ui.nav_panel("Overview",
                        ui.layout_columns(
                            ui.markdown(
                                "Welcome!<br> Here you can explore the data related to our literature review: "\
                                "***Development and application of methylation profile scores in pediatric research: "\
                                "A systematic review and primer*** [DOI](todo).<br>" \
                                "On this page, you can search and filter the literature table.<br>"),
                            add_value_box('paper_count', 'publications', 'file-circle-check'),
                            add_value_box('mpss_count', 'unique MPSs', 'dna'),
                            add_value_box('phenotype_count', 'unique Phenotypes', 'stethoscope'),
                            ui.value_box('Last update:', ui.span('14/08/2025', style='font-size: 26px'),
                                         showcase=fa.icon_svg("arrow-rotate-right", width='0.5em', height='0.7em'),
                                         showcase_layout="top right",
                                         max_height='120px'),
                            col_widths = [4, 2, 2, 2, 2]),
                         ui.markdown("Use the selection pane below to filter the data. Rows in the table are colored "\
                                     "based on the *Category* assigned to the corresponding MPS. Navigate to the other "\
                                     "tabs for interactive visualizations of these data."),
                         ui.layout_columns(
                             var_selector(page_id, "Category"),
                             var_selector(page_id, "Phenotype"),
                             var_selector(page_id, "Developmental period"),
                             var_slider(page_id, "Year", "Publication year"), 
                             var_checkbox(page_id, "Based on"),
                             col_widths=(3, 3, 2, 2, 2),
                             gap='10px',
                             style=style.SELECTION_PANE),
                        
                        ui.input_radio_buttons(id='overview_page_which_table', label="", 
                                               choices={'mps_table': 'Show all MPSs',
                                                        'pub_table': 'Group by publication'},
                                               selected="mps_table",
                                               inline=True),
                        ui.output_data_frame("overview_page_table"),
                        )

def phenotypes_page(page_id='phenotype_page'):
    return ui.nav_panel("Explore phenotypes",
                        ui.markdown("In just a moment, you will see a couple of ways to visualize the distribution "\
                                    "of phenotypes across the literature review."),
                        ui.navset_card_underline(
                            ui.nav_panel("Category | Phenotype pie chart", 
                                         ui.markdown("In the multilevel pie chart below, you can see the number of MPSs per "\
                                                     "` Category | Pehnotype `. Hover over the slices to get more information. "\
                                                     "Click on one of the *inner* macro-categories to zoom into its phenotypes."),
                                                     output_widget('multilevel_piechart')),
                            ui.nav_panel("Number of MPSs per publication", 
                                        ui.markdown("In the histogram below, you can see the number of MPSs calculated in each "\
                                                    "individual publication, colored by phenotype category. "\
                                                    "Hover over the bars to get more information (e.g publication title)."),
                                                    output_widget('mps_count_histogram')),                         
                            ui.nav_panel("Categories over time", 
                                         ui.markdown("The stacked histogram below shows the interest in different phenotype categories over " \
                                                     "the years."),
                                                     output_widget('category_over_years'))
                        ))

def publications_page():
    return ui.nav_panel("Publications / Author network",
                        ui.markdown("In just a moment, you will see some publication metadata. Explore the network of authors and publications."),
                        ui.card(ui.card_header("Most prolific authors"),
                                ui.markdown("The histogram below shows the number of publications per author, when more than " \
                                    "1 publication was included in the review. These are colored by phenotype category. Hover over the bars to get more info.<br>" \
                                    "Use the zoom and pan tools on the top-right corner of the plot to explore the data.<br>"),
                                output_widget('publication_histogram')),     
                        ui.card(ui.card_header("Publication network"),
                                ui.markdown("The network graph below shows the connections between all authors and publications " \
                                    "included in the review. Squares represent publications and they are colored by phenotype category. "\
                                    "The light-blue dots represent individual authors. Hover over the nodes to get more info.<br>"),
                                output_widget('publication_network'))
                        )

def target_base_comparison_page():
    pubs_count = mps_table.groupby('Based on')['Title'].nunique().reset_index().set_index('Based on')

    # Add a column for the percentage of all titles
    total_titles = pubs_count['Title'].sum()
    pubs_count['Percent'] = round((pubs_count['Title'] / total_titles) * 100)

    return ui.nav_panel("Development vs. application dataset",
                        ui.markdown(f"**{int(100 - pubs_count.loc['Raw individual-level data', 'Percent'])}%** of publications in this review "\
                                    f"use existing resources to compute their MPSs.<br>{int(pubs_count.loc['Published summary statistics (semi-supervised)', 'Percent'])}% "\
                                    f"use published EWAS summary statistics, and {int(pubs_count.loc['Pre-established MPS', 'Percent'])}% "\
                                    f"use a Pre-established MPS. Here we explore these publications further."),
                        ui.card(ui.card_header(ui.layout_columns(
                                                "Development vs. application sample match on:",
                                                ui.input_selectize(id='comparison_selected_variable',
                                                                  label='',
                                                                  choices=['Array', 'Tissue', 'Ancestry', 'Developmental period'],
                                                                  selected='Array'),
                                                ui.input_selectize(id='comparison_selected_base_type',
                                                                  label='',
                                                                  choices=['All application studies', 'Published summary statistics (semi-supervised)', 'Pre-established MPS']),
                                                col_widths=[4, 3, 3, -2])),
                                ui.markdown("The Sankey diagram below shows how well the sample(s) used to develop the MPS " \
                                            "match the sample in which the MPS was applied.<br>"),
                                ui.output_plot("sankey_target_base"),
                                style="min-height: 800px;"))

def sample_size_page():
    return ui.nav_panel("Sample size over time",
                        ui.markdown("Here you take a look at sample sizes over publication date.<br>"),
                        output_widget('sample_size_over_time')),

