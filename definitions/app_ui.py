from shiny import ui
from shinywidgets import output_widget
import faicons as fa

from definitions.table_reactivity import mps_table
from definitions.submission_module import pub_block_ui, mps_block_ui


def add_value_box(value, label, icon, title='', value_style=None):
    _label = ui.markdown(label + "<br>") if label else ''

    _value = (
        ui.span(ui.output_text(value), style=value_style)
        if value_style
        else ui.output_text(value)
    )

    # icon can be a FA name (str) or a path to a PNG/SVG file (Path or str ending in an extension)
    _icon_str = str(icon)
    if _icon_str.endswith((".png", ".svg")):
        _showcase = ui.img(src=_icon_str,
            style="width: 3.2em; height: 3.2em; object-fit: contain;")
    else:
        _width  = '0.5em' if _icon_str == 'dna' else '0.8em'
        _showcase = fa.icon_svg(_icon_str, width=_width, height='0.7em', margin_right='0.1em')

    return ui.value_box(title, _value, _label,
        showcase = _showcase, showcase_layout = "top right", max_height='120px',
        style='display: flex; padding-bottom: 2px; padding-top: 5px; margin-right: 2px;')


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
                                   selected=_options, inline = True)


def overview_page(page_id='overview_page'):
    return ui.nav_panel("Overview",
                        ui.layout_columns(
                            ui.markdown(
                                "Welcome!<br>This is the **Developmental Methylation Risk Atlas** (*DeMetRA*), an interactive resource "\
                                "for finding MPSs that have been developed or applied to paediatric samples.<br>The atlas is originally "\
                                "based on this systematic literature review:"),
                            add_value_box('paper_count', 'publications', 'file-circle-check'),
                            add_value_box('mpss_count', 'unique MPSs', icon = 'methyl-icon.png'), # 'dna'
                            add_value_box('phenotype_count', 'unique Phenotypes', 'stethoscope'),
                            add_value_box("last_updated", "", "arrow-rotate-right",
                                          title="Last update:", value_style="font-size: 26px"),
                            col_widths = [4, 2, 2, 2, 2]),
                         ui.markdown(
                            "[*Methylation profile scores in early life: a systematic review and developmental risk atlas*]"\
                            "(http://dx.doi.org/10.2139/ssrn.5852502), but is meant as a collaborative, live resource.<br><br>"\
                            "On this page, you can ***search the atlas***, navigate to the other tabs to explore ***interactive visualizations*** of these "\
                            "data and to ***upload your MPS*** for others to find!"),
                         ui.layout_columns(
                             var_selector(page_id, "Category"),
                             var_selector(page_id, "Phenotype"),
                             var_selector(page_id, "Developmental period"),
                             var_slider(page_id, "Year", "Publication year"), 
                             var_checkbox(page_id, "Based on"),
                             col_widths=(3, 3, 2, 2, 2),
                             gap='9px',
                             class_="selection-pane"),
                        ui.input_radio_buttons(id='overview_page_which_table', label="", 
                                               choices={'mps_table': 'Show all MPSs',
                                                        'pub_table': 'Group by publication'},
                                               selected="mps_table",
                                               inline=True),
                        ui.output_data_frame("overview_page_table"),
                        ui.markdown(
                            "*<ins>Note</ins>: Rows in the table are colored "\
                            "based on the *Category* assigned to the corresponding MPS."),
                        )

def explore_page(page_id='explore_page'):
    return ui.nav_panel("Explore insights from the atlas",

        ui.markdown("On this page, we collected different ways to visualize insights from the DeMetRA resource.<br>"\
            "The interactive plots below are based on the entire atlas (i.e. latest systematic review (2025) + user uploads)."),
        
        # Phenotype category information
        ui.layout_columns(
            ui.card(
                ui.card_header("Category | Phenotype pie chart"),
                ui.markdown("In the multilevel pie chart below, you can see the number of MPSs per "\
                            "` Category | Pehnotype `. Hover over the slices to get more information. "\
                            "Click on one of the *inner* macro-categories to zoom into its phenotypes."),
                output_widget('multilevel_piechart'),
                full_screen=True, style="padding: 0; overflow: hidden;"
            ),
            ui.card(
                ui.card_header("Phenotype publication count"),
                ui.markdown("Which phenotypes got the most publications? Hover over the bars to find out.<br>"),
                output_widget('phenotype_pub_count'),
                full_screen=True, style="padding: 0; overflow: hidden;"
            ),
             col_widths=[8, 4]
        ),
        
        # Publications
        ui.markdown("Take a look at some publication metadata. Explore the network of authors and publications."),

        ui.navset_card_underline(
            ui.nav_panel("Publication network", 
                ui.markdown("The network graph below shows the connections between all authors and publications " \
                        "included in the review. Squares represent publications and they are colored by phenotype category. "\
                        "The light-blue dots represent individual authors. Hover over the nodes to get more info.<br>"),
                output_widget('publication_network')),
            ui.nav_panel("Number of MPSs per publication", 
                ui.markdown("In the histogram below, you can see the number of MPSs calculated in each "\
                            "individual publication, colored by phenotype category. "\
                            "Hover over the bars to get more information (e.g publication title)."),
                output_widget('mps_count_histogram')),                         
            ui.nav_panel("Most prolific authors", 
                ui.markdown("The histogram below shows the number of publications per author, when more than " \
                        "1 publication was included in the review. These are colored by phenotype category. Hover over the bars to get more info.<br>" \
                        "Use the zoom and pan tools on the top-right corner of the plot to explore the data.<br>"),
            output_widget('publication_histogram')),
        ),

        # Timelines
        ui.markdown("How is Developmental MPS research evolving over time?"),

        ui.navset_card_underline(
            ui.nav_panel("Sample size over time", 
                ui.markdown("Take a look at sample sizes over publication date.<br>"),
                output_widget('sample_size_over_time')),
             ui.nav_panel("Categories over time", 
                ui.markdown("The stacked histogram below shows the interest in different phenotype categories "\
                            "over the years."),
                output_widget('category_over_years')),
        )
    )      

def review_page():
    pubs_count = mps_table.groupby('Based on')['Title'].nunique().reset_index().set_index('Based on')

    # Add a column for the percentage of all titles
    total_titles = pubs_count['Title'].sum()
    pubs_count['Percent'] = round((pubs_count['Title'] / total_titles) * 100)

    return ui.nav_panel("Explore insights from our latest systematic review (2025)",
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
                                full_screen=True, style="min-height: 900px;padding: 0; overflow: hidden;"
                                ))

def submit_page(page_id="submit_page"):
    return ui.nav_panel(
        "Submit your MPS to DeMetRA!",
        ui.markdown(
            "Fill in as many information as you can about the MPS(s) you have developed or "
            "applied to in a developmental context, then hit the **Submit for review** button. We will "
            "process your submission as soon as possible and get back to you.<br>"
            "Note: something is wrong or missing? Please open an [issue](https://github.com/inDEPTHlab/DeMetRA-review/issues) to let us know!"
        ),
        # ── Publication-level (filled once) ─────────────────────────────
        ui.div(pub_block_ui(page_id)),
        
        # ── First MPS block is static, then depend on + add MPS ───
        ui.div(mps_block_ui("mps_1", 1), id=f"{page_id}_mps_container"),
        
        # ── Buttons ───────────────────────────────────────────────
        ui.input_action_button(f"{page_id}_add_mps", "➕ add MPS", class_="btn-demetra"),

        ui.div(
            ui.input_action_button(f"{page_id}_submit",  "📨 Submit for review", class_="btn-submit"), 
            style="display: flex; justify-content: flex-end; margin-top: 10px;",
        ),
        ui.output_ui(f"{page_id}_result")
    )
