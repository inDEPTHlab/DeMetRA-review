from shiny import ui
import faicons as fa

from shinywidgets import output_widget

def overview_page():
    return ui.nav_panel("Overview",
                        ui.markdown("Welcome, here you can find an overview of the literature review data.<br>" \
                                    "You can filter the data by clicking on the column headers.<br>" \
                                    "You can also reset the filters by clicking the button below."),

                        ui.layout_column_wrap(
                            ui.value_box('', ui.output_text('paper_count'), 'publications',
                                         showcase=fa.icon_svg('file-circle-check')),
                            ui.value_box('', ui.output_text("mpss_count"), 'unique MPSs',
                                            showcase=fa.icon_svg("dna")),
                            ui.value_box('', ui.output_text("phenotype_count"), 'unique Phenotypes',
                                            showcase=fa.icon_svg("stethoscope")),
                            ui.value_box('Last update:', '14/02/2025',
                                            showcase=fa.icon_svg("arrow-rotate-right")),
                            fill=False),

                        ui.input_action_button("reset_filter_df", "Reset filters"),
                        ui.output_data_frame("litreview_df")
                        )

def phenotypes_page():
    return ui.nav_panel("Explore phenotypes",
                        ui.markdown("Here you can explore the distribution of phenotypes across the literature review.<br>" \
                                    "In the multilevel pie chart, click on one of the inner categories to select it. <br>"),
                        output_widget('multilevel_piechart'),
                         ui.markdown("The stacked histogram below show the interest in different phenotype categories over " \
                                     "the years.<br>"),
                        output_widget('category_over_years')),

def publications_page():
    return ui.nav_panel("Publications / Author network",
                        ui.markdown("Here you can explore the network of authors and publications.<br>" \
                                    "The histogram shows the number of publications per author, when more than" \
                                    "one publication was included in the review. Hover over the bars to get more" \
                                    "info.<br>" ),
                        output_widget('publication_histogram'),
                        ui.markdown("The network graph shows the connections between all authors and publications" \
                                    " included in the review. Hover over the nodes to get more info.<br>"),
                        output_widget('publication_network'))

def target_base_comparison_page():
    return ui.nav_panel("Target vs. base comparison",
                        ui.markdown("Some paper use existing resource to compute MPSs."),
                        ui.output_plot("sankey_target_base")),

def sample_size_page():
    return ui.nav_panel("Sample size over time",
                        ui.markdown("Here you take a look at sample sizes over publication date.<br>"),
                        output_widget('sample_size_over_time')),

