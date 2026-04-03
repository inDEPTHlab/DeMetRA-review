from shiny import module, ui, reactive, render
import faicons as fa
import pandas as pd

def field_name(label: str = "Name", bold: bool = True, blank: bool = False, mb: str = "4px",
               required: bool = False, info: str | None = None):
    
    # Build style string
    styles = []
    if bold:
        styles.append("font-weight:bold;")
    if blank:
        styles.append("color:white;")
    if mb is not None:
        styles.append(f"margin-bottom:{mb};")
    style_str = " ".join(styles)

    # Build label contents
    children = [label]

    # Red asterisk for required fields
    if required and not blank:
        children.append(ui.span(" *", style="color:#d9534f;"))

    # Info icon with hover tooltip
    if info and not blank:
        children.append(ui.tooltip(
            ui.span(fa.icon_svg("circle-info", width="13px", height="13px", fill="var(--demetra-purple)"),
                                style="margin-left: 5px; margin-bottom: 4px"), 
            info, placement='right'))

    return ui.h6(*children, style=style_str)


# Load existing phenotypes/tissues/arrays from the current
mps_df = pd.read_csv("assets/mps_table.csv")
obs_phenotypes = sorted(mps_df["Phenotype"].dropna().unique().tolist())
obs_categories = mps_df["Category"].dropna().unique().tolist()
obs_tissues = mps_df["Tissue"].dropna().unique().tolist()
obs_arrays = mps_df["Array"].dropna().unique().tolist()
pub_df = pd.read_csv("assets/pub_table.csv")
obs_journals = sorted(pub_df["Journal"].dropna().unique().tolist())

def input_text_suggest(id, label, suggestions, placeholder, required = False, info = None):

    return ui.input_selectize(id, field_name(label, required=required, info=info),
                choices=suggestions, selected=None, multiple=True,
                options={"create": True,  # allow new values not in the list
                         "placeholder": placeholder, "maxItems": 1, 
                         "closeAfterSelect": True}, width="100%")

def pub_block_ui(page_id = 'submit_page'):
    return ui.card(
            ui.card_header("🗞️ Publication or pre-print info"),
            
            ui.input_text(f"{page_id}_title", field_name("Title", required=True),
                          placeholder="Full paper title", width="100%"),
            ui.layout_columns(
                ui.input_text(f"{page_id}_author", field_name("Author(s)", required=True, 
                              info='Follow Harvard format e.g. "Surname, N., Surname, N.N., ..."'),
                              placeholder="e.g. Scott, M.J., Schrute, D.", width="100%"),
                ui.input_text(f"{page_id}_contact", field_name("Contact", required=True),
                              placeholder="e.g. m.scott@dundermifflin.com", width="100%"),
                col_widths=[8, 4], style="align-items: stretch;"
            ),
            ui.layout_columns(
                ui.input_text(f"{page_id}_doi", field_name("DOI", required=True),
                              placeholder="e.g. 10.1000/xyz123", width="100%"),
                input_text_suggest(f"{page_id}_journal", label="Journal", 
                                   suggestions=obs_journals, 
                                   placeholder="e.g. medRxiv, Journal of Epigenetics"),
                ui.input_date(f"{page_id}_year", field_name("Publication date"),
                              value=None, # Defaults to today
                              min="2000-01-01", max=None, width="100%"),
                col_widths=[5, 5, 2], style="align-items: stretch;")
        )

@module.ui
def mps_block_ui(idx: int):
    """Self-contained UI for one MPS entry. idx is just for the card header label."""
    return ui.card(
        ui.card_header(f"🧬\t\tMPS #{idx}"),
        ui.layout_columns(
            input_text_suggest("phenotype", label="Phenotype", required = True,
                               suggestions=obs_phenotypes, 
                               placeholder="e.g. ADHD, BMI"),
            input_text_suggest("category", label="Category", required = True,
                               suggestions=obs_phenotypes, 
                               placeholder="e.g. Biological markers"),                      
            ui.input_numeric("n_cpgs", field_name("N CpGs", required = True),
                             value=None, min=2, step=1),
            ui.input_numeric("sample_size", field_name("Sample size", required = True, 
                             info="Internal validation sample size or total sample size (for train-test split)"),
                             value=None, min=2, step=1),
            col_widths=[4, 4, 2, 2], style="align-items: stretch;"
        ),
        ui.layout_columns(
            ui.input_selectize("based_on", field_name("Based on"),
                               choices=["Raw individual-level data", 
                                        "Published summary statistics (semi-supervised)",
                                        "Pre-established MPS"],
                               multiple=False,
                               selected="Raw individual-level data"),
            input_text_suggest("array", label="Array", required = True,
                               suggestions=obs_arrays, 
                               placeholder="e.g. EPICv2"),
            input_text_suggest("tissue", label="Tissue", required = True,
                               suggestions=obs_tissues, 
                               placeholder="e.g. Whole blood"),
            input_text_suggest("developmental_period", label="Developmental period", required = True,
                               suggestions=["Birth", "Early childhood", "Mid childhood",
                                            "Late childhood", "Adolescence"], 
                               placeholder="e.g. median age [age range], Adolescence"),
            ui.input_text("ancestry", field_name("Ancestry"),
                          placeholder="e.g. European, Mixed"),
            col_widths=[3, 2, 2, 3, 2], style="align-items: stretch;"
        ),
        ui.output_ui("dev_reference_block"),
        ui.layout_columns(
            input_text_suggest("method", label="Method", 
                               suggestions=["Elastic net", "LASSO", "Lassosum",
                               "Linear Regression", "Linear mixed model", "Logistic regression",
                               "Support vector machine (SVM)", "Random forest (RF)", "gradient boosting machine (GBM)",
                               "Partial least squares-discriminant analysis", "Neural Network", "Naive Bayes"], 
                               placeholder="e.g. Elastic net, LASSO, PCA"),
            ui.div(field_name("Predictive performance"),
                   ui.layout_columns(
                        ui.input_selectize("performance_metric", '', # field_name(blank=True),
                                           choices=["R²", "AUC", "Accuracy", "F1 score", "Pearson r"],
                                           selected="R²", multiple=False),
                        ui.input_text("performance_value", '', # field_name(blank=True), 
                                      placeholder="e.g. 0.05", width="100%"),
                        col_widths=[7, 5], gap="5px"), style="margin-bottom: 0; padding-top: 4px;"
            ),
            ui.input_text("weights_link", field_name("Link to code / MPS weights")),
            col_widths=[6, 3, 3], style="align-items: stretch;",
        ),
        style="border-left: 2px solid var(--demetra-lightpurple); border-bottom: 4px solid var(--demetra-lightpurple); margin-bottom: 10px;",
    )


@module.server
def mps_block_server(input, output, session):
    """Exposes a reactive that returns this block's values as a dict."""

    @render.ui
    def dev_reference_block():

        if input.based_on() == "Raw individual-level data" or input.based_on() is None:
            return None
        
        return ui.layout_columns(
            ui.input_text("dev_reference", field_name("Development dataset reference (DOI)"),
                          placeholder="e.g. 10.1000/xyz123", width="100%"),
            ui.input_checkbox_group("dev_match", field_name("Matching:"),
                                choices=["Array", "Tissue", "Developmental period", "Ancestry"],
                                inline=True),
            col_widths=[6, 6], style="align-items: stretch;")

    @reactive.Calc
    def values():
        return {
            "Phenotype": input.phenotype(),
            "Category": input.category(),
            "n CpGs": input.n_cpgs(),
            "Sample size": input.sample_size(),
            "Array": input.array(),
            "Tissue": input.tissue(),
            "Developmental period": input.developmental_period(),
            "Ancestry": input.ancestry(),
            "Based on": input.based_on(),
            "Development dataset reference": input.dev_reference() if input.based_on() != "Raw individual-level data" else None,
            "Development dataset match": list(input.dev_match()) if input.based_on() != "Raw individual-level data" else [],
            "Method": input.method(),
            "Performance metric": input.performance_metric(),
            "Performance value": input.performance_value(),
            "MPS link": input.weights_link()
        }

    return values  # parent server can call this to collect the row
