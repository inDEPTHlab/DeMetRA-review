from shiny import module, ui, reactive

def pub_block_ui(page_id = 'submit_page'):
    return ui.card(
            ui.card_header("🗞️ Publication or pre-print info"),
            
            ui.input_text(f"{page_id}_title", ui.h6("Title", style="font-weight:bold"),
                          placeholder="Full paper title", width="100%"),
            ui.input_text(f"{page_id}_author", ui.h6("Author(s)", style="font-weight:bold"),
                              placeholder="e.g. Smith et al.", width="100%"),
            
            ui.layout_columns(
                ui.input_text(f"{page_id}_doi", ui.h6("DOI", style="font-weight:bold"),
                              placeholder="e.g. 10.1000/xyz123", width="100%"),
                ui.input_text(f"{page_id}_journal", ui.h6("Journal", style="font-weight:bold"),
                              placeholder="e.g. medRxiv, Journal of Epigenetics", width="100%"),
                ui.input_date(f"{page_id}_year", ui.h6("Publication date", style="font-weight:bold"),
                                 value=None, # Defaults to today
                                 min="2000-01-01", max=None),
                col_widths=[5, 4, 3])
        )

@module.ui
def mps_block_ui(idx: int):
    """Self-contained UI for one MPS entry. idx is just for the card header label."""
    return ui.card(
        ui.card_header(f"🧬\t\tMPS #{idx}"),
        ui.layout_columns(
            ui.input_text("phenotype", ui.h6("Phenotype", style="font-weight:bold"),
                          placeholder="e.g. ADHD, BMI", width="100%"),
            ui.input_selectize("category", ui.h6("Category", style="font-weight:bold"),
                               choices=["Biological markers", "Genetic syndromes",
                                        "Lifestyle and environment",
                                        "Physical health indicators",
                                        "Neuro-psychiatric health indicators",
                                        "Cancer"],
                               selected=None),
            ui.input_numeric("n_cpgs", ui.h6("N CpGs", style="font-weight:bold"),
                             value=None, min=2),
            col_widths=[6, 4, 2], style="align-items: stretch;",
        ),
        ui.layout_columns(
            ui.input_selectize("based_on", ui.h6("Bared on", style="font-weight:bold"),
                               choices=["Raw individual-level data", 
                                        "Published summary statistics (semi-supervised)",
                                        "Pre-established MPS"],
                               selected=None),
            ui.input_text("array", ui.h6("Array", style="font-weight:bold"),
                          placeholder="e.g. EPICv2"),
            ui.input_text("tissue",   ui.h6("Tissue",   style="font-weight:bold"),
                          placeholder="e.g. Whole blood"),
            ui.input_text("method", ui.h6("Method", style="font-weight:bold"),
                          placeholder="e.g. LASSO, elastic net, PCA"),
            col_widths=[3, 3, 3, 3], style="align-items: stretch;",
        ),
        ui.layout_columns(
            ui.input_numeric("sample_size", ui.h6("Sample size", style="font-weight:bold"),
                             value=None, min=1),
            ui.input_selectize("developmental_period",
                               ui.h6("Developmental period", style="font-weight:bold"),
                               choices=["Birth", "Early childhood", "Mid childhood",
                                        "Late childhood", "Adolescence"],
                               selected=None),
            ui.input_text("ancestry", ui.h6("Ancestry", style="font-weight:bold"),
                          placeholder="e.g. European"),
            col_widths=[4, 4, 4], style="align-items: stretch;",
        ),
        style="border-left: 2px solid var(--demetra-darkpink); border-bottom: 4px solid var(--demetra-darkpink); margin-bottom: 10px;",
    )


@module.server
def mps_block_server(input, output, session):
    """Exposes a reactive that returns this block's values as a dict."""

    @reactive.Calc
    def values():
        return {
            "Phenotype": input.phenotype(),
            "Category": input.category(),
            "Tissue": input.tissue(),
            "Array": input.array(),
            "Ancestry": input.ancestry(),
            "Developmental period": input.developmental_period(),
            "Sample size": input.sample_size(),
            "n CpGs": input.n_cpgs(),
            "Method": input.method(),
            "Based on": input.based_on()
        }

    return values  # parent server can call this to collect the row
