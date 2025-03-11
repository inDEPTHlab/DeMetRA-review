import pandas as pd

import plotly.express as px
import textwrap

import definitions.layout_styles as styles

from pathlib import Path


main_dir = Path(__file__).parent.parent

assets_directory = main_dir / 'assets'

data = pd.read_csv(f'{assets_directory}/MPS_literature_cleaned.csv', parse_dates=['Date'])
data['Sample size'] = pd.to_numeric(data['Sample_size_total'], errors='coerce')

data_subset = data[['Author', 'Year', 'Title', 'Journal', 'DOI',
                    'Category', 'Phenotype', 'Developmental_period', 'Sample_size_total', 
                    'Tissue', 'Array', 'Ancestry']]
# ================


def _count_papers(d=data):
    n = int(len(pd.unique(d['Identifier'])))
    return n


def _count_mpss(d=data):
    return d.shape[0]

def _count_phenotypes(d=data):
    n = int(len(pd.unique(d['Phenotype'])))
    return n


# ================ PLOTTING ================


def _sample_size_over_time(data=data, color_by='Category', 
                           log_sample_size=True, model_type="ols", scope="overall"):
    """
    Scatterplot of sample size (y axis) vs. publication date (x axis)
    """
    # Color set-up
    if color_by in styles.COLOR_MAPS.keys():
        color_map = styles.COLOR_MAPS[color_by]
    else:
        color_map = 'Virdis'

    line_color = 'grey' if scope == 'overall' else None
    model_options = dict(log_y=log_sample_size) if model_type == "ols" else dict(frac=1)

    # Plot
    fig = px.scatter(data, x='Date', y='Sample size', log_y=log_sample_size,
                     color=color_by, color_discrete_map=color_map,
                     hover_name=data.Title.apply(lambda t: "<br>".join(textwrap.wrap(t, width=80))),
                     hover_data='Phenotype',
                     trendline=model_type, trendline_scope=scope, trendline_color_override=line_color,
                     trendline_options=model_options,
                     title='Sample size over time')
    # Make it pretty
    fig.update_traces(marker=dict(size=10, opacity=.5))
    axes_style = dict(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    ylabel_note = ' (log scale)' if log_sample_size else ''
    fig.update_yaxes(title_text=f'<b>Sample size</b>{ylabel_note}', **axes_style)
    fig.update_xaxes(title_text='Publication date', **axes_style)
    fig.update_layout(plot_bgcolor='whitesmoke', width=1300, height=400, margin=dict(l=10, r=10, t=25, b=10))

    return fig


def _category_over_years(data=data, color_by='Category', percent=True, width=1300, height=400):
    """
       Histogram of counts (y axis) vs. publication year (x axis)
    """
    # Color set-up
    if color_by in styles.COLOR_MAPS.keys():
        color_map = styles.COLOR_MAPS[color_by]
    else:
        color_map = "Virdis"

    norm = 'percent' if percent else None

    fig = px.histogram(data, x='Year', color=color_by, barnorm=norm,
                       color_discrete_map=color_map,
                       title=f'{color_by} over time', width=width, height=height)

    return fig

def _multilevel_piechart(lvl1='Category', lvl2='Phenotype'):

    counts = data[[lvl1, lvl2]].value_counts().reset_index()

    # Color set-up
    color_by = lvl1
    if color_by in styles.COLOR_MAPS.keys():
        color_map = styles.COLOR_MAPS[color_by]
    else:
        color_map = 'Virdis'

    fig = px.sunburst(counts,
                      path=[lvl1, lvl2], values='count',
                      color = color_by, color_discrete_map = color_map,
                      width=1200, height=750,
                      title='Number of MPSs per phenotype')

    return fig