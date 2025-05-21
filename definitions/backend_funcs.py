import pandas as pd
import numpy as np
import pickle

from shiny import ui

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import networkx as nx
import ast 
import re
import textwrap

import definitions.layout_styles as styles

from pathlib import Path


main_dir = Path(__file__).parent.parent

assets_directory = main_dir / 'assets'

# ================== DATA LOADING =================
mps_table = pd.read_csv(f'{assets_directory}/MPS_table_cleaned.csv', parse_dates=['Date'])
pub_table = pd.read_csv(f'{assets_directory}/Publication_table_cleaned.csv', parse_dates=['Date'])


# Turn DOI into a clickable link
mps_table[' '] = [ui.markdown(f'<a href="https://doi.org/{doi}" target="_blank">DOI</a>') for doi in mps_table['DOI']]
pub_table[' '] = [ui.markdown(f'<a href="https://doi.org/{doi}" target="_blank">DOI</a>') for doi in pub_table['DOI']]

pub_table['Phenotype(s)'] = pub_table['Phenotype']

# Display lists inside cells as bulletpoints
def list_to_html(cell):
    ul_element = '<ul style="margin-left:0; padding-left:0;">'
    if isinstance(cell, str): # lists are turned into strings by the csv reader
        # print(cell)
        try:
            value = ast.literal_eval(cell)
            # print(value)
            if isinstance(value, list):
                # print(value)
                return ui.HTML(ul_element + ''.join([f'<li>{item}</li>' for item in value]) + '</ul>')
            else: #if isinstance(value, float):
                return str(int(value)) if not np.isnan(value) else ''
            
        except (ValueError, SyntaxError):
            # Fallback: parse np.float64-style strings
            matches = re.findall(r'np\.float64\((nan|[\d\.]+)\)', cell)
            items = []
            for m in matches:
                if not m == 'nan':
                    f = float(m)
                    # items.append(str(int(f)) if f.is_integer() else str(f))
                    # Instead of a bulletpoint list I show range
                    items.append(int(f))
            
            if items:
                # return ui.HTML(ul_element + ''.join([f'<li>{item}</li>' for item in items]) + '</ul>')
                if len(items) == 2:
                    return f'{items[0]}, {items[1]}'
                else: 
                    return f'{min(items)}-{max(items)}'

    elif isinstance(cell, list):
        # print(cell)
        items = [str(int(x)) if isinstance(x, float) and x.is_integer() else str(x) for x in cell]
        return ui.HTML(ul_element + ''.join([f"<li>{item}</li>" for item in items]) + '</ul>')
    
    elif isinstance(cell, float) or isinstance(cell, int):
        return str(int(cell)) if not np.isnan(cell) else ''
    
    else: print(cell)

    return cell

mps_table_show = mps_table[['Phenotype', 'Category', 'n CpGs', 'Based on',
                            'Author', 'Year', 'Title', ' ',  
                            'Sample type', 'Sample size', 'n Cases', 'n Controls', 
                            'Developmental period', 'Tissue', 'Array', 'Ancestry',]]
                            # 'Including_CpGs_1', 'Including_CpGs_2', 'Including_CpGs_3', 'Including_CpGs_4', 'Including_CpGs_5',
                            # 'Sample_overlap_target_base', 'Determining_weights_1', 'Train_test',
                            # 'Independent_validation', 'Comparison', 'Missing_value_note',
                            # 'Reflect_phenotype']]

pub_table_show = pub_table[['Author', 'Year', 'Title', 'Journal',' ','n MPSs', 'Phenotype(s)', 'Category', 
                            'n CpGs', 'Based on', 'Sample type', 'Sample size', 'n Cases', 'n Controls',
                            'Developmental period', 'Tissue', 'Array', 'Ancestry',
                            'Publication type',]] # 'Keywords', 'Abstract',
                            # 'Including_CpGs_1', 'Including_CpGs_2', 'Including_CpGs_3', 'Including_CpGs_4', 'Including_CpGs_5',
                            # 'Sample_overlap_target_base', 'Determining_weights_1', 'Train_test',
                            # 'Independent_validation', 'Comparison', 'Missing_value_note',
                            # 'Reflect_phenotype', 'Covariates']]

base_targ_data = pd.read_csv(f'{assets_directory}/MPS_base_target_cleaned.csv')

# ==========================================


def _count_papers():
    return pub_table.shape[0]


def _count_mpss():
    return mps_table.shape[0]


def _count_phenotypes(d=mps_table):
    n = int(len(pd.unique(d['Phenotype'])))
    return n

# =============== FILTERING & STYLING TABLES ================

def _filter_litreview_table(selected_category, selected_phenotype, selected_period,
                            selected_year_range, based_on_filter, which_table):
    
    if which_table == 'mps_table':
        table = mps_table_show.copy()
    else:
        table = pub_table_show.copy()

    if len(selected_category) > 0:
        table = table.loc[table['Category'].str.contains('|'.join(list(selected_category)), na=False, regex=True), ]

    if len(selected_phenotype) > 0:
        v_name = 'Phenotype' if which_table == 'mps_table' else 'Phenotype(s)'
        table = table.loc[table[v_name].str.contains('|'.join(list(selected_phenotype)), na=False, regex=True), ]

    if len(selected_period) > 0:
        table = table.loc[table['Developmental period'].str.contains('|'.join(list(selected_period)), na=False, regex=True), ]

    if (selected_year_range[0] > table['Year'].min()) or (selected_year_range[1] < table['Year'].max()):
        table = table.loc[(table['Year'] >= selected_year_range[0]) & (table['Year'] <= selected_year_range[1]), ]

    if len(based_on_filter) < 3: 
        table = table.loc[table['Based on'].str.contains('|'.join(list(based_on_filter)), na=False, regex=True), ]
    
    # Sort by phenotype 
    if which_table == 'mps_table':
        table = table.sort_values(by='Phenotype')
    else:
        table = table.sort_values(by=['Author', 'Year'])
        # Do some extra styling for collapsed values
        for var_to_restyle in ['Phenotype(s)', 'Category', 'Based on', 
                               'Developmental period', 'Tissue', 'Array', 'Ancestry',
                               'n CpGs', 'Sample size','n Cases', 'n Controls']:
            table[var_to_restyle] = table[var_to_restyle].apply(list_to_html)

    table_style = _style_litreview_table(table.reset_index(), which_table=which_table)

    return table, table_style


def _style_litreview_table(table, which_table):
    
    table_style = [
        {'cols': ['Category', 'Author', 'Based on'],
        'style': {'width': '130px', 'max-width': '150px', 'min-width': '110px'}},
        {'cols': ['Title'],
        'style': {'width': '600px', 'max-width': '700px', 'min-width': '500px'}},
        {'cols': [' ', 'Sample size', 'n Cases', 'n Controls'],
        'style': {'width': '30px', 'max-width': '30px', 'min-width': '30px', 'text-align': 'right'}},
        ]
    
    if which_table == 'pub_table':
        table_style.extend([
            {'cols': ['Phenotype(s)'],
             'style': {'width': '200px', 'max-width': '500px', 'min-width': '200px'}},
            {'cols': ['n MPSs'],
             'style': {'text-align': 'center'}}
             ])
    else:
        table_style.extend([
            {'cols': ['Phenotype', 'Category'],
             'style': {'width': '200px', 'max-width': '300px', 'min-width': '180px'}},
             ])
    
    for cat in table['Category'].unique():
        fetch_color = f'var(--light-Category-{cat.replace(" ", "_")})'

        row_color = {'rows': table.index[table['Category'] == cat].tolist(),
                     'style': {'background-color': fetch_color}}
        table_style.append(row_color)
    
    
    return table_style

# ================ PLOTTING ================


def _multilevel_piechart(lvl1='Category', lvl2='Phenotype', color_by="Category",
                         fig_width=1300, fig_height=800, data=mps_table):
    """
        Multilevel pie chart of Category | Phenotypes
    """

    counts = data[[lvl1, lvl2]].value_counts().reset_index()

    # Count publications
    unique_papers = data.groupby(lvl2)['Title'].nunique().reset_index()
    counts = counts.merge(unique_papers, on=lvl2, how='left')

    counts['hover_label'] = counts.apply(
        lambda row: f"<b>{row[lvl2]}</b>"\
                    f"<br>Unique MPSs: {row['count']}"\
                    f"<br>Unique publications: {row['Title']}"\
                    f"<br>Category: {row[lvl1]}", axis=1
    )

    # Color set-up
    if color_by in styles.COLOR_MAPS.keys():
        color_map = styles.COLOR_MAPS[color_by]
    else:
        color_map = "Virdis"
    
    fig = px.sunburst(counts,
                      path=[lvl1, lvl2], values='count',
                      hover_data={'hover_label': True},
                      color = lvl1, color_discrete_map = color_map,
                      width=fig_width, height=fig_height,
                      title='')

    fig.update_traces(hovertemplate='%{customdata[0]}')

    return fig

def _mps_count_histogram(fig_width=1700, fig_height=390, data=mps_table):

    # Group by publication to extract publication category
    pub_category = data.groupby("Title")['Category'].apply(list).reset_index()
    pub_category['Publication category'] = [p[0] if len(set(p)) == 1 else 'Mixed' for p in pub_category['Category']]

    data_with_category = data.merge(pub_category, on='Title', how='left')

    color_map = {'Biological markers': '#113ab7',
                     'Genetic syndromes': '#008080',
                     'Lifestyle and environment': '#ffd000',
                     'Physical health indicators': '#fc9ead',
                     'Neuro-psychiatric health indicators': '#7e04b3',
                     'Cancer': '#a21414'}
    # styles.COLOR_MAPS['Category']
    color_map['Mixed'] = 'grey'

    fig = px.histogram(
            data_with_category,
            x='Title',
            y=None, 
            nbins=len(pd.unique(data.Title)),
            color='Publication category',
            color_discrete_map=color_map)
    
    fig.update_xaxes(showticklabels=False,
                     categoryorder='total descending')
    
    fig.update_layout(
        width=fig_width,
        height=fig_height,
        xaxis_title='<b>Publication</b>',
        yaxis_title='<b>MPS count</b>',
        showlegend=False
    )

    return fig


def _category_over_years(data=mps_table, color_by='Category', 
                         fig_width=1300, fig_height=450,
                         percent=True):
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
                       title='', 
                       width=fig_width, height=fig_height)

    return fig

def _publication_histogram(min_count=1, fig_width=1700, fig_height=350, data=mps_table):

    # Group by publication to extract publication category
    pub_category = data.groupby("Title")['Category'].apply(list).reset_index()
    pub_category['Publication category'] = [p[0] if len(set(p)) == 1 else 'Mixed' for p in pub_category['Category']]
    
    # Match with authors
    pub_authors = data.groupby("Title")['Author_list'].first().reset_index()
    pubs = pd.merge(pub_category.drop('Category', axis=1), pub_authors, on='Title')

    # Explode so that each author-publication pair is a row
    pubs['Author_list'] = pubs['Author_list'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)  # Ensure it's a list
    pubs_by_author = pubs.explode('Author_list').reset_index(drop=True)

    # Prettyfy 
    pubs_by_author.rename(columns={'Author_list': 'Author'}, inplace=True)
    pubs_by_author = pubs_by_author[['Author', 'Title','Publication category']].sort_values(by='Author')

    # Filter authors that appear more than `min_count`
    author_counts = pubs_by_author['Author'].value_counts()
    prolific_authors = author_counts[author_counts > min_count].index
    filtered_pubs_by_author = pubs_by_author[pubs_by_author['Author'].isin(prolific_authors)].reset_index(drop=True)

    # Sort authors by total publication count
    author_order = (
        filtered_pubs_by_author.groupby('Author')
        .size()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # Plot histogram 
    color_map = styles.COLOR_MAPS['Category']
    color_map['Mixed'] = 'grey'

    # Create the histogram
    fig = px.histogram(
        filtered_pubs_by_author,
        x='Author',
        y=None,  # Automatically counts occurrences
        category_orders={'Author': author_order},
        color='Publication category',
        color_discrete_map=color_map,
        # hover_data={'Titles': True, 'Author': False},  # Show titles in hover, hide redundant author info
        title='',
        
    )

    fig.update_layout(
        width=fig_width,
        height=fig_height,
        xaxis_title='<b>Author</b>',
        yaxis_title='<b>Number of publications</b>',
        showlegend=False
    )
    # Rotate x-tick labels 65 degrees and reduce font-size ensure all labels are shown
    fig.update_xaxes(tickangle=65, tickmode='linear', tickfont=dict(size=8))

    # Update hover template
    fig.update_traces(hovertemplate='Author: <b>%{x}</b><br>Publication count: %{y}')

    return fig

    
def _publication_network(fig_width=1300, fig_height=900, data=mps_table):

    # Read the graph object from file
    with open(f'{assets_directory}/Publications_network.pkl', 'rb') as file:
        G = pickle.load(file)

    # Wrap text for hover info
    def wrap_text(text, width=35):
        return '<br>'.join(textwrap.wrap(text, width))
    
    # List categories by publication
    pub_category = data.groupby("Title")['Category'].apply(list).reset_index()

    # Check that no publications cover more that 1 category
    pub_category['Publication category'] = [p[0] if len(set(p)) == 1 else 'Mixed' for p in pub_category['Category']]
    # Clean titles from ":" to match the entries in the network
    pub_category['Title'] = pub_category['Title'].str.replace(":", " ")

    author_node_x = []
    author_node_y = []
    paper_node_x = []
    paper_node_y = []
    paper_node_color = []

    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        if node.startswith('Author/'):
            author_node_x.append(x)
            author_node_y.append(y)
        else:
            paper_node_x.append(x)
            paper_node_y.append(y)

            node_title = node.split('Paper/')[1]
            try: 
                node_category = pub_category.loc[pub_category['Title']==node_title, 'Publication category'].iloc[0]
                node_color = 'grey' if node_category == 'Mixed' else styles.COLOR_MAPS['Category'][node_category]
            except:
                # Matching went wrong, check input 
                print(node_title)
                node_color = 'black'
            
            paper_node_color.append(node_color)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    author_node_trace = go.Scatter(x=author_node_x, y=author_node_y, mode='markers',
                                   marker=dict(size=5, line_width=0.3, color='skyblue'),
                                   hoverinfo='text',
                                   text=[n.split('/')[-1] for n in G.nodes() if n.startswith('Author/')])

    paper_node_trace = go.Scatter(x=paper_node_x, y=paper_node_y, mode='markers',
                                  marker=dict(size=7, line_width=0, color=paper_node_color, symbol='square'),
                                  hoverinfo='text',
                                  text=[f"<b>{wrap_text(n.split('/')[-1])}</b><br>Category: "
                                        f"{pub_category.loc[pub_category['Title'] == n.split('/')[-1], 'Publication category'].squeeze()}" 
                                        for n in G.nodes() if n.startswith('Paper/')])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines',
                            hoverinfo='none',
                            line=dict(width=0.4, color='grey'))

    fig = go.Figure(data=[edge_trace, author_node_trace, paper_node_trace],
                    layout=go.Layout(
                        # title=dict(text="Publication network graph", font=dict(size=16)),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=5,l=5,r=5,t=5),
                        width=fig_width,
                        height=fig_height,
                        plot_bgcolor='white',
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    return fig

# ================== SANKY DIAGRAMS ==================
def sankey(ax, var, left_labels, right_labels, d=base_targ_data, left=' - base', right=' - targ',
           title_left='Base', title_right='Target', spacer=10, fss={'sm': 14, 'l': 15, 'xl': 25}):
    
    counts = pd.DataFrame(d[[f'{var}{left}',f'{var}{right}']].value_counts(dropna=False)).reset_index()
    
    total = counts['count'].sum()
    
    def size_esimator(label_dict, side):
    
        size_list = list()
        
        for label in label_dict.keys():
            label_count = int(counts.loc[counts[f'{var}{side}']==label, 'count'].sum())
            label_dict[label]['size'] = label_count
            size_list.append(label_count)
    
        cumulative_sum = np.cumsum(size_list).tolist()
        
        top_pos = [0] + [c+(spacer*(i+1))for i,c in enumerate(cumulative_sum[:-1])]
        bottom_pos = [c+(spacer*(i))for i,c in enumerate(cumulative_sum)]
    
        for i, label in enumerate(label_dict.keys()):
            label_dict[label]['top'] = top_pos[i]
            label_dict[label]['bottom'] = bottom_pos[i]
        
        return label_dict
    
    left_dict = size_esimator(left_labels, side=left)
    right_dict = size_esimator(right_labels, side=right)
    
    def label_y(label_dict, label):
        if label_dict[label]['size'] > spacer:
            y = label_dict[label]['top']+1
            va='top'
        else:
            y = label_dict[label]['top'] + label_dict[label]['size']*0.5
            va='center'

        string_spacer = '\n' if label_dict[label]['size'] > 5 else ' '
        percent_count = round(label_dict[label]['size'] / total * 100)
        percent_count = percent_count if percent_count > 0 else '<1'
        
        s = f"{label}{string_spacer}({percent_count}%)"
        
        return dict(y=y, s=s, va=va)
        
    # Draw left counts
    for label in left_dict.keys():
        ax.fill_between(x=[0, 1], y1=left_dict[label]['top'], y2=left_dict[label]['bottom'], 
                        color=left_dict[label]['color'], edgecolor=None)
        ax.text(x=-0.1, **label_y(left_dict, label), ha='right', fontsize=fss['sm'])
    
    # Draw right counts
    for label in right_dict.keys():
        ax.fill_between(x=[9, 10], y1=right_dict[label]['top'], y2=right_dict[label]['bottom'], 
                        color=right_dict[label]['color'], alpha=1, edgecolor=None)
        ax.text(x=10.1, **label_y(right_dict, label), ha='left', fontsize=fss['sm'])

    # Add titles on each side
    titlespecs = dict(y=-10, va='center',ha='center', fontweight='bold', fontsize=fss['l'])
    ax.text(x=0.5, s=title_left, **titlespecs)
    ax.text(x=9.5, s=title_right, **titlespecs)
    
    # Draw strips 
    for left_label in left_dict.keys():
        
        for right_label in right_dict.keys():
            
            strip_color = left_dict[left_label]['color'] # Color strip according to the left side
            
            strip_size = counts.loc[(counts[f'{var}{left}']==left_label) & (counts[f'{var}{right}']==right_label), 'count']
    
            
            if  len(strip_size) > 0:
                strip_size = int(strip_size.iloc[0])
    
                # Create array of y values for each strip, half at left value, half at right, convolve
                ys_d = np.array(50 * [left_dict[left_label]['top']] + 50 * [right_dict[right_label]['top']])
                ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode='valid')
                ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode='valid')
                
                ys_u = np.array(50 * [left_dict[left_label]['top'] + strip_size] + 50 * [right_dict[right_label]['top'] + strip_size])
                ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode='valid')
                ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode='valid')
    
                # Update bottom edges at each label so next strip starts at the right place
                left_dict[left_label]['top'] += strip_size
                right_dict[right_label]['top'] += strip_size
                
                ax.fill_between(np.linspace(1, 9, len(ys_d)), ys_d, ys_u, alpha=0.4, color=strip_color, edgecolor=None)
    
    largest_count = max(left_dict[list(left_dict.keys())[-1]]['bottom'], right_dict[list(right_dict.keys())[-1]]['bottom'])
    ax.set_xlim(-0.1, 10.1)
    ax.set_ylim(-10, largest_count+10)
    ax.invert_yaxis()
    ax.axis('off')

    # Add superior title
    ax.set_title(' '.join(var.split('_')), fontweight='bold', fontsize=fss['xl'], pad=25)
    
    # Also return overall overla 
    color_counts = counts.copy()
    color_counts[f'{var}{left}'] = [left_labels[i]['color'] for i in counts[f'{var}_{left}']]
    color_counts[f'{var}{right}'] = [right_labels[i]['color'] for i in counts[f'{var}_{right}']]
    
    match = int(color_counts.loc[(color_counts[f'{var}{left}'] == color_counts[f'{var}{right}']) & (color_counts[f'{var}{left}'] != 'grey'), 
                'count'].sum())
    match_percent = match / total * 100

    return match_percent


def display_match(ax, match, fs=22): 
    ax.text(x=.5, y=.95, s=f'Match: {round(match)}%',fontsize=fs, 
            ha='center', va='center', transform=ax.transAxes)
    ax.axis('off')


def _target_base_sankey(fig_width=200, fig_height=200):

    fss={'sm': 8, 'l': 11, 'xl': 21}
    
    fig, axs = plt.subplot_mosaic('AB;ab;CD;cd', figsize=(fig_height, fig_width),
                                  height_ratios=[1,.2, 1,.2], gridspec_kw=dict(hspace=0, wspace=1.5))
   
    a = sankey(axs['A'], var='Array', 
            right_labels = {'450K': {'color': 'darkgreen'}, 
                        'EPICv1': {'color': 'mediumpurple'},
                        'Multiple (450K, EPICv1)': {'color': 'orange'},
                        'Multiple (450K, GMEL (~3000 CpGs from EPICv1))': {'color': 'orange'},
                        'Multiple (450K, EPICv2)': {'color': 'orange'}},
            left_labels = {'450K': {'color': 'darkgreen'}, 
                        'EPICv1': {'color': 'mediumpurple'}, 
                        'Multiple (450K, EPICv1)': {'color': 'orange'},
                        'Multiple (450K, EPICv2)': {'color': 'orange'},
                        'Multiple (450K, EPICv3)': {'color': 'orange'},
                        'Multiple (450K, EPICv4)': {'color': 'orange'}}, fss=fss)

    display_match(axs['a'], a, fs=fss['l'])

    b = sankey(axs['B'], var='Tissue',
            right_labels = {'Peripheral blood': {'color':'crimson'},
                            'Whole blood': {'color':'crimson'},
                            'Dried bloodspot': {'color':'crimson'},
                            'Blood-clots': {'color':'crimson'},
                            'Cord blood': {'color':'darkred'},
                            'Saliva': {'color':'lightblue'},
                            'Buccal cells': {'color':'teal'},
                            'Tumour cells': {'color':'orange'},
                            'Not reported': {'color':'grey'}}, 
            left_labels = {'Peripheral blood': {'color':'crimson'},
                            'Whole blood': {'color':'crimson'},
                            'Cord blood': {'color':'darkred'},
                            'Multiple (Cord blood, Dried bloodspot)': {'color':'crimson'},
                            'Multiple (Cord blood, Whole blood)': {'color':'crimson'},
                            'Multiple (Whole blood, HPCs)': {'color':'crimson'},
                            'Buccal cells': {'color':'teal'},
                            'Leukocytes': {'color':'mediumpurple'},
                            'Tumour cells': {'color':'orange'}}, fss=fss)

    display_match(axs['b'], b, fs=fss['l'])

    c = sankey(axs['C'], var='Ancestry',
            right_labels = {'White': {'color':'pink'}, 
                            'European': {'color':'pink'}, 
                            'Mixed': {'color':'purple'}, 
                            'Hispanic': {'color': 'orange'},
                            'African': {'color':'crimson'},
                            'Not reported': {'color': 'grey'}}, 
            left_labels = {'White': {'color':'pink'},
                            'European': {'color':'pink'},
                            'Mixed': {'color':'purple'}, 
                            'Hispanic': {'color': 'orange'},
                            'Not reported': {'color':'grey'}}, fss=fss)

    display_match(axs['c'], c, fs=fss['l'])


    dp = sankey(axs['D'], var='Developmental_period',
            right_labels = {'Birth': {'color':'darkblue'}, 
                'Very early childhood': {'color':'#4132d4'}, 
                'Early childhood': {'color':'#4132d4'},
                'Mid childhood': {'color':'#7566ff'}, 
                'Late childhood': {'color':'#beb7ff'}, 
                'Childhood and adolescence': {'color':'#f0cdff'}, 
                'Adolescence': {'color':'purple'},
                'Not reported': {'color':'grey'}},
            left_labels = {'Birth': {'color':'darkblue'}, 
                    'Mid childhood': {'color':'#7566ff'}, 
                    'Late childhood': {'color':'#beb7ff'}, 
                    'Childhood': {'color':'blue'},
                    'Childhood and adolescence': {'color':'#f0cdff'}, 
                    'Birth, Childhood and adolescence': {'color':'#7b07d0'},
                    'Adolescence': {'color':'purple'},
                    'Adults':{'color':'teal'},
                    'Not reported': {'color':'grey'}}, fss=fss)

    display_match(axs['d'], dp, fs=fss['l'])

    return fig


def _sample_size_over_time(data=mps_table, color_by='Category', 
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
        

