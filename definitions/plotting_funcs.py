from pathlib import Path

import pandas as pd
import numpy as np
import pickle

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import ast 
import textwrap

import definitions.layout_styles as styles

main_dir = Path(__file__).parent.parent

assets_directory = main_dir / 'assets'

# Review data 

base_targ_data = pd.read_csv(f'{assets_directory}/review_2025/MPS_base_matched_cleaned.csv')

fig_margins = dict(l=0, r=0, t=25, b=0)  # t=25 keeps just enough for the toolbar

def _multilevel_piechart(data, lvl1='Category', lvl2='Phenotype', color_by="Category",
                         fig_width=800, fig_height=800):

    counts = data[[lvl1, lvl2]].value_counts().reset_index()
    counts = counts.merge(data.groupby(lvl2)['Title'].nunique().reset_index(), on=lvl2)

    color_map = styles.COLOR_MAPS.get(color_by, {})

    cat_totals = counts.groupby(lvl1)['count'].sum().reset_index()

    parents = pd.DataFrame({
        'id':     cat_totals[lvl1],
        'label':  cat_totals[lvl1],
        'parent': '',
        'value':  cat_totals['count'],
        'color':  cat_totals[lvl1].map(color_map),
        'hover':  cat_totals.apply(lambda r:
            f"<b>{r[lvl1]}</b><br>Unique MPSs: {r['count']}<br>"
            f"<i>Click to zoom into {r[lvl1]}</i>", axis=1),
    })

    leaves = pd.DataFrame({
        'id':     counts[lvl1] + '/' + counts[lvl2],
        'label':  counts[lvl2],
        'parent': counts[lvl1],
        'value':  counts['count'],
        'color':  counts[lvl1].map(color_map),
        'hover':  counts.apply(lambda r:
            f"<b>{r[lvl2]}</b><br>Unique MPSs: {r['count']}<br>"
            f"Unique publications: {r['Title']}<br>Category: {r[lvl1]}", axis=1),
    })

    nodes = pd.concat([parents, leaves], ignore_index=True)

    fig = go.Figure(go.Sunburst(
        ids=nodes['id'], labels=nodes['label'], parents=nodes['parent'],
        values=nodes['value'], branchvalues='total',
        marker=dict(colors=nodes['color']),
        customdata=nodes['hover'],
        hovertemplate='%{customdata}<extra></extra>',
    ))

    fig.update_layout(width=fig_width, height=fig_height, margin=fig_margins)
    return fig

def _phenotype_pub_counts(data, min_publications=3, fig_width=300, fig_height=800):
    """
    Horizontal bar chart of publication count per phenotype, colored by category.
    Only phenotypes in at least `min_publications` publications are shown.
    """
    color_map = styles.COLOR_MAPS.get('Category', {})

    agg = (
        data.groupby('Phenotype')
        .agg(publications=('Title', 'nunique'), mps_count=('Phenotype', 'count'), category=('Category', 'first'))
        .reset_index()
        .query('publications >= @min_publications')
        .sort_values('publications', ascending = True)
    )

    fig = px.bar(
        agg, x='publications', y='Phenotype', orientation='h',
        color='category', color_discrete_map=color_map,
        custom_data=['category', 'mps_count'],
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Category: %{customdata[0]}<br>"
            "Publications: %{x}<br>"
            "Unique MPSs: %{customdata[1]}"
            "<extra></extra>"
        )
    )

    axes_style = dict(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig.update_xaxes(title_text='<b>Number of publications</b>', **axes_style)
    fig.update_yaxes(title_text='', showticklabels=False, **axes_style, 
                     categoryorder='total ascending')
    fig.update_layout(width=fig_width, height=fig_height, margin=fig_margins, 
                      showlegend=False, plot_bgcolor='whitesmoke')
    return fig


def _mps_count_histogram(data, fig_width=1700, fig_height=390):

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
        showlegend=False,
        margin=fig_margins
    )

    return fig


def _category_over_years(data, color_by='Category', 
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

def _publication_histogram(data: pd.DataFrame, min_count: int = 1,
                           fig_width: int = 1700, fig_height: int = 350):
    """
    Parameters
    ----------
    data : one row per publication, must have columns:
          'Title', 'Author_list', 'Category'
    """

    # ── Resolve publication category ──────────────────────────────
    # Category can vary across MPS rows for same publication — keep first
    # (data should already be deduplicated, but guard anyway)
    pubs = data.drop_duplicates(subset="Title").copy()
    pubs["Publication category"] = pubs["Category"].apply(
        lambda c: c if pd.notna(c) and c else "Unknown"
    )

    # ── Parse Author_list ─────────────────────────────────────────
    def _to_list(x) -> list:
        if isinstance(x, list):
            return x
        try:
            return ast.literal_eval(x)
        except Exception:
            return [x] if isinstance(x, str) else list(x)

    pubs = pubs.copy()
    pubs["Author_list"] = pubs["Author_list"].apply(_to_list)

    # ── Explode to one row per author-publication pair ────────────
    pubs_by_author = (
        pubs[["Author_list", "Title", "Publication category"]]
        .explode("Author_list")
        .rename(columns={"Author_list": "Author"})
        .sort_values("Author")
        .reset_index(drop=True)
    )

    # ── Filter to authors with more than min_count publications ───
    author_counts    = pubs_by_author["Author"].value_counts()
    prolific_authors = author_counts[author_counts > min_count].index
    filtered         = pubs_by_author[pubs_by_author["Author"].isin(prolific_authors)].reset_index(drop=True)

    # ── Sort by total publication count ───────────────────────────
    author_order = (
        filtered.groupby("Author")
        .size()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # ── Plot ──────────────────────────────────────────────────────
    color_map = {**styles.COLOR_MAPS["Category"], "Mixed": "grey", "Unknown": "lightgrey"}

    fig = px.histogram(
        filtered,
        x="Author",
        color="Publication category",
        category_orders={"Author": author_order},
        color_discrete_map=color_map,
        title="",
    )

    fig.update_layout(
        width=fig_width,
        height=fig_height,
        xaxis_title="<b>Author</b>",
        yaxis_title="<b>Number of publications</b>",
        showlegend=False,
        margin=fig_margins
    )
    fig.update_xaxes(tickangle=65, tickmode="linear", tickfont=dict(size=8))
    fig.update_traces(hovertemplate="Author: <b>%{x}</b><br>Publication count: %{y}")

    return fig


def _publication_network(data, nx_file, fig_width=1300, fig_height=900):

    # Read the graph object from file
    with open(nx_file, 'rb') as file:
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


def _sample_size_over_time(data, color_by='Category', 
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
                     title='')
    # Make it pretty
    fig.update_traces(marker=dict(size=10, opacity=.5))
    axes_style = dict(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    ylabel_note = ' (log scale)' if log_sample_size else ''
    fig.update_yaxes(title_text=f'<b>Sample size</b>{ylabel_note}', **axes_style)
    fig.update_xaxes(title_text='Publication date', **axes_style)
    fig.update_layout(plot_bgcolor='whitesmoke', width=1300, height=400, margin=fig_margins)

    return fig
        

# ===== SANKY DIAGRAMS =====================================================

def sankey(ax, var, left_labels, right_labels, data, 
           left = ' [development]', right = ' [application]',
           title_left='Development\ndataset', title_right='Application\ndataset', 
           spacer=10, fss={'sm': 14, 'l': 15, 'xl': 25}):
    
    counts = pd.DataFrame(data[[f'{var}{left}',
                                f'{var}{right}']].value_counts(dropna=False)).reset_index()

    # Check specified labels are correct and complete
    def check_labels(label_dict, side):
        labels_found = set(counts[f'{var}{side}'])
        labels_requested = set(label_dict.keys())
        
        # Elements found but not requested
        others = list(set(labels_found) - set(labels_requested))
        if len(others) > 0:
            print(f'{var}{side}: {len(others)} labels were found in data but not specified ' 
                  f'in label dict, these will be grouper under "Other": {others}')

            counts[f'{var}{side}'] = counts[f'{var}{side}'].replace({l: "Other" for l in others})
            label_dict['Other'] = {'color': 'grey'}
        
        missing = list(set(labels_requested) - set(labels_found))
        if len(missing) > 0:
            print(f'{var}{side}: {len(missing)} labels specified in label dictionary'
                  f'were not found in data: {missing}')
            
            label_dict = {k: v for k, v in label_dict.items() if k not in missing}

        return counts, label_dict
    
    counts, left_labels = check_labels(left_labels, side=left)
    counts, right_labels = check_labels(right_labels, side=right)

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
        
        # TMP case specific string handling 
        if label == 'Multiple (450K, GMEL (~3000 CpGs from EPICv1))':
            label = 'Multiple (450K, GMEL*)'

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
    titlespecs = dict(y=-15, va='center',ha='center', fontweight='bold', fontsize=fss['l'])
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
    ax.set_title(' '.join(var.split('_')), fontweight='bold', fontsize=fss['xl'], pad=28)
    
    # Also return overall overlap
    color_counts = counts.copy()
    color_counts[f'{var}{left}'] = [left_labels[i]['color'] for i in counts[f'{var}{left}']]
    color_counts[f'{var}{right}'] = [right_labels[i]['color'] for i in counts[f'{var}{right}']]
    
    match = int(color_counts.loc[(color_counts[f'{var}{left}'] == color_counts[f'{var}{right}']) & (color_counts[f'{var}{left}'] != 'grey'), 
                'count'].sum())
    match_percent = match / total * 100

    return match_percent


def display_match(ax, match, fs=22, note=None):
            
    ax.text(x=.4, y=.80, s='Match:',fontsize=fs, 
            ha='center', va='center', transform=ax.transAxes)
    ax.text(x=.6, y=.80, s=f'{round(match)}%',fontsize=fs+4, fontweight='bold',
            ha='center', va='center', transform=ax.transAxes)
    if note:
        ax.text(x=.9, y=.99, s=note,fontsize=fs-3, fontstyle='italic',
            ha='left', va='center', transform=ax.transAxes)
    ax.axis('off')


def _single_sankey(var, right_label_order, left_label_order, df = base_targ_data, note=None,
                   filter = None, color_maps = styles.COLOR_MAPS, 
                   fig_width=8, fig_height=10):
    '''Draw a single sankey diagram for a given variable'''

    if filter != "All application studies":
        data = df.loc[df['Based on'] == filter, ]
    else:
        data = df

    fss={'sm': 8, 'l': 11, 'xl': 21}

    color_dict = color_maps[var]

    # ── Fall back to full color map if no order specified ─────────
    right_order = right_label_order if right_label_order else list(color_dict.keys())
    left_order  = left_label_order  if left_label_order  else list(color_dict.keys())

    right_labels = {label: {'color': color_dict[label]} for label in right_order if label in color_dict}
    left_labels  = {label: {'color': color_dict[label]} for label in left_order  if label in color_dict}

    fig, axs = plt.subplot_mosaic('A;a', figsize=(fig_width, fig_height),
                                  height_ratios=[1,.27], gridspec_kw=dict(hspace=0, wspace=1.2))
    
    main_plot = sankey(axs['A'], var=var, right_labels = right_labels, left_labels = left_labels, 
                       data=data, fss=fss)

    display_match(axs['a'], main_plot, fs=fss['l'], note=note)

    # Add this to ensure proper layout in Shiny
    fig.subplots_adjust(left=0.25, right=0.75, bottom=0.1, top=0.9)

    return fig

