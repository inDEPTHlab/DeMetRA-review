# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "openpyxl==3.1.5",
#     "upsetplot==0.9.0",
# ]
# ///

import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import pandas as pd
    import numpy as np
    import openpyxl

    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import colorsys
    from upsetplot import UpSet, from_indicators

    assets_directory = './assets/'
    output_directory = './static_output/'
    return (
        UpSet,
        assets_directory,
        colorsys,
        from_indicators,
        mcolors,
        mo,
        np,
        output_directory,
        pd,
        plt,
    )


@app.cell
def _(assets_directory, pd):
    mps_table = pd.read_csv(f'{assets_directory}MPS_table_cleaned.csv')
    pub_table = pd.read_csv(f'{assets_directory}Publication_table_cleaned.csv')

    mps_base_matched = pd.read_csv(f'{assets_directory}MPS_base_matched_cleaned.csv')
    return mps_base_matched, mps_table


@app.cell
def _(plt):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Verdana"]
    return


@app.cell
def _(mps_table, np):

    # Number of publication line
    pub_count = mps_table.groupby('Year').Title.nunique()
    f0 = pub_count.plot(figsize=(10, 5), marker='o', ms=9, color='royalblue')

    # Add projection?

    # Histogram
    years = mps_table['Year']
    min_year, max_year = (years.min(), years.max())
    _counts, bin_edges = np.histogram(years, bins=np.arange(min_year, max_year + 2))
    max_count = _counts.max()
    scaled_counts = _counts * ((max(pub_count)+1) / max_count) if max_count > 0 else _counts
    nmps_bars = f0.bar(bin_edges[:-1], scaled_counts, width=0.4, color='pink', alpha=0.5, edgecolor=None, zorder=1, align='center')

    for bar, count in zip(nmps_bars, _counts):
        f0.text(bar.get_x() + bar.get_width() / 2, -3, str(count), ha='center', va='bottom', color='crimson', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='pink', edgecolor='none', alpha=0.5))

    f0.text(max_year+1, -5, '*Number of MPSs', ha='center', va='bottom', fontsize=10, color='crimson', fontdict={'fontstyle': 'italic'})

    # Axes 
    f0.set_xticks([int(x) for x in pub_count.index])
    f0.tick_params(axis='x', length=10, width=1, direction='in', color='black')
    f0.hlines(-8.35, min_year, max_year, colors='black', linewidth=1)

    f0.set_yticks([1] + [int(x) for x in range(5, max(pub_count), 5)] + [max(pub_count)+1])
    for _spine in f0.spines.values():
        _spine.set_visible(False)
    f0.set_ylim(-10, 35)
    f0.tick_params(axis='y', length=0, labelcolor='royalblue', labelsize=10)
    f0.grid(axis='y', color='royalblue', alpha=0.3)

    f0.set_ylabel('Number of publications', fontsize=12, fontweight='bold', labelpad=10)
    f0.set_xlabel('Year', fontsize=12, fontweight='bold', labelpad=10)

    # f0.figure.savefig(f'./static_output/Figure0_MPS-pubs-per-year.png', dpi=300, bbox_inches='tight')
    f0
    return


@app.cell
def _(plt, single_sankey):
    # Figure 3 

    f3, axs = plt.subplot_mosaic('AB;ab;CD;cd', figsize=(20, 25),
                                 height_ratios=[1,.27]*2, gridspec_kw=dict(hspace=0, wspace=1.2))

    fss={'sm': 11, 'l': 14, 'xl':16, 'xxl': 25}

    single_sankey([axs['A'], axs['a']], var='Array', 
                   left_label_order = ['450K', 'EPICv1',
                                       'Multiple (450K, EPICv1)', 
                                       'Multiple (450K, EPICv1, PCR)', 
                                       'Multiple (450K, PCR)'], 
                    right_label_order = ['450K', 'EPICv1', 'EPICv2', 'Nanopore sequencing',
                                       'Multiple (450K, EPICv1)',
                                       'Multiple (450K, GMEL (~3000 CpGs from EPICv1))',
                                       'Multiple (450K, EPICv2)', 'Not reported'],
                    note='* ~3000 CpGs from EPICv1', fss=fss)

    single_sankey([axs['B'], axs['b']], var='Tissue', 
                   left_label_order = ['Peripheral blood', 'Whole blood', 'Blood', 
                                       'Cord blood', 'Placenta', 'Multiple (Placenta, Cord blood)', 
                                       'Multiple (Cord blood, Dried bloodspot)',
                                       'Multiple (Cord blood, Whole blood)',
                                       'Multiple (Whole blood, HPCs)',
                                       'Multiple (Whole blood, Nasal epithelial cells)',
                                       'Buccal cells',
                                       'Leukocytes',
                                       'Tumour cells'], 
                    right_label_order = ['Peripheral blood', 'Whole blood', 'Blood', 'Dried bloodspot', 'Blood-clots',
                                       'Cord blood', 'Placenta',
                                       'Saliva', 'Buccal cells', 'Nasal epithelial cells', 'Tumour cells',
                                       'Not reported'], fss=fss)

    single_sankey([axs['C'], axs['c']], var='Ancestry', 
                   left_label_order = ['White', 'European', 'Mixed', 'Hispanic', 'Not reported'],
                   right_label_order = ['White', 'European', 'Mixed', 'Hispanic', 'African', 'Not reported'], fss=fss)

    single_sankey([axs['D'], axs['d']], var='Developmental period', 
                  left_label_order = ['Birth', 'Mid childhood', 'Late childhood', 'Childhood',
                                      'Childhood and adolescence', 'Birth, Childhood and adolescence',
                                      'Multiple (Birth to Adolescence)',
                                      'Adolescence', 'Adulthood', 'Not reported'],
                   right_label_order = ['Birth', 'Very early childhood', 'Early childhood', 'Mid childhood',
                                      'Late childhood', 'Childhood', 'Childhood and adolescence', 'Adolescence', 
                                      'Not reported'], fss=fss)

    # f3.savefig('./static_output/Figure3_dev-app-sankey.png', dpi=300, bbox_inches='tight')
    return


@app.cell
def _(from_indicators, mo, mps_table, np, pd):
    drs = mps_table[[f'Dimension reduction ({i})' for i in range(1, 6)]]

    def safe_split(cell):
        if isinstance(cell, str):
            parts = cell.split(' | ')
            # Pad with np.nan if not enough parts
            while len(parts) < 3:
                parts.append(np.nan)
            return parts
        else:
            return [np.nan, np.nan, np.nan]  # or other defaults

    split_drs = pd.DataFrame([
        {'orig_col': col, 'category': parts[0], 'method': parts[1], 'cutoff': parts[2], 'row': idx}
        for idx, row in drs.iterrows()
        for col, cell in row.items()
        for parts in [safe_split(cell)]
    ])

    split_drs['method'] = split_drs['method'].replace('? ', np.nan)

    # Drop rows where method is nan
    split_drs_clean = split_drs.dropna(subset=['method'])

    # Pivot: rows = samples, columns = methods
    upset_df = split_drs_clean.pivot_table(
        index='row', # ['row', 'orig_col']
        columns='method',
        aggfunc='size',
        fill_value=0
    )
    upset_df = upset_df.astype(bool)
    drs_membership = from_indicators(upset_df)

    mo.ui.tabs({'dimred': split_drs_clean, 'dimred_bool': upset_df})
    return drs_membership, split_drs_clean


@app.cell
def _(UpSet, drs_membership, mcolors, output_directory, split_drs_clean):
    f4a = UpSet(drs_membership, subset_size='count', sort_by='cardinality', sort_categories_by='-cardinality')

    category_methods = split_drs_clean.groupby('category').method.unique().to_dict()
    category_colors = {'Association DNAm phenotype': 'crimson',
                       'Biological relevance': 'navy',
                       'Pruning': 'purple',
                       'Reproducibility': 'green'}
    for category in category_methods.keys():
        f4a.style_categories(categories=category_methods[category], bar_facecolor=category_colors[category],
                             shading_facecolor = mcolors.to_rgba(category_colors[category], alpha=0.2))
    axplot = f4a.plot()['matrix']

    axplot.figure.savefig(f'{output_directory}Figure4A.png', dpi = 300)
    return (category_colors,)


@app.cell
def _(category_colors, plt):
    fig, ax = plt.subplots(figsize=(4, len(category_colors) * 0.7))

    for i, (label, color) in enumerate(category_colors.items()):
        ax.scatter(0, i, s=400, color=color)  # s=400 for big dot
        ax.text(0.2, i, label, va="center", fontsize=15)

    ax.set_xlim(-0.1, 1)
    ax.set_ylim(-0.5, len(category_colors) - 0.5)
    ax.invert_yaxis()
    ax.axis("off")  # Hide axes

    fig
    return


@app.cell
def _(colorsys, mcolors, mps_table, np, output_directory, plt):
    def generate_palette(base_color, n, min_light=0.35, max_light=0.85):
        """Generate n visually distinct colors from a base color by varying lightness."""
        base = mcolors.to_rgb(base_color)
        colors = []
        h, l, s = colorsys.rgb_to_hls(*base)
        for _i in range(n):
            light = min_light + (max_light - min_light) * _i / max(1, n - 1)
            rgb = colorsys.hls_to_rgb(h, light, s)
            colors.append(rgb)
        return colors

    weights_split = mps_table['Weights estimation'].dropna().str.split('|', expand=True)
    weights_split.columns = ['Main', 'Sub']
    weights_split['Main'] = weights_split['Main'].str.strip()
    weights_split['Sub'] = weights_split['Sub'].str.strip()
    stacked_counts = weights_split.groupby(['Main', 'Sub']).size().unstack(fill_value=0)
    sub_order = stacked_counts.sum(axis=0).sort_values(ascending=False).index
    stacked_counts = stacked_counts[sub_order]
    main_base_colors = ['#ff6600', '#3ca55c', '#205493']
    main_groups = stacked_counts.index.tolist()
    colormaps = [plt.cm.Oranges_r, plt.cm.Greens_r, plt.cm.Blues_r]
    _color_dict = {}
    all_subs = stacked_counts.columns.tolist()
    for _i, main in enumerate(main_groups):
        subs_for_main = [sub for sub in all_subs if stacked_counts.loc[main, sub] > 0]
        palette = generate_palette(main_base_colors[_i], len(subs_for_main))
        for j, sub in enumerate(subs_for_main):
            _color_dict[main, sub] = palette[j]
    _fig, (_ax, leg) = plt.subplots(2, 1, figsize=(15, 6), gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.3})
    gap = 0.5
    bottom = np.zeros(len(main_groups))
    for _idx, sub in enumerate(all_subs):
        vals = stacked_counts[sub].values
        _color = [_color_dict.get((main, sub), '#cccccc') for main in main_groups]
        _ax.barh(main_groups, vals, left=bottom, color=_color, label=sub)
        vals_with_gap = vals.astype(float)
        vals_with_gap[vals != 0] = vals_with_gap[vals != 0] + gap
        bottom = bottom + vals_with_gap
    _ax.set_xlabel('MPS count', fontweight='bold')
    _ax.set_ylabel('Method', fontweight='bold')
    _ax.set_yticklabels(['Discovery\nEWAS', 'Other\nmachine-learning\nalgorithm', 'Penalized\nregression'])
    _ax.set_title('Overview of methods for determining weights')
    _ax.set_xlim(0, 350)
    _ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=True)
    for _spine in ['top', 'right', 'left']:
        _ax.spines[_spine].set_visible(False)
    n_main = len(main_groups)
    x_gap = 2.5
    for _i, main in enumerate(main_groups):
        subs_for_main = [sub for sub in all_subs if stacked_counts.loc[main, sub] > 0]
        for j, sub in enumerate(subs_for_main):
            y = j + 1
            x = _i * x_gap + x_gap
            _color = _color_dict[main, sub]
            leg.scatter(x, y, color=_color, s=60)
            leg.text(x - 0.2, y, sub, va='center', fontsize=10)
    max_rows = max([sum(stacked_counts.loc[main] > 0) for main in main_groups])
    leg.set_xlim(-0.5, n_main * x_gap + 0.1)
    leg.set_ylim(0, max_rows + 2)
    leg.invert_yaxis()
    leg.invert_xaxis()
    leg.axis('off')
    _fig.savefig(f'{output_directory}/Figure4B.png', dpi=300, bbox_inches='tight')

    _fig

    return


@app.cell
def _():
    array_color_map = {'450K': 'darkgreen',
                       'EPICv1': 'mediumpurple',
                       'EPICv2': 'darkblue',
                       'Nanopore sequencing': 'orangered',
                       'Multiple (450K, EPICv1)': 'orange',  #'seagreen',
                       'Multiple (450K, GMEL (~3000 CpGs from EPICv1))': 'orange',
                       'Multiple (450K, EPICv2)': 'orange', 
                       'Multiple (450K, EPICv1, PCR)': 'orange',
                       'Multiple (450K, PCR)': 'orange',
                       'Not reported': 'grey'}

    # TODO: read colors directly from CSS file
    category_color_map = {'Biological markers': '#113ab7',
                         'Genetic syndromes': '#008080',
                         'Lifestyle and environment': '#ffd000',
                         'Physical health indicators': '#fc9ead',
                         'Neuro-psychiatric health indicators': '#7e04b3',
                         'Cancer': '#a21414'}

    tissue_color_map = {'Peripheral blood': 'crimson',
                        'Whole blood': 'crimson',
                        'Blood': 'crimson',
                        'Dried bloodspot': 'crimson', # 'indianred',
                        'Blood-clots': 'crimson', #'blueviolet',
                        'Cord blood': 'darkred', # 'brown',
                        'Placenta': 'pink',
                        'Saliva': 'lightblue',
                        'Buccal cells': 'teal', # 'steelblue',
                        'Leukocytes': 'mediumpurple',
                        'Tumour cells': 'orange', #'darkgreen',
                        'Multiple (Cord blood, Dried bloodspot)': 'crimson',
                        'Multiple (Cord blood, Whole blood)': 'crimson',
                        'Multiple (Whole blood, HPCs)': 'crimson',
                        'Multiple (Placenta, Cord blood)': 'darkred',
                        'Multiple (Whole blood, Nasal epithelial cells)': 'orange',
                        'Nasal epithelial cells': 'teal',
                        'Cervical cells': 'magenta',
                        'Not reported': 'grey'}

    ancestry_color_map = {'White': 'pink',
                          'European': 'pink',
                          'Mixed': 'purple',
                          'African': 'crimson',
                          'Hispanic': 'orange',
                          'Not reported': 'grey'}

    period_color_map = {'Birth': 'darkblue', 
                        'Very early childhood': '#4132d4', 
                        'Early childhood': '#4132d4',
                        'Mid childhood': '#7566ff',
                        'Childhood': 'blue',
                        'Late childhood': '#beb7ff', 
                        'Childhood and adolescence':'#f0cdff', 
                        'Birth, Childhood and adolescence': '#7b07d0',
                        'Multiple (Birth to Adolescence)': '#7b07d0',
                        'Adolescence': 'purple',
                        'Adulthood': 'teal',
                        'Not reported': 'grey'}

    color_maps = {'Category': category_color_map,
                  'Tissue': tissue_color_map,
                  'Array': array_color_map, 
                  'Ancestry': ancestry_color_map,
                  'Developmental period': period_color_map}
    return (color_maps,)


@app.cell
def _(color_maps, mps_base_matched, np, pd):
    # ================== SANKY DIAGRAMS ==================
    def sankey(ax, var, left_labels, right_labels, data=mps_base_matched, 
               left = ' [development]', right = ' [application]',
               title_left='Development\ndataset', title_right='Application\ndataset', 
               spacer=10, fss={'sm': 14, 'l': 15, 'xxl': 25}):
    
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
        ax.set_title(' '.join(var.split('_')), fontweight='bold', fontsize=fss['xxl'], pad=30)
    
        # Also return overall overlap
        color_counts = counts.copy()
        color_counts[f'{var}{left}'] = [left_labels[i]['color'] for i in counts[f'{var}{left}']]
        color_counts[f'{var}{right}'] = [right_labels[i]['color'] for i in counts[f'{var}{right}']]
    
        match = int(color_counts.loc[(color_counts[f'{var}{left}'] == color_counts[f'{var}{right}']) & (color_counts[f'{var}{left}'] != 'grey'), 
                    'count'].sum())
        match_percent = match / total * 100

        return match_percent


    def display_match(ax, match, fss, note=None):
            
        ax.text(x=.4, y=.80, s='Match:',fontsize=fss['l'], 
                ha='center', va='center', transform=ax.transAxes)
        ax.text(x=.6, y=.80, s=f'{round(match)}%',fontsize=fss['xl'], fontweight='bold',
                ha='center', va='center', transform=ax.transAxes)
        if note:
            ax.text(x=.9, y=.99, s=note,fontsize=fss['sm'], fontstyle='italic',
                ha='left', va='center', transform=ax.transAxes)
        ax.axis('off')

    def single_sankey(axs, var, right_label_order, left_label_order, note=None,
                       color_maps = color_maps, fss={'sm': 8, 'l': 11, 'xl': 21}):
        '''Draw a single sankey diagram for a given variable'''

        color_dict = color_maps[var]

        right_labels = {label: {'color': color_dict[label]} for label in right_label_order}
        left_labels = {label: {'color': color_dict[label]} for label in left_label_order}

        # fig, axs = plt.subplot_mosaic('A;a', figsize=(fig_width, fig_height),
        #                               height_ratios=[1,.27], gridspec_kw=dict(hspace=0, wspace=1.2))
    
        main_plot = sankey(axs[0], var=var, right_labels = right_labels, left_labels = left_labels, 
                           data=mps_base_matched, fss=fss)

        display_match(axs[1], main_plot, fss=fss, note=note)

        return axs
    return (single_sankey,)


if __name__ == "__main__":
    app.run()
