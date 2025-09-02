# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "openpyxl==3.1.5",
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

    assets_directory = './assets/'
    output_directory = './static_output/'
    return assets_directory, colorsys, go, mcolors, np, pd, plt


@app.cell
def _(assets_directory, pd):
    mps_table = pd.read_csv(f'{assets_directory}MPS_table_cleaned-tmp.csv')
    pub_table = pd.read_csv(f'{assets_directory}Publication_table_cleaned-tmp.csv')

    d_targ_base = pd.read_csv(f'{assets_directory}MPS_base_target_cleaned.csv')
    return (mps_table,)


@app.cell
def _(mps_table, np, plt):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Verdana"]

    f0 = mps_table.groupby('Year').Title.nunique().plot(figsize=(10, 5), marker='o', ms=9, color='royalblue')
    f0.set_xticklabels([int(x) for x in f0.get_xticks()], rotation=60)
    f0.set_yticks([1] + [int(x) for x in range(5, 26, 5)])
    years = mps_table['Year']
    min_year, max_year = (years.min(), years.max())
    _counts, bin_edges = np.histogram(years, bins=np.arange(mps_table['Year'].min(), mps_table['Year'].max() + 2))
    max_count = _counts.max()
    scaled_counts = _counts * (30 / max_count) if max_count > 0 else _counts
    nmps_bars = f0.bar(bin_edges[:-1], scaled_counts, width=0.4, color='pink', alpha=0.5, edgecolor=None, zorder=1, align='center')
    for bar, count in zip(nmps_bars, _counts):
        f0.text(bar.get_x() + bar.get_width() / 2, -3, str(count), ha='center', va='bottom', color='crimson', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='pink', edgecolor='none', alpha=0.5))
    f0.text(2025, -3, '*Number of MPSs', ha='center', va='bottom', fontsize=10, color='crimson', fontdict={'fontstyle': 'italic'})
    for _spine in f0.spines.values():
        _spine.set_visible(False)
    f0.set_ylim(-10, 35)
    f0.tick_params(axis='y', length=0, labelcolor='royalblue', labelsize=10)
    f0.set_ylabel('Number of publications', fontsize=12, fontweight='bold', labelpad=10)
    f0.grid(axis='y', color='royalblue', alpha=0.3)
    f0.tick_params(axis='x', length=10, width=1, direction='in', color='black')
    f0.hlines(-8.4, 2016.999, 2024, colors='black', linewidth=1)
    f0.set_xlabel('Year', fontsize=12, fontweight='bold', labelpad=10)
    # f0.figure.savefig(f'./static_output/Figure0_MPS-pubs-per-year.png', dpi=300, bbox_inches='tight')

    f0
    return


@app.cell
def _(go, mps_table, pd):

    steps = ['Including_CpGs_1', 'Including_CpGs_2', 'Including_CpGs_3', 'Including_CpGs_4', 'Including_CpGs_5']
    _color_dict = {'Association DNAm phenotype ': 'crimson', 'Biological relevance ': 'darkblue', 'Pruning ': 'purple', 'Reproducibility ': 'forestgreen'}
    colors = dict()
    order = dict()
    for step in steps:
        strategies = mps_table[step].dropna().str.split('|', expand=True)[[0, 1, 2]]
        _counts = pd.DataFrame(strategies[[0, 1]].value_counts(dropna=False)).reset_index()
        order[step] = _counts[1].to_list()
        for _i, _row in _counts.iterrows():
            if _row[1] not in colors.keys():
                colors[_row[1]] = _color_dict[_row[0]] if _row[0] in _color_dict else 'gray'
    df = pd.DataFrame({step: mps_table[step].str.split('|', expand=True)[1] for step in steps})
    df = df[~df.isna().all(axis=1)]
    df = df.fillna('')
    df['dummy'] = ''
    steps = df.columns.tolist()
    labels = []
    xpos = []
    step_label_map = {}
    n_steps = len(steps)
    for _i, step in enumerate(steps):
        step_labels = df[step].unique().astype(str)
        step_map = {}
        for l in step_labels:
            label = f'[{_i + 1}]{l}'
            labels.append(label)
            xpos.append(_i / (n_steps - 1) if n_steps > 1 else 0)
            step_map[l] = len(labels) - 1
        step_label_map[_i] = step_map
    sources = []
    targets = []
    values = []
    for _i in range(len(steps) - 1):
        left = df[steps[_i]].astype(str)
        right = df[steps[_i + 1]].astype(str)
        links = pd.DataFrame({'source': left, 'target': right})
        _counts = links.value_counts().reset_index(name='count')
        for _, _row in _counts.iterrows():
            if _row['source'] in step_label_map[_i] and _row['target'] in step_label_map[_i + 1]:
                src_idx = step_label_map[_i][_row['source']]
                tgt_idx = step_label_map[_i + 1][_row['target']]
                sources.append(src_idx)
                targets.append(tgt_idx)
                values.append(_row['count'])
    node_labels = []
    node_colors = []
    hidden_indices = set()
    for _i, label in enumerate(labels):
        if len(label) < 4:
            node_colors.append('rgba(0,0,0,0)')
            node_labels.append('')
            hidden_indices.add(_i)
        else:
            node_colors.append(colors[label.split(']')[1]])
            node_labels.append(label)
    link_colors = []
    link_hovertemplate = []
    for s, _t in zip(sources, targets):
        if s in hidden_indices or _t in hidden_indices:
            link_colors.append('rgba(0,0,0,0)')
            link_hovertemplate.append(' ')
        else:
            link_colors.append('rgba(128,128,128,0.2)')
            link_hovertemplate.append(f'%{node_labels[s]} → {node_labels[_t]}<extra></extra>')
    _fig = go.Figure(data=[go.Sankey(node=dict(pad=15, thickness=20, x=xpos, label=node_labels, color=node_colors, line=dict(width=0)), link=dict(source=sources, target=targets, value=values, color=link_colors, hovertemplate=['']))])
    for label, _color in _color_dict.items():
        _fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=15, color=_color), legendgroup=label, showlegend=True, name=label))
    _fig.update_layout(title_text='Sankey diagram of CpG inclusion strategies', font_size=10, width=1100, plot_bgcolor='white', paper_bgcolor='white', xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), margin=dict(l=10, r=0, t=50, b=0), legend=dict(orientation='h', y=-0.2, yanchor='bottom', x=0.5, xanchor='center'))
    return


@app.cell
def _(colorsys, mcolors, mps_table, np, plt):
    mps_table.Determining_weights_1.value_counts()


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
    
    weights_split = mps_table['Determining_weights_1'].dropna().str.split('|', expand=True)
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
    # _fig.savefig(f'../static_output/Figure4B.png', dpi=300, bbox_inches='tight')

    _fig
    return


if __name__ == "__main__":
    app.run()
