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
def _(mo):
    mo.md(
        r"""
    # DeMetRA - literature review

    This the descriptives pipeline for the metadata included in the DeMetRA review. Inputs to the pipeline are: 

    - The `MPS_table_cleaned.csv`
    - The `Publication_table_cleaned.csv`

    Both are created by the `preprocessing/data_preprocessing.py` script.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import openpyxl

    from pyprojroot.here import here

    from scipy.stats import spearmanr
    import matplotlib.pyplot as plt

    proj_folder = str(here())
    assets_directory = f'{proj_folder}/assets/'

    # import os
    # os.getcwd()
    return assets_directory, mo, np, pd, plt, spearmanr


@app.cell
def _(assets_directory, pd):
    mps_table = pd.read_csv(f'{assets_directory}MPS_table_cleaned.csv')
    pub_table = pd.read_csv(f'{assets_directory}Publication_table_cleaned.csv')
    mps_base_matched = pd.read_csv(f'{assets_directory}MPS_base_matched_cleaned.csv')
    return mps_base_matched, mps_table, pub_table


@app.cell
def _(mo):
    mo.md(r"""### General descriptives""")
    return


@app.cell
def _(mps_table, pub_table):
    print('Total number of publications:', pub_table.shape[0])
    print('Total number of unique MPSs:', mps_table.shape[0])

    pubs_count = mps_table.groupby('Based on')['Title'].nunique().reset_index().set_index('Based on')
    pubs_count['%'] = round(pubs_count['Title'] / pub_table.shape[0] * 100, 1)
    print('\n', pub_table['Based on'].value_counts())
    pubs_count
    return


@app.cell
def _(pub_table):
    print(pub_table['n MPSs'].describe())

    _counts = pub_table['n MPSs'].value_counts().sort_index(ascending=True)
    _percent = _counts / _counts.sum() * 100
    n_MPS_count = _counts.to_frame(name='count')
    n_MPS_count['%'] = _percent.apply(lambda x: f'{x:.1f}%')

    print('\n', round(100 - sum(_percent.iloc[0:10])), '% of publications report more than 10 MPSs.')
    n_MPS_count.T
    return


@app.cell
def _(mps_table):
    pheno_count = mps_table.Phenotype.value_counts()
    n_pheno = len(pheno_count)
    pheno_count_rep = pheno_count[pheno_count > 1]
    n_pheno_rep = len(pheno_count_rep)
    print('Total umber of phenotypes:', n_pheno)
    print(f'Number of phenotypes with more than one MPS: {n_pheno_rep} ({round(n_pheno_rep / n_pheno * 100)}%)')
    pheno_count_rep[pheno_count_rep > 9]


    print(pheno_count.loc[pheno_count.index.str.contains('smoking'), ])
    print(pheno_count.loc[pheno_count.index.str.contains('Kabuki'), ])

    pheno_count.sort_index()

    return


@app.cell
def _(mps_base_matched, mps_table, pd, pub_table):
    def get_count_percent(variable, dataset, precision=0):
        _counts = pd.DataFrame(dataset[variable].value_counts())
        _total = _counts.sum()
        _counts['%'] = round(_counts / _total * 100, precision)
        _counts.index = _counts.index.map(lambda x: str(x) if isinstance(x, list) else x)

        return(_counts)

    def get_mps_pub_counts(variable, precision=0, pheno_count = True):

        mps_counts = get_count_percent(variable, mps_table, precision)
        pub_counts = get_count_percent(variable, pub_table, precision)

        # print('\n')
        for _idx, _row in mps_counts.iterrows():
            fmt_mps_count = int(_row['count'])
            fmt_mps_percent = int(_row['%']) if precision == 0 else _row['%']

            _pub_count = pub_counts.loc[pub_counts.index.str.contains(_idx, regex=False), ]
            fmt_pub_total = int(_pub_count['count'].sum())
            fmt_pub_percent = int(_pub_count['%'].sum())

            if pheno_count:
                # Also count number of unique phenotypes
                pheno_counts = mps_table.loc[mps_table[variable]==_idx, 
                    'Phenotype'].value_counts()
                pheno_total = len(pheno_counts.index)
                pheno_top = pheno_counts.index[:3]

                addon = f'\n\tUnique pheno: {pheno_total}\ttop: {list(pheno_top)}\n'
            else:
                addon = '\n'

            print(f'{_idx}\n'
                  f'\tMPS: {fmt_mps_count} ({fmt_mps_percent}%)'
                  f'\tPubs: {fmt_pub_total} ({fmt_pub_percent}%){addon}')

    def get_base_mismatch(variable, df = mps_base_matched, exclude_from_mismatch=[],
                          x=' [development]', y=' [application]'):

        df_subset = df.loc[df[f"{variable}{x}"] != df[f"{variable}{y}"], :]

        value_counts = pd.DataFrame((df_subset[f"{variable}{x}"].astype(str) +" --- "+ 
                        df_subset[f"{variable}{y}"].astype(str)).value_counts())

        value_counts['%'] = round(value_counts / df.shape[0] * 100,1)

        if len(exclude_from_mismatch) > 0: 
            value_counts = value_counts.drop(index=exclude_from_mismatch)

        print(f'Total mismatching: {round(value_counts['%'].sum(), 1)}%')

        return value_counts
    return get_base_mismatch, get_count_percent, get_mps_pub_counts


@app.cell
def _(get_mps_pub_counts):
    get_mps_pub_counts('Category')
    return


@app.cell
def _(get_mps_pub_counts):
    get_mps_pub_counts('Based on')
    return


@app.cell
def _(
    get_base_mismatch,
    get_count_percent,
    get_mps_pub_counts,
    mps_base_matched,
):
    get_mps_pub_counts('Developmental period', precision=1)
    print(get_count_percent('Developmental period [development]', mps_base_matched))
    get_base_mismatch('Developmental period')
    return


@app.cell
def _(get_base_mismatch, get_mps_pub_counts):
    get_mps_pub_counts("Tissue")
    get_base_mismatch('Tissue', 
                      exclude_from_mismatch=['Whole blood --- Peripheral blood',
                                             'Whole blood --- Blood-clots',
                                             'Whole blood --- Dried bloodspot',
                                             'Whole blood --- Blood',
                                             'Peripheral blood --- Blood-clots',
                                             'Peripheral blood --- Whole blood',
                                             'Blood --- Peripheral blood',
                                             'Peripheral blood --- Blood'])
    return


@app.cell
def _(get_base_mismatch, get_mps_pub_counts):
    get_mps_pub_counts('Array')
    get_base_mismatch('Array')
    return


@app.cell
def _(
    get_base_mismatch,
    get_count_percent,
    get_mps_pub_counts,
    mps_base_matched,
):
    get_mps_pub_counts('Ancestry')
    print(get_count_percent('Ancestry [development]', mps_base_matched))
    get_base_mismatch('Ancestry', exclude_from_mismatch=[
        'European --- White',
        'Mixed --- European',
        'Mixed --- White',
        'Hispanic --- European',
        'White --- European'
    ])
    return


@app.cell
def _(get_mps_pub_counts):
    get_mps_pub_counts('Sample type')
    return


@app.cell
def _():
    # get_mps_pub_counts('Sample_overlap_target_base')
    return


@app.cell
def _(mo, mps_table):
    print(mps_table[['n CpGs','Sample size']].describe())

    n_cpg_hist = mps_table['n CpGs'].hist(bins=100, figsize=(15, 4))
    n_obs_hist = mps_table['Sample size'].hist(bins=100, figsize=(15, 4))

    mo.ui.tabs({"n CpG": n_cpg_hist, "Sample size": n_obs_hist})
    return


@app.cell
def _(mo, mps_table, np, plt, spearmanr):
    from statsmodels.nonparametric.smoothers_lowess import lowess

    def assess_relationship(x_var = 'Sample size', y_var ='n CpGs', data = mps_table, 
                           outlier_exclusion=0, lowess_frac=0.4): 

        x = data[x_var]
        y = data[y_var]

        if outlier_exclusion:
            x_thresh = x.quantile(outlier_exclusion)
            y_thresh = y.quantile(outlier_exclusion)

            x = x.where(x < x_thresh, np.nan)
            y = y.where(y < y_thresh, np.nan)

        r, p = spearmanr(x, y, nan_policy='omit')
        print('r =', round(r, 3), ', P = ', round(p, 3))

        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(x, y, 'o', alpha=0.5, color = 'royalblue')
        ax.set_xlabel(x_var, fontweight = 'bold')
        ax.set_ylabel(y_var, fontweight = 'bold')

        smoothed = lowess(y, x, frac=lowess_frac)  # frac = smoothing span
        ax.plot(smoothed[:, 0], smoothed[:, 1], color='crimson', lw=1.5)

        return(fig)

    mo.ui.tabs({"Sample size vs. n CpG": assess_relationship(), 
                "Sample size vs. n CpG (filtered)": assess_relationship(outlier_exclusion=0.95)}) 
    return


@app.cell
def _():
    # sample size
    # covariates
    return


@app.cell
def _(mps_table, pd):
    strategy_count = mps_table[[f'Dimension reduction ({i})' for i in range(1, 6)]].notna().sum(axis=1)
    print(strategy_count.describe())
    strategy_count = strategy_count.value_counts().sort_index()
    _percent = (strategy_count / strategy_count.sum() * 100).round(1)
    print(pd.DataFrame({'count': strategy_count, '%': _percent}))
    print(round(_percent[2:].sum(), 1), '% of MPSs use > 1 strategy to include CpGs in the analysis.')

    mps_table[[f'Dimension reduction ({i})' for i in range(1, 2)]].value_counts()
    return


@app.cell
def _(mo):
    mo.md(r"""### Analytical methods""")
    return


@app.cell
def _(mo, mps_table, pd, pub_table):
    def count_dimension_reduction_strategies(df = mps_table):

        strategies = [f'Dimension reduction ({i})' for i in range(1, 6)]

        df = df[strategies]

        df['Dimension reduction (1)'] = df['Dimension reduction (1)'].rename({'None': pd.NA})

        # How many use more than 1 strategy
        strategy_count = df[strategies].notna().sum(axis=1)

        print(strategy_count.describe())

        strategy_count = strategy_count.value_counts().sort_index()
        _percent = (strategy_count / strategy_count.sum() * 100).round(1)
        print(pd.DataFrame({'count': strategy_count, '%': _percent}))
        print(round(_percent[2:].sum(), 1), '% of MPSs / pubs use > 1 strategy to include CpGs in the analysis.')


    def summarize_dimension_reduction_strategies(df = mps_table):

        strategies = [f'Dimension reduction ({i})' for i in range(1, 6)]

        tab_dict = {s: df[s].value_counts(dropna=False).sort_index() for s in strategies}

        stacked_strategies = df[strategies].stack().reset_index(drop=True)
        conbined_strategies = df[strategies].astype(str).agg(" --- ".join, axis=1)

        tab_dict["All_strategies"] = stacked_strategies.value_counts(dropna=False).sort_index()

        tab_dict["Combined_strategies"] = conbined_strategies.value_counts(dropna=False)

        tabs = mo.ui.tabs(tab_dict)
        return(tabs)


    def filter_dimension_reduction_strategies(strategy, returns = 'None'):

        strategies = [f'Dimension reduction ({i})' for i in range(1, 6)]

        dfs = {'MPSs': mps_table, 'pubs': pub_table}
        masked_dfs = {}

        for k, df in dfs.items():
            mask = df[strategies].apply(lambda col: col.str.contains(strategy, na=False), axis=0).any(axis=1)
            print(f'{df[mask].shape[0]} ({round(df[mask].shape[0] / df.shape[0] * 100, 1)}%) of {k} adopt {strategy}')
            masked_dfs[k] = df.loc[mask, ['Title']+strategies]

        if returns != 'None': 
            return masked_dfs[returns]

    # count_dimension_reduction_strategies()
    count_dimension_reduction_strategies(pub_table)

    summarize_dimension_reduction_strategies(pub_table)
    return (
        filter_dimension_reduction_strategies,
        summarize_dimension_reduction_strategies,
    )


@app.cell
def _(
    filter_dimension_reduction_strategies,
    summarize_dimension_reduction_strategies,
):
    summarize_dimension_reduction_strategies(
        df = filter_dimension_reduction_strategies('Association DNAm phenotype', returns='pubs')
    )
    return


@app.cell
def _(filter_dimension_reduction_strategies):
    print('Statistical significance')
    filter_dimension_reduction_strategies('Association DNAm phenotype \\| P-value')
    print('\nStatistical significance | top-ranking')
    filter_dimension_reduction_strategies('Association DNAm phenotype \\| P-value \\| top-ranking')
    print('\nEffect size')
    filter_dimension_reduction_strategies('Association DNAm phenotype \\| Effect size \\| | Methylation change')
    print('\nAUC')
    filter_dimension_reduction_strategies('Association DNAm phenotype \\| AUC')
    print('\nEffect size * p-value')
    filter_dimension_reduction_strategies('Association DNAm phenotype \\| Effect size \\* P-value')

    print('\nMultiple')
    filter_dimension_reduction_strategies('Multiple | \\[')

    return


@app.cell
def _(
    filter_dimension_reduction_strategies,
    summarize_dimension_reduction_strategies,
):
    summarize_dimension_reduction_strategies(
        df = filter_dimension_reduction_strategies('Pruning', returns='pubs')
    )
    return


@app.cell
def _(filter_dimension_reduction_strategies):
    print('Correlations')
    filter_dimension_reduction_strategies('Pruning \\| Pairwise correlation|Pruning \\| R-squared')

    print('\nCoMeBack')
    filter_dimension_reduction_strategies('Pruning \\| CoMeBack')

    print('\nStepwise selection')
    filter_dimension_reduction_strategies('Pruning \\| Stepwise selection')

    # 'Pruning | Multivariate regression'
    # 'Pruning | Feature selection ML'
    return


@app.cell
def _(
    filter_dimension_reduction_strategies,
    summarize_dimension_reduction_strategies,
):
    summarize_dimension_reduction_strategies(
        df = filter_dimension_reduction_strategies('Reproducibility', returns='pubs')
    )
    return


@app.cell
def _(filter_dimension_reduction_strategies):
    filter_dimension_reduction_strategies('Reproducibility \\| Array')
    filter_dimension_reduction_strategies('Reproducibility \\| Tissue')
    return


@app.cell
def _(
    filter_dimension_reduction_strategies,
    summarize_dimension_reduction_strategies,
):
    print('Functional annotation')
    filter_dimension_reduction_strategies('Biological relevance \\| Functional annotation')

    summarize_dimension_reduction_strategies(
        df = filter_dimension_reduction_strategies('Biological relevance', returns='pubs')
    )
    return


@app.cell
def _(mps_table, pub_table):
    def filter_models(model, data=mps_table):
        mask = data['Weights estimation'].str.contains(model, na=False)

        print(f'{data[mask].shape[0]} ({round(data[mask].shape[0] / data.shape[0] * 100, 1)}%)')
        return data[mask]

    for _tab_entry in ['Discovery EWAS', 'Machine learning', 'Penalized regression']:
        print(_tab_entry)
        filter_models(_tab_entry)
        filter_models(_tab_entry, data = pub_table)

    mps_table['Weights estimation'].value_counts().sort_index()
    # pub_table['Weights estimation'].value_counts().sort_index()
    return


@app.cell
def _():
    # Check use of different techniques (dichotomous vs. continuous outcomes)
    # mps_table.loc[mps_table.Determining_weights_1 == "Penalized regression | Elastic net", "n Cases"].notna().sum()
    # mps_table.loc[(mps_table.Determining_weights_1 == "Machine learning | Support vector machine") & mps_table["n Cases"].isna(), ]
    return


@app.cell
def _(mo):
    mo.md(r"""### Validation""")
    return


@app.cell
def _(mps_table, pub_table):
    def crosstab_count(df = mps_table):
        no_validation = df.loc[(df['Internal validation'] == 'No') & (df['External validation'] == 'No'),].shape[0]
        print(f'{no_validation} ({round(no_validation / df.shape[0] * 100, 1)}%) do not use any validation method')

    crosstab_count()
    crosstab_count(pub_table)
    return


@app.cell
def _(count_containing, pub_table):
    count_containing('Yes', 'External validation')
    count_containing('Yes', 'External validation', pub_table)
    # get_mps_pub_counts('External validation')
    return


@app.cell
def _(get_mps_pub_counts, mps_table, pub_table):
    def count_containing(term, var, df = mps_table):
        count = df.loc[df[var].str.contains(term, na=False),]
        print(f'{count.shape[0]} ({round(count.shape[0] / df.shape[0] * 100, 1)}%) have {var} ~= {term}')

    # Internal validation
    count_containing('Yes|validation|split', 'Internal validation')
    count_containing('Yes|validation|split', 'Internal validation', pub_table)
    get_mps_pub_counts('Internal validation')
    return (count_containing,)


@app.cell
def _(mo):
    mo.md(r"""## Table""")
    return


@app.cell
def _(mps_table, pub_table):
    print(pub_table.shape[0], mps_table.shape[0])
    return


@app.cell
def _(mps_table, pd):
    def get_simple_counts(var, precision = 0, df = mps_table, groups=None):

        if groups is not None: 
            group_map = {item: group for group, items in groups.items() for item in items}
            var_col = df[var].replace(group_map)
        else: 
            var_col = df[var]

        counts = pd.DataFrame(var_col.value_counts())
        _perc = round(counts['count'] / df.shape[0] * 100, precision)
        if precision == 0:
            _perc = [int(i) for i in _perc]
        counts['count'] = [f'{c} ({p}%)' for c, p in zip(counts['count'], _perc)]
        return counts

    def get_median_range(var, precision = 0, df = mps_table):
        desc = pd.DataFrame(df[var].describe()[['min','50%','max']])
        if precision == 0:
            desc[var] = desc[var].astype(int)
        return f'{desc.loc['50%',var]} [{desc.loc['min',var]}; {desc.loc['max',var]}]'
    return get_median_range, get_simple_counts


@app.cell
def _(get_simple_counts, pub_table):
    get_simple_counts('Sample type', df = pub_table)
    return


@app.cell
def _(get_median_range, pub_table):
    get_median_range('n MPSs', df = pub_table)
    return


@app.cell
def _(get_simple_counts):
    get_simple_counts('Category')
    return


@app.cell
def _(mps_table, np, pd):

    def validation_crosstab(df = mps_table):
        df['internal'] = np.where(df['Internal validation'].str.contains('Yes|validation|split', na=False), 'yes', 'no')
        df['external'] = np.where(df['External validation'].str.contains('Yes', na=False), 'yes', 'no')

        counts = pd.DataFrame(df[['internal','external']].value_counts())
        _perc = round(counts['count'] / df.shape[0] * 100, 0)
        _perc = [int(i) for i in _perc]
        counts['count'] = [f'{c} ({p}%)' for c, p in zip(counts['count'], _perc)]
        return(counts)

        # print(f'{count.shape[0]} ({round(count.shape[0] / df.shape[0] * 100, 1)}%) have {var} ~= {term}')
    validation_crosstab()
    return


@app.cell
def _(get_simple_counts):
    get_simple_counts('Based on')
    return


@app.cell
def _(get_simple_counts):
    get_simple_counts('Tissue', groups = {'Blood derived': ['Peripheral blood', 'Whole blood', 'Dried bloodspot', 'Blood-clots', 'Blood', 'Leukocytes', 'Cord blood'],
                                          'Saliva and buccal': ['Saliva', 'Buccal cells'],
                                          'Other': ['Placenta', 'Tumour cells', 'Nasal epithelial cells', 'Cervical cells', 'Urine']
                                         })
    return


@app.cell
def _(get_simple_counts):
    get_simple_counts('Array', groups = {'Multiple': ['Multiple (450K, EPICv1)', 
                                                      'Multiple (450K, GMEL (~3000 CpGs from EPICv1))', 'Multiple (450K, EPICv2)']})
    return


@app.cell
def _(get_simple_counts):
    get_simple_counts('Ancestry', groups = {'European / White': ['European', 'White'], 
                                            'Latinx / Hispanic': ['Latinx', 'Hispanic']})
    return


@app.cell
def _(get_simple_counts):
    get_simple_counts('Developmental period')
    return


@app.cell
def _(filter_dimension_reduction_strategies):
    drgroups = ['Association DNAm phenotype', 'Biological relevance', 'Pruning', 'Reproducibility']

    for g in drgroups: 
        filter_dimension_reduction_strategies(g)


    return


@app.cell
def _(get_median_range):
    get_median_range('n CpGs')
    return


@app.cell
def _(get_simple_counts, mps_table):

    def make_dict(labels, var = mps_table['Weights estimation']): 
        out_dict = {}
        for lab in labels:
            out_dict[lab] = list(var.loc[var.str.contains(lab, na=False)].unique())
        return out_dict

    wegroups = make_dict(['Penalized regression', 'Discovery EWAS', 'Machine learning'])
    get_simple_counts('Weights estimation', groups = wegroups)
    return


@app.cell
def _(mps_table, pub_table):
    print(mps_table.Comparison.value_counts().sort_index(), '\n')
    print(pub_table.Comparison.value_counts().sort_index())
    return


@app.cell
def _(mps_table, pub_table):
    print(mps_table.Missing_value_note.value_counts().sort_index(), '\n')
    print(pub_table.Missing_value_note.value_counts().sort_index())
    return


@app.cell
def _(mps_table, pub_table):
    print(mps_table.Covariates.value_counts().sort_index(), '\n')
    print(pub_table.Covariates.value_counts().sort_index())
    return


if __name__ == "__main__":
    app.run()
