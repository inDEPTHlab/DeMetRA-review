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

    Both are created by the `data_preprocessing.py` script.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import openpyxl

    from scipy.stats import spearmanr

    assets_directory = './assets/'
    return assets_directory, mo, pd, spearmanr


@app.cell
def _(assets_directory, pd):
    mps_table = pd.read_csv(f'{assets_directory}MPS_table_cleaned-tmp.csv')
    pub_table = pd.read_csv(f'{assets_directory}Publication_table_cleaned-tmp.csv')
    return mps_table, pub_table


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
    pheno_count.sort_index()
    return


@app.cell
def _(mps_table, pd):
    def get_count_percent(variable, data=mps_table, precision=0, print_count=True):
        _counts = pd.DataFrame(data[variable].value_counts())
        _total = _counts.sum()
        _counts['%'] = round(_counts / _total * 100, precision)
        _counts.index = _counts.index.map(lambda x: str(x) if isinstance(x, list) else x)
        print('\n')
        for _idx, _row in _counts.iterrows():
            fmtcount = int(_row['count']) if print_count else ''
            fmtpercent = int(_row['%']) if precision == 0 else _row['%']
            print(f'{_idx} {fmtcount} ({fmtpercent}%)')
    return (get_count_percent,)


@app.cell
def _(get_count_percent, pub_table):
    get_count_percent('Category')
    get_count_percent('Category', data=pub_table)
    return


@app.cell
def _(get_count_percent):
    get_count_percent('Based on')
    return


@app.cell
def _(get_count_percent, pub_table):
    get_count_percent('Developmental period', precision=1)
    get_count_percent('Developmental period', precision=1, data=pub_table)
    return


@app.cell
def _(get_count_percent, mps_table):
    get_count_percent("Tissue") #, print_count=False)

    mps_table['Tissue'].astype(str).value_counts(dropna=False)

    # a = sum([457, 35, 45, 26, 26, 3])
    # b = sum([120, 20])
    # c = sum([13, 6, 4, 1])
    # d = 1
    # 1
    # for n in [a, b, c, d]:
    #     print(f'{n} ({round(n / mps_table_1.shape[0] * 100, 1)}%)')
    return


@app.cell
def _(get_count_percent, mps_table, pd, pub_table):
    print('MPS count')
    get_count_percent('Category', precision=1)
    print('\nPhenotype count\n', mps_table.groupby('Category')['Phenotype'].nunique())
    pub_count_category = pd.DataFrame(mps_table.groupby('Category')['Title'].nunique())
    pub_count_category['%'] =  round(pub_count_category / pub_table.shape[0] * 100, 1)

    print('\nPublication count\n', pub_count_category)
    mps_table[['Category', 'Phenotype']].value_counts().sort_index().reset_index()
    return


@app.cell
def _(get_count_percent, pub_table):
    get_count_percent('Sample type')
    get_count_percent('Sample type', data=pub_table)
    return


@app.cell
def _(get_count_percent, pub_table):
    get_count_percent('Array')
    get_count_percent('Array', data=pub_table)
    return


@app.cell
def _(get_count_percent, pub_table):
    get_count_percent('Ancestry', precision=1)
    get_count_percent('Ancestry', precision=1, data=pub_table)
    return


@app.cell
def _(get_count_percent):
    get_count_percent('Sample_overlap_target_base')

    # get_count_percent('Sample_overlap_target_base', data=pub_table)
    return


@app.cell
def _(mps_table):
    print(mps_table['n CpGs'].describe())
    mps_table['n CpGs'].hist(bins=100, figsize=(15, 4))
    return


@app.cell
def _(mps_table):
    print(mps_table['Sample size'].describe())
    mps_table['Sample size'].hist(bins=100, figsize=(15, 4))
    return


@app.cell
def _(mps_table, plt, spearmanr):
    r, p = spearmanr(mps_table['Sample size'], mps_table['n CpGs'], nan_policy='omit')
    print('r =', round(r, 3), ', P = ', round(p, 3))

    _fig, _ax = plt.subplots(figsize=(15, 6))
    _ax.plot(mps_table['Sample size'], mps_table['n CpGs'], 'o')
    return


@app.cell
def _():
    # sample size
    # covariates
    return


@app.cell
def _(mps_table, pd):
    strategy_count = mps_table[['Including_CpGs_1', 'Including_CpGs_2', 'Including_CpGs_3', 'Including_CpGs_4', 'Including_CpGs_5']].notna().sum(axis=1)
    print(strategy_count.describe())
    strategy_count = strategy_count.value_counts().sort_index()
    _percent = (strategy_count / strategy_count.sum() * 100).round(1)
    print(pd.DataFrame({'count': strategy_count, '%': _percent}))
    print(round(_percent[2:].sum(), 1), '% of MPSs use > 1 strategy to include CpGs in the analysis.')
    return


@app.cell
def _(mo):
    mo.md(r"""### Analytical methods""")
    return


@app.cell
def _(mps_table):
    def summarize_strategies(data=mps_table):
        strategies = ['Including_CpGs_1', 'Including_CpGs_2', 'Including_CpGs_3', 'Including_CpGs_4', 'Including_CpGs_5']
        for strategy in strategies:
            print(data[strategy].value_counts().sort_index(), '\n')
    summarize_strategies()
    return (summarize_strategies,)


@app.cell
def _(mps_table):
    def filter_strategies(strategy, data=mps_table):
        mask = data[['Including_CpGs_1', 'Including_CpGs_2', 'Including_CpGs_3', 'Including_CpGs_4', 'Including_CpGs_5']].apply(lambda col: col.str.contains(strategy, na=False), axis=0).any(axis=1)
        print(f'{data[mask].shape[0]} ({round(data[mask].shape[0] / data.shape[0] * 100, 1)}%)')
        return data[mask]
    return (filter_strategies,)


@app.cell
def _(filter_strategies, summarize_strategies):
    redundancy = filter_strategies('Pruning')
    summarize_strategies(redundancy)

    func_annot = filter_strategies('Functional annotation')
    summarize_strategies(func_annot)

    reproducible = filter_strategies('Reproducibility')
    summarize_strategies(reproducible)

    for _tab_entry in ['Association DNAm phenotype', 'Biological relevance', 'Pruning', 'Reproducibility']:
        print('\n', _tab_entry)
        filter_strategies(_tab_entry)
    return


@app.cell
def _(filter_strategies, pub_table, summarize_strategies):
    stat_sign = filter_strategies('p-value|top-ranking significant probes', data=pub_table)
    stat_sign = filter_strategies('top-ranking significant probes')
    summarize_strategies(stat_sign)
    return


@app.cell
def _(filter_strategies, pub_table, summarize_strategies):
    effect_size = filter_strategies('Actual change in methylation|Effect size|logFC', data=pub_table)
    effect_size = filter_strategies('Actual change in methylation|Effect size|logFC')
    summarize_strategies(effect_size)
    return


@app.cell
def _(filter_strategies, pub_table, summarize_strategies):
    auc = filter_strategies('AUC', data=pub_table)
    auc = filter_strategies('AUC')
    summarize_strategies(auc)
    return


@app.cell
def _(mps_table):
    print(mps_table.Determining_weights_1.value_counts().sort_index())

    def filter_models(model, data=mps_table):
        mask = data['Determining_weights_1'].str.contains(model, na=False)
        print(model)
        print(f'{data[mask].shape[0]} ({round(data[mask].shape[0] / data.shape[0] * 100, 1)}%)')
        return data[mask]
    for _tab_entry in ['Discovery EWAS', 'Machine learning', 'Penalized regression']:
        filter_models(_tab_entry)
    return


@app.cell
def _():
    # Check use of different techniques (dichotomous vs. continuous outcomes)
    # mps_table.loc[mps_table.Determining_weights_1 == "Penalized regression | Elastic net", "n Cases"].notna().sum()
    # mps_table.loc[(mps_table.Determining_weights_1 == "Machine learning | Support vector machine") & mps_table["n Cases"].isna(), ]
    return


@app.cell
def _(mps_table, pub_table):
    print(mps_table.Sample_overlap_target_base.value_counts().sort_index(), '\n')
    print(pub_table.Sample_overlap_target_base.value_counts().sort_index())
    return


@app.cell
def _(mo):
    mo.md(r"""### Validation""")
    return


@app.cell
def _(mps_table):
    no_validation = mps_table.loc[(mps_table.Independent_validation == 'No') & (mps_table.Train_test == 'No'),].shape[0]
    print(f'{no_validation} ({round(no_validation / mps_table.shape[0] * 100, 1)}%) do not use any validation method')
    ext_validation = mps_table.loc[mps_table.Independent_validation.str.contains('Yes', na=False),]
    print(f'{ext_validation.shape[0]} ({round(ext_validation.shape[0] / mps_table.shape[0] * 100, 1)}%) MPSs perform external validation')
    int_validation = mps_table.loc[(mps_table.Train_test != 'No') & mps_table.Train_test.notna(),]
    print(f'{int_validation.shape[0]} ({round(int_validation.shape[0] / mps_table.shape[0] * 100, 1)}%) MPSs perform internal validation')
    return


@app.cell
def _(get_count_percent):
    get_count_percent('Train_test')
    # print(pub_table.Train_test.value_counts().sort_index())
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
    print(mps_table.Reflect_phenotype.value_counts().sort_index(), '\n')
    print(pub_table.Reflect_phenotype.value_counts().sort_index())
    return


@app.cell
def _(mps_table, pub_table):
    print(mps_table.Covariates.value_counts().sort_index(), '\n')
    print(pub_table.Covariates.value_counts().sort_index())
    return


if __name__ == "__main__":
    app.run()
