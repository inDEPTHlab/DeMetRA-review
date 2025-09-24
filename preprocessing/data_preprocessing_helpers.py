import marimo as mo
import pandas as pd
import openpyxl

from dateutil.parser import parse as parse_date

def read_sys_review(assets_directory, sys_review_file):
    '''Read in the systematic review data from the excel file.'''
    lit_raw, base_ss, base_va = pd.read_excel(f'{assets_directory}/Annotations_raw_files/{sys_review_file}',
        sheet_name=[0, 1, 2]).values()

    lit_raw = lit_raw.loc[lit_raw.Include == 'Yes'].drop('Include', axis=1) # All included

    # Drop any empty rows 
    base_ss = base_ss.loc[base_ss.Identifier.notna() & (base_ss.Identifier.str.strip() != "")]
    base_va = base_va.loc[base_va.Identifier.notna() & (base_va.Identifier.str.strip() != "")]

    if 'Base use' not in base_ss.columns:
        base_ss.insert(1, 'Base use', 'Summary statistics')

    if 'Base use' not in base_va.columns:
        base_va.insert(1, 'Base use', 'Validated algorithm')

    # print(base_ss.shape, base_va.shape)

    print(lit_raw.shape, '\n', list(lit_raw.columns))
    print(base_ss.shape, '\n', list(base_ss.columns))
    print(base_va.shape, '\n', list(base_va.columns))

    lit_base_raw = pd.concat([base_ss, base_va], axis=0, ignore_index=True, sort=False)

    return lit_raw, lit_base_raw


def inspect_variable_levels(Main, Base, var, var_base = None, sort_index = True):
    '''Inspect all levels of a variable in both Main and Base datasets.'''

    if not var_base: 
        var_base = var

    allobs = pd.concat([Main[var], Base[var_base]])

    counts = allobs.value_counts(dropna=False)

    if sort_index: 
        counts = counts.sort_index()

    counts.index = [f"~{i}~" for i in counts.index]

    var_multiple = f'Multiple_{var.lower()}'

    try: 
        allobs_multiple = pd.concat([Main[var_multiple], Base[var_multiple]])
        counts_multiple = allobs_multiple.value_counts(dropna=False)
        counts_multiple.index = [f"~{i}~" for i in counts_multiple.index]

        return counts, counts_multiple

    except KeyError:
        # print(f"No {var_multiple} found.")
        return counts


def summarize_dimension_reduction_strategies(Main, Base):
    '''Inspect all levels of dimension reduction variables in both Main and Base datasets.'''

    strategies = [f"Including_CpGs_{i}" for i in range(1, 6)] # five levels

    df = pd.concat([Main[strategies], Base[strategies]])

    tab_dict = {s: df[s].value_counts(dropna=False).sort_index() for s in strategies}

    stacked_strategies = df.stack().reset_index(drop=True)
    conbined_strategies = df.astype(str).agg(" --- ".join, axis=1)

    tab_dict["All_strategies"] = stacked_strategies.value_counts(dropna=False).sort_index()
    tab_dict["Combined_strategies"] = conbined_strategies.value_counts(dropna=False)

    tabs = mo.ui.tabs(tab_dict)
    return(tabs)


def count_categories_per_phenotype(Main, Base):
    '''Check that no phenotypes are assigned to multiple categories '''

    counts = []

    for df in [Main, Base]:
        count = df.groupby('Phenotype')['Category'].nunique().sort_values(ascending=False)
        if count.iloc[0] > 1: 
             counts.append(count)

    return counts

def aggregate_values(series):
    if series.apply(lambda x: isinstance(x, list)).sum() > 0:
        return series.iloc[0]
    unique_values = series.unique()
    if len(unique_values) == 1:
        return unique_values[0]
    else:
        return f"Multiple ({', '.join(str(v) for v in unique_values)})" 


def replace_multiples(df, drop_multiple_cols=True):
    for var in ['Tissue', 'Array', 'Ancestry']:
        if var == 'Ancestry':
            df.loc[df[var] == 'Multiple', var] = 'Mixed'
        else:
            df.loc[df[var] == 'Multiple', var] = [f'Multiple ({values})' for values in df.loc[
                df[var] == 'Multiple', f'Multiple_{var.lower()}']]

        if drop_multiple_cols:
            df = df.drop([f'Multiple_{var.lower()}'], axis=1)
    return df


def clean_n_CpGs(df, replace_with_df = None):

    n_cpg_var = "Number of CpGs"
    new_n_cpg_var = "n CpGs"

    empty_mps_count = df[df[n_cpg_var] == 0].shape[0]
    if empty_mps_count > 0:
        print(f"Dropping {empty_mps_count} MPSs which did no identify a solution (0 CpGs included)")
        df = df[df[n_cpg_var] != 0].reset_index(drop=True)

    na_count = df[n_cpg_var].isna().sum()
    print(f"{na_count} NA values in `{n_cpg_var}`")

    if replace_with_df is not None:
        print("    --> Replacing with info from base, where possible...")

        for _idx, _row in df.iterrows():
            if pd.isna(_row[n_cpg_var]) & (_row['What_is_available'] != "Only phenotype"):
                match = replace_with_df[(replace_with_df['Identifier'] == _row['Identifier_base']) & 
                                        (replace_with_df['Phenotype'] == _row['Phenotype'])]
                if match.empty:
                    print(f"\t({_row['What_is_available']}) {_row['Identifier_base']} {_row['Phenotype']} not found.")
                else:
                    df.at[_idx, n_cpg_var] = match.iloc[0][n_cpg_var]

        print(f"{df[n_cpg_var].isna().sum()} `NA` values left +",
              f"{df.loc[df[n_cpg_var] == 'Not reported',].shape[0]} `Not reported` values.")

    df = coerce_to_numeric(df, n_cpg_var, new_n_cpg_var)

    return df

def coerce_to_numeric(df, old_var, new_var):

    df[new_var] = pd.to_numeric(df[old_var], errors='coerce')

    coerced_mask = df[new_var].isna() & df[old_var].notna()
    coerced_values = df.loc[coerced_mask, old_var].value_counts()

    if coerced_values.shape[0] > 0:
        print('\nThese values are coerced to NA:\n', coerced_values)

    return df

# Bibliography ===============================================================

ris_tags = {'TY': 'Reference Type', 
            'AU': 'Author_list', 
            'PY': 'Year', 
            'TI': 'Title', 
            'T2': 'Journal', 
            'J2': 'Journal', 
            'AB': 'Abstract', 
            'DO': 'DOI', 
            # 'UR': 'URL', 
            'KW': 'Keywords', 
            'DA': 'Date'}

def parse_ris(file_path):
    with open(file_path, 'r', encoding='utf-8') as _file:
        references = []
        current_entry = {}
        current_tag = None
        for line in _file:
            if not line.strip():
                continue
            elif line.strip() == 'ER  -':
                references.append(current_entry)
                current_entry = {}
                current_tag = None
            else:
                tag = line[:5]
                if tag in [f'{_t}  -' for _t in ris_tags.keys()]:
                    current_tag = tag
                    value = line[5:].strip()
                    key = ris_tags[tag[:2]]
                    if key in current_entry:
                        if isinstance(current_entry[key], list):
                            current_entry[key].append(value)
                        else:
                            current_entry[key] = [current_entry[key], value]
                    else:
                        current_entry[key] = value
                elif tag in [f'{y}-' for y in range(2000, 2025)]:
                    value = line.strip()
                    if 'Date' not in current_entry:
                        current_entry['Date'] = value
                elif current_tag == 'KW  -':
                    value = line.strip()
                    if isinstance(current_entry['Keywords'], list):
                        current_entry['Keywords'].append(value)
                    else:
                        current_entry['Keywords'] = [current_entry['Keywords'], value]
        return references


def read_bibliography(assets_directory, bibliograp_file, title_list):
    '''Read in and clean the bibliography data from the RIS file.'''

    parsed_data = parse_ris(f"{assets_directory}/Bibliography_raw_files/{bibliograp_file}")

    bib = pd.DataFrame(parsed_data)[['Author_list', 'Year', 'Title', 'Journal', 'Keywords', 
                                     'Abstract', 'Date', 'DOI']] # 'URL' are not reliable, I will use DOI 
    print('Full bibliography:', bib.shape)

    bib_incl = bib.loc[bib.Title.isin(title_list),].reset_index(drop=True)
    print('Included in review:', bib_incl.shape)

    # Cleaning Dates
    missing_date = bib_incl.Date.isna().sum()
    if missing_date > 0:
        print('Note', missing_date, 'NaN Date values will be set to 01/06 of respective year')
    
    date_tmp = pd.Series([' '.join([d, y]) if y not in d else d for d, y in zip(bib_incl.Date.map(str), bib_incl.Year.map(str))])
    # date_tmp[date_tmp=="May-Jun 2025"] = "Jun 2025"
    bib_incl.loc[:, 'Date'] = date_tmp.apply(lambda date: 
                                            parse_date(date).strftime('%Y-%m-%d') if 'nan' not in date else 
                                            parse_date('01 06' + date[3:]).strftime('%Y-%m-%d'))

    # Cleaning Authors
    bib_incl['Author'] = [f"{fa[0].split(',')[0]} et al." if isinstance(fa, list) else f"{fa.split(',')[0]}" 
                        for fa in bib_incl['Author_list']]
    
    # Reorder columns
    bib_incl = bib_incl[['Author', 'Year', 'Title', 'Journal', 'DOI', 'Date', 
                         'Keywords', 'Abstract', 'Author_list']]

    return(bib_incl)