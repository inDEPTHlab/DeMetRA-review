from pathlib import Path

import pandas as pd
import numpy as np
import re
import ast 

from shiny import ui

proj_dir = Path(__file__).parent.parent

assets_dir = proj_dir / 'assets'

mps_table = pd.read_csv(f'{assets_dir}/mps_table.csv', parse_dates=['Date'])
pub_table = pd.read_csv(f'{assets_dir}/pub_table.csv', parse_dates=['Date'])

# Turn DOI into a clickable link
mps_table[' '] = [ui.markdown(f'<a href="https://doi.org/{doi}" target="_blank">DOI  </a>') for doi in mps_table['DOI']]
pub_table[' '] = [ui.markdown(f'<a href="https://doi.org/{doi}" target="_blank">DOI  </a>') for doi in pub_table['DOI']]

# Display lists inside cells as bullet points
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
                    # Instead of a bullet point list I show range
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
    
    else: 
        print(cell)

    return cell

mps_table_show = mps_table[['Phenotype', 'Category', 'n CpGs',
                            'Author', 'Year', 'Title', ' ', 'Based on',
                            'Sample type', 'Sample size', # 'n Cases', 'n Controls', 
                            'Developmental period', 'Tissue', 'Array', 'Ancestry']]
                           
pub_table_show = pub_table[['Author', 'Year', 'Title', 'Journal',' ','n MPSs', 'Phenotype(s)', 'Category', 
                            'n CpGs', 'Based on', 'Sample type', 'Sample size', # 'n Cases', 'n Controls',
                            'Developmental period', 'Tissue', 'Array', 'Ancestry']]


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
                               'n CpGs', 'Sample size', #'n Cases', 'n Controls'
                               ]:
            table[var_to_restyle] = table[var_to_restyle].apply(list_to_html)

    table_style = _style_litreview_table(table.reset_index(), which_table = which_table)

    return table, table_style


def _style_litreview_table(table, which_table):
    
    table_style = [
        {'cols': ['Category', 'Author', 'Based on'],
        'style': {'width': '130px', 'max-width': '150px', 'min-width': '110px'}},
        {'cols': ['Title'],
        'style': {'width': '600px', 'max-width': '700px', 'min-width': '500px'}},
        {'cols': ['Sample size'], # 'n Cases', 'n Controls'],
        'style': {'width': '30px', 'max-width': '30px', 'min-width': '30px', 'text-align': 'right'}},
        {'cols': [' '],
        'style': {'width': '30px', 'max-width': '30px', 'min-width': '30px', 'text-align': 'center'}},
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