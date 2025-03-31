
PAGE_PADDING = 'padding-top: 50px; padding-right: 50px; padding-bottom: 50px; padding-left: 50px'
PAGE_GAP = '20px'

INFO_MESSAGE = 'text-align: center; padding-top: 10px; padding-bottom: 10px'

NAVBAR_STYLE = "background-color: #4a235a; color: white;"  # Dark purple, you can change this to any color

# ------ PLOTTING ------------
# Specify colors to use for categorical variables

array_color_map = {'EPICv1': 'mediumpurple',
                   '450k': 'darkgreen',
                   'Multiple (450K, EPICv1)': 'orange',  #'seagreen',
                   'Multiple (450K, EPICv2)': 'darkblue',
                   'Multiple (450K, GMEL (~3000 CpGs from EPICv1))': 'orange',
                   'WGBS': 'magenta'}

# category_color_map = {'Syndrome': 'royalblue',
#                       'Disease': 'firebrick',
#                       'Protein': 'darkviolet',
#                       'Lifestyle': 'gold',
#                       'Psychiatric': 'teal',
#                       'Environmental exposure': 'limegreen',
#                       'Birth outcome': 'lightblue',
#                       'Tumour': 'pink',
#                       'Aging phenotype': 'darkgrey'}

category_color_map = {'Biological markers': 'royalblue',
                      'Genetic syndromes': 'teal',
                      'Lifestyle and environment': 'gold',
                      'Physical health indicators': 'pink',
                      'Neuro-psychiatric health indicators': 'darkviolet',
                      'Cancer': 'firebrick'}

# tissue_color_map = {'Peripheral blood': 'crimson',
                    # 'Whole blood': 'gold',
                    # 'Cord blood': 'brown',
                    # 'Placenta': 'pink',
                    # 'Blood-clots': 'blueviolet',
                    # 'Dried bloodspot': 'indianred',
                    # 'Saliva': 'lightblue',
                    # 'Buccal cells': 'steelblue',
                    # 'Nasal epithelial cells': 'teal',
                    # 'Tumour cells': 'darkgreen',
                    # 'Leukocytes': 'yellow',
                    # 'Leukocytes': 'yellow',
                    # 'Bone marrow granulocytes': 'magenta',
                    # 'Not reported': 'darkgrey'}

tissue_color_map = {'Peripheral blood': 'crimson',
                    'Whole blood': 'crimson',
                    'Multiple (Whole blood, HPCs)': 'crimson',
                    'Cord blood': 'darkred', # 'brown',
                    'Placenta': 'pink',
                    'Blood-clots': 'indianred', #'blueviolet',
                    'Dried bloodspot': 'indianred',
                    'Saliva': 'lightblue',
                    'Buccal cells': 'teal', # 'steelblue',
                    'Nasal epithelial cells': 'teal',
                    'Tumour cells': 'orange', #'darkgreen',
                    'Leukocytes': 'mediumpurple',
                    'Cervical cells': 'magenta',
                    'Not reported': 'grey'}

COLOR_MAPS = {'Category': category_color_map,
              'Tissue': tissue_color_map,
              'Array': array_color_map}

DATATABLE_STYLE = [
    {'cols': ['Author'],
     'style': {'width': '120px', 'max-width': '130px', 'min-width': '90px'}},
    {'cols': ['Title'],
     'style': {'width': '600px', 'max-width': '700px', 'min-width': '500px'}}
     ]

SELECTION_PANE = 'padding-top: 15px; padding-bottom: 15px; padding-right: 25px; padding-left: 25px; ' \
                 'border-radius: 15px; ' \
                 'background-color: #efe7f6'  # light pink

