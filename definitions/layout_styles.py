
PAGE_PADDING = 'padding-top: 50px; padding-right: 50px; padding-bottom: 50px; padding-left: 50px'
PAGE_GAP = '20px'

INFO_MESSAGE = 'text-align: center; padding-top: 10px; padding-bottom: 10px'

NAVBAR_STYLE = "background-color: #4a235a; color: white;"  # Dark purple, you can change this to any color

# ------ PLOTTING ------------
# Specify colors to use for categorical variables

array_color_map = {'450K': 'darkgreen',
                   'EPICv1': 'mediumpurple',
                   'EPICv2': 'darkblue',
                   'Nanopore sequencing': 'black',
                   'Multiple (450K, EPICv1)': 'orange',  #'seagreen',
                   'Multiple (450K, GMEL (~3000 CpGs from EPICv1))': 'orange',
                   'Multiple (450K, EPICv2)': 'orange', 
                   'Multiple (450K, EPICv1, PCR)': 'orange',
                   'Multiple (450K, PCR)': 'orange'}

# TODO: read colors directly from CSS file
category_color_map = {'Biological markers': '#113ab7',
                     'Genetic syndromes': '#008080',
                     'Lifestyle and environment': '#ffd000',
                     'Physical health indicators': '#fc9ead',
                     'Neuro-psychiatric health indicators': '#7e04b3',
                     'Cancer': '#a21414'}

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

SELECTION_PANE = 'padding-top: 15px; padding-bottom: 15px; padding-right: 25px; padding-left: 25px; ' \
                 'border-radius: 15px; ' \
                 'background-color: #efe7f6'  # light pink

