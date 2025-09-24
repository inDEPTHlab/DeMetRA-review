
PAGE_PADDING = 'padding-top: 50px; padding-right: 50px; padding-bottom: 50px; padding-left: 50px'
PAGE_GAP = '20px'

INFO_MESSAGE = 'text-align: center; padding-top: 10px; padding-bottom: 10px'

NAVBAR_STYLE = "background-color: #4a235a; color: white;"  # Dark purple, you can change this to any color

# ------ PLOTTING ------------
# Specify colors to use for categorical variables

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

COLOR_MAPS = {'Category': category_color_map,
              'Tissue': tissue_color_map,
              'Array': array_color_map, 
              'Ancestry': ancestry_color_map,
              'Developmental period': period_color_map}

SELECTION_PANE = 'padding-top: 15px; padding-bottom: 15px; padding-right: 25px; padding-left: 25px; ' \
                 'border-radius: 15px; ' \
                 'background-color: #efe7f6'  # light pink

