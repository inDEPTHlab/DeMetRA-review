
PAGE_PADDING = 'padding-top: 50px; padding-right: 50px; padding-bottom: 50px; padding-left: 50px'
PAGE_GAP = '20px'

SELECTION_PANE = 'padding-top: 10px; padding-right: 20px; padding-left: 20px; ' \
                 'border-radius: 35px; ' \
                 'background-color: #DCE3F0'  # light grey

INFO_MESSAGE = 'text-align: center; padding-top: 10px; padding-bottom: 10px'

# ------ PLOTTING ------------
# Specify colors to use for categorical variables
category_color_map = {'Syndrome': 'royalblue',
                      'Disease': 'firebrick',
                      'Protein': 'darkviolet',
                      'Lifestyle': 'gold',
                      'Psychiatric': 'teal',
                      'Environmental exposure': 'limegreen',
                      'Birth outcome': 'lightblue',
                      'Tumour': 'pink',
                      'Aging phenotype': 'darkgrey'}

tissue_color_map = {'Peripheral blood': 'crimson',
                    'Whole blood': 'gold',
                    'Cord blood': 'brown',
                    'Placenta': 'pink',
                    'Blood-clots': 'blueviolet',
                    'Dried bloodspot': 'indianred',
                    'Saliva': 'lightblue',
                    'Buccal cells': 'steelblue',
                    'Nasal epithelial cells': 'teal',
                    'Tumour cells': 'darkgreen',
                    'Leukocytes': 'yellow',
                    'Leukocytes': 'yellow',
                    'Bone marrow granulocytes': 'magenta',
                    'Not reported': 'darkgrey'}

array_color_map = {'EPICv2': 'darkblue',
                   'EPICv1': 'mediumblue',
                   '450K, EPICv1': 'seagreen',
                   '450k': 'orange',
                   '450K, GMEL (~3000 CpGs from EPICv1)':'orange',
                   'WGBS': 'magenta'}

COLOR_MAPS = {'Category': category_color_map,
              'Tissue': tissue_color_map,
              'Array': array_color_map}

# BETA_COLORMAP = 'viridis'
# CLUSTER_COLORMAP = 'turbo'
#
# OVLP_COLOR1 = '#88CCEE'  # light blue
# OVLP_COLOR2 = '#DDCC77'  # yellow
# OVLP_COLOR3 = '#CC6677'  # light red
#
# OVLP_COLORS = [OVLP_COLOR1, OVLP_COLOR2, OVLP_COLOR3]



