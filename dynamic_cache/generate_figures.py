from pathlib import Path

import pandas as pd
import plotly.io as pio

from definitions.plotting_funcs import (
    _multilevel_piechart,
    _phenotype_pub_counts,
    _mps_count_histogram,
    _category_over_years,
    _publication_histogram,
    _publication_network,
    _sample_size_over_time,
)

proj_dir = Path(__file__).parent.parent.resolve()

input_dir =  proj_dir / "assets"

output_dir = proj_dir / "dynamic_cache" / "figures"


mps_table = pd.read_csv(f'{input_dir}/mps_table.csv', parse_dates=['Date'])
pub_table = pd.read_csv(f'{input_dir}/pub_table.csv', parse_dates=['Date'])

pub_nx_file = proj_dir / "dynamic_cache" / "metadata" / "publication_network.pkl"

figures = {
    "multilevel_piechart":   _multilevel_piechart(mps_table),
    "phenotype_pub_count":   _phenotype_pub_counts(mps_table),
    "mps_count_histogram":   _mps_count_histogram(mps_table),
    "category_over_years":   _category_over_years(mps_table),
    "publication_histogram": _publication_histogram(pub_table), # Author_list
    "publication_network":   _publication_network(mps_table, pub_nx_file),
    "sample_size_over_time": _sample_size_over_time(mps_table), # Date
}

for name, fig in figures.items():
    if fig is None:
        print(f"WARNING: {name} returned None — skipping")
        continue
    path = output_dir / f"{name}.json"
    pio.write_json(fig, str(path))
    print(f"Saved: {path}")