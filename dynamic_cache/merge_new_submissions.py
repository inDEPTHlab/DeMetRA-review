from pathlib import Path
import pandas as pd
from datetime import datetime
import networkx as nx
import pickle
import json

proj_dir = Path(__file__).parent.parent.resolve()

main_files_dir =  proj_dir / "assets"
submission_dir = main_files_dir / "submissions"
archive_dir = submission_dir / "archive"

cache_dir = proj_dir / "dynamic_cache" / "metadata" 

new_submissions = sorted(submission_dir.glob("update_*.csv"))

if not new_submissions:
    print("No new submissions found.")
    exit(0)

# NOTE: I only expect 1 submission file per PR
submission_file = new_submissions[0] 

# Parse update date from filename: update_author_date.csv
timestamp_str = submission_file.stem.split("_")[-1]
last_updated  = datetime.strptime(timestamp_str, "%Y%m%d").strftime("%d/%m/%Y")

# Load pending submissions
mps_update = pd.read_csv(submission_file)
print(f"Processing {last_updated} submission — {len(mps_update)} new MPS(s).")

mps_update['Author'] = [f"{fa[0].split(',')[0]} et al." if isinstance(fa, list) else f"{fa.split(',')[0]}" 
                        for fa in mps_update['Author_list']]
mps_update['Year'] = mps_update.loc[0, 'Date'][:4]

# TMP 
mps_update['n Cases'] = pd.NA
mps_update['n Controls'] = pd.NA

# Also in preprocessing_helpers.py
def aggregate_values(series):
    if series.apply(lambda x: isinstance(x, list)).sum() > 0:
        return series.iloc[0]
    unique_values = series.unique()
    if len(unique_values) == 1:
        return unique_values[0]
    else:
        return f"Multiple ({', '.join(str(v) for v in unique_values)})" 

pub_update = mps_update.groupby('Title').agg(aggregate_values).reset_index()
# Assume only one publication per update 
pub_update['n MPSs'] = len(mps_update)

pub_update = pub_update.rename(columns={'Phenotype': 'Phenotype(s)'})
# Reorder and clean 

mps_update = mps_update[['Phenotype', 'Category', 'n CpGs',
                         'Author', 'Year', 'Title', 'DOI', 'Based on',
                         'Sample type', 'Sample size', 'Developmental period', 
                         'Tissue', 'Array', 'Ancestry', 'Author_list', 'Date']]

pub_update = pub_update[['Author', 'Year', 'Title', 'Journal','DOI','n MPSs', 'Phenotype(s)', 'Category', 
                         'n CpGs', 'Based on', 'Sample type', 'Sample size', 'n Cases', 'n Controls',
                         'Developmental period', 'Tissue', 'Array', 'Ancestry', 'Author_list', 'Date']]

# # Append to main table
mps_table = pd.read_csv(f'{main_files_dir}/mps_table.csv')
pub_table = pd.read_csv(f'{main_files_dir}/pub_table.csv', parse_dates=['Date'])

mps_merged = pd.concat([mps_table, mps_update], ignore_index=True)
pub_merged = pd.concat([pub_table, pub_update], ignore_index=True)

mps_merged.to_csv(f'{main_files_dir}/mps_table.csv', index=False)
pub_merged.to_csv(f'{main_files_dir}/pub_table.csv', index=False)


# ── Stats cache ───────────────────────────────────────────────────

def _count_papers(pub_table):
    return pub_table.shape[0]


def _count_mpss(mps_table):
    return mps_table.shape[0]


def _count_phenotypes(mps_table):
    return int(len(pd.unique(mps_table['Phenotype'])))

stats = {
    "publication_count": _count_papers(pub_merged),
    "mps_count": _count_mpss(mps_merged),
    "phenotype_count": _count_phenotypes(mps_merged),
    "last_updated": last_updated
}

stats_path =  cache_dir / "stats.json"

with open(stats_path, "w") as f:
     json.dump(stats, f, indent=2)
print(f"Saved: {stats_path}")

# Archive processed files
submission_file.rename(archive_dir / submission_file.name)
print(f"Archived: {submission_file.name}")

# ── Publication network cache ───────────────────────────────────────────────────

def update_publication_network(title: str, authors: str | list[str], cache_dir) -> nx.Graph:
    """
    Loads the cached network, adds a new publication and its authors,
    re-runs the layout with a warm-start, and saves back to pkl.

    Parameters
    ----------
    title   : publication title string
    authors : single author string or list of 'Surname, N.' strings
    """
    cached_network_file = cache_dir / "publication_network.pkl"

    with open(cached_network_file, "rb") as f:
        G = pickle.load(f)

    # ── Add nodes and edges ───────────────────────────────────────
    node_title = f"Paper/{title.replace(':', ' ')}"

    if isinstance(authors, str):
        authors = [authors]

    for author in authors:
        node_author = f"Author/{author.replace('. ', '.')}"
        G.add_edge(node_author, node_title)

    # ── Re-run layout with warm-start ─────────────────────────────
    initial_pos = {
        node: data["pos"]
        for node, data in G.nodes(data=True)
        if "pos" in data # new nodes have no pos yet → excluded
    }
    
    print('Updating network layout...')
    pos = nx.fruchterman_reingold_layout(G, seed=3108, k=0.05, pos=initial_pos or None,)

    nx.set_node_attributes(G, pos, "pos")

    # ── Save ──────────────────────────────────────────────────────
    with open(cached_network_file, "wb") as f:
        pickle.dump(G, f)

    return G

g = update_publication_network(pub_update.loc[0, 'Title'], pub_update.loc[0, 'Author_list'], 
                               cache_dir = cache_dir)

print("── Publication network updated.")