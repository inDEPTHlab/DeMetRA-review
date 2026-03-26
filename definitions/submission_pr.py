import unicodedata
import re

import os
from datetime import datetime

import pandas as pd
from github import Github

def extract_author_slug(author_raw: str) -> str:
    # "Smith, J., Van Den Berg, A. B." → "smith"
    first_author = author_raw.split(",")[0].strip()   # "Smith, J."

    first_author = unicodedata.normalize("NFKD", first_author).encode("ascii", "ignore").decode()
    first_author = re.sub(r"[^a-zA-Z\s]", "", first_author)
    return re.sub(r"\s+", "", first_author).lower() or "unknown"

def _open_pr(df: pd.DataFrame, contact: str) -> str:
    """
    Writes df as a new submission CSV on a fresh branch and opens a PR.
    Returns the PR URL.

    Parameters
    ----------
    df          : validated, structured DataFrame from validate_and_structure()
    author_raw  : raw author string (used for branch/filename slugs)
    title       : paper title (used for PR title and commit message)
    """
    g    = Github(os.environ["GITHUB_PAT"])
    repo = g.get_repo("inDEPTHlab/DeMetRA-review")

    # ── Slugs for branch name and filename ────────────────────────
    timestamp   = datetime.now().strftime("%Y%m%d")
    # First author Surname, clean of odd characters
    author_slug = extract_author_slug(df.loc[0, "Author_list"][0])
    filename = f"assets/submissions/update_{author_slug}_{timestamp}.csv"
    branch_name = f"submission/{author_slug}-{timestamp}"

    # ── Create branch off main ────────────────────────────────────
    main_sha = repo.get_branch("main").commit.sha
    repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha = main_sha)

    # ── Push submission CSV ───────────────────────────────────────
    repo.create_file(
        path = filename,
        message = f"Submission: {title[:60]}",
        content = df.to_csv(index=False),
        branch = branch_name,
    )

    # ── Build PR body ─────────────────────────────────────────────
    n_mps = len(df)
    pr_body = (
        f"**Submitted via app** — {n_mps} new MPS(s)\n\n"
        f"| Field | Value |\n|---|---|\n"
        f"| Author | {(', ').join(df['Author_list'].iloc[0])} |\n"
        f"| Year | {df['Date'].iloc[0][:4]} |\n"
        f"| DOI | https://doi.org/{df['DOI'].iloc[0]} |\n"
        f"| Rows added | {n_mps} |\n"
        f"| Contact | {contact} |\n"
        f"| File | `{filename}` |\n\n"
        + "\n\n".join(
            f"### MPS #{i + 1}\n"
            + "\n".join(
                f"- **{col}**: {row[col]}"
                for col in ["Phenotype", "Category", "n CpGs"]
            )
            for i, row in df.iterrows()
        )
    )

    # ── Open PR with label ────────────────────────────────────────
    pr = repo.create_pull(
        title = f"[Submission] {author_slug} ({timestamp}) — {n_mps} MPS(s)",
        body = pr_body,
        head = branch_name,
        base = "main",
    )

    try:
        label = repo.get_label("submission")
    except Exception:
        label = repo.create_label(
            name="submission",
            color="f9d0d8",
            description="New MPS submission via app",
        )
    pr.add_to_labels(label)

    return pr.html_url