from __future__ import annotations

import re
import requests
import pandas as pd

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# ── Constants ─────────────────────────────────────────────────────────────────

VALID_CATEGORIES: list[str] = [
    "Biological markers",
    "Genetic syndromes",
    "Lifestyle and environment",
    "Physical health indicators",
    "Neuro-psychiatric health indicators",
    "Cancer",
]

# Matches one complete author: "Surname, N." or "Van Den Berg, A. B." or "O'Brien, M."
# Surname words contain lowercase letters; initials are single capital + period
_AUTHOR_FIND_RE = re.compile(
    r"[A-Za-zÀ-ÿ\'][A-Za-zÀ-ÿ\-\']*"              # first surname word
    r"(?:\s[A-Za-zÀ-ÿ\'][A-Za-zÀ-ÿ\-\']*)*"        # optional extra surname words
    r",\s*(?:[A-Z]\.\s*)+",                          # comma + one or more initials
    re.UNICODE,
)

# Accepted date input formats, in order of preference
_DATE_FORMATS: list[str] = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y"]


# ── Validation result ─────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """
    Accumulates validation errors across all fields.
    `valid` is False as soon as any error is added.
    """
    valid:  bool                    = True
    errors: dict[str, list[str]]    = field(default_factory=dict)

    def add_error(self, field_name: str, message: str) -> None:
        self.valid = False
        self.errors.setdefault(field_name, []).append(message)

    def flat_errors(self) -> list[str]:
        """Flat list of 'Field: message' strings, suitable for UI display."""
        return [
            f"**{name}**: {msg}"
            for name, messages in self.errors.items()
            for msg in messages
        ]


# ── Field-level validators ────────────────────────────────────────────────────

def _check_required_string(
    value:      Any,
    field_name: str,
    result:     ValidationResult,
    min_length: int = 1,
) -> str | None:
    """
    Validates a required non-empty string.
    Returns the stripped string on success, None on failure.
    """
    if not isinstance(value, str) or len(value.strip()) < min_length:
        result.add_error(field_name, "Required — cannot be empty.")
        return None
    return value.strip()


def _check_optional_string(value: Any) -> str:
    """Returns a stripped string or empty string for optional fields."""
    if isinstance(value, str):
        return value.strip()
    return ""


def _check_author_list(raw: Any, result: ValidationResult) -> list[str] | str |  None:
    """
    Parses a comma-separated author string where each author is 'Surname, N.'

    Accepted input:
        'Smith, J.'
        'García-López, J. M.'
        'Smith, J., Van Den Berg, A. B., O\\'Brien, M.'
    """
    if not isinstance(raw, str) or not raw.strip():
        result.add_error(
            "Author list",
            "Required — provide authors in 'Surname, N.' format, "
            "e.g. 'Smith, J., Van Den Berg, A. B., O\\'Brien, M.'",
        )
        return None

    authors = [m.strip() for m in _AUTHOR_FIND_RE.findall(raw)]

    if not authors:
        result.add_error(
            "Author list",
            f"Could not parse any authors from '{raw}'. "
            "Expected format: 'Surname, N.' — e.g. 'Smith, J., Van Den Berg, A. B.'",
        )
        return None

    # Coverage check: warn if large parts of the string were not matched
    matched_chars = sum(len(a) for a in authors)
    total_chars   = len(re.sub(r"[\s,.]", "", raw))
    coverage      = matched_chars / total_chars if total_chars else 1

    if coverage < 0.75:
        result.add_error(
            "Author list",
            f"Some names could not be parsed — found {authors}. "
            "Check that each author follows 'Surname, N.' format.",
        )
        return None

    return authors[0] if len(authors) == 1 else authors


def _check_doi(raw: Any, result: ValidationResult) -> str | None:
    """
    Validates a DOI string by:
      1. Stripping common URL prefixes.
      2. Checking the DOI resolves via an HTTP HEAD request to doi.org.
    Returns the cleaned DOI (without URL prefix) on success, None on failure.
    """
    if not isinstance(raw, str) or not raw.strip():
        result.add_error("DOI", "Required — cannot be empty.")
        return None

    doi = (
        raw.strip()
        .removeprefix("https://doi.org/")
        .removeprefix("http://doi.org/")
        .removeprefix("doi:")
        .strip()
    )

    if not doi:
        result.add_error("DOI", "Invalid DOI — nothing left after stripping prefix.")
        return None

    try:
        response = requests.head(
            f"https://doi.org/{doi}",
            allow_redirects=True,
            timeout=6,
            headers={"User-Agent": "DeMetRA-review/1.0 (submission validator)"},
        )
        if response.status_code >= 400:
            result.add_error(
                "DOI",
                f"DOI 'https://doi.org/{doi}' returned HTTP {response.status_code} — "
                "the link appears broken. Please double-check."
            )
            return None
    except requests.Timeout:
        result.add_error("DOI", "DOI validation timed out — please check your connection and try again.")
        return None
    except requests.RequestException as exc:
        result.add_error("DOI", f"Could not verify DOI: {exc}")
        return None

    return doi


def _check_date(raw: Any, result: ValidationResult) -> str | None:
    """
    Parses a date string in any of the accepted formats.
    Returns a normalised 'YYYY-MM-DD' string on success, None on failure.
    """
    if not raw:
        result.add_error("Date", "Required.")
        return None

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(str(raw).strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    result.add_error(
        "Date",
        f"Unrecognised date format '{raw}'. "
        f"Accepted formats: YYYY-MM-DD, DD/MM/YYYY, MM/DD/YYYY, YYYY.",
    )
    return None


def _check_non_negative_int(
    value:      Any,
    field_name: str,
    result:     ValidationResult,
    minimum:    int = 0,
) -> int | None:
    """
    Validates an integer value with an optional minimum.
    Returns the integer on success, None on failure.
    """
    if value is None:
        result.add_error(field_name, f"Required — must be an integer ≥ {minimum}.")
        return None
    try:
        v = int(value)
    except (TypeError, ValueError):
        result.add_error(field_name, f"Must be an integer (got '{value}').")
        return None
    if v < minimum:
        result.add_error(field_name, f"Must be ≥ {minimum} (got {v}).")
        return None
    return v


def _check_category(value: Any, field_name: str, result: ValidationResult) -> str:
    """
    Validates an optional category string against the allowed list.
    An empty/None value is accepted. Returns the value as-is or empty string.
    """
    if not value or not isinstance(value, str) or not value.strip():
        return ""
    stripped = value.strip()
    if stripped not in VALID_CATEGORIES:
        result.add_error(
            field_name,
            f"'{stripped}' is not a recognised category. "
            f"Valid options: {', '.join(VALID_CATEGORIES)} — or leave blank.",
        )
    return stripped


# ── Public API ────────────────────────────────────────────────────────────────

def validate_and_structure(
    pub:      dict[str, Any],
    mps_list: list[dict[str, Any]],
) -> tuple[ValidationResult, pd.DataFrame | None]:
    """
    Validates all publication-level and MPS-level fields and returns a
    structured DataFrame with one row per MPS entry.

    Parameters
    ----------
    pub : dict
        Publication-level fields. Expected keys:
        'Author_list', 'Date', 'Title', 'DOI', 'Based_on',
        'Sample_type', 'Sample_size', 'Tissue', 'Array', 'Ancestry'

    mps_list : list of dict
        One dict per MPS block. Expected keys per block:
        'Phenotype', 'Category', 'n_CpGs', 'Developmental_period', 'Method'

    Returns
    -------
    (ValidationResult, DataFrame | None)
        DataFrame is None if validation failed.
        DataFrame has one row per MPS, with all publication columns repeated.
    """
    result = ValidationResult()

    # ── Publication-level validation ──────────────────────────────────────────
    author_list = _check_author_list(pub.get("Author_list"), result)
    date        = _check_date(pub.get("Date"), result)
    doi         = _check_doi(pub.get("DOI"), result)
    title       = _check_required_string(pub.get("Title"),       "Title",       result)
    based_on    = _check_required_string(pub.get("Based_on"),    "Based on",    result)
    sample_type = _check_required_string(pub.get("Sample_type"), "Sample type", result)
    sample_size = _check_non_negative_int(pub.get("Sample_size"), "Sample size", result, minimum=0)

    # Optional publication-level strings
    tissue   = _check_optional_string(pub.get("Tissue"))
    array    = _check_optional_string(pub.get("Array"))
    ancestry = _check_optional_string(pub.get("Ancestry"))

    # ── MPS-level validation ──────────────────────────────────────────────────
    if not mps_list:
        result.add_error("MPS", "At least one MPS entry is required.")

    cleaned_mps: list[dict[str, Any]] = []

    for i, mps in enumerate(mps_list, start=1):
        prefix = f"MPS #{i}"

        phenotype  = _check_required_string(mps.get("Phenotype"),            f"{prefix} — Phenotype",             result)
        category   = _check_category(        mps.get("Category"),             f"{prefix} — Category",              result)
        n_cpgs     = _check_non_negative_int( mps.get("n_CpGs"),              f"{prefix} — n CpGs",                result, minimum=3)
        dev_period = _check_required_string( mps.get("Developmental_period"), f"{prefix} — Developmental period",  result)
        method     = _check_required_string( mps.get("Method"),               f"{prefix} — Method",                result)

        cleaned_mps.append({
            "Phenotype":            phenotype   or "",
            "Category":             category,
            "n CpGs":               n_cpgs,
            "Developmental period": dev_period  or "",
            "Method":               method      or "",
        })

    # ── Abort if any errors were found ────────────────────────────────────────
    if not result.valid:
        return result, None

    # ── Build output DataFrame ────────────────────────────────────────────────
    pub_cols: dict[str, Any] = {
        "Author_list": author_list,
        "Date":        date,
        "Title":       title,
        "DOI":         doi,
        "Based on":    based_on,
        "Sample type": sample_type,
        "Sample size": sample_size,
        "Tissue":      tissue,
        "Array":       array,
        "Ancestry":    ancestry,
    }

    df = pd.DataFrame([{**pub_cols, **mps} for mps in cleaned_mps])

    return result, df

# TEST -----------------------------------------------------------------------------------
# pub = {
#     "Author_list":  "Smith, J., Van Den Berg, A. B., O'Brien, M.",
#     "Date":         "2026-03-15",
#     "Title":        "Methylation profile scores in early childhood: a test submission",
#     "DOI":          "10.1038/s41398-022-02195-3", 
#     "Journal": "Journal of Epi",
#     "Based_on":     "Raw individual-level data",
#     "Sample_type":  "Population cohort",
#     "Sample_size":  1200,
#     "Tissue":       "Whole blood",
#     "Array":        "EPICv2",
#     "Ancestry":     "European",
# }

# mps_list = [
#     {
#         "Phenotype":            "ADHD",
#         "Category":             "Neuro-psychiatric health indicators",
#         "n_CpGs":               450,
#         "Developmental_period": "Mid childhood",
#         "Method":               "Elastic net",
#     },
#     {
#         "Phenotype":            "BMI",
#         "Category":             "Physical health indicators",
#         "n_CpGs":               112,
#         "Developmental_period": "Birth",
#         "Method":               "LASSO",
#     },
# ]
# result, df = validate_and_structure(pub, mps_list)