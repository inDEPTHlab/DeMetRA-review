from __future__ import annotations

import pandas as pd
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any
import requests


@dataclass
class ValidationResult:
    """
    Accumulates validation errors across all fields.
    `valid` is False as soon as any error is added.
    """
    valid:  bool                 = True
    errors: dict[str, list[str]] = field(default_factory=dict)

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

def _unwrap(value: Any) -> str:
    """Unwrap selectize single-value tuples/lists → plain string."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else ""
    return value if value is not None else ""


def _check_string(
    element: Any,
    field_name: str,
    result: ValidationResult,
    context: str = "",
    required: bool = False,
    min_length: int = 1) -> str | None:
    """
    Validates a required non-empty string.
    Returns the stripped string on success, None on failure.
    """
    value = element.get(field_name)
   
    s = _unwrap(value).strip()

    if required: 
        return s
    
    if len(s) < min_length:

        if context != '':
            field_name = f"{context} - {field_name}"

        result.add_error(field_name, "Required — cannot be empty.")

        return ""
       
    return s


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
            "Required — provide authors in Harvard format, "
            "e.g. 'Smith, J., Van Den Berg, A. B., O\\'Brien, M.'",
        )
        return None
    
    # Matches one complete author: "Surname, N." or "Van Den Berg, A. B." or "O'Brien, M."
    # Surname words contain lowercase letters; initials are single capital + period
    _AUTHOR_FIND_RE = re.compile(
        r"[A-Za-zÀ-ÿ\'][A-Za-zÀ-ÿ\-\']*"          # first surname word
        r"(?:\s[A-Za-zÀ-ÿ\'][A-Za-zÀ-ÿ\-\']*)*"   # optional extra surname words
        r",\s*(?:[A-Z]\.\s*)+",                   # comma + one or more initials
        re.UNICODE,
    )

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
    total_chars = len(re.sub(r"[\s,.]", "", raw))
    
    if total_chars and (matched_chars / total_chars) < 0.75:
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


def _check_contact(value, result):
    _EMAIL_RE = re.compile(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$')

    s = _unwrap(value).strip()

    if not s:
        result.add_error("Contact", "Required — cannot be empty.")
        return None
    if not _EMAIL_RE.match(s):
        result.add_error("Contact", f"'{s}' is not a valid email address.")
        return None
    return s

def _check_date(raw, result):
    if raw is None:
        result.add_error("Date", "Required.")
        return None
    if raw > datetime.today().date():
        result.add_error("Date", "Publication date cannot be in the future.")
        return None
    return raw.strftime("%Y-%m-%d")


def _check_non_negative_int(
    value:      Any,
    field_name: str,
    result:     ValidationResult,
    minimum:    int = 1,
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

    pub_cols: dict[str, Any] = {
        "Title": _check_string(pub, "Title", result, required = True),
        "Author_list": _check_author_list(pub.get("Author_list"), result),
        "Contact": _check_contact(pub.get("Contact"), result),
        "DOI": _check_doi(pub.get("DOI"), result),
        "Journal": _check_string(pub, "Journal", result, required = False),
        "Date": _check_date(pub.get("Date"), result),
    }

    # ── MPS-level validation ──────────────────────────────────────────────────
    if not mps_list:
        result.add_error("MPS", "At least one MPS entry is required.")

    cleaned_mps: list[dict[str, Any]] = []

    for i, mps in enumerate(mps_list, start=1):
        p = f"MPS #{i}"
    
        cleaned_mps.append({
            "Phenotype": _check_string(mps, "Phenotype", result, required = True, context = p),
            "Category": _check_string(mps, "Category", result, required = True, context = p),
            "N CpGs": _check_non_negative_int(mps.get("n CpGs"), 
                        f"{p} — N CpGs", result, minimum=2),
            "Sample size": _check_non_negative_int(mps.get("Sample size"), 
                        f"{p} — Sample size", result, minimum=1),
            "Array": _check_string(mps, "Array", result, required = True, context = p),
            "Tissue": _check_string(mps, "Tissue", result, required = True, context = p),
            "Developmental period": _check_string(mps, "Developmental period", result, 
                        required = True, context = p),
            "Ancestry": _check_string(mps, "Ancestry", result, required = False, context = p),
            "Based on": _check_string(mps, "Based on", result, required = True, context = p),
            "Reference": _check_string(mps, "Reference", result, required = False, context = p),
            "Reference DOI": _check_doi(mps.get("Reference DOI"), result),
            "Reference match": mps.get("Reference match") or [],
            "Method": _check_string(mps, "Method", result, required = False, context = p),
            "Performance metric": _check_string(mps, "Performance metric", result, required = False, context = p),
            "Performance value": _check_string(mps, "Performance value", result, required = False, context = p),
            "MPS link": _check_string(mps, "MPS link", result, required = False, context = p), 
        })

    # ── Abort if any errors were found ────────────────────────────────────────
    if not result.valid:
        return result, None

    # ── Build output DataFrame ────────────────────────────────────────────────

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