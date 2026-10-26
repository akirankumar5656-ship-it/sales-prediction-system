"""
src/file_utils.py

Shared file-loading helpers used by data_preprocessing.py, eda.py, and
analytics.py — kept in one place so encoding-handling logic doesn't drift
between the three.
"""

import logging
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Tried in order. utf-8 first (the correct/expected case), then the two
# encodings real-world "exported from Excel on Windows" CSVs almost always
# turn out to be, then latin-1 last as a catch-all — latin-1 can decode
# ANY byte sequence without raising, so it guarantees the file loads, at
# the cost of possibly mis-rendering a handful of special characters if
# the true encoding was something more exotic.
CANDIDATE_ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]


def read_csv_robust(path: str, encodings: Optional[List[str]] = None, **read_csv_kwargs) -> pd.DataFrame:
    """
    Read a CSV trying several encodings in turn, instead of letting
    pandas's UTF-8 default raise UnicodeDecodeError on a Windows/Excel
    -exported file (Kaggle sales datasets are very commonly cp1252 or
    latin-1, not UTF-8). Logs which encoding actually worked.
    """
    encodings = encodings or CANDIDATE_ENCODINGS
    last_error = None

    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding, **read_csv_kwargs)
            if encoding != "utf-8":
                logger.warning(
                    "'%s' is not valid UTF-8 — loaded successfully using '%s' instead. "
                    "If any text columns look garbled, re-save the CSV as UTF-8 in Excel "
                    "('Save As' -> 'CSV UTF-8').",
                    path, encoding,
                )
            return df
        except UnicodeDecodeError as exc:
            last_error = exc
            continue

    # Should be unreachable since latin-1 never raises UnicodeDecodeError,
    # but re-raise defensively if every candidate somehow failed.
    raise last_error
