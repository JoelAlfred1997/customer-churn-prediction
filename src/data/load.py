"""Download and cache the IBM Telco Customer Churn dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Ordered list of mirrors; first success wins.
_DEFAULT_URLS: list[str] = [
    # IBM's own GitHub repository for the ICP4D demo
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    # IBM Watson Analytics asset mirror
    "https://raw.githubusercontent.com/IBM/invoke-wml-using-cognos-custom-control-api/master/assets/WA_Fn-UseC_-Telco-Customer-Churn.csv",
]

_KAGGLE_FALLBACK = (
    "kaggle datasets download -d blastchar/telco-customer-churn --unzip"
)


def download_telco_data(
    dest: Path | str,
    urls: list[str] | None = None,
    *,
    force: bool = False,
) -> Path:
    """Download the Telco churn CSV to *dest*, skipping if already present.

    Tries each URL in *urls* in order, falling back on failure.

    Args:
        dest: Destination file path (e.g. ``data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv``).
        urls: Ordered list of source URLs.  Defaults to well-known IBM mirrors.
        force: Re-download even when the file already exists.

    Returns:
        Resolved path of the saved file.

    Raises:
        RuntimeError: When every URL fails.
    """
    import requests  # soft import — not needed if file is already present

    dest = Path(dest).resolve()

    if dest.exists() and not force:
        logger.info("Dataset already present at %s — skipping download.", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    candidates = urls if urls is not None else _DEFAULT_URLS
    last_exc: Exception | None = None

    for url in candidates:
        try:
            logger.info("Downloading from %s", url)
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            dest.write_bytes(response.content)
            logger.info("Saved %d bytes to %s", len(response.content), dest)
            return dest
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            last_exc = exc

    raise RuntimeError(
        f"All {len(candidates)} download URL(s) failed.  Last error: {last_exc}\n"
        f"Manual Kaggle download:\n    {_KAGGLE_FALLBACK}"
    )


def load_raw(path: Path | str) -> pd.DataFrame:
    """Load the raw Telco CSV from *path*.

    TotalCharges is forced to ``str`` on read because ~11 rows in the source
    file contain blank strings that pandas would otherwise coerce to NaN and
    infer as float, masking the original data quality issue.

    Args:
        path: Path to the raw CSV file.

    Returns:
        DataFrame with 21 columns and ~7,043 rows.

    Raises:
        FileNotFoundError: When the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}.  "
            "Run `download_telco_data()` or see data/README.md for manual steps."
        )

    return pd.read_csv(path, dtype={"TotalCharges": str})
