"""Download the LBNL "Queued Up" interconnection-queue workbooks into this folder.

The raw ``.xlsx`` files are large and re-downloadable, so they are git-ignored;
this module fetches them on demand. ``model/train.py`` calls :func:`ensure` for
the workbook it needs, so the pipeline is self-bootstrapping; you can also run
this file directly to download every registered workbook:

    python data/fetch_workbooks.py

URLs are pinned per edition (the LBNL publication landing page blocks automated
requests, so auto-discovering "the latest" is unreliable). When LBNL ships a new
edition, add its direct URL to ``WORKBOOKS`` below. A browser ``User-Agent`` is
required — LBNL's CDN returns 403 to urllib's default agent.
"""

import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent


def _ssl_context():
    """An SSL context that can verify LBNL's certificate.

    The python.org macOS Python build does not use the system CA store, so a
    plain default context raises CERTIFICATE_VERIFY_FAILED (even though ``curl``
    works). Use certifi's CA bundle when available — that is what makes urllib
    match curl — and fall back to the default context otherwise.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


_SSL_CONTEXT = _ssl_context()

# Destination filename (as referenced elsewhere in the repo) -> direct URL.
WORKBOOKS = {
    "LBNL_Ix_Queue_Data_File_thru2025.xlsx":
        "https://emp.lbl.gov/sites/default/files/2026-05/"
        "LBNL_Ix_Queue_Data_File_thru2025.xlsx",
    "LBNL_Ix_Queue_Data_File_thru2024_v2.xlsx":
        "https://eta-publications.lbl.gov/sites/default/files/2025-08/"
        "lbnl_ix_queue_data_file_thru2024_v2.xlsx",
}

_HEADERS = {"User-Agent": "Mozilla/5.0"}


def download(name):
    """Download the workbook registered under ``name`` into ``DATA_DIR``.

    Args:
        name: A key of ``WORKBOOKS`` (also the destination filename).

    Returns:
        The ``Path`` of the downloaded file.
    """
    url = WORKBOOKS[name]
    dest = DATA_DIR / name
    print(f"Downloading {name} from {url} ...", flush=True)
    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, context=_SSL_CONTEXT) as resp:
            payload = resp.read()
    except urllib.error.URLError as exc:  # network error / 403 / moved URL / SSL
        hint = ""
        if isinstance(getattr(exc, "reason", None), ssl.SSLCertVerificationError):
            hint = (" This is a missing CA bundle; `pip install certifi` usually "
                    "fixes it on macOS python.org builds.")
        raise RuntimeError(
            f"Could not download {name} from {url} ({exc}).{hint} The URL may "
            f"also have changed — update WORKBOOKS in data/fetch_workbooks.py."
        ) from exc
    dest.write_bytes(payload)
    print(f"  saved {dest} ({len(payload) / 1e6:.1f} MB)")
    return dest


def ensure(path):
    """Return ``path``, downloading the workbook first if it is missing.

    Args:
        path: Path to a workbook. Its filename must be a key of ``WORKBOOKS``.

    Returns:
        The ``Path`` to the present-on-disk workbook.
    """
    path = Path(path)
    if path.exists():
        return path
    if path.name not in WORKBOOKS:
        raise FileNotFoundError(
            f"{path} is missing and no download URL is registered for "
            f"{path.name!r} in data/fetch_workbooks.py."
        )
    return download(path.name)


def main():
    """Download every registered workbook that is not already present."""
    for name in WORKBOOKS:
        if (DATA_DIR / name).exists():
            print(f"Have {name}")
        else:
            download(name)


if __name__ == "__main__":
    sys.exit(main())
