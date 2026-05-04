#!/usr/bin/env python3
"""Download LncRNADisease v3.0 bulk files from the official site.

Source: http://www.rnanut.net/lncrnadisease/index.php/home/info/download

Please cite the LncRNADisease v3.0 paper when using this data, e.g.
Zhang et al., "LncRNADisease v3.0: an updated database of long non-coding
RNA-associated diseases", Nucleic Acids Research (2024), among others
as listed on the database site.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request

FILES = {
    "simple": (
        "http://www.rnanut.net/lncrnadisease/static/download/website_simple_data.csv",
        "website_simple_data.csv",
    ),
    "alldata": (
        "http://www.rnanut.net/lncrnadisease/static/download/website_alldata.tsv",
        "website_alldata.tsv",
    ),
    "causal": (
        "http://www.rnanut.net/lncrnadisease/static/download/website_causal_data.tsv",
        "website_causal_data.tsv",
    ),
}


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research; +https://github.com/)"})
    print(f"Downloading\n  {url}\n  -> {dest}")
    with urllib.request.urlopen(req, timeout=600) as resp, open(dest, "wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    print(f"Done ({dest.stat().st_size} bytes)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "raw",
        help="Directory to write downloaded files",
    )
    p.add_argument(
        "--which",
        nargs="+",
        choices=list(FILES.keys()),
        default=["simple"],
        help="Which official file(s) to fetch (default: simple CSV)",
    )
    args = p.parse_args()
    for key in args.which:
        url, name = FILES[key]
        download(url, args.out_dir / name)


if __name__ == "__main__":
    main()
