"""Download the three modeling batches from the public MIT-Stanford dataset.

Files are saved beside this script by default. Existing complete files are
skipped, and an incomplete file is resumed when the server supports ranges.
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


DEFAULT_DATA_DIR = Path(__file__).resolve().parent

FILES = {
    "2017-05-12_batchdata_updated_struct_errorcorrect.mat": (
        "https://data.matr.io/1/api/v1/file/5c86c0b5fa2ede00015ddf66/download",
        3_025_320_241,
    ),
    "2018-02-20_batchdata_updated_struct_errorcorrect.mat": (
        "https://s3.amazonaws.com/publications.matr.io/1/final_data/2018-02-20_batchdata_updated_struct_errorcorrect.mat",
        2_022_599_329,
    ),
    "2018-04-12_batchdata_updated_struct_errorcorrect.mat": (
        "https://data.matr.io/1/api/v1/file/5c86bd64fa2ede00015ddbb2/download",
        3_236_690_412,
    ),
}


def download(url: str, destination: Path, expected_size: int) -> None:
    current = destination.stat().st_size if destination.exists() else 0
    if current == expected_size:
        print(f"skip {destination.name}: already complete")
        return
    request = urllib.request.Request(url, headers={"Range": f"bytes={current}-"} if current else {})
    print(f"download {destination.name}: {current:,}/{expected_size:,} bytes")
    response = urllib.request.urlopen(request)
    # A server may ignore Range and return the full file with HTTP 200.
    # In that case overwrite instead of appending a duplicate payload.
    mode = "ab" if current and getattr(response, "status", 200) == 206 else "wb"
    with response, destination.open(mode) as target:
        while chunk := response.read(8 * 1024 * 1024):
            target.write(chunk)
    actual = destination.stat().st_size
    if actual != expected_size:
        raise IOError(f"Size mismatch for {destination}: expected {expected_size}, got {actual}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, (url, size) in FILES.items():
        download(url, args.output_dir / filename, size)


if __name__ == "__main__":
    main()
