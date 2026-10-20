"""Fetch the official research dataset locally. Never include data/ in a shared archive."""
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile
import io

ROOT=Path(__file__).resolve().parent
URL="https://files.grouplens.org/datasets/movielens/ml-1m.zip"
if __name__=="__main__":
    with urllib.request.urlopen(URL,timeout=60) as response: raw=response.read()
    folder=ROOT/"data";folder.mkdir(exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for name in ["ratings.dat","movies.dat","README"]:
            (folder/name).write_bytes(archive.read("ml-1m/"+name))
    (folder/"download.json").write_text(json.dumps({"source":URL,"archive_sha256":hashlib.sha256(raw).hexdigest()},indent=2))
    print("Downloaded MovieLens 1M. Read data/README before use. Raw data must not be redistributed.")
