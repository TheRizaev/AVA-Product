"""Download CC0 Poly Haven textures (diffuse + normal GL, 2k JPG) and their metadata into assets/."""
import json, sys, urllib.request
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets"
IDS = sys.argv[1:]
UA = {"User-Agent": "AVA-demo-examples/1.0"}

def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()

meta_path = ASSETS / "assets_meta.json"
meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
for aid in IDS:
    files = json.loads(get(f"https://api.polyhaven.com/files/{aid}"))
    info = json.loads(get(f"https://api.polyhaven.com/info/{aid}"))
    rec = {"id": aid, "name": info["name"], "authors": list(info["authors"].keys()),
           "url": f"https://polyhaven.com/a/{aid}", "license": "CC0 1.0", "files": {}}
    for key, out in (("Diffuse", "diff"), ("nor_gl", "nor_gl")):
        if key not in files:
            continue
        for res in ("2k", "1k"):
            if res in files[key] and "jpg" in files[key][res]:
                url = files[key][res]["jpg"]["url"]
                dst = ASSETS / f"{aid}_{out}_{res}.jpg"
                if not dst.exists():
                    dst.write_bytes(get(url))
                rec["files"][out] = {"path": dst.name, "url": url}
                break
    meta[aid] = rec
    print(aid, rec["authors"], list(rec["files"]))
meta_path.write_text(json.dumps(meta, indent=1))
