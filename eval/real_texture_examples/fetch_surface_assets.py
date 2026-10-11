"""Download Poly Haven (CC0) texture maps + metadata used by make_surface_examples.py."""
import json, os, sys, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets")
UA = {"User-Agent": "AVA-hackathon-demo/1.0"}

def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return r.read()

def fetch(aid, maps=("Diffuse", "nor_gl", "Displacement", "Rough")):
    os.makedirs(os.path.join(OUT, aid), exist_ok=True)
    meta_p = os.path.join(OUT, aid, "info.json")
    files = json.loads(get(f"https://api.polyhaven.com/files/{aid}"))
    info = json.loads(get(f"https://api.polyhaven.com/info/{aid}"))
    rec = {"id": aid, "name": info.get("name"), "authors": list(info.get("authors", {}).keys()),
           "url": f"https://polyhaven.com/a/{aid}", "licence": "CC0 1.0", "scale": info.get("scale"),
           "files": {}}
    for m in maps:
        alt = {"Diffuse": ["Diffuse", "diff", "coll1", "col1"], "Displacement": ["Displacement", "disp"]}.get(m, [m])
        src = next((files[a] for a in alt if a in files), None)
        if src is None:
            continue
        res = "2k" if "2k" in src else "1k"
        url = src[res]["jpg"]["url"]
        p = os.path.join(OUT, aid, os.path.basename(url))
        if not os.path.exists(p):
            open(p, "wb").write(get(url))
        rec["files"][m] = {"url": url, "path": os.path.relpath(p, HERE).replace("\\", "/")}
    json.dump(rec, open(meta_p, "w"), indent=1)
    return rec

if __name__ == "__main__":
    for aid in sys.argv[1:]:
        r = fetch(aid); print(aid, r["authors"], list(r["files"]))
