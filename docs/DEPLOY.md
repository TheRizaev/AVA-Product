# Deploying AVA

**Rule from the brief:** deploy **only from the submitted repository**. GitHub and the Hugging Face Space must run **the same commit**. After the deadline, only that submitted commit may be redeployed.

| Remote | URL |
|---|---|
| `origin` | `https://github.com/TheRizaev/AVA-Product.git` (public) |
| `space` | `https://huggingface.co/spaces/<USER>/<SPACE>` (Docker SDK) — **decide it first, see below** |
| Live app | `https://<user>-<space>.hf.space` (lower case, `/` → `-`) |

### Decide the target Space first (state on 2026-10-10)

- **`TheRizaev/ava` is NOT ours to push to.** That name resolves to `TheRizaev/AVA`, an existing *static* Space of an earlier project ("AVA - traffic CCTV analytics", served at `therizaev-ava.static.hf.space`). `https://therizaev-ava.hf.space` returns 404. **Never `--force` push there**: it would wipe that project.
- **The Space the current `space` remote points to** (`git remote -v`) is public and still serves an old prototype build: Russian UI, the old working name, uncalibrated thresholds (`/api/health`: threshold 0.45, threshold_pixdiff 0.08).
- **Recommended:** rename that old Space in its HF *Settings* to a free name (for example `TheRizaev/ava-inspect`). It is already a Docker Space on CPU Basic, which is free. Creating a *new* Docker Space instead needs a paid HF plan (PRO, $9/month; HF docs, checked 2026-10-11, see docs/SOURCES.md "Hosting"). Then point the remote at it and push the submitted commit:

  ```bash
  git remote -v
  git remote set-url space https://huggingface.co/spaces/<USER>/<SPACE>
  ```

- If the old prototype Space is not renamed and reused, make it private or delete it after the new deploy.
- Then put the final URL into: the README "Live demo" line, the repository's *About → Website*, the deck and the QR code. Check `curl https://<user>-<space>.hf.space/api/health` before the defence.

---

## 0. Before every deploy

Run these from the repository root.

1. **Regenerate the demo examples** only if the synthetic generator or the thresholds changed. Then review the five images in `web/examples/`.

   ```bash
   python -m eval.make_examples
   ```

2. **Re-run the full test suite.** It must pass, and `tests/test_demo_examples.py` in particular must pass: it runs the real pipeline on the five demo images and checks the four expected verdicts.

   ```bash
   pip install -r requirements-dev.txt
   python -m pytest -q
   ```

3. **Commit.** The user makes the commit; the same commit then goes to both remotes.

## 1. Hugging Face Space (Docker)

**What the Space runs:** the repository's `Dockerfile`.
- `python:3.11-slim`, running as uid 1000;
- copies only `ava/`, `server/`, `web/` and `requirements.txt` (see `.dockerignore`);
- one uvicorn worker on port 7860, because the reference cache is in-process;
- no disk writes.

The YAML front matter at the top of `README.md` (`sdk: docker`, `app_port: 7860`) tells HF how to build and expose the container.

**Hardware:** *CPU Basic*, 2 vCPU.
- A CPU Basic Space sleeps after 48 hours without visitors, and the first visit wakes it.
- Plans and prices are on <https://huggingface.co/pricing> and <https://huggingface.co/docs/hub/spaces-gpus>. Check them there for the cost model.

### Push steps

1. **Token.** Go to <https://huggingface.co/settings/tokens> and create a token with **write** access to `TheRizaev/ava`. Never commit it, and never paste it into a shared chat.
2. **Push the same commit to both remotes.** Use your HF username and the token as the password; Git Credential Manager stores it.

   ```bash
   git checkout main
   git push origin main
   git push space main
   ```

   - If the Space still holds only its own template commit (a new Space or the renamed old prototype Space), the first push needs `git push --force space main`. Never force-push to a Space that holds another project.
   - **HF rejects any binary file that is not tracked with LFS/Xet, whatever its size** ("Your push was rejected because it contains binary files"), and it checks every pushed commit. The five demo JPEGs in `web/examples/` were committed as plain blobs (commit `cebc21c`), so they must be moved to LFS before the first push, which rewrites `cebc21c` and later commits:

     ```bash
     git lfs install
     git lfs track "*.jpg" "*.jpeg" "*.png"        # writes .gitattributes; drop the old "*.jpg binary" / "*.png binary" lines
     git add .gitattributes && git commit -m "Track images with LFS"
     git lfs migrate import --include="*.jpg,*.jpeg,*.png" --include-ref=refs/heads/main
     git lfs ls-files                              # the five web/examples/*.jpg must be listed
     git push --force-with-lease origin main       # origin already has the old hashes
     git push --force space main
     ```

     Optionally test the push against a throwaway Space first. PDFs and dataset images are git-ignored.
3. **Check the build logs.** On the Space page the badge goes *Building* → *Running*. Open **Logs**:
   - *Build* should end with the `pip install -r requirements.txt` step;
   - *Container* should show `Application startup complete` and `Uvicorn running on http://0.0.0.0:7860`. Startup includes a short OpenCV warm-up.
   - If the build fails, read the first red line in *Build*. It is usually a dependency version.
4. **Smoke test:**

   ```bash
   curl https://<user>-<space>.hf.space/api/health        # {"ok":true,"version":"0.2.0","params":{...}}
   ```

   The `params` must match `ava/config.py` (for example `threshold`, `threshold_pixdiff`, `mismatch_frac` 0.15). Old values mean an old build.

5. **Test on a phone over HTTPS.** Open the **direct** URL `https://<user>-<space>.hf.space`, not the `huggingface.co/spaces/...` page, which wraps the app in an iframe.
   - [ ] **Surface check** → "Try an example: Defect" → FAIL with a zone. "No defect" → PASS.
   - [ ] **Compare with reference** → "Defect" → FAIL with a zone on the tape. "Brightness change" → AVA PASS, and the "Plain reference subtraction" panel shows "FAIL — a large part of the frame changed".
   - [ ] Camera: the permission prompt appears, and the rear camera shows. In compare mode the reference overlay slider works.
   - [ ] The QR button opens a code that another phone can scan.
   - If the camera is denied: on iOS, Settings → Safari → Camera → Allow, then reload. On Android Chrome, the lock icon → Permissions.

### Verify that it is the same commit everywhere

```bash
git rev-parse HEAD
git ls-remote origin main
git ls-remote space main
```

All three must print the same SHA. Put this SHA in the submission form.

### On the day of the defence

- **Open the live app URL at least 2 minutes before the defence.** That wakes the Space and warms up the process. Run one example in each mode, and keep the page open.
- Do not push anything after the deadline. To redeploy the submitted commit:

  ```bash
  git push --force space <SUBMITTED_COMMIT_SHA>:main
  ```

  A *Factory rebuild* from the Space settings also works.

## 2. GitHub

1. `TheRizaev/AVA-Product` must be **public**.
2. Push with `git push origin main`.
3. On github.com, check that:
   - `README.md` renders (the HF front matter shows as a small table, which is expected) and the Mermaid diagrams draw;
   - the demo JPEGs in `web/examples/` open (they are LFS objects after the migration);
   - `LICENSE` is detected as MIT;
   - `docs/`, `eval/RESULTS.md` and `eval/RESULTS_SURFACE.md` are present.
4. Put the Space URL in the repository's "About" → Website.

## 3. Fallback: local server + Cloudflare quick tunnel

Use this if the Space is down or the venue network blocks it. The phone needs HTTPS for the live camera, and a quick tunnel provides it without an account.

```bash
# laptop (Windows: winget install --id Cloudflare.cloudflared)
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000        # no --reload during the demo
cloudflared tunnel --url http://localhost:8000
# prints https://<random-words>.trycloudflare.com -> open it on the phones; the QR button follows it
```

- The URL changes on every run, so use the QR button on the running page.
- Keep the laptop on mains power, with sleep disabled.
- The Docker image also works locally: `docker build -t ava . && docker run --rm -p 8000:7860 ava`, then tunnel port 8000.
- **Without any network:** the laptop's own browser on `http://localhost:8000` has camera access, and the built-in examples need no camera at all.

## 4. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `503` or a "Starting" page | The Space is waking up or rebuilding. Wait about a minute, then reload. |
| No live camera, only the gallery | The page is not on HTTPS or `localhost`, or permission was denied. |
| `too_large` toast | The photo is over 15 MB. Lower the camera resolution, or use the live camera. |
| `RETAKE: could not align` | The Check photo is too far from the reference pose, or the part is featureless. Re-shoot from the same angle and distance. |
| `RETAKE: different part or part missing` | Over 15 % of the checked area changed. Check that it is the same part, and the same side of it. |
| `RETAKE: glare` with "A bright, overexposed spot could not be checked" | A white or clipped blob appeared that the reference does not have: glare, or a white label / tape / bare metal. Change the angle or lower the exposure; if it stays, check the spot by eye. |
| False zones in Surface check | Edges, holes, logos or background are in the frame. Fill the frame with the surface only, or use Compare with reference. |
| Slow responses | CPU Basic has 2 vCPU, and at most 2 checks run at once. The first check against a new reference also prepares it. |
