# Deploying AVA

**Rule from the brief:** deploy **only from the submitted repository**. GitHub and the Hugging Face Space must run **the same commit**. After the deadline, only that submitted commit may be redeployed.

| Remote | URL |
|---|---|
| `origin` | `https://github.com/TheRizaev/AVA-Product.git` (public) |
| Space | `https://huggingface.co/spaces/TheRizaev/ava-qa` (Docker SDK, CPU Basic) |
| Live app | **https://therizaev-ava-qa.hf.space** |

Do **not** deploy to `TheRizaev/ava`: that is a different, earlier static Space of the team.

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

1. **Token.** Log in once with a token that has **write** access to `TheRizaev/ava-qa`: `huggingface-cli login`. Never commit the token.
2. **Push to GitHub, then upload exactly that commit to the Space.** HF rejects plain-git pushes of binary files (the demo JPEGs), so the Space is updated through the Hub API from a clean export of the committed tree; the Space commit message names the GitHub commit.

   ```bash
   git push origin main
   rm -rf /tmp/ava-deploy && mkdir /tmp/ava-deploy && git archive HEAD | tar -x -C /tmp/ava-deploy
   python - <<'PY'
   import subprocess
   from huggingface_hub import HfApi
   sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
   HfApi().upload_folder(folder_path="/tmp/ava-deploy", repo_id="TheRizaev/ava-qa", repo_type="space",
                         commit_message=f"Deploy GitHub commit {sha[:7]}",
                         commit_description=f"Exact contents of https://github.com/TheRizaev/AVA-Product/commit/{sha}",
                         delete_patterns=["*"])
   PY
   ```
3. **Check the build logs.** On the Space page the badge goes *Building* → *Running*. Open **Logs**:
   - *Build* should end with the `pip install -r requirements.txt` step;
   - *Container* should show `Application startup complete` and `Uvicorn running on http://0.0.0.0:7860`. Startup includes a short OpenCV warm-up.
   - If the build fails, read the first red line in *Build*. It is usually a dependency version.
4. **Smoke test:**

   ```bash
   curl https://therizaev-ava-qa.hf.space/api/health        # {"ok":true,"version":"0.2.0","params":{...}}
   ```

   The `params` must match `ava/config.py` (for example `threshold`, `threshold_pixdiff`, `mismatch_frac` 0.15). Old values mean an old build.

5. **Test on a phone over HTTPS.** Open the **direct** URL `https://therizaev-ava-qa.hf.space`, not the `huggingface.co/spaces/...` page, which wraps the app in an iframe.
   - [ ] **Surface check** → "Try an example: Defect" → FAIL with a zone. "No defect" → PASS.
   - [ ] **Compare with reference** → "Defect" → FAIL with a zone on the crack. "Brightness change" → AVA PASS, and the "Plain reference subtraction" panel shows "FAIL — a large part of the frame changed".
   - [ ] Camera: the permission prompt appears, and the rear camera shows. In compare mode the reference overlay slider works.
   - [ ] The QR button opens a code that another phone can scan.
   - If the camera is denied: on iOS, Settings → Safari → Camera → Allow, then reload. On Android Chrome, the lock icon → Permissions.

### Verify that it is the same commit everywhere

```bash
git rev-parse --short HEAD                     # e.g. 61b37b2
git ls-remote origin main
# the latest commit on https://huggingface.co/spaces/TheRizaev/ava-qa/commits/main is "Deploy GitHub commit <same sha>"
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
   - the demo JPEGs in `web/examples/` open;
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
