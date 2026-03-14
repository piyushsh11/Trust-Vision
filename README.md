# Trust Vision – Adversarial Robustness Lab

Unified demo with two surfaces:
- **Pixel attack lab** (`index2.html`, FastAPI backend).
- **Multimodal adversarial medical simulator** (`para meter based attack/index.html`) with Grad-CAM overlay, disease probability bars, findings, vitals support, and adversarial comparison.

## Project layout
- `backend.py` – FastAPI pixel-attack API (FGSM/PGD/CW/AutoAttack, defenses, cert radii).
- `server.py` – ASGI entrypoint: mounts API at `/api` and serves static files (all HTML/JS/CSS) from repo root.
- `index1.html` – Overview + navigation.
- `index2.html` – Pixel attack UI.
- `para meter based attack/index.html` – Multimodal UI (visual/analytic only; no in-browser PyTorch).
- `requirements.txt` – Python deps.
- `Dockerfile`, `hf.yaml` – Deployable container spec for Hugging Face Spaces (port 8000).
- `logs/` – runtime logs (ignored in git).
- Large assets/datasets (ignored via `.gitignore`): checkpoints, COVID/Chest sample images, zips.

## Quickstart (local, unified frontend + API)
```bash
cd adversarial-web-demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# run unified server (serves static + /api)
uvicorn server:app --host 0.0.0.0 --port 8000
# open in browser:
# http://0.0.0.0:8000/index1.html
# http://0.0.0.0:8000/index2.html
# http://0.0.0.0:8000/para%20meter%20based%20attack/index.html
```

If you prefer a separate static server:
```bash
python -m http.server 3000 --bind 0.0.0.0
uvicorn backend:app --host 0.0.0.0 --port 8000
```
Update JS endpoints to point to `http://localhost:8000` if needed.

## Deploy free on Hugging Face Spaces (recommended)
1. Ensure `Dockerfile` and `hf.yaml` are in the repo root.
2. Push to GitHub.
3. Create a new HF Space → “Docker” → connect the repo.
4. Build; Space exposes port 8000 from the container. Static pages are served by `server.py` and API is under `/api`.
   - Open: `https://<your-space>.hf.space/index1.html`, `.../index2.html`, `.../para%20meter%20based%20attack/index.html`.

## Notes on assets
- Checkpoints (`*.pt/*.pth`) and large datasets are git-ignored. Keep them locally or upload to a storage bucket if needed.
- The multimodal UI shows Grad-CAM-style overlays and synthetic vitals/predictions; wiring real Grad-CAM requires a model checkpoint + backend route.

## SDG scope
- Primary: **SDG 3** (health) and **SDG 9** (industry/innovation).
- SDG 4/5 are not implemented in messaging/UX.

## Next improvements (optional)
- Add true Grad-CAM API endpoint and hook UI to it.
- Add epsilon sweep & AutoAttack batch summary table.
- Expose certified radius badge per sample.
- CI guardrail: run `robustness_check.py` in CI to fail when robust acc drops.
