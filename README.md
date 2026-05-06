# Hairstyle Recommender Try-On

Current repository state: reorganized around four backend subsystems behind a single gateway app.

Current focus:

- static auto try-on
- generative try-on
- live 2D try-on
- live 3D try-on placeholder

## Backend structure

- `backend/app` - gateway backend entrypoint that mounts the subsystem apps
- `backend/systems/static_auto_tryon` - separated automatic static try-on subsystem shell
- `backend/systems/generative_tryon` - separated generative try-on subsystem shell
- `backend/systems/live_2d` - dedicated live 2D webcam subsystem
- `backend/systems/live_3d` - reserved live 3D subsystem
- `backend/models` - shared trained and runtime model store for all systems
- `backend/data` - shared data root (kept intact)
- `backend/outputs` - shared runtime/output root

## Backend run

Gateway backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --app-dir backend
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

Open:

- `http://127.0.0.1:5173`
- `http://127.0.0.1:8000/docs`

Subsystem mounts:

- `http://127.0.0.1:8000/api/static-auto`
- `http://127.0.0.1:8000/api/generative`
- `http://127.0.0.1:8000/api/live-2d`
- `http://127.0.0.1:8000/api/live-3d`

Install Python dependencies from:

- `requirements.txt`

The current static system uses the reviewed full-hair asset bank under:

- `backend/data/processed/celeba_full_hair_assets`
