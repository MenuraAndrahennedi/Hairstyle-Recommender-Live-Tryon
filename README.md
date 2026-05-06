# Hairstyle Recommender Try-On

Current repository state: focused on the static 2D try-on backend.

Current focus:

- static image analysis
- static hairstyle recommendation
- static 2D try-on

## Backend structure

- `backend/systems/static_2d` - active static 2D backend
- `backend/data` - shared data root (kept intact)
- `backend/outputs` - shared runtime/output root

## Static 2D run

Backend:

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

Install Python dependencies from:

- `backend/systems/static_2d/requirements.txt`

The current static system uses the reviewed asset bank at:

- `backend/data/processed/stage1_asset_bank`
