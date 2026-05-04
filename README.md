# Hairstyle Recommender Live Tryon

Stage-based implementation of a hairstyle recommendation and virtual try-on system.

Current focus:

- Version 1: static image analysis, recommendation, and overlay try-on
- Version 2: dataset-based hairstyle asset bank preparation

## Stage 1 run

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

Stage 1 uses the reviewed asset bank at:

- `backend/data/processed/stage1_asset_bank`
