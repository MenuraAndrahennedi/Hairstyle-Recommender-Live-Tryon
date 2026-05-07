# Hairstyle Recommender Live Tryon

Multi-system hairstyle try-on project with:

- Static Auto Try-On
- Generative Try-On
- Live 2D Try-On
- Live 3D placeholder

## First-Time Setup

### 1. Download the project

```powershell
git clone https://github.com/MenuraAndrahennedi/Hairstyle-Recommender-Live-Tryon.git
cd Hairstyle-Recommender-Live-Tryon
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install Python dependencies

```powershell
pip install -r requirements.txt
```

### 4. Install frontend dependencies

```powershell
cd frontend
npm install
cd ..
```

## Run the Project

### Backend gateway

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --app-dir backend
```

### Frontend

```powershell
cd frontend
npm run dev
```

### Open in browser

- Frontend: `http://127.0.0.1:5173`
- Backend docs: `http://127.0.0.1:8000/docs`

### Backend subsystem mounts

- `http://127.0.0.1:8000/api/static-auto`
- `http://127.0.0.1:8000/api/generative`
- `http://127.0.0.1:8000/api/live-2d`
- `http://127.0.0.1:8000/api/live-3d`

### Run live 2D webcam directly

```powershell
.\.venv\Scripts\python.exe backend/systems/live_2d/scripts/run_webcam_demo.py
```

## Project Structure

- `backend/app`
  Gateway backend that mounts all subsystems under one FastAPI app.

- `backend/systems/static_auto_tryon`
  Static automatic pipeline: upload image, detect face, segment hair, recommend hairstyles, and render 2D try-on outputs.

- `backend/systems/generative_tryon`
  Generative workflow: uses the same recommendation and segmentation pipeline to prepare reference packages and generate edited hairstyle outputs.

- `backend/systems/live_2d`
  Webcam-based live 2D try-on system using MediaPipe face tracking plus the project’s own recommendation, segmentation, and hair assets.

- `backend/systems/live_3d`
  Reserved for future live 3D work.

- `backend/models`
  Shared trained model store for the whole project.
  Includes face attribute, hair segmentation, mapper, hairstyle attribute, and MediaPipe face landmarker assets.

- `backend/data/raw`
  Original source datasets kept for preprocessing and reproducibility.

- `backend/data/datasets`
  Training-ready datasets and manifests created from the raw data.

- `backend/data/processed`
  Processed asset banks used by the running systems.

- `backend/outputs`
  Runtime outputs such as uploads, predictions, static try-on results, and generative package outputs.

- `notebooks`
  End-to-end research, preprocessing, dataset creation, training, evaluation, and try-on experiment notebooks.

- `frontend`
  Vite + React frontend with a single home page for the available systems.

## Datasets and How They Were Used

### 1. CelebA

Used for:

- face attribute dataset creation
- hairstyle asset extraction
- recommendation asset bank creation

Main role in the project:

- face photos from CelebA were filtered and processed into hairstyle assets
- these assets were labeled, reviewed, and converted into the final static/generative asset banks
- CelebA-derived records were also used to train the face attribute model and support recommendation mapping

Important outputs created from it:

- `backend/data/datasets/celeba_face_basic`
- `backend/data/datasets/celeba_full_hair`
- `backend/data/datasets/celeba_hair_rich`
- `backend/data/processed/celeba_full_hair_assets`
- `backend/data/processed/celeba_hair_rich_assets`

### 2. CelebAMask-HQ

Used for:

- hair segmentation dataset preparation
- reviewed hairstyle attribute labeling support

Main role in the project:

- segmentation masks from CelebAMask-HQ were used to create the training data for the hair segmentation model
- reviewed samples also supported hairstyle attribute work

Important outputs created from it:

- `backend/data/datasets/hair_segmentation`
- `backend/data/processed/celebamask_hq_hair_assets`

### 3. K-Hairstyle

Used for:

- hairstyle attribute dataset preparation
- hairstyle label translation and normalization

Main role in the project:

- K-Hairstyle labels were translated into the project’s normalized hairstyle taxonomy
- those normalized labels were used to build training records for hairstyle-related models and experiments

Important outputs created from it:

- `backend/data/datasets/hairstyle_attribute`
- preprocessing notebooks that translate and normalize hairstyle labels

## How Assets Were Built

The runtime hairstyle assets mainly come from the CelebA full-hair pipeline.

High-level flow:

1. Extract usable hairstyle regions from source images.
2. Save image, mask, and metadata for each asset.
3. Build richer labeled metadata with predicted and reviewed hairstyle attributes.
4. Review and keep only better assets for runtime use.
5. Create cleaner render-safe variants for try-on.
6. Build `tryon_clean` image/mask pairs for static and live rendering.

Most important runtime asset folders:

- `backend/data/processed/celeba_full_hair_assets/metadata`
- `backend/data/processed/celeba_full_hair_assets/reviewed_render_safe/kept_assets.jsonl`
- `backend/data/processed/celeba_full_hair_assets/tryon_clean/images`
- `backend/data/processed/celeba_full_hair_assets/tryon_clean/masks`

How systems use them:

- Static Auto Try-On:
  recommends from the reviewed render-safe asset bank and renders with `tryon_clean` when available.

- Generative Try-On:
  uses the same reviewed asset bank and `tryon_clean` assets as reference inputs for generation.

- Live 2D:
  uses the same recommendation pipeline and overlays `tryon_clean` assets in real time.

## Main Models

Shared under `backend/models`:

- `celeba_face_basic/face_attribute_model.pt`
  Predicts face attributes used for recommendation.

- `hair_segmentation/v2_unet_product.pt`
  Segments the subject hair region.

- `reviewed_hairstyle_basic/basic_attribute_model.pt`
  Hairstyle attribute model used in data/model workflows.

- `face_to_hair_mapper_light/mapper_config.json`
  Maps face attributes to likely hairstyle attributes for recommendation.

- `face_landmarker.task`
  MediaPipe model used for landmark detection, especially in live 2D.

## Main Libraries Used

- `FastAPI`
  Backend APIs and subsystem gateway.

- `React + Vite`
  Frontend UI.

- `PyTorch`
  Training and inference for the project’s segmentation and attribute models.

- `MediaPipe`
  Face landmark detection and geometry estimation, especially for live 2D tracking.

- `OpenCV`
  Webcam capture and real-time image compositing.

- `Pillow`
  Image loading, masking, cleaning, and asset preparation.

- `NumPy`
  Image and mask array operations.

- `Diffusers`, `Transformers`, `Accelerate`
  Generative try-on pipeline and inpainting workflow.

- `Pandas`
  Dataset preparation, metadata processing, and notebook analysis.

## Notes

- The project is organized as a development-focused repository, so raw data, dataset-building notebooks, and training notebooks are kept for reproducibility.
- Running systems mainly use `backend/data/processed` and `backend/models`.
- `live_3d` is still a placeholder for future work.
