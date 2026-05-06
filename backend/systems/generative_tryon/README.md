# Generative Try-On Backend

This is the standalone backend system for the generative try-on approach.

It keeps its own copied support logic and prepares a standalone generative package containing:

- subject input image
- predicted subject hair mask
- erased subject image
- cleaned reference hairstyle image and mask
- preview panel
- manifest with anchors and generation instructions
- optional final inpainted output produced from the package

Outputs are written under `backend/outputs/generative_tryon`.

## Recommended environment

Use the project virtual environment:

```powershell
.\.venv\Scripts\python.exe
```

The final generative edit path uses:

- `torch`
- `torchvision`
- `diffusers`
- `transformers`
- `accelerate`
- `safetensors`

## Run

From this folder:

```powershell
uvicorn generative_app.main:app --reload
```

## Main endpoint

- `POST /api/generative-tryon/package`

Form fields:

- `image`: input portrait image
- `asset_id`: reference hairstyle asset id

## Notebook / CLI final generation

After a package is prepared, you can produce a final inpainted image with:

```powershell
cd backend/systems/generative_tryon
..\..\..\.venv\Scripts\python.exe -m generative_app.core.generative_inpaint_runner --manifest "D:\path\to\package_manifest.json"
```

Default model:

- `runwayml/stable-diffusion-inpainting`

Notes:

- The first run downloads model weights unless you point `--model-id` at a local model directory.
- The runner writes `final_generated.png` and `final_generation_metadata.json` into the same package folder.
