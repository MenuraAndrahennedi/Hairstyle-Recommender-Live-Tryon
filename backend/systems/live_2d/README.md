# Live 2D Try-On

This subsystem now works only with the project runtime and project assets.

Engine stack:

- MediaPipe face landmarker
- project face-attribute and hairstyle recommendation logic
- project hair segmentation model
- `celeba_full_hair_assets/tryon_clean` hairstyle assets
- original live overlay logic for webcam try-on
- OpenCV webcam capture and live frame rendering

Key runtime assets:

- `backend/models/face_landmarker.task`
- `backend/data/processed/celeba_full_hair_assets/*`
- `backend/data/processed/celeba_hair_rich_assets/*`
- `backend/models/*`

Run the exact webcam demo engine:

```powershell
.\.venv\Scripts\python.exe backend/systems/live_2d/scripts/run_webcam_demo.py
```

API info endpoint:

```text
/api/live-2d/info
```
