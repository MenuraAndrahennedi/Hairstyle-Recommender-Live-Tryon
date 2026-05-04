# Dataset Setup

This document captures the dataset layout currently present in the workspace and the parts of each dataset that matter for this project.

## Raw dataset locations

- `backend/data/raw/celeba`
- `backend/data/raw/celebamask_hq`
- `backend/data/raw/celebahairmask_hq`
- `backend/data/raw/khairstyle`

## Current workspace status

- `CelebA` is present and populated.
- `CelebAMask-HQ` is present and populated.
- `CelebAHairMask-HQ` folder currently appears empty in this workspace.
- `K-Hairstyle` is present and populated.

## CelebA

### Present files

- `img_align_celeba/`
- `list_attr_celeba.csv`
- `list_bbox_celeba.csv`
- `list_eval_partition.csv`
- `list_landmarks_align_celeba.csv`

### Observed annotation schema

- `list_attr_celeba.csv`
  - one row per image
  - `image_id`
  - 40 binary attributes with values `1` and `-1`
  - directly useful fields for this project include:
    - `Bangs`
    - `Bald`
    - `Black_Hair`
    - `Blond_Hair`
    - `Brown_Hair`
    - `Gray_Hair`
    - `Eyeglasses`
    - `Male`
    - `No_Beard`
    - `Mustache`
    - `Sideburns`
    - `Straight_Hair`
    - `Wavy_Hair`
    - `Wearing_Hat`
    - `Young`

- `list_bbox_celeba.csv`
  - face crop box per image
  - columns: `image_id`, `x_1`, `y_1`, `width`, `height`

- `list_landmarks_align_celeba.csv`
  - 5 aligned facial landmarks per image
  - columns:
    - `lefteye_x`, `lefteye_y`
    - `righteye_x`, `righteye_y`
    - `nose_x`, `nose_y`
    - `leftmouth_x`, `leftmouth_y`
    - `rightmouth_x`, `rightmouth_y`

- `list_eval_partition.csv`
  - train/val/test split
  - `partition` values appear to follow standard CelebA split ids

### Project relevance

- Not a direct hairstyle recommendation dataset.
- Useful later for:
  - optional face-attribute modeling
  - face-analysis experiments
  - validation of face-related preprocessing

### Practical note

- The CelebA images are present in this workspace.
- The current layout is nested one level deeper than expected:
  - `backend/data/raw/celeba/img_align_celeba/img_align_celeba/*.jpg`
- A direct scan of `backend/data/raw/celeba/img_align_celeba` can therefore look empty if the code assumes images are stored at the first level only.
- Sample image size observed: `178 x 218`

## CelebAMask-HQ

### Present files

- `CelebA-HQ-img/`
- `CelebAMask-HQ-mask-anno/`
- `CelebA-HQ-to-CelebA-mapping.txt`
- `CelebAMask-HQ-attribute-anno.txt`
- `CelebAMask-HQ-pose-anno.txt`
- `README.txt`

### Observed structure

- HQ images are named like `0.jpg`, `1.jpg`, `2.jpg`
- sample image size: `1024 x 1024`
- masks are stored in numbered folders such as `0/`, `1/`, ...
- mask filenames are per-part, for example:
  - `00000_hair.png`
  - `00000_skin.png`
  - `00000_l_eye.png`
  - `00000_r_eye.png`
  - `00000_nose.png`
  - `00000_mouth.png`
- sample hair mask size: `512 x 512`
- sampled mask pixels are binary with values equivalent to black/white

### Observed annotation schema

- `CelebA-HQ-to-CelebA-mapping.txt`
  - maps HQ ids back to original CelebA image ids/files

- `CelebAMask-HQ-pose-anno.txt`
  - first line indicates `30000`
  - rows contain `Yaw`, `Pitch`, `Raw` values

- `CelebAMask-HQ-attribute-anno.txt`
  - first line indicates `30000`
  - rows contain CelebA-style binary attributes for each HQ image

### Project relevance

- This is now the main `Stage 1` source for segmented hairstyle assets.
- Strong early dataset for:
  - hair mask understanding
  - face/hair boundary handling
  - segmentation and occlusion experiments
  - generating transparent hairstyle PNG cutouts for static try-on

- Especially useful for:
  - validating mask reading code
  - testing cutout generation logic
  - building a manually reviewed Stage 1 asset bank
  - later hair segmentation training/fine-tuning

### Practical note

- The dataset stores per-part masks instead of one combined parsing map, so our scripts should be prepared to gather the `*_hair.png` mask specifically.
- The hair masks are `512 x 512` while the paired HQ images are `1024 x 1024`, so preprocessing must resize masks to the image resolution with nearest-neighbor sampling before asset export.

## CelebAHairMask-HQ

### Current status

- The folder exists but currently appears empty in this workspace.

### Project relevance

- Optional for later stages:
  - better hair-mask quality experiments
  - hair matting / segmentation refinement

### Practical note

- Not required to start `V1-V2`.

## K-Hairstyle

### Present structure

- `mqset/images/`
- `mqset/labels/`
- zipped source archives are also still present:
  - `images_mqset001.zip`
  - `labels_mqset.zip`

### Observed counts

- discovered `.jpg` images: `48252`
- discovered `.json` label files: `48252`

### Observed image properties

- sampled image size: `512 x 512`
- image paths are nested deeply under grouped folders

### Observed label schema

Each image has a matching JSON annotation file. Sample fields include:

- identity and source:
  - `id`
  - `source`
  - `filename`
  - `path`
- hairstyle attributes:
  - `basestyle`
  - `basestyle-type`
  - `length`
  - `curl`
  - `bang`
  - `loss`
  - `side`
  - `color`
  - `partition`
  - `sex`
  - `before-after`
- quality / pose style fields:
  - `front`
  - `horizontal`
  - `vertical`
- additional hair condition fields:
  - `hair-width`
  - `natural-curl`
  - `damage`
- geometry / segmentation fields:
  - `polygon1`
  - `polygon2`
  - `width`
  - `height`
- extra capture metadata:
  - phone/device and EXIF-like fields

### Important annotation details

- The JSON files are valid UTF-8 and can be read correctly in Python.
- `polygon1` and `polygon2` are stored as stringified lists of point dictionaries, not as already-parsed arrays.
- These polygon strings need parsing before mask rasterization.
- A sampled record had:
  - `front = true`
  - `horizontal = 0`
  - `vertical = 상`
  - both `polygon1` and `polygon2` present

### Project relevance

- This is the most important dataset for `Stage 2` and later DL work.
- It supports:
  - hairstyle taxonomy definition
  - hairstyle metadata normalization
  - training the custom hairstyle attribute classifier
  - later scaling of the internal asset bank through predicted attributes

### Practical notes

- Folder names contain Korean characters, so scripts should use robust path handling.
- The label field names and values should be normalized into our internal schema before indexing.
- The `length` field should not be trusted semantically without inspection across more records, because a sampled record contained `length = 남자`, which suggests some fields may encode style/gender conventions rather than our desired normalized length taxonomy.

## What matters first for Stage 1

### Stage 1 asset source

- Use `CelebAMask-HQ` as the main segmented-asset source.
- Build a preprocessing workflow that:
  - pairs HQ images with `*_hair.png` masks
  - resizes masks to the HQ image size
  - crops padded transparent hair PNG assets
  - saves reusable metadata for manual review

### Stage 1 labeling strategy

- Use the `K-Hairstyle` taxonomy as the labeling reference.
- Start with a small manually reviewed subset and record:
  - `length`
  - `curl`
  - `bang`
  - `volume`
  - `side_hair`
  - `color`
  - `style_family`

### Stage 2 preparation

- Keep `K-Hairstyle` as the main training source for the custom hairstyle attribute classifier.

### Not blocking now

- Full CelebA image payload
- CelebAHairMask-HQ
- any custom-trained model
