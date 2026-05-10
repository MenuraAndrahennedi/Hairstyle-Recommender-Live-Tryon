from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image

from generative_app.config import MODEL_CACHE_DIR


DEFAULT_MODEL_ID = "runwayml/stable-diffusion-inpainting"
DEFAULT_IP_ADAPTER_REPO = "h94/IP-Adapter"
DEFAULT_IP_ADAPTER_SUBFOLDER = "models"
DEFAULT_IP_ADAPTER_WEIGHT = "ip-adapter_sd15.bin"
DEFAULT_IP_ADAPTER_SCALE = 0.85
DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, deformed hairline, distorted face, extra face, duplicate face, "
    "bad anatomy, hat, wig cap, artifacts, cropped head, broken forehead, unrealistic hair"
)


def _load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def _build_runtime_prompt(
    manifest: dict[str, Any],
    extra_prompt: str | None = None,
) -> str:
    reference = manifest.get("reference", {})
    attrs = reference.get("normalized_attributes", {})
    base_prompt = (
        "Photorealistic portrait of the same person. "
        "Preserve face identity, skin tone, pose, clothing, and background. "
        "Edit only the hair and hairline. "
        f"{attrs.get('color', 'black')} {attrs.get('length', 'medium')} {attrs.get('curl', 'straight')} hair, "
        f"{attrs.get('volume', 'medium')} volume, {attrs.get('bang', 'none')} bangs, "
        f"{attrs.get('side_hair', 'covered')} sides, {attrs.get('style_family', 'other')} style. "
        "Natural forehead transition, realistic strands, studio lighting."
    )
    if extra_prompt:
        return f"{base_prompt} {extra_prompt.strip()}"
    return base_prompt


def _resolve_device_and_dtype() -> tuple[str, str]:
    import torch

    if torch.cuda.is_available():
        return "cuda", "float16"
    return "cpu", "float32"


def _round_to_multiple(value: int, multiple: int = 8) -> int:
    return max((value // multiple) * multiple, multiple)


def _prepare_inpaint_inputs(
    image_path: str | Path,
    mask_path: str | Path,
    target_max_side: int,
) -> tuple[Image.Image, Image.Image, tuple[int, int]]:
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    original_size = image.size

    width, height = original_size
    scale = min(1.0, float(target_max_side) / max(width, height))
    resized_width = _round_to_multiple(int(width * scale))
    resized_height = _round_to_multiple(int(height * scale))

    resized_image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    resized_mask = mask.resize((resized_width, resized_height), Image.Resampling.NEAREST)
    return resized_image, resized_mask, original_size

def _load_reference_image_for_ip_adapter(reference_path: str | Path) -> Image.Image:
    reference = Image.open(reference_path).convert("RGBA")

    # IP-Adapter expects a normal RGB image.
    # Your hairstyle assets may contain transparency, so place them on white background.
    background = Image.new("RGBA", reference.size, (255, 255, 255, 255))
    background.alpha_composite(reference)

    return background.convert("RGB")

def run_generative_inpaint(
    manifest_path: str | Path,
    model_id: str = DEFAULT_MODEL_ID,
    output_name: str = "final_generated.png",
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    strength: float = 0.92,
    max_side: int = 768,
    extra_prompt: str | None = None,
    negative_prompt: str | None = DEFAULT_NEGATIVE_PROMPT,
    use_ip_adapter: bool = True,
    ip_adapter_scale: float = DEFAULT_IP_ADAPTER_SCALE,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = _load_manifest(manifest_path)

    subject = manifest["subject"]
    reference = manifest.get("reference", {})
    generation = manifest.get("generation", {})

    subject_image_path = Path(subject["input_image_path"])
    mask_path = Path(subject["inpaint_mask_path"])

    reference_image_path = reference.get("image_path")
    reference_image = None

    if use_ip_adapter and reference_image_path:
        reference_image = _load_reference_image_for_ip_adapter(reference_image_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Inpaint mask not found: {mask_path}")

    prompt = _build_runtime_prompt(manifest, extra_prompt=extra_prompt)
    image, mask, original_size = _prepare_inpaint_inputs(subject_image_path, mask_path, target_max_side=max_side)

    import torch
    from diffusers import AutoPipelineForInpainting

    device, dtype_name = _resolve_device_and_dtype()
    torch_dtype = torch.float16 if dtype_name == "float16" else torch.float32

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODEL_CACHE_DIR))

    try:
        pipe = AutoPipelineForInpainting.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            cache_dir=str(MODEL_CACHE_DIR),
        )
        used_safetensors = True
    except OSError:
        pipe = AutoPipelineForInpainting.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            use_safetensors=False,
            cache_dir=str(MODEL_CACHE_DIR),
        )
        used_safetensors = False

    if device == "cuda":
        pipe = pipe.to(device)
    else:
        pipe = pipe.to(device)
        pipe.enable_attention_slicing()

    if use_ip_adapter and reference_image is not None:
        pipe.load_ip_adapter(
            DEFAULT_IP_ADAPTER_REPO,
            subfolder=DEFAULT_IP_ADAPTER_SUBFOLDER,
            weight_name=DEFAULT_IP_ADAPTER_WEIGHT,
            cache_dir=str(MODEL_CACHE_DIR),
        )
        pipe.set_ip_adapter_scale(ip_adapter_scale)

    pipe_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": image,
        "mask_image": mask,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "strength": strength,
    }

    if use_ip_adapter and reference_image is not None:
        pipe_kwargs["ip_adapter_image"] = reference_image

    result = pipe(**pipe_kwargs)

    generated = result.images[0].resize(original_size, Image.Resampling.LANCZOS)
    output_path = manifest_path.parent / output_name
    generated.save(output_path)

    metadata = {
        "manifest_path": str(manifest_path),
        "output_image_path": str(output_path),
        "model_id": model_id,
        "device": device,
        "torch_dtype": dtype_name,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "strength": strength,
        "max_side": max_side,
        "cache_dir": str(MODEL_CACHE_DIR),
        "used_safetensors": used_safetensors,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "use_ip_adapter": use_ip_adapter,
        "ip_adapter_repo": DEFAULT_IP_ADAPTER_REPO if use_ip_adapter else None,
        "ip_adapter_weight": DEFAULT_IP_ADAPTER_WEIGHT if use_ip_adapter else None,
        "ip_adapter_scale": ip_adapter_scale if use_ip_adapter else None,
        "reference_image_path": str(reference_image_path) if reference_image_path else None,
    }
    metadata_path = manifest_path.parent / "final_generation_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata["metadata_path"] = str(metadata_path)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the separate generative try-on inpainting step.")
    parser.add_argument("--manifest", required=True, help="Path to package_manifest.json")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id or local model path")
    parser.add_argument("--output-name", default="final_generated.png", help="Output filename inside the package dir")
    parser.add_argument("--steps", type=int, default=30, help="Number of denoising steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--strength", type=float, default=0.99, help="Inpaint strength")
    parser.add_argument("--max-side", type=int, default=768, help="Resize longer image side before generation")
    parser.add_argument("--extra-prompt", default=None, help="Additional prompt text to append")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt")
    args = parser.parse_args()

    result = run_generative_inpaint(
        manifest_path=args.manifest,
        model_id=args.model_id,
        output_name=args.output_name,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        max_side=args.max_side,
        extra_prompt=args.extra_prompt,
        negative_prompt=args.negative_prompt,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
