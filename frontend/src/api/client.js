export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
export const SYSTEM_BASE_PATHS = {
  staticAuto: "/api/static-auto",
  generative: "/api/generative",
  live2d: "/api/live-2d",
  live3d: "/api/live-3d"
};

function mediaUrl(path) {
  if (!path) return "";
  if (path.startsWith("http")) return path;
  return `${API_BASE_URL}${path}`;
}

export function resolveMediaUrl(path) {
  return mediaUrl(path);
}

function subsystemUrl(basePath, route) {
  return `${API_BASE_URL}${basePath}${route}`;
}

export async function analyzeFace(image, basePath = SYSTEM_BASE_PATHS.staticAuto) {
  const formData = new FormData();
  formData.append("image", image);
  const response = await fetch(subsystemUrl(basePath, "/api/analyze-face"), {
    method: "POST",
    body: formData
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function recommendHairstyles(
  faceAttributes,
  targetGender,
  topK = 6,
  basePath = SYSTEM_BASE_PATHS.staticAuto
) {
  const response = await fetch(subsystemUrl(basePath, "/api/recommend"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      face_attributes: faceAttributes,
      preferences: {
        allow_bangs: true,
        target_gender: targetGender
      },
      top_k: topK
    })
  });
  if (!response.ok) throw new Error(await response.text());
  const payload = await response.json();
  return payload.recommendations;
}

export async function generateTryOn(image, assetId, basePath = SYSTEM_BASE_PATHS.staticAuto) {
  const formData = new FormData();
  formData.append("image", image);
  formData.append("asset_id", assetId);
  const response = await fetch(subsystemUrl(basePath, "/api/tryon"), {
    method: "POST",
    body: formData
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function fetchAssetBankSummary(basePath = SYSTEM_BASE_PATHS.staticAuto) {
  const response = await fetch(subsystemUrl(basePath, "/api/assets/summary"));
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function fetchLiveTopTierAssets(basePath = SYSTEM_BASE_PATHS.staticAuto) {
  const response = await fetch(subsystemUrl(basePath, "/api/assets/live-top-tier"));
  if (!response.ok) throw new Error(await response.text());
  const payload = await response.json();
  return payload.assets;
}
