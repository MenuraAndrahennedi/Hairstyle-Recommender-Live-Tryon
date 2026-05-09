export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export const SYSTEM_BASE_PATHS = {
  staticAuto: "/api/static-auto",
  generative: "/api/generative",
  live2d: "/api/live-2d",
  live3d: "/api/live-3d",
};

function subsystemUrl(basePath, route) {
  return `${API_BASE_URL}${basePath}${route}`;
}

export function live2dWsUrl() {
  const wsBaseUrl = API_BASE_URL.replace(/^http/, "ws");
  return `${wsBaseUrl}${SYSTEM_BASE_PATHS.live2d}/ws`;
}

export function resolveMediaUrl(path, basePath = "") {
  if (!path) return "";
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }
  if (path.startsWith("/media/") && basePath) {
    return `${API_BASE_URL}${basePath}${path}`;
  }
  return `${API_BASE_URL}${path}`;
}

async function expectJson(response) {
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

export async function analyzeFace(
  image,
  basePath = SYSTEM_BASE_PATHS.staticAuto,
) {
  const formData = new FormData();
  formData.append("image", image);
  const response = await fetch(subsystemUrl(basePath, "/api/analyze-face"), {
    method: "POST",
    body: formData,
  });
  return expectJson(response);
}

export async function recommendHairstyles(
  faceAttributes,
  targetGender,
  topK = 5,
  basePath = SYSTEM_BASE_PATHS.staticAuto,
) {
  const response = await fetch(subsystemUrl(basePath, "/api/recommend"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      face_attributes: faceAttributes,
      preferences: {
        allow_bangs: true,
        target_gender: targetGender,
      },
      top_k: topK,
    }),
  });
  const payload = await expectJson(response);
  return payload.recommendations ?? [];
}

export async function generateTryOn(
  image,
  assetId,
  basePath = SYSTEM_BASE_PATHS.staticAuto,
) {
  const formData = new FormData();
  formData.append("image", image);
  formData.append("asset_id", assetId);
  const response = await fetch(subsystemUrl(basePath, "/api/tryon"), {
    method: "POST",
    body: formData,
  });
  return expectJson(response);
}

export async function generateGenerativePackage(image, assetId) {
  const formData = new FormData();
  formData.append("image", image);
  formData.append("asset_id", assetId);
  const response = await fetch(
    subsystemUrl(SYSTEM_BASE_PATHS.generative, "/api/generative-tryon/package"),
    {
      method: "POST",
      body: formData,
    },
  );
  return expectJson(response);
}

export async function fetchAssetBankSummary(
  basePath = SYSTEM_BASE_PATHS.staticAuto,
) {
  const response = await fetch(subsystemUrl(basePath, "/api/assets/summary"));
  return expectJson(response);
}

export async function fetchAssets(basePath = SYSTEM_BASE_PATHS.staticAuto) {
  const response = await fetch(subsystemUrl(basePath, "/api/assets"));
  const payload = await expectJson(response);
  return payload.assets ?? [];
}

export async function fetchLiveTopTierAssets() {
  const response = await fetch(
    subsystemUrl(SYSTEM_BASE_PATHS.staticAuto, "/api/assets/live-top-tier"),
  );
  const payload = await expectJson(response);
  return payload.assets ?? [];
}

export async function fetchLive2dInfo() {
  const response = await fetch(subsystemUrl(SYSTEM_BASE_PATHS.live2d, "/info"));
  return expectJson(response);
}

export async function processLive2dFrame(image, selectedAssetId = "") {
  const formData = new FormData();
  formData.append("image", image);
  if (selectedAssetId) {
    formData.append("selected_asset_id", selectedAssetId);
  }
  const response = await fetch(
    subsystemUrl(SYSTEM_BASE_PATHS.live2d, "/frame"),
    {
      method: "POST",
      body: formData,
    },
  );
  return expectJson(response);
}
