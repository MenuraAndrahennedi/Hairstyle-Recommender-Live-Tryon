import { useEffect, useMemo, useRef, useState } from "react";
import { fetchLiveTopTierAssets, resolveMediaUrl } from "../api/client.js";

const DEFAULT_TUNING = {
  widthScale: 1.42,
  heightScale: 1,
  xOffset: 0,
  yOffset: -0.68,
  anchorLift: 0.35,
  opacity: 0.96,
  rotationOffset: 0
};

let webARRocksLoaderPromise = null;

function loadWebARRocksFace() {
  if (window.WEBARROCKSFACE) {
    return Promise.resolve(window.WEBARROCKSFACE);
  }
  if (webARRocksLoaderPromise) {
    return webARRocksLoaderPromise;
  }

  webARRocksLoaderPromise = new Promise((resolve, reject) => {
    const existing = document.querySelector('script[data-webarrocks-face="true"]');
    if (existing) {
      existing.addEventListener("load", () => resolve(window.WEBARROCKSFACE), { once: true });
      existing.addEventListener("error", () => reject(new Error("Could not load WebAR.rocks script.")), {
        once: true
      });
      return;
    }

    const script = document.createElement("script");
    script.src = "/webarrocks/dist/WebARRocksFace.js";
    script.async = true;
    script.dataset.webarrocksFace = "true";
    script.onload = () => {
      if (window.WEBARROCKSFACE) {
        resolve(window.WEBARROCKSFACE);
      } else {
        reject(new Error("WebAR.rocks loaded but did not expose a tracker global."));
      }
    };
    script.onerror = () => reject(new Error("Could not load WebAR.rocks script."));
    document.head.appendChild(script);
  });

  return webARRocksLoaderPromise;
}

function getDetection(detectState) {
  const state = Array.isArray(detectState) ? detectState[0] : detectState;
  if (!state) return null;
  const confidence = Number(state.detected ?? state.detectedRaw ?? 0);
  const isDetected = Boolean(state.isDetected) || confidence > 0.72;
  return isDetected ? { ...state, confidence } : null;
}

function resizeCanvases(videoCanvas, overlayCanvas) {
  if (!videoCanvas || !overlayCanvas) return;
  const container = videoCanvas.parentElement;
  const rect = container?.getBoundingClientRect();
  const cssWidth = Math.max(320, Math.round(rect?.width || 960));
  const cssHeight = Math.round(cssWidth * 9 / 16);
  const dpr = window.devicePixelRatio || 1;
  const width = Math.round(cssWidth * dpr);
  const height = Math.round(cssHeight * dpr);

  for (const canvas of [videoCanvas, overlayCanvas]) {
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    canvas.style.width = `${cssWidth}px`;
    canvas.style.height = `${cssHeight}px`;
  }
}

function drawDebug(ctx, x, y, faceSize) {
  ctx.save();
  ctx.strokeStyle = "rgba(78, 220, 255, 0.88)";
  ctx.lineWidth = Math.max(2, faceSize * 0.012);
  ctx.beginPath();
  ctx.arc(x, y, faceSize * 0.05, 0, Math.PI * 2);
  ctx.stroke();
  ctx.strokeRect(x - faceSize * 0.5, y - faceSize * 0.5, faceSize, faceSize);
  ctx.restore();
}

export default function WebcamTryOn() {
  const videoCanvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const webarRef = useRef(null);
  const hairImageRef = useRef(null);
  const trackingRef = useRef({ frames: 0, detected: false, confidence: 0 });
  const tuningRef = useRef(DEFAULT_TUNING);

  const [assets, setAssets] = useState([]);
  const [selectedAssetId, setSelectedAssetId] = useState("");
  const [status, setStatus] = useState("idle");
  const [message, setMessage] = useState("Start the camera to track your head and render the selected PNG hair asset.");
  const [debug, setDebug] = useState(false);
  const [tuning, setTuning] = useState(DEFAULT_TUNING);
  const [tracking, setTracking] = useState(trackingRef.current);

  const selectedAsset = useMemo(
    () => assets.find((asset) => asset.asset_id === selectedAssetId) ?? assets[0],
    [assets, selectedAssetId]
  );

  useEffect(() => {
    let ignore = false;

    async function loadAssets() {
      try {
        const nextAssets = await fetchLiveTopTierAssets();
        if (!ignore) {
          setAssets(nextAssets);
          setSelectedAssetId(nextAssets[0]?.asset_id ?? "");
        }
      } catch (err) {
        if (!ignore) {
          setMessage(err instanceof Error ? err.message : "Could not load live top-tier assets.");
        }
      }
    }

    loadAssets();
    return () => {
      ignore = true;
    };
  }, []);

  useEffect(() => {
    tuningRef.current = tuning;
  }, [tuning]);

  useEffect(() => {
    if (!selectedAsset?.image_url) {
      hairImageRef.current = null;
      return;
    }

    const image = new Image();
    image.crossOrigin = "anonymous";
    image.onload = () => {
      hairImageRef.current = image;
      setMessage(`Loaded ${selectedAsset.asset_id}.`);
    };
    image.onerror = () => {
      hairImageRef.current = null;
      setMessage(`Could not load ${selectedAsset.asset_id}.`);
    };
    image.src = resolveMediaUrl(selectedAsset.image_url);
  }, [selectedAsset]);

  useEffect(() => {
    const handleResize = () => {
      resizeCanvases(videoCanvasRef.current, overlayCanvasRef.current);
      webarRef.current?.resize?.();
    };

    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  useEffect(() => {
    return () => {
      webarRef.current?.destroy?.();
      webarRef.current = null;
    };
  }, []);

  function updateTuning(key, value) {
    setTuning((current) => ({ ...current, [key]: Number(value) }));
  }

  function renderHair(detectState) {
    const overlayCanvas = overlayCanvasRef.current;
    const ctx = overlayCanvas?.getContext("2d");
    if (!overlayCanvas || !ctx) return;

    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    const detection = getDetection(detectState);
    const image = hairImageRef.current;
    if (!detection || !image) {
      trackingRef.current = {
        frames: trackingRef.current.frames + 1,
        detected: false,
        confidence: detection?.confidence ?? 0
      };
      return;
    }

    const width = overlayCanvas.width;
    const height = overlayCanvas.height;
    const faceSize = Math.max(80, Number(detection.s || 0.35) * width);
    const x = (0.5 + Number(detection.x || 0) * 0.5) * width + faceSize * tuningRef.current.xOffset;
    const y = (0.5 - Number(detection.y || 0) * 0.5) * height + faceSize * tuningRef.current.yOffset;
    const targetWidth = faceSize * tuningRef.current.widthScale;
    const targetHeight = targetWidth * (image.height / image.width) * tuningRef.current.heightScale;
    const rotation = -(Number(detection.rz || 0) + tuningRef.current.rotationOffset * Math.PI / 180);

    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(rotation);
    ctx.globalAlpha = tuningRef.current.opacity;
    ctx.drawImage(
      image,
      -targetWidth / 2,
      -targetHeight * tuningRef.current.anchorLift,
      targetWidth,
      targetHeight
    );
    ctx.restore();

    if (debug) drawDebug(ctx, x, y, faceSize);

    trackingRef.current = {
      frames: trackingRef.current.frames + 1,
      detected: true,
      confidence: detection.confidence
    };
  }

  async function startTracking() {
    if (status === "loading" || status === "ready" || status === "tracking") return;

    const videoCanvas = videoCanvasRef.current;
    const overlayCanvas = overlayCanvasRef.current;
    if (!videoCanvas || !overlayCanvas) return;

    resizeCanvases(videoCanvas, overlayCanvas);
    setStatus("loading");
    setMessage("Loading WebAR.rocks face tracker...");

    try {
      const WebARRocksFace = await loadWebARRocksFace();
      webarRef.current = WebARRocksFace;

      WebARRocksFace.init({
        canvas: videoCanvas,
        NNCPath: "/webarrocks/neuralNets/NN_FACE_3.json",
        maxFacesDetected: 1,
        followZRot: true,
        videoSettings: {
          facingMode: "user",
          idealWidth: 1280,
          idealHeight: 720
        },
        scanSettings: {
          threshold: 0.74,
          nScaleLevels: 4
        },
        stabilizationSettings: {
          translationFactorRange: [0.0015, 0.006],
          rotationFactorRange: [0.10, 0.22],
          qualityFactorRange: [0.82, 0.94],
          alphaRange: [0.05, 0.88]
        },
        callbackReady: (error) => {
          if (error) {
            setStatus("error");
            setMessage(`WebAR tracker failed: ${error}`);
            return;
          }
          setStatus("ready");
          setMessage("Camera ready. Move slowly and tune the overlay until the hairline locks to your head.");
        },
        callbackTrack: (detectState) => {
          renderHair(detectState);
          if (trackingRef.current.frames % 12 === 0) {
            setTracking({ ...trackingRef.current });
            setStatus(trackingRef.current.detected ? "tracking" : "ready");
          }
        }
      });
    } catch (err) {
      setStatus("error");
      setMessage(err instanceof Error ? err.message : "Could not start WebAR.rocks.");
    }
  }

  async function stopTracking() {
    const ctx = overlayCanvasRef.current?.getContext("2d");
    if (ctx && overlayCanvasRef.current) {
      ctx.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
    }
    await webarRef.current?.destroy?.();
    webarRef.current = null;
    setStatus("idle");
    setMessage("Camera stopped.");
  }

  return (
    <section className="live-tool">
      <div className="live-stage">
        <canvas ref={videoCanvasRef} className="tracking-canvas" aria-label="Webcam face tracking canvas" />
        <canvas ref={overlayCanvasRef} className="hair-overlay-canvas" aria-label="PNG hairstyle overlay canvas" />
      </div>

      <aside className="live-controls">
        <div className="live-status">
          <strong>{status}</strong>
          <span>
            {tracking.detected ? "face tracked" : "waiting for face"} / confidence{" "}
            {Math.round((tracking.confidence || 0) * 100)}%
          </span>
        </div>
        <p className="muted">{message}</p>

        <div className="control-row">
          <button onClick={startTracking} disabled={status === "loading" || status === "tracking" || status === "ready"}>
            Start camera
          </button>
          <button className="secondary-button" onClick={stopTracking} disabled={status === "idle" || status === "loading"}>
            Stop
          </button>
        </div>

        <label className="select-field">
          <span>Live PNG asset</span>
          <select value={selectedAsset?.asset_id ?? ""} onChange={(event) => setSelectedAssetId(event.target.value)}>
            {assets.map((asset) => (
              <option key={asset.asset_id} value={asset.asset_id}>
                {asset.asset_id} / {asset.normalized_attributes.style_family.replace("_", " ")}
              </option>
            ))}
          </select>
        </label>

        {selectedAsset?.image_url && (
          <img className="live-asset-preview" src={resolveMediaUrl(selectedAsset.image_url)} alt={selectedAsset.asset_id} />
        )}

        <div className="tuning-grid">
          <label>
            <span>Width</span>
            <input type="range" min="0.8" max="2.4" step="0.02" value={tuning.widthScale} onChange={(event) => updateTuning("widthScale", event.target.value)} />
            <output>{tuning.widthScale.toFixed(2)}</output>
          </label>
          <label>
            <span>Height</span>
            <input type="range" min="0.7" max="1.6" step="0.02" value={tuning.heightScale} onChange={(event) => updateTuning("heightScale", event.target.value)} />
            <output>{tuning.heightScale.toFixed(2)}</output>
          </label>
          <label>
            <span>Horizontal</span>
            <input type="range" min="-0.45" max="0.45" step="0.01" value={tuning.xOffset} onChange={(event) => updateTuning("xOffset", event.target.value)} />
            <output>{tuning.xOffset.toFixed(2)}</output>
          </label>
          <label>
            <span>Vertical</span>
            <input type="range" min="-1.25" max="0.1" step="0.01" value={tuning.yOffset} onChange={(event) => updateTuning("yOffset", event.target.value)} />
            <output>{tuning.yOffset.toFixed(2)}</output>
          </label>
          <label>
            <span>Hairline</span>
            <input type="range" min="0.05" max="0.85" step="0.01" value={tuning.anchorLift} onChange={(event) => updateTuning("anchorLift", event.target.value)} />
            <output>{tuning.anchorLift.toFixed(2)}</output>
          </label>
          <label>
            <span>Opacity</span>
            <input type="range" min="0.25" max="1" step="0.01" value={tuning.opacity} onChange={(event) => updateTuning("opacity", event.target.value)} />
            <output>{tuning.opacity.toFixed(2)}</output>
          </label>
        </div>

        <label className="toggle-field">
          <input type="checkbox" checked={debug} onChange={(event) => setDebug(event.target.checked)} />
          <span>Show anchor diagnostics</span>
        </label>
      </aside>
    </section>
  );
}
