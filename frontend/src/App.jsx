import { useEffect, useMemo, useRef, useState } from "react";
import {
  SYSTEM_BASE_PATHS,
  analyzeFace,
  fetchAssetBankSummary,
  generateTryOn,
  recommendHairstyles,
  resolveMediaUrl
} from "./api/client.js";

const SYSTEMS = [
  {
    id: "staticManual",
    label: "Static Manual Try-On",
    badge: "Active",
    eyebrow: "Subsystem 1",
    summary:
      "Upload, segment, recommend, and prepare hair assets for manual drag-and-place positioning on the head."
  },
  {
    id: "staticAuto",
    label: "Static Auto Try-On",
    badge: "Active",
    eyebrow: "Subsystem 2",
    summary:
      "Upload, segment, recommend, and automatically generate static 2D try-on results with the current renderer."
  },
  {
    id: "generative",
    label: "Generative Try-On",
    badge: "Experimental",
    eyebrow: "Subsystem 3",
    summary:
      "Uses the separate generative package-building workflow and final inpainting experiments without changing the underlying logic."
  },
  {
    id: "live2d",
    label: "Live 2D Try-On",
    badge: "Empty",
    eyebrow: "Subsystem 4",
    summary: "Reserved backend slot for future live 2D try-on work."
  },
  {
    id: "live3d",
    label: "Live 3D Try-On",
    badge: "Empty",
    eyebrow: "Subsystem 5",
    summary: "Reserved backend slot for future live 3D try-on work."
  }
];

function PlaceholderPanel({ title, body, note }) {
  return (
    <section className="panel subsystem-placeholder">
      <h2>{title}</h2>
      <p className="muted">{body}</p>
      {note ? <p className="status-pill subtle-pill">{note}</p> : null}
    </section>
  );
}

export default function App() {
  const resultRef = useRef(null);
  const [activeSystem, setActiveSystem] = useState("staticAuto");
  const [imageFile, setImageFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [analysis, setAnalysis] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [selectedAssetId, setSelectedAssetId] = useState("");
  const [targetGender, setTargetGender] = useState("male");
  const [tryOn, setTryOn] = useState(null);
  const [step, setStep] = useState("idle");
  const [error, setError] = useState("");
  const [assetBankSummary, setAssetBankSummary] = useState(null);

  const selectedSystem = useMemo(
    () => SYSTEMS.find((item) => item.id === activeSystem) ?? SYSTEMS[1],
    [activeSystem]
  );
  const subsystemBasePath =
    SYSTEM_BASE_PATHS[activeSystem] ?? SYSTEM_BASE_PATHS.staticAuto;
  const supportsSharedStaticFlow =
    activeSystem === "staticManual" || activeSystem === "staticAuto";
  const supportsAutoTryOn = activeSystem === "staticAuto";

  useEffect(() => {
    if ((tryOn?.output_image_url || step === "tryon") && resultRef.current) {
      resultRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [step, tryOn]);

  useEffect(() => {
    setAnalysis(null);
    setRecommendations([]);
    setSelectedAssetId("");
    setTryOn(null);
    setError("");
    setStep("idle");
  }, [activeSystem]);

  useEffect(() => {
    let ignore = false;

    async function loadSummary() {
      if (!supportsSharedStaticFlow) {
        setAssetBankSummary(null);
        return;
      }

      try {
        const summary = await fetchAssetBankSummary(subsystemBasePath);
        if (!ignore) {
          setAssetBankSummary(summary);
        }
      } catch {
        if (!ignore) {
          setAssetBankSummary(null);
        }
      }
    }

    loadSummary();
    return () => {
      ignore = true;
    };
  }, [subsystemBasePath, supportsSharedStaticFlow]);

  function handleFileChange(file) {
    setImageFile(file);
    setAnalysis(null);
    setRecommendations([]);
    setSelectedAssetId("");
    setTryOn(null);
    setError("");
    setStep("idle");
    setPreviewUrl(file ? URL.createObjectURL(file) : "");
  }

  async function runAnalysis() {
    if (!imageFile || !supportsSharedStaticFlow) return;
    try {
      setStep("analyzing");
      setError("");
      const nextAnalysis = await analyzeFace(imageFile, subsystemBasePath);
      setAnalysis(nextAnalysis);
      if (!nextAnalysis.face_detected || !nextAnalysis.face_attributes) {
        setStep("error");
        setError(nextAnalysis.message);
        return;
      }

      setStep("recommending");
      const nextRecommendations = await recommendHairstyles(
        nextAnalysis.face_attributes,
        targetGender,
        6,
        subsystemBasePath
      );
      setRecommendations(nextRecommendations);
      setSelectedAssetId(nextRecommendations[0]?.asset_id ?? "");
      setStep("done");
    } catch (err) {
      setStep("error");
      setError(err instanceof Error ? err.message : "The selected subsystem failed.");
    }
  }

  async function runTryOn(assetId = selectedAssetId) {
    if (!imageFile || !assetId || !supportsAutoTryOn) return;
    try {
      setStep("tryon");
      setError("");
      setSelectedAssetId(assetId);
      setTryOn(null);
      const nextTryOn = await generateTryOn(imageFile, assetId, subsystemBasePath);
      setTryOn(nextTryOn);
      setStep("done");
    } catch (err) {
      setStep("error");
      setError(err instanceof Error ? err.message : "Try-on failed.");
    }
  }

  const isBusy = step === "analyzing" || step === "recommending" || step === "tryon";

  return (
    <main className="shell">
      <section className="hero">
        <p className="eyebrow">Multi-system hairstyle platform</p>
        <h1>One home page, five try-on subsystems</h1>
        <p>
          The backend is now separated into static manual try-on, static auto try-on,
          generative try-on, live 2D try-on, and live 3D try-on. The existing logic stays
          in place and each subsystem now has its own backend mount.
        </p>
      </section>

      <section className="system-grid">
        {SYSTEMS.map((system) => (
          <button
            key={system.id}
            type="button"
            className={`system-card ${activeSystem === system.id ? "active" : ""}`}
            onClick={() => setActiveSystem(system.id)}
          >
            <span className="system-eyebrow">{system.eyebrow}</span>
            <strong>{system.label}</strong>
            <span className="system-badge">{system.badge}</span>
            <p>{system.summary}</p>
          </button>
        ))}
      </section>

      <section className="panel subsystem-header">
        <p className="eyebrow">{selectedSystem.eyebrow}</p>
        <h2>{selectedSystem.label}</h2>
        <p className="muted">{selectedSystem.summary}</p>
        <p className="status-pill subtle-pill">Mounted backend path: {subsystemBasePath}</p>
        {assetBankSummary && (
          <p className="status-pill">
            Active asset bank: {assetBankSummary.asset_count} reviewed hairstyle assets
          </p>
        )}
      </section>

      {supportsSharedStaticFlow ? (
        <>
          <section className="workspace">
            <div className="panel upload-panel">
              <label className="file-drop">
                <span>Choose a face image</span>
                <input
                  type="file"
                  accept="image/png,image/jpeg,image/webp"
                  onChange={(event) => handleFileChange(event.target.files?.[0] ?? null)}
                />
              </label>
              <label className="select-field">
                <span>Recommendation gender</span>
                <select value={targetGender} onChange={(event) => setTargetGender(event.target.value)}>
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                  <option value="any">Any</option>
                </select>
              </label>
              {previewUrl && <img className="preview" src={previewUrl} alt="Uploaded preview" />}
              <button disabled={!imageFile || isBusy} onClick={runAnalysis}>
                {isBusy ? "Working..." : "Analyze and recommend"}
              </button>
              {error && <p className="error">{error}</p>}
            </div>

            <div className="panel">
              <h2>Face analysis</h2>
              {analysis?.face_attributes ? (
                <div className="metric-grid">
                  {Object.entries(analysis.face_attributes).map(([key, value]) => (
                    <div key={key} className="metric">
                      <span>{key.replace("_", " ")}</span>
                      <strong>{value}</strong>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="muted">Analysis appears after upload.</p>
              )}
            </div>
          </section>

          <section className="recommendations">
            <div className="section-heading">
              <h2>Recommended assets</h2>
              <p>
                {supportsAutoTryOn
                  ? "Ready for automatic static 2D generation."
                  : "Ready for manual positioning in the dedicated manual try-on workflow."}
              </p>
            </div>
            <div className="cards">
              {recommendations.map((item) => (
                <article
                  key={item.asset_id}
                  className={`card ${selectedAssetId === item.asset_id ? "selected" : ""}`}
                  onClick={() => setSelectedAssetId(item.asset_id)}
                >
                  {item.image_url && (
                    <img src={resolveMediaUrl(item.image_url)} alt={`Hair asset ${item.asset_id}`} />
                  )}
                  <div>
                    <strong>{item.asset_id}</strong>
                    <span>{Math.round(item.score * 100)}% match</span>
                  </div>
                  <p>{item.reason}</p>
                  <small>
                    {item.gender_suitability ?? "neutral"} recommendation / {item.normalized_attributes.length} /{" "}
                    {item.normalized_attributes.curl} /{" "}
                    {item.normalized_attributes.style_family.replace("_", " ")}
                  </small>
                  {supportsAutoTryOn ? (
                    <button onClick={() => runTryOn(item.asset_id)} disabled={isBusy}>
                      {step === "tryon" && selectedAssetId === item.asset_id
                        ? "Generating..."
                        : "Try this"}
                    </button>
                  ) : (
                    <button
                      type="button"
                      className="secondary-button"
                      onClick={() => setSelectedAssetId(item.asset_id)}
                    >
                      Select asset
                    </button>
                  )}
                </article>
              ))}
            </div>
          </section>

          <section ref={resultRef} className="result panel">
            <h2>{supportsAutoTryOn ? "Try-on result" : "Manual workflow slot"}</h2>
            {supportsAutoTryOn ? (
              step === "tryon" ? (
                <p className="muted">Generating try-on preview...</p>
              ) : tryOn?.output_image_url ? (
                <>
                  <p className="muted">{tryOn.message}</p>
                  <img src={resolveMediaUrl(tryOn.output_image_url)} alt="Generated try-on result" />
                  {tryOn.segmentation_mask_url && (
                    <div className="result-meta">
                      <p className="muted">Segmentation-guided placement is active for this result.</p>
                      <img
                        className="preview"
                        src={resolveMediaUrl(tryOn.segmentation_mask_url)}
                        alt="Predicted hair mask"
                      />
                    </div>
                  )}
                </>
              ) : error ? (
                <p className="error">{error}</p>
              ) : (
                <p className="muted">Select a recommendation and generate the overlay.</p>
              )
            ) : recommendations.length ? (
              <>
                <p className="muted">
                  This subsystem keeps the existing recommendation and segmentation flow, but the
                  final drag-and-place interface remains a separate manual workspace.
                </p>
                <p className="status-pill subtle-pill">
                  Selected asset: {selectedAssetId || "Choose one of the recommended assets"}
                </p>
              </>
            ) : (
              <p className="muted">
                Upload an image and run the shared analysis pipeline to prepare manual asset selection.
              </p>
            )}
          </section>
        </>
      ) : activeSystem === "generative" ? (
        <PlaceholderPanel
          title="Generative Try-On Backend"
          body="The generative subsystem is now separated in backend routing and still uses the dedicated experimental package-building and notebook flow you created. The frontend home page exposes it as a distinct option without changing that logic."
          note="Use the separate generative notebook and mounted backend under /api/generative for current experiments."
        />
      ) : activeSystem === "live2d" ? (
        <PlaceholderPanel
          title="Live 2D Try-On"
          body="This subsystem has an isolated backend slot and is intentionally empty for now."
          note="Reserved for future live 2D implementation."
        />
      ) : (
        <PlaceholderPanel
          title="Live 3D Try-On"
          body="This subsystem has an isolated backend slot and is intentionally empty for now."
          note="Reserved for future live 3D implementation."
        />
      )}
    </main>
  );
}
