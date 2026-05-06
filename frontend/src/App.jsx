import { useEffect, useRef, useState } from "react";
import {
  analyzeFace,
  fetchAssetBankSummary,
  generateTryOn,
  recommendHairstyles,
  resolveMediaUrl
} from "./api/client.js";

export default function App() {
  const resultRef = useRef(null);
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

  useEffect(() => {
    if ((tryOn?.output_image_url || step === "tryon") && resultRef.current) {
      resultRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [step, tryOn]);

  useEffect(() => {
    let ignore = false;

    async function loadSummary() {
      try {
        const summary = await fetchAssetBankSummary();
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
  }, []);

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
    if (!imageFile) return;
    try {
      setStep("analyzing");
      setError("");
      const nextAnalysis = await analyzeFace(imageFile);
      setAnalysis(nextAnalysis);
      if (!nextAnalysis.face_detected || !nextAnalysis.face_attributes) {
        setStep("error");
        setError(nextAnalysis.message);
        return;
      }

      setStep("recommending");
      const nextRecommendations = await recommendHairstyles(nextAnalysis.face_attributes, targetGender);
      setRecommendations(nextRecommendations);
      setSelectedAssetId(nextRecommendations[0]?.asset_id ?? "");
      setStep("done");
    } catch (err) {
      setStep("error");
      setError(err instanceof Error ? err.message : "The static 2D pipeline failed.");
    }
  }

  async function runTryOn(assetId = selectedAssetId) {
    if (!imageFile || !assetId) return;
    try {
      setStep("tryon");
      setError("");
      setSelectedAssetId(assetId);
      setTryOn(null);
      const nextTryOn = await generateTryOn(imageFile, assetId);
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
        <p className="eyebrow">Static 2D try-on system</p>
        <h1>Hairstyle recommendation and virtual try-on</h1>
        <p>
          Upload a face image, analyze face geometry, rank reviewed render-safe hairstyle assets,
          predict a cleaned hair mask, and render a stronger static 2D try-on preview.
        </p>
        {assetBankSummary && (
          <p className="status-pill">
            Active asset bank: {assetBankSummary.asset_count} reviewed hairstyle assets
          </p>
        )}
      </section>

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
          <p>Backed by the reviewed render-safe live asset bank.</p>
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
                {item.normalized_attributes.curl} / {item.normalized_attributes.style_family.replace("_", " ")}
              </small>
              <button onClick={() => runTryOn(item.asset_id)} disabled={isBusy}>
                {step === "tryon" && selectedAssetId === item.asset_id ? "Generating..." : "Try this"}
              </button>
            </article>
          ))}
        </div>
      </section>

      <section ref={resultRef} className="result panel">
        <h2>Try-on result</h2>
        {step === "tryon" ? (
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
        )}
      </section>
    </main>
  );
}
