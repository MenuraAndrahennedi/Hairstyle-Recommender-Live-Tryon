import { useEffect, useRef, useState } from "react";
import {
  SYSTEM_BASE_PATHS,
  analyzeFace,
  fetchAssetBankSummary,
  fetchLive2dInfo,
  generateGenerativePackage,
  generateTryOn,
  live2dWsUrl,
  recommendHairstyles,
  resolveMediaUrl,
} from "./api/client.js";

const RECOMMENDATION_COUNT = 3;

const HOME_CARDS = [
  {
    id: "static",
    title: "Static Tryon",
    subtitle:
      "Upload a photo, get recommendations, and generate 2D try-on results.",
    accent: "violet",
    disabled: false,
  },
  {
    id: "generative",
    title: "Generative Tryon",
    subtitle: "Create AI-guided hairstyle try-on packages and previews.",
    accent: "pink",
    disabled: false,
  },
  {
    id: "live2d",
    title: "Live 2D Tryon",
    subtitle:
      "Use your webcam with real-time recommendation-driven preview tools.",
    accent: "blue",
    disabled: false,
  },
  {
    id: "live3d",
    title: "Live 3D Tryon",
    subtitle: "Reserved for future work.",
    accent: "cyan",
    disabled: true,
  },
];

const ABOUT_POINTS = [
  "Static Auto Try-On generates direct 2D hairstyle previews from uploaded portraits.",
  "Generative Try-On prepares and serves the AI-driven hairstyle editing workflow.",
  "Live 2D Try-On uses browser webcam input while staying connected to the project backend.",
];

const CONTACT_POINTS = [
  "GitHub: MenuraAndrahennedi/Hairstyle-Recommender-Live-Tryon",
  "Backend docs: http://127.0.0.1:8000/docs",
  "Frontend expects the backend gateway to be running locally by default.",
];

function LogoMark() {
  return (
    <div className="brand-mark" aria-hidden="true">
      <span className="brand-mark-core" />
    </div>
  );
}

function SparkleIcon() {
  return (
    <svg viewBox="0 0 24 24" className="icon" aria-hidden="true">
      <path d="M12 2l1.8 5.2L19 9l-5.2 1.8L12 16l-1.8-5.2L5 9l5.2-1.8L12 2zM18.5 14l.9 2.6 2.6.9-2.6.9-.9 2.6-.9-2.6-2.6-.9 2.6-.9.9-2.6z" />
    </svg>
  );
}

function PhotoIcon() {
  return (
    <svg viewBox="0 0 24 24" className="icon" aria-hidden="true">
      <path d="M4 5h16a2 2 0 012 2v10a2 2 0 01-2 2H4a2 2 0 01-2-2V7a2 2 0 012-2zm0 2v10h16V7H4zm3 8l3-4 2.4 3 1.8-2.2L18 15H7zm2-6.2A1.8 1.8 0 1110.8 7 1.8 1.8 0 019 8.8z" />
    </svg>
  );
}

function CameraIcon() {
  return (
    <svg viewBox="0 0 24 24" className="icon" aria-hidden="true">
      <path d="M12 5a7 7 0 017 7v5a2 2 0 01-2 2H7a2 2 0 01-2-2v-5a7 7 0 017-7zm0 2a5 5 0 00-5 5v5h10v-5a5 5 0 00-5-5zm0-5l2.5 2.5h-5L12 2z" />
    </svg>
  );
}

function GlobeIcon() {
  return (
    <svg viewBox="0 0 24 24" className="icon" aria-hidden="true">
      <path d="M12 2a10 10 0 100 20 10 10 0 000-20zm6.9 9h-3.2a15.3 15.3 0 00-1.4-5A8 8 0 0118.9 11zM12 4c1 1.2 1.9 3.8 2 7h-4c.1-3.2 1-5.8 2-7zM5.1 13h3.2a15.3 15.3 0 001.4 5A8 8 0 015.1 13zm0-2A8 8 0 019.7 6a15.3 15.3 0 00-1.4 5H5.1zm6.9 9c-1-1.2-1.9-3.8-2-7h4c-.1 3.2-1 5.8-2 7zm2.3-2a15.3 15.3 0 001.4-5h3.2a8 8 0 01-4.6 5z" />
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg viewBox="0 0 24 24" className="icon" aria-hidden="true">
      <path d="M12 3l4 4h-3v6h-2V7H8l4-4zm-7 9h2v6h10v-6h2v6a2 2 0 01-2 2H7a2 2 0 01-2-2v-6z" />
    </svg>
  );
}

function DownloadIcon() {
  return (
    <svg viewBox="0 0 24 24" className="icon" aria-hidden="true">
      <path d="M11 3h2v9l3-3 1.4 1.4L12 16l-5.4-5.6L8 9l3 3V3zm-7 14h16v4H4v-4z" />
    </svg>
  );
}

function ShareIcon() {
  return (
    <svg viewBox="0 0 24 24" className="icon" aria-hidden="true">
      <path d="M15 8a3 3 0 10-2.8-4H12a3 3 0 00.2 1L7.9 7.5a3 3 0 100 9l4.3 2.5A3 3 0 1013 17l-4.3-2.5a3 3 0 000-3L13 9a3 3 0 002 .8z" />
    </svg>
  );
}

function RefreshIcon() {
  return (
    <svg viewBox="0 0 24 24" className="icon" aria-hidden="true">
      <path d="M18 7V3l-1.7 1.7A9 9 0 103 12h2a7 7 0 117 7 6.9 6.9 0 01-4.6-1.7L10 15H4v6l1.9-1.9A9 9 0 1021 12a9 9 0 00-3-6.7L18 7z" />
    </svg>
  );
}

function FlipIcon() {
  return (
    <svg viewBox="0 0 24 24" className="icon" aria-hidden="true">
      <path d="M7 7h10l-2.5-2.5L16 3l5 5-5 5-1.5-1.5L17 9H7V7zm10 10H7l2.5 2.5L8 21l-5-5 5-5 1.5 1.5L7 15h10v2z" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg viewBox="0 0 24 24" className="icon small-icon" aria-hidden="true">
      <path d="M9.6 16.2L5.4 12l1.4-1.4 2.8 2.8 7-7 1.4 1.4-8.4 8.4z" />
    </svg>
  );
}

function ThumbUpIcon() {
  return (
    <svg viewBox="0 0 24 24" className="icon small-icon" aria-hidden="true">
      <path d="M14 3l-4 5v13h8a2 2 0 002-1.7l1-7A2 2 0 0019 10h-5l1-5-2-2zM3 10h5v11H3V10z" />
    </svg>
  );
}

function ThumbDownIcon() {
  return (
    <svg viewBox="0 0 24 24" className="icon small-icon" aria-hidden="true">
      <path d="M10 21l4-5V3H6a2 2 0 00-2 1.7l-1 7A2 2 0 005 14h5l-1 5 2 2zm11-8h-5V2h5v11z" />
    </svg>
  );
}

function buildObjectUrl(file) {
  return file ? URL.createObjectURL(file) : "";
}

function downloadFile(url, name) {
  if (!url) return;
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function useObjectUrl(file) {
  const [objectUrl, setObjectUrl] = useState("");

  useEffect(() => {
    if (!file) {
      setObjectUrl("");
      return undefined;
    }
    const nextUrl = buildObjectUrl(file);
    setObjectUrl(nextUrl);
    return () => URL.revokeObjectURL(nextUrl);
  }, [file]);

  return objectUrl;
}

function TopNavigation({ currentView, onNavigate }) {
  const activeNav =
    currentView === "about" || currentView === "contact" ? currentView : "home";

  return (
    <header className="topbar">
      <button
        type="button"
        className="brand"
        onClick={() => onNavigate("home")}
      >
        <LogoMark />
        <div className="brand-copy">
          <strong>Hairstyles</strong>
          <span>Recommender and Tryon</span>
        </div>
      </button>

      <nav className="nav">
        <button
          type="button"
          className={`nav-link ${activeNav === "home" ? "active" : ""}`}
          onClick={() => onNavigate("home")}
        >
          Home
        </button>
        <button
          type="button"
          className={`nav-link ${activeNav === "about" ? "active" : ""}`}
          onClick={() => onNavigate("about")}
        >
          About Us
        </button>
        <button
          type="button"
          className={`nav-link ${activeNav === "contact" ? "active" : ""}`}
          onClick={() => onNavigate("contact")}
        >
          Contact Us
        </button>
      </nav>
    </header>
  );
}

function SceneBackdrop() {
  return (
    <>
      <div className="ambient ambient-left-top" />
      <div className="ambient ambient-left-bottom" />
      <div className="ambient ambient-right-top" />
      <div className="ambient ambient-right-bottom" />
      <div className="ambient-grid" />
    </>
  );
}

function SystemCard({ card, onOpen }) {
  return (
    <button
      type="button"
      className={`home-card ${card.accent} ${card.disabled ? "disabled" : ""}`}
      onClick={() => !card.disabled && onOpen(card.id)}
      disabled={card.disabled}
    >
      <div className="home-card-icon">
        {card.id === "static" ? <PhotoIcon /> : null}
        {card.id === "generative" ? <SparkleIcon /> : null}
        {card.id === "live2d" ? <CameraIcon /> : null}
        {card.id === "live3d" ? <GlobeIcon /> : null}
      </div>
      <div className="home-card-copy">
        <strong>{card.title}</strong>
        <span>{card.subtitle}</span>
      </div>
      <span className="home-card-arrow">{card.disabled ? "Soon" : "›"}</span>
    </button>
  );
}

function HomeScreen({ onOpenSystem }) {
  return (
    <section className="page home-page">
      <div className="page-head home-head">
        <h1>Hairstyles Recommender and Tryon</h1>
        <p>
          Discover the perfect hairstyle for you with AI-powered recommendations
          and realistic try-on experiences.
        </p>
      </div>

      <div className="home-grid">
        {HOME_CARDS.map((card) => (
          <SystemCard key={card.id} card={card} onOpen={onOpenSystem} />
        ))}
      </div>
    </section>
  );
}

function InfoScreen({ title, text, items }) {
  return (
    <section className="page info-page">
      <div className="page-head compact-head">
        <h1>{title}</h1>
        <p>{text}</p>
      </div>

      <div className="glass-card info-card">
        {items.map((item) => (
          <p key={item}>{item}</p>
        ))}
      </div>
    </section>
  );
}

function GenderToggle({ value, onChange }) {
  return (
    <div className="gender-toggle">
      <button
        type="button"
        className={value === "male" ? "active" : ""}
        onClick={() => onChange("male")}
      >
        Male
      </button>
      <button
        type="button"
        className={value === "female" ? "active" : ""}
        onClick={() => onChange("female")}
      >
        Female
      </button>
    </div>
  );
}

function LiveGenderToggle({ value, onChange }) {
  return (
    <div className="gender-toggle gender-toggle-triple">
      <button
        type="button"
        className={value === "any" ? "active" : ""}
        onClick={() => onChange("any")}
      >
        Any
      </button>
      <button
        type="button"
        className={value === "male" ? "active" : ""}
        onClick={() => onChange("male")}
      >
        Male
      </button>
      <button
        type="button"
        className={value === "female" ? "active" : ""}
        onClick={() => onChange("female")}
      >
        Female
      </button>
    </div>
  );
}

function UploadCard({
  title,
  subtitle,
  file,
  previewUrl,
  gender,
  onFileChange,
  onClear,
  onGenderChange,
  uploadLabel,
  helper,
}) {
  return (
    <section className="glass-card upload-card">
      <div className="section-title">
        <div className="section-icon">
          <UploadIcon />
        </div>
        <div>
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
      </div>

      <div className="image-frame portrait-frame">
        {previewUrl ? (
          <img
            src={previewUrl}
            alt="Uploaded preview"
            className="cover-image"
          />
        ) : (
          <div className="empty-panel">Upload an image to begin</div>
        )}
        {file ? (
          <button type="button" className="floating-close" onClick={onClear}>
            ×
          </button>
        ) : null}
      </div>

      <label className="upload-cta">
        <input
          type="file"
          accept="image/png,image/jpeg,image/webp"
          onChange={(event) => onFileChange(event.target.files?.[0] ?? null)}
        />
        <div className="upload-cta-copy">
          <strong>{uploadLabel}</strong>
          <span>JPG, PNG up to 10MB</span>
        </div>
      </label>

      <div className="field-group">
        <span className="field-label">Select Gender</span>
        <GenderToggle value={gender} onChange={onGenderChange} />
      </div>

      {helper ? <div className="soft-note">{helper}</div> : null}
    </section>
  );
}

function RecommendationGrid({
  title,
  subtitle,
  recommendations,
  selectedAssetId,
  onSelect,
  footer,
  action,
  compact = false,
  mediaBasePath = SYSTEM_BASE_PATHS.staticAuto,
}) {
  return (
    <section className="glass-card recommendation-panel">
      <div className="section-title">
        <div className="section-icon pink-icon">
          <SparkleIcon />
        </div>
        <div>
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
      </div>

      <div className={`recommendation-grid ${compact ? "compact-grid" : ""}`}>
        {recommendations.map((item) => (
          <button
            key={item.asset_id}
            type="button"
            className={`style-card ${
              selectedAssetId === item.asset_id ? "selected" : ""
            }`}
            onClick={() => onSelect(item)}
          >
            <div className="style-thumb">
              {item.image_url ? (
                <img
                  src={resolveMediaUrl(item.image_url, mediaBasePath)}
                  alt={item.asset_id}
                  className="cover-image"
                />
              ) : (
                <div className="empty-panel">No preview</div>
              )}
              {selectedAssetId === item.asset_id ? (
                <span className="selected-badge">
                  <CheckIcon />
                </span>
              ) : null}
            </div>
            <strong>
              {formatAssetName(item.asset_id, item.normalized_attributes)}
            </strong>
          </button>
        ))}
      </div>

      {action ? <div className="panel-action-row">{action}</div> : null}
      {footer ? <p className="panel-footer">{footer}</p> : null}
    </section>
  );
}

function ResultCard({
  title,
  subtitle,
  imageUrl,
  children,
  emptyMessage = "Your result will appear here",
  downloadLabel = "Download Image",
  onDownload,
  extraActions,
}) {
  return (
    <section className="glass-card result-card">
      <div className="section-title">
        <div className="section-icon result-icon">
          <SparkleIcon />
        </div>
        <div>
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
      </div>

      <div className="image-frame result-frame">
        {imageUrl ? (
          <img src={imageUrl} alt={title} className="cover-image" />
        ) : (
          <div className="empty-panel">{emptyMessage}</div>
        )}
      </div>

      <div className="result-actions">
        <div className="result-actions-right"></div>
        <button
          type="button"
          className="button download-button"
          onClick={onDownload}
          disabled={!imageUrl}
        >
          <DownloadIcon />
          {downloadLabel}
        </button>
        {extraActions}
      </div>

      {children}
    </section>
  );
}

function StatusBanner({ state, error, message }) {
  if (error) {
    return <div className="status-banner error-banner">{error}</div>;
  }
  if (!message && !state) return null;
  return <div className="status-banner">{message ?? state}</div>;
}

function formatAssetName(assetId, attrs) {
  if (attrs?.style_family && attrs.style_family !== "other") {
    return attrs.style_family
      .split("_")
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(" ");
  }
  return assetId.replace("celeba_full_hair_", "").replaceAll("_", " ");
}

function chooseGenerativeImageUrl(result) {
  return (
    resolveMediaUrl(result?.final_image_url, SYSTEM_BASE_PATHS.generative) ||
    resolveMediaUrl(result?.preview_url, SYSTEM_BASE_PATHS.generative) ||
    resolveMediaUrl(
      result?.reference_image_url,
      SYSTEM_BASE_PATHS.generative,
    ) ||
    resolveMediaUrl(result?.input_image_url, SYSTEM_BASE_PATHS.generative) ||
    ""
  );
}

function StaticTryOnScreen() {
  const [gender, setGender] = useState("female");
  const [imageFile, setImageFile] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [selectedAssetId, setSelectedAssetId] = useState("");
  const [preparedResults, setPreparedResults] = useState({});
  const [busyState, setBusyState] = useState("");
  const [error, setError] = useState("");
  const [summary, setSummary] = useState(null);
  const previewUrl = useObjectUrl(imageFile);

  useEffect(() => {
    let cancelled = false;
    fetchAssetBankSummary(SYSTEM_BASE_PATHS.staticAuto)
      .then((payload) => {
        if (!cancelled) {
          setSummary(payload);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setSummary(null);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  async function runRecommendations(file, nextGender) {
    setBusyState("Processing, please wait. This may take about 1 minute...");
    setError("");
    setPreparedResults({});
    const nextAnalysis = await analyzeFace(file, SYSTEM_BASE_PATHS.staticAuto);
    if (!nextAnalysis.face_detected || !nextAnalysis.face_attributes) {
      throw new Error(nextAnalysis.message || "Face not detected.");
    }
    setAnalysis(nextAnalysis);
    const nextRecommendations = await recommendHairstyles(
      nextAnalysis.face_attributes,
      nextGender,
      RECOMMENDATION_COUNT,
      SYSTEM_BASE_PATHS.staticAuto,
    );
    setRecommendations(nextRecommendations);
    setSelectedAssetId(nextRecommendations[0]?.asset_id ?? "");

    if (!nextRecommendations.length) {
      setBusyState("");
      return;
    }

    setBusyState("Processing, please wait. This may take about 1 minute...");
    const generatedEntries = await Promise.all(
      nextRecommendations.map(async (item) => [
        item.asset_id,
        await generateTryOn(file, item.asset_id, SYSTEM_BASE_PATHS.staticAuto),
      ]),
    );
    setPreparedResults(Object.fromEntries(generatedEntries));
    setBusyState("");
  }

  async function handleFileChange(file) {
    setImageFile(file);
    setAnalysis(null);
    setRecommendations([]);
    setSelectedAssetId("");
    setPreparedResults({});
    setBusyState("");
    setError("");

    // Do not auto-run static try-on after upload.
    // User must click "See Results".
  }

  async function handleSeeResults() {
    if (!imageFile) {
      setError("Please upload an image first.");
      return;
    }

    setRecommendations([]);
    setSelectedAssetId("");
    setPreparedResults({});
    setError("");

    try {
      await runRecommendations(imageFile, gender);
    } catch (nextError) {
      setBusyState("");
      setError(
        nextError instanceof Error
          ? nextError.message
          : "Static try-on failed.",
      );
    }
  }

  async function handleSelect(item, file = imageFile) {
    if (!file || !item?.asset_id) return;
    setSelectedAssetId(item.asset_id);
    if (preparedResults[item.asset_id]) {
      return;
    }
    setBusyState("Generating static try-on...");
    setError("");
    try {
      const nextResult = await generateTryOn(
        file,
        item.asset_id,
        SYSTEM_BASE_PATHS.staticAuto,
      );
      setPreparedResults((current) => ({
        ...current,
        [item.asset_id]: nextResult,
      }));
    } catch (nextError) {
      setError(
        nextError instanceof Error
          ? nextError.message
          : "Static try-on failed.",
      );
    } finally {
      setBusyState("");
    }
  }

  async function handleGenderChange(nextGender) {
    setGender(nextGender);
    if (imageFile && analysis?.face_attributes) {
      try {
        setBusyState("Refreshing recommendations...");
        const nextRecommendations = await recommendHairstyles(
          analysis.face_attributes,
          nextGender,
          RECOMMENDATION_COUNT,
          SYSTEM_BASE_PATHS.staticAuto,
        );
        setRecommendations(nextRecommendations);
        setSelectedAssetId(nextRecommendations[0]?.asset_id ?? "");
        setPreparedResults({});
        if (!nextRecommendations.length) {
          setBusyState("");
          return;
        }
        setBusyState("Preparing try-on results...");
        const generatedEntries = await Promise.all(
          nextRecommendations.map(async (item) => [
            item.asset_id,
            await generateTryOn(
              imageFile,
              item.asset_id,
              SYSTEM_BASE_PATHS.staticAuto,
            ),
          ]),
        );
        setPreparedResults(Object.fromEntries(generatedEntries));
        setBusyState("");
      } catch (nextError) {
        setBusyState("");
        setError(
          nextError instanceof Error
            ? nextError.message
            : "Could not refresh recommendations.",
        );
      }
    }
  }

  const result = selectedAssetId
    ? (preparedResults[selectedAssetId] ?? null)
    : null;

  return (
    <section className="page system-page">
      <StatusBanner
        state={busyState}
        error={error}
        message={
          summary
            ? `Runtime asset bank: ${summary.asset_count} styles`
            : busyState
        }
      />

      <div className="three-panel-layout">
        <UploadCard
          title="Upload Image"
          subtitle="Upload a clear front-facing photo"
          file={imageFile}
          previewUrl={previewUrl}
          gender={gender}
          onFileChange={handleFileChange}
          onClear={() => handleFileChange(null)}
          onGenderChange={handleGenderChange}
          uploadLabel={imageFile ? "Change Photo" : "Upload Photo"}
          helper={
            <button
              type="button"
              className="primary-gradient see-results"
              onClick={handleSeeResults}
              disabled={!imageFile || Boolean(busyState)}
            >
              See Results
            </button>
          }
        />

        <ResultCard
          title="Static Try-On Result"
          subtitle="See how the hairstyle looks on you"
          imageUrl={resolveMediaUrl(
            result?.output_image_url,
            SYSTEM_BASE_PATHS.staticAuto,
          )}
          emptyMessage={
            busyState
              ? "Processing, please wait. This may take about 1 minute..."
              : imageFile
                ? "Click See Results to generate your static try-on."
                : "Upload an image to begin."
          }
          onDownload={() =>
            downloadFile(
              resolveMediaUrl(
                result?.output_image_url,
                SYSTEM_BASE_PATHS.staticAuto,
              ),
              `${selectedAssetId || "static-tryon"}.png`,
            )
          }
        />
        <RecommendationGrid
          title="Recommended Hairstyles"
          recommendations={busyState ? [] : recommendations}
          selectedAssetId={selectedAssetId}
          onSelect={handleSelect}
          footer="Click a hairstyle to see the try-on result"
          mediaBasePath={SYSTEM_BASE_PATHS.staticAuto}
        />
      </div>
    </section>
  );
}

function GenerativeTryOnScreen() {
  const [gender, setGender] = useState("male");
  const [imageFile, setImageFile] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [selectedAssetId, setSelectedAssetId] = useState("");
  const [preparedPackages, setPreparedPackages] = useState({});
  const [busyState, setBusyState] = useState("");
  const [error, setError] = useState("");
  const [feedback, setFeedback] = useState("");
  const previewUrl = useObjectUrl(imageFile);

  async function refreshRecommendations(file = imageFile, nextGender = gender) {
    if (!file) return;
    setBusyState("Analyzing image and refreshing generative suggestions...");
    setError("");
    const nextAnalysis = await analyzeFace(file, SYSTEM_BASE_PATHS.staticAuto);
    if (!nextAnalysis.face_detected || !nextAnalysis.face_attributes) {
      throw new Error(nextAnalysis.message || "Face not detected.");
    }
    setAnalysis(nextAnalysis);
    const nextRecommendations = await recommendHairstyles(
      nextAnalysis.face_attributes,
      nextGender,
      RECOMMENDATION_COUNT,
      SYSTEM_BASE_PATHS.staticAuto,
    );
    setRecommendations(nextRecommendations);
    setSelectedAssetId(nextRecommendations[0]?.asset_id ?? "");
    if (!nextRecommendations.length) {
      setPreparedPackages({});
      setBusyState("");
      return;
    }
    setBusyState("Preparing generative previews...");
    const generatedEntries = await Promise.all(
      nextRecommendations.map(async (item) => [
        item.asset_id,
        await generateGenerativePackage(file, item.asset_id),
      ]),
    );
    setPreparedPackages(Object.fromEntries(generatedEntries));
    setBusyState("");
  }

  async function handleFileChange(file) {
    setImageFile(file);
    setAnalysis(null);
    setRecommendations([]);
    setSelectedAssetId("");
    setPreparedPackages({});
    setError("");
    setFeedback("");
    if (!file) return;
    try {
      await refreshRecommendations(file, gender);
    } catch (nextError) {
      setBusyState("");
      setError(
        nextError instanceof Error
          ? nextError.message
          : "Could not prepare generative try-on.",
      );
    }
  }

  async function handleSelect(item, file = imageFile) {
    if (!file || !item?.asset_id) return;
    setSelectedAssetId(item.asset_id);
    if (preparedPackages[item.asset_id]) {
      return;
    }
    setBusyState("Building generative package...");
    setError("");
    try {
      const nextResult = await generateGenerativePackage(file, item.asset_id);
      setPreparedPackages((current) => ({
        ...current,
        [item.asset_id]: nextResult,
      }));
    } catch (nextError) {
      setError(
        nextError instanceof Error
          ? nextError.message
          : "Generative try-on failed.",
      );
    } finally {
      setBusyState("");
    }
  }

  async function handleGenderChange(nextGender) {
    setGender(nextGender);
    if (imageFile && analysis?.face_attributes) {
      try {
        setBusyState("Refreshing recommendations...");
        const nextRecommendations = await recommendHairstyles(
          analysis.face_attributes,
          nextGender,
          RECOMMENDATION_COUNT,
          SYSTEM_BASE_PATHS.staticAuto,
        );
        setRecommendations(nextRecommendations);
        setSelectedAssetId(nextRecommendations[0]?.asset_id ?? "");
        if (!nextRecommendations.length) {
          setPreparedPackages({});
          setBusyState("");
          return;
        }
        setBusyState("Preparing generative previews...");
        const generatedEntries = await Promise.all(
          nextRecommendations.map(async (item) => [
            item.asset_id,
            await generateGenerativePackage(imageFile, item.asset_id),
          ]),
        );
        setPreparedPackages(Object.fromEntries(generatedEntries));
        setBusyState("");
      } catch (nextError) {
        setBusyState("");
        setError(
          nextError instanceof Error
            ? nextError.message
            : "Could not refresh recommendations.",
        );
      }
    }
  }

  const result = selectedAssetId
    ? (preparedPackages[selectedAssetId] ?? null)
    : null;
  const displayImageUrl = chooseGenerativeImageUrl(result);
  const generativeMessage = result?.final_generation_completed
    ? "Final generative result created successfully."
    : result?.final_generation_error
      ? `Final generation failed, showing prototype preview instead. ${result.final_generation_error}`
      : result?.message || busyState;

  return (
    <section className="page system-page">
      <StatusBanner
        state={busyState}
        error={error}
        message={generativeMessage}
      />

      <div className="three-panel-layout">
        <UploadCard
          title="Upload Image"
          subtitle="Upload a clear portrait for AI try-on"
          file={imageFile}
          previewUrl={previewUrl}
          gender={gender}
          onFileChange={handleFileChange}
          onClear={() => handleFileChange(null)}
          onGenderChange={handleGenderChange}
          uploadLabel={imageFile ? "Upload Another Image" : "Upload Image"}
          helper="Your photos are secure and private. We don't store your images."
        />

        <RecommendationGrid
          title="Recommended Hairstyles"
          subtitle="AI suggestions tailored for you"
          recommendations={recommendations}
          selectedAssetId={selectedAssetId}
          onSelect={handleSelect}
          footer=""
          mediaBasePath={SYSTEM_BASE_PATHS.staticAuto}
          action={
            <button
              type="button"
              className="secondary-pill"
              onClick={() => refreshRecommendations()}
              disabled={!imageFile || Boolean(busyState)}
            >
              <RefreshIcon />
              Refresh Recommendations
            </button>
          }
        />

        <ResultCard
          title="Generative Result"
          subtitle={
            result?.final_generation_completed
              ? "AI-generated final hairstyle result"
              : "AI-generated try-on preview"
          }
          imageUrl={displayImageUrl}
          onDownload={() =>
            downloadFile(
              displayImageUrl,
              `${selectedAssetId || "generative-tryon"}.png`,
            )
          }
          extraActions={
            <button
              type="button"
              className="secondary-pill"
              disabled={!displayImageUrl}
              onClick={async () => {
                if (!displayImageUrl) return;
                if (navigator.share) {
                  await navigator.share({
                    title: result?.final_generation_completed
                      ? "Generative Try-On Result"
                      : "Generative Try-On Preview",
                    url: displayImageUrl,
                  });
                  return;
                }
                await navigator.clipboard.writeText(displayImageUrl);
              }}
            >
              <ShareIcon />
              Share
            </button>
          }
        >
          <div className="feedback-row">
            <span>Happy with the result?</span>
            <button
              type="button"
              className={`icon-pill ${feedback === "up" ? "active" : ""}`}
              onClick={() => setFeedback("up")}
            >
              <ThumbUpIcon />
            </button>
            <button
              type="button"
              className={`icon-pill ${feedback === "down" ? "active" : ""}`}
              onClick={() => setFeedback("down")}
            >
              <ThumbDownIcon />
            </button>
          </div>
        </ResultCard>
      </div>
    </section>
  );
}

function Live2DTryOnScreen() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const socketRef = useRef(null);
  const selectedAssetRef = useRef("");
  const targetGenderRef = useRef("any");
  const pendingFrameRef = useRef(false);
  const latestFrameUrlRef = useRef("");
  const sendLoopTimerRef = useRef(null);
  const frameCanvasRef = useRef(null);

  const [liveInfo, setLiveInfo] = useState(null);
  const [cameraError, setCameraError] = useState("");
  const [cameraReady, setCameraReady] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [recommendations, setRecommendations] = useState([]);
  const [liveStats, setLiveStats] = useState(null);
  const [selectedAssetId, setSelectedAssetId] = useState("");
  const [targetGender, setTargetGender] = useState("any");
  const [facingMode, setFacingMode] = useState("user");
  const [processedFrameUrl, setProcessedFrameUrl] = useState("");
  const [wsConnected, setWsConnected] = useState(false);

  useEffect(() => {
    let cancelled = false;

    fetchLive2dInfo()
      .then((payload) => {
        if (!cancelled) {
          setLiveInfo(payload);

          if (payload.status !== "project_ready") {
            setCameraError(
              "Live 2D backend is not ready. Check models and assets.",
            );
          }
        }
      })
      .catch((error) => {
        if (!cancelled) {
          console.error("Live 2D info failed:", error);
          setLiveInfo(null);
          setCameraError("Could not connect to the Live 2D backend.");
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let active = true;

    async function startCamera() {
      if (!navigator.mediaDevices?.getUserMedia) {
        setCameraError("Camera access is not supported in this browser.");
        return;
      }

      try {
        setCameraError("");
        setCameraReady(false);

        if (streamRef.current) {
          streamRef.current.getTracks().forEach((track) => track.stop());
        }

        const nextStream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode,
            width: { ideal: 640, max: 640 },
            height: { ideal: 480, max: 480 },
            frameRate: { ideal: 15, max: 15 },
          },
          audio: false,
        });

        if (!active) {
          nextStream.getTracks().forEach((track) => track.stop());
          return;
        }

        streamRef.current = nextStream;

        if (videoRef.current) {
          videoRef.current.srcObject = nextStream;
          await videoRef.current.play();
          setCameraReady(true);
        }
      } catch (error) {
        setCameraError(
          error instanceof Error
            ? error.message
            : "Could not start the camera.",
        );
      }
    }

    startCamera();

    return () => {
      active = false;

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, [facingMode]);

  useEffect(() => {
    selectedAssetRef.current = selectedAssetId;
  }, [selectedAssetId]);

  useEffect(() => {
    targetGenderRef.current = targetGender;
  }, [targetGender]);

  useEffect(() => {
    return () => {
      if (latestFrameUrlRef.current) {
        URL.revokeObjectURL(latestFrameUrlRef.current);
        latestFrameUrlRef.current = "";
      }
    };
  }, []);

  async function captureFrameBlob() {
    const video = videoRef.current;

    if (!video || !video.videoWidth || !video.videoHeight) {
      throw new Error("Camera frame is not ready yet.");
    }

    const maxSide = 640;
    const scale = Math.min(
      1,
      maxSide / Math.max(video.videoWidth, video.videoHeight),
    );

    const canvas = frameCanvasRef.current ?? document.createElement("canvas");
    frameCanvasRef.current = canvas;

    canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale));

    const context = canvas.getContext("2d");

    if (!context) {
      throw new Error("Could not initialize the live frame capture canvas.");
    }

    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", 0.7),
    );

    if (!blob) {
      throw new Error("Could not capture a live frame.");
    }

    return blob;
  }

  function sendControlAction(action) {
    const socket = socketRef.current;

    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return;
    }

    socket.send(
      JSON.stringify({
        type: "control",
        action,
        selected_asset_id: selectedAssetRef.current || "",
        target_gender: targetGenderRef.current || "any",
      }),
    );

    sendLiveFrame();
  }

  async function sendLiveFrame(nextSelectedAssetId = selectedAssetRef.current) {
    const socket = socketRef.current;

    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return;
    }

    // Important: drop frames while backend is still processing previous one.
    if (pendingFrameRef.current) {
      return;
    }

    try {
      pendingFrameRef.current = true;
      setIsRefreshing(true);

      socket.send(
        JSON.stringify({
          type: "control",
          selected_asset_id: nextSelectedAssetId || "",
          target_gender: targetGenderRef.current || "any",
        }),
      );

      const blob = await captureFrameBlob();
      socket.send(blob);
    } catch (error) {
      pendingFrameRef.current = false;
      setIsRefreshing(false);
      setCameraError(
        error instanceof Error ? error.message : "Could not send live frame.",
      );
    }
  }

  useEffect(() => {
    function handleKeyDown(event) {
      const key = event.key.toLowerCase();

      if (
        [
          "arrowup",
          "arrowdown",
          "arrowleft",
          "arrowright",
          "w",
          "s",
          "a",
          "d",
          "t",
          "r",
          "i",
          "j",
          "k",
          "l",
        ].includes(key)
      ) {
        event.preventDefault();
      }

      if (key === "arrowleft" || key === "j") {
        sendControlAction("move_left");
      } else if (key === "arrowright" || key === "l") {
        sendControlAction("move_right");
      } else if (key === "arrowup" || key === "i") {
        sendControlAction("move_up");
      } else if (key === "arrowdown" || key === "k") {
        sendControlAction("move_down");
      } else if (key === "w") {
        sendControlAction("scale_up");
      } else if (key === "s") {
        sendControlAction("scale_down");
      } else if (key === "a") {
        sendControlAction("rotate_left");
      } else if (key === "d") {
        sendControlAction("rotate_right");
      } else if (key === "t") {
        sendControlAction("reset");
      } else if (key === "r") {
        sendControlAction("cycle");
      }
    }

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [cameraReady, wsConnected]);

  useEffect(() => {
    if (!cameraReady || liveInfo?.status !== "project_ready") {
      return;
    }

    let closedByEffect = false;
    const socket = new WebSocket(live2dWsUrl());

    socket.binaryType = "blob";
    socketRef.current = socket;

    socket.onopen = () => {
      setWsConnected(true);
      setCameraError("");

      socket.send(
        JSON.stringify({
          type: "control",
          selected_asset_id: selectedAssetRef.current || "",
          target_gender: targetGenderRef.current || "any",
        }),
      );

      sendLoopTimerRef.current = window.setInterval(() => {
        sendLiveFrame();
      }, 100);
    };

    socket.onmessage = (event) => {
      if (typeof event.data === "string") {
        try {
          const payload = JSON.parse(event.data);

          if (payload.type === "error") {
            pendingFrameRef.current = false;
            setIsRefreshing(false);
            setCameraError(payload.message || "Live 2D backend error.");
            return;
          }

          if (payload.type === "metadata") {
            setRecommendations(payload.recommendations ?? []);

            setLiveStats({
              selectedScore: payload.selected_score ?? 0,
              topPicks: payload.top_picks ?? 0,
              cleanBank: payload.clean_bank ?? 0,
              tuning: payload.tuning ?? {
                x: 0,
                y: 0,
                scale: 1,
                rotation: 0,
              },
            });

            const nextSelected =
              payload.selected_asset_id ??
              payload.recommendations?.[0]?.asset_id ??
              "";

            setSelectedAssetId(nextSelected);
          }
        } catch (error) {
          console.error("Could not parse Live 2D metadata:", error);
        }

        return;
      }

      const blob =
        event.data instanceof Blob
          ? event.data
          : new Blob([event.data], { type: "image/jpeg" });

      const nextUrl = URL.createObjectURL(blob);

      if (latestFrameUrlRef.current) {
        URL.revokeObjectURL(latestFrameUrlRef.current);
      }

      latestFrameUrlRef.current = nextUrl;
      setProcessedFrameUrl(nextUrl);

      pendingFrameRef.current = false;
      setIsRefreshing(false);
    };

    socket.onerror = () => {
      pendingFrameRef.current = false;
      setIsRefreshing(false);
      setCameraError("Live 2D WebSocket connection failed.");
    };

    socket.onclose = () => {
      pendingFrameRef.current = false;
      setIsRefreshing(false);
      setWsConnected(false);

      if (!closedByEffect) {
        setCameraError("Live 2D WebSocket disconnected.");
      }
    };

    return () => {
      closedByEffect = true;

      if (sendLoopTimerRef.current) {
        window.clearInterval(sendLoopTimerRef.current);
        sendLoopTimerRef.current = null;
      }

      pendingFrameRef.current = false;
      setIsRefreshing(false);
      setWsConnected(false);

      socket.close();
    };
  }, [cameraReady, liveInfo?.status]);

  const selectedRecommendation =
    recommendations.find((item) => item.asset_id === selectedAssetId) ??
    recommendations[0] ??
    null;

  return (
    <section className="page live-page">
      <div className="page-head live-head">
        <div className="headline-with-icon">
          <div className="headline-icon">
            <CameraIcon />
          </div>
          <h1>Live 2D Tryon</h1>
        </div>
        <p>See hairstyles on you in real-time using your webcam.</p>
      </div>

      <StatusBanner
        state={
          isRefreshing
            ? "Streaming live frame..."
            : wsConnected
              ? "Live WebSocket connected"
              : ""
        }
        error={cameraError}
        message={
          liveInfo?.status === "project_ready"
            ? `Live 2D backend ready with ${
                liveInfo.readiness?.tryon_clean_candidate_count ?? 0
              } clean overlay assets`
            : ""
        }
      />
      <section className="glass-card live-stage-card live-wide-card">
        <div className="stage-topbar">
          <span className="live-chip">
            <span className={`live-dot ${wsConnected ? "on" : ""}`} />
            Live 2D Try-On
          </span>
          <span className="camera-chip">
            <span className={`live-dot ${cameraReady ? "on" : ""}`} />
            Camera: {cameraReady ? "On" : "Off"}
          </span>
        </div>

        <div className="live-2d-layout">
          <aside className="live-side-panel">
            <div className="live-instruction-card">
              <h3>Move Hair</h3>

              <div className="live-instruction-list">
                <p>
                  <strong>Arrow keys</strong> or <strong>I/J/K/L</strong> — move
                  hair
                </p>
                <p>
                  <strong>W / S</strong> — increase / decrease size
                </p>
                <p>
                  <strong>A / D</strong> — rotate left / right
                </p>
                <p>
                  <strong>R</strong> — cycle hairstyle
                </p>
                <p>
                  <strong>T</strong> — reset tuning
                </p>
              </div>
            </div>
          </aside>
          <div className="live-main-panel">
            <div className="live-stage">
              <video
                ref={videoRef}
                className="live-video live-video-source"
                playsInline
                muted
              />

              {processedFrameUrl ? (
                <img
                  src={processedFrameUrl}
                  alt="Live 2D backend output"
                  className="live-video"
                />
              ) : (
                <div className="empty-panel">
                  Waiting for backend live frame...
                </div>
              )}

              <button
                type="button"
                className="flip-camera"
                onClick={() =>
                  setFacingMode((current) =>
                    current === "user" ? "environment" : "user",
                  )
                }
              >
                <FlipIcon />
                Flip Camera
              </button>
            </div>

            <div className="live-toolbar">
              <div className="gender-inline">
                <span>Backend engine:</span>
                <strong>WebSocket live 2D try-on</strong>
                <span>Filter:</span>
                <strong>
                  {targetGender === "any"
                    ? "Any"
                    : targetGender === "male"
                      ? "Male"
                      : "Female"}
                </strong>
              </div>

              <button
                type="button"
                className="secondary-pill"
                onClick={() => sendLiveFrame()}
                disabled={!cameraReady || !wsConnected || isRefreshing}
              >
                <RefreshIcon />
                Refresh Live Frame
              </button>
            </div>
          </div>

          <aside className="live-side-panel">
            <div className="live-recommend-card">
              <h3>Hairstyles</h3>

              <div className="live-mini-grid">
                {recommendations.map((item) => (
                  <button
                    key={item.asset_id}
                    type="button"
                    className={`live-mini-card ${
                      selectedAssetId === item.asset_id ? "active" : ""
                    }`}
                    onClick={() => {
                      setSelectedAssetId(item.asset_id);
                      selectedAssetRef.current = item.asset_id;

                      const socket = socketRef.current;

                      if (socket && socket.readyState === WebSocket.OPEN) {
                        socket.send(
                          JSON.stringify({
                            type: "control",
                            selected_asset_id: item.asset_id,
                            target_gender: targetGenderRef.current || "any",
                          }),
                        );
                      }

                      if (cameraReady && wsConnected) {
                        sendLiveFrame(item.asset_id);
                      }
                    }}
                  >
                    {item.image_url ? (
                      <img
                        src={resolveMediaUrl(
                          item.image_url,
                          SYSTEM_BASE_PATHS.live2d,
                        )}
                        alt={item.asset_id}
                      />
                    ) : null}

                    <span className="live-mini-meta">
                      <strong>
                        {formatAssetName(
                          item.asset_id,
                          item.normalized_attributes,
                        )}
                      </strong>
                      <small>
                        {Math.round((item.score ?? 0) * 100)}% match
                      </small>
                    </span>
                  </button>
                ))}
              </div>
              <div className="live-gender-filter">
                <h4>Recommendation Filter</h4>

                <div className="gender-toggle">
                  <button
                    type="button"
                    className={targetGender === "any" ? "active" : ""}
                    onClick={() => setTargetGender("any")}
                  >
                    Any
                  </button>

                  <button
                    type="button"
                    className={targetGender === "male" ? "active" : ""}
                    onClick={() => setTargetGender("male")}
                  >
                    Male
                  </button>

                  <button
                    type="button"
                    className={targetGender === "female" ? "active" : ""}
                    onClick={() => setTargetGender("female")}
                  >
                    Female
                  </button>
                </div>
              </div>
            </div>
          </aside>
        </div>
      </section>

      <section className="glass-card live-details-card">
        <div className="section-title">
          <div className="section-icon pink-icon">
            <SparkleIcon />
          </div>
          <div>
            <h2>Selected Hair Match Details</h2>
            <p>
              Recommendation and manual tuning details are shown here instead of
              on the camera window.
            </p>
          </div>
        </div>

        <div className="live-stats-grid">
          <div className="live-stat-box">
            <span>Selected hairstyle</span>
            <strong>{selectedRecommendation?.asset_id ?? "None"}</strong>
          </div>

          <div className="live-stat-box">
            <span>Recommendation score</span>
            <strong>
              {Math.round((liveStats?.selectedScore ?? 0) * 100)}%
            </strong>
          </div>

          <div className="live-stat-box">
            <span>Top picks</span>
            <strong>{liveStats?.topPicks ?? recommendations.length}</strong>
          </div>

          <div className="live-stat-box">
            <span>Clean bank</span>
            <strong>
              {liveStats?.cleanBank ??
                liveInfo?.readiness?.tryon_clean_candidate_count ??
                0}
            </strong>
          </div>

          <div className="live-stat-box">
            <span>X offset</span>
            <strong>{liveStats?.tuning?.x ?? 0}</strong>
          </div>

          <div className="live-stat-box">
            <span>Y offset</span>
            <strong>{liveStats?.tuning?.y ?? 0}</strong>
          </div>

          <div className="live-stat-box">
            <span>Scale</span>
            <strong>{liveStats?.tuning?.scale ?? 1}</strong>
          </div>

          <div className="live-stat-box">
            <span>Rotation</span>
            <strong>{liveStats?.tuning?.rotation ?? 0}</strong>
          </div>
        </div>

        {selectedRecommendation?.reason ? (
          <p className="panel-footer live-reason">
            <strong>Reason:</strong> {selectedRecommendation.reason}
          </p>
        ) : null}
      </section>
    </section>
  );
}

function App() {
  const [currentView, setCurrentView] = useState("home");

  function openSystem(systemId) {
    if (systemId === "static") {
      setCurrentView("static");
    } else if (systemId === "generative") {
      setCurrentView("generative");
    } else if (systemId === "live2d") {
      setCurrentView("live2d");
    }
  }

  return (
    <div className="app-shell">
      <SceneBackdrop />
      <TopNavigation currentView={currentView} onNavigate={setCurrentView} />

      {currentView === "home" ? <HomeScreen onOpenSystem={openSystem} /> : null}

      {currentView === "about" ? (
        <InfoScreen
          title="About Us"
          text="This project combines recommendation, segmentation, rendering, and live experimentation in one hairstyle try-on platform."
          items={ABOUT_POINTS}
        />
      ) : null}

      {currentView === "contact" ? (
        <InfoScreen
          title="Contact Us"
          text="Use the repository and local backend gateway as the main entry points for this project."
          items={CONTACT_POINTS}
        />
      ) : null}

      {currentView === "static" ? <StaticTryOnScreen /> : null}
      {currentView === "generative" ? <GenerativeTryOnScreen /> : null}
      {currentView === "live2d" ? <Live2DTryOnScreen /> : null}
    </div>
  );
}

export default App;
