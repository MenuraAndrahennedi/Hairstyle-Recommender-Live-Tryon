import WebcamTryOn from "../components/WebcamTryOn.jsx";

export default function LiveTryOn() {
  return (
    <>
      <section className="live-heading">
        <p className="eyebrow">WebAR.rocks local engine</p>
        <h1>Browser live PNG hairstyle try-on</h1>
        <p>
          WebAR.rocks tracks the head in the browser while this app renders our reviewed PNG hairstyle assets
          directly on a canvas overlay.
        </p>
      </section>
      <WebcamTryOn />
    </>
  );
}
