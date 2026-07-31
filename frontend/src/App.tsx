import { useState, useCallback } from "react";
import { StyleGallery } from "./components/StyleGallery";
import { UploadZone } from "./components/UploadZone";
import { CompareView } from "./components/CompareView";
import { StyleMixSlider } from "./components/StyleMixSlider";
import type { Style } from "./types";

type AppState = "idle" | "uploading" | "stylizing" | "result" | "error";

export default function App() {
  const [state, setState] = useState<AppState>("idle");
  const [styles, setStyles] = useState<Style[]>([]);
  const [selectedStyle, setSelectedStyle] = useState<number | null>(null);
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mixAlpha, setMixAlpha] = useState(1.0);
  const [mixStyleA, setMixStyleA] = useState<number | null>(null);
  const [mixStyleB, setMixStyleB] = useState<number | null>(null);

  const loadStyles = useCallback(async () => {
    try {
      const res = await fetch("/api/styles");
      const data = await res.json();
      setStyles(data);
    } catch {
      setStyles([]);
    }
  }, []);

  useState(() => {
    loadStyles();
  });

  const handleUpload = useCallback((file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      setOriginalImage(e.target?.result as string);
      setResultImage(null);
      setState("uploading");
      setError(null);
    };
    reader.readAsDataURL(file);
  }, []);

  const handleStylize = useCallback(async () => {
    if (!originalImage || selectedStyle === null) return;

    setState("stylizing");
    setError(null);

    try {
      const blob = await fetch(originalImage).then((r) => r.blob());
      const form = new FormData();
      form.append("image", blob, "upload.jpg");

      if (mixStyleA !== null && mixStyleB !== null) {
        form.append("style_a", String(mixStyleA));
        form.append("style_b", String(mixStyleB));
        form.append("alpha", String(mixAlpha));
      } else {
        form.append("style_id", String(selectedStyle));
      }

      const res = await fetch("/api/stylize", { method: "POST", body: form });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Stylization failed" }));
        throw new Error(err.detail || "Stylization failed");
      }

      const contentType = res.headers.get("content-type") || "";
      if (contentType.includes("image")) {
        const resultBlob = await res.blob();
        const url = URL.createObjectURL(resultBlob);
        setResultImage(url);
        setState("result");
      } else {
        const data = await res.json();
        if (data.job_id) {
          pollJob(data.job_id);
        }
      }
    } catch (e) {
      setState("error");
      setError(e instanceof Error ? e.message : "Unknown error");
    }
  }, [originalImage, selectedStyle, mixStyleA, mixStyleB, mixAlpha]);

  const pollJob = useCallback(async (jobId: string) => {
    const poll = async () => {
      try {
        const res = await fetch(`/api/jobs/${jobId}`);
        const data = await res.json();
        if (data.status === "complete") {
          const blob = await res.blob();
          const url = URL.createObjectURL(blob);
          setResultImage(url);
          setState("result");
        } else if (data.status === "error") {
          setState("error");
          setError(data.error || "Processing failed");
        } else {
          setTimeout(poll, 1000);
        }
      } catch {
        setState("error");
        setError("Connection lost");
      }
    };
    poll();
  }, []);

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>StyleForge</h1>
          <span className="subtitle">Neural Style Transfer Engine</span>
        </div>
      </header>

      <main className="main">
        <aside className="sidebar">
          <UploadZone onUpload={handleUpload} />

          <StyleGallery
            styles={styles}
            selected={selectedStyle}
            onSelect={setSelectedStyle}
          />

          <StyleMixSlider
            styles={styles}
            styleA={mixStyleA}
            styleB={mixStyleB}
            alpha={mixAlpha}
            onStyleAChange={setMixStyleA}
            onStyleBChange={setMixStyleB}
            onAlphaChange={setMixAlpha}
          />

          <button
            className="btn btn-primary"
            disabled={state === "stylizing" || !originalImage || (selectedStyle === null && mixStyleA === null)}
            onClick={handleStylize}
          >
            {state === "stylizing" ? "Stylizing..." : "Stylize"}
          </button>

          {error && <div className="error-banner">{error}</div>}
        </aside>

        <section className="canvas-area">
          {state === "idle" && (
            <div className="empty-state">
              <p>Upload an image and select a style to begin.</p>
            </div>
          )}

          {(state === "uploading" || state === "stylizing" || state === "result") && (
            <CompareView
              original={originalImage}
              result={resultImage}
              loading={state === "stylizing"}
            />
          )}

          {state === "result" && resultImage && (
            <div className="actions">
              <a
                className="btn btn-primary"
                href={resultImage}
                download="styleforge-result.jpg"
              >
                Download
              </a>
              <button
                className="btn btn-secondary"
                onClick={() => {
                  setResultImage(null);
                  setState("uploading");
                }}
              >
                New Style
              </button>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
