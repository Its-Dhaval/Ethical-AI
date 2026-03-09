"use client";

import { Icon } from "@iconify/react";
import { useEffect, useMemo, useRef, useState } from "react";

type MediaType = "image" | "video" | "audio";

type ModelResult = {
  model_name: string;
  fake_probability: number;
};

type DetectionResponse = {
  media_type: MediaType;
  label: string;
  fake_probability: number;
  real_probability: number;
  confidence: number;
  method?: string;
  analyzed_frames?: number;
  model_results: ModelResult[];
  details?: Record<string, unknown>;
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

function inferMediaType(file: File): MediaType | null {
  const mime = file.type ?? "";
  const name = file.name.toLowerCase();
  if (mime.startsWith("image/") || /\.(png|jpg|jpeg|bmp|webp)$/.test(name)) {
    return "image";
  }
  if (mime.startsWith("video/") || /\.(mp4|mov|avi|mkv|webm)$/.test(name)) {
    return "video";
  }
  if (mime.startsWith("audio/") || /\.(wav|mp3|flac|ogg|m4a)$/.test(name)) {
    return "audio";
  }
  return null;
}

function prettyModelName(name: string): string {
  return name
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .trim();
}

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function modelPredictionLabel(modelValue: number): {
  label: string;
  textClass: string;
  barClass: string;
} {
  if (modelValue >= 50) {
    return {
      label: "Predicts Deepfake",
      textClass: "text-red-300/90",
      barClass: "bg-red-500"
    };
  }
  if (modelValue >= 45) {
    return {
      label: "Likely Deepfake",
      textClass: "text-amber-300/90",
      barClass: "bg-amber-400"
    };
  }
  return {
    label: "Predicts Real",
    textClass: "text-cyan-300/90",
    barClass: "bg-cyan-500"
  };
}

export default function Home() {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const mainRef = useRef<HTMLElement | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [cursor, setCursor] = useState({ x: 50, y: 20 });

  const mediaType = useMemo(() => (file ? inferMediaType(file) : null), [file]);
  const hasResult = result !== null;
  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  useEffect(() => {
    let rafId = 0;
    const handlePointerMove = (event: PointerEvent) => {
      if (rafId) {
        cancelAnimationFrame(rafId);
      }
      rafId = requestAnimationFrame(() => {
        const x = (event.clientX / window.innerWidth) * 100;
        const y = (event.clientY / window.innerHeight) * 100;
        setCursor({ x, y });
      });
    };

    window.addEventListener("pointermove", handlePointerMove);
    return () => {
      if (rafId) {
        cancelAnimationFrame(rafId);
      }
      window.removeEventListener("pointermove", handlePointerMove);
    };
  }, []);

  useEffect(() => {
    const nodes = Array.from(document.querySelectorAll<HTMLElement>("[data-reveal]"));
    if (nodes.length === 0) {
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            entry.target.classList.add("in-view");
            observer.unobserve(entry.target);
          }
        }
      },
      { threshold: 0.18 }
    );

    for (const node of nodes) {
      observer.observe(node);
    }

    return () => observer.disconnect();
  }, []);

  const modelRows = result?.model_results ?? [];
  const isFake = result?.label?.toLowerCase() === "fake";
  const isAnalysisLocked = !file;

  async function runAnalysisForFile(uploadedFile: File) {
    const detectedType = inferMediaType(uploadedFile);
    if (!detectedType) {
      setError("Unsupported media type. Use JPG, PNG, MP4, MOV, WAV, or MP3.");
      return;
    }

    setError(null);
    setIsAnalyzing(true);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", uploadedFile);

      const response = await fetch(`${API_BASE_URL}/detect?media_type=${detectedType}`, {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        let detail = "Analysis failed.";
        try {
          const data = (await response.json()) as { detail?: string };
          if (data.detail) {
            detail = data.detail;
          }
        } catch {
          // no-op
        }
        throw new Error(detail);
      }

      const data = (await response.json()) as DetectionResponse;
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to analyze media.");
    } finally {
      setIsAnalyzing(false);
    }
  }

  function moveToResultSection() {
    const productSection = document.getElementById("product");
    if (productSection) {
      productSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function onDropFile(droppedFile: File) {
    setFile(droppedFile);
    setResult(null);
    setError(null);
    moveToResultSection();
    void runAnalysisForFile(droppedFile);
  }

  function downloadReport() {
    if (!result) {
      return;
    }
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "deepfake-report.json";
    anchor.click();
    URL.revokeObjectURL(url);
  }

  function applyInteractiveGlow(event: React.MouseEvent<HTMLElement>) {
    const el = event.currentTarget;
    const rect = el.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * 100;
    const y = ((event.clientY - rect.top) / rect.height) * 100;
    el.style.setProperty("--px", `${x}%`);
    el.style.setProperty("--py", `${y}%`);
  }

  function resetInteractiveGlow(event: React.MouseEvent<HTMLElement>) {
    event.currentTarget.style.setProperty("--px", "50%");
    event.currentTarget.style.setProperty("--py", "50%");
  }

  return (
    <main
      ref={mainRef}
      className="relative"
      style={
        {
          "--mx": `${cursor.x}%`,
          "--my": `${cursor.y}%`
        } as React.CSSProperties
      }
    >
      <div className="page-grid" />
      <div className="cursor-orb" />
      <div className="ambient-glow bg-purple-600/20 w-[600px] h-[600px] top-[-200px] left-[-100px]" />
      <div
        className="ambient-glow bg-cyan-600/10 w-[800px] h-[800px] top-[20%] right-[-200px]"
        style={{ animationDelay: "-4s" }}
      />

      <nav className="fixed top-0 w-full z-50 glass-panel border-b-0 border-white/5 reveal in-view">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="text-xl font-semibold tracking-tighter text-white">EAI</div>
          <div className="hidden md:flex items-center gap-8 text-sm font-medium text-slate-400">
            <a href="#product" className="hover:text-white transition-colors">
              Product
            </a>
            <a href="#how-it-works" className="hover:text-white transition-colors">
              How it Works
            </a>
            <a href="#pricing" className="hover:text-white transition-colors">
              Pricing
            </a>
          </div>
          <a
            href="#demo"
            className="magnetic-btn px-4 py-2 rounded-full bg-white text-slate-950 text-sm font-medium hover:bg-slate-200 transition-colors"
          >
            Try Detector
          </a>
        </div>
      </nav>

      <section className="relative pt-40 pb-20 px-6 min-h-screen flex flex-col justify-center overflow-hidden">
        <div className="max-w-5xl mx-auto text-center z-10">
          <div
            data-reveal
            className="reveal inline-flex items-center gap-2 px-3 py-1 rounded-full border border-cyan-500/30 bg-cyan-500/10 text-cyan-400 text-xs font-medium mb-8"
          >
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500" />
            </span>
            Model v4.2 now available
          </div>

          <h1 data-reveal className="reveal delay-1 text-5xl md:text-7xl font-semibold tracking-tight text-white mb-6 leading-tight">
            Detect Deepfakes Instantly
            <br className="hidden md:block" />
            <span className="bg-gradient-to-r from-purple-400 to-cyan-400 gradient-text">
              with Ethical AI
            </span>
          </h1>

          <p data-reveal className="reveal delay-2 text-lg md:text-xl text-slate-400 max-w-2xl mx-auto mb-10 font-light">
            Upload an image, video, or audio file and let our neural networks analyze authenticity in
            milliseconds.
          </p>

          <div id="demo" data-reveal className="reveal delay-3 max-w-3xl mx-auto w-full group">
            <div
              role="button"
              tabIndex={0}
              onMouseMove={applyInteractiveGlow}
              onMouseLeave={resetInteractiveGlow}
              onClick={() => inputRef.current?.click()}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  inputRef.current?.click();
                }
              }}
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={(e) => {
                e.preventDefault();
                setIsDragging(false);
              }}
              onDrop={(e) => {
                e.preventDefault();
                setIsDragging(false);
                const dropped = e.dataTransfer.files?.[0];
                if (dropped) {
                  onDropFile(dropped);
                }
              }}
              className={`interactive-card relative rounded-2xl border-2 border-dashed bg-slate-900/40 backdrop-blur-sm p-12 transition-all cursor-pointer overflow-hidden ${
                isDragging
                  ? "border-cyan-400 bg-slate-900/70"
                  : "border-slate-700/50 group-hover:border-cyan-500/50 group-hover:bg-slate-900/60"
              }`}
            >
              <input
                ref={inputRef}
                type="file"
                accept=".png,.jpg,.jpeg,.bmp,.webp,.mp4,.mov,.avi,.mkv,.webm,.wav,.mp3,.flac,.ogg,.m4a"
                className="hidden"
                onChange={(e) => {
                  const selected = e.target.files?.[0];
                  if (selected) {
                    onDropFile(selected);
                  }
                }}
              />
              <div className="absolute inset-0 bg-gradient-to-b from-cyan-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
              <div className="relative z-10 flex flex-col items-center justify-center gap-4">
                <div className="w-16 h-16 rounded-full bg-slate-800 flex items-center justify-center group-hover:scale-110 transition-transform duration-300 shadow-xl border border-white/5">
                  <Icon icon="solar:cloud-upload-linear" width={32} className="text-cyan-400" />
                </div>
                <h3 className="text-xl font-medium text-white tracking-tight">Drag and drop media here</h3>
                <p className="text-sm text-slate-400">Supports JPG, PNG, MP4, MOV, WAV, MP3 up to 50MB</p>
                <div className="magnetic-btn mt-2 px-6 py-2 rounded-full bg-white/5 text-sm font-medium text-white border border-white/10 group-hover:bg-white/10 transition-colors">
                  Browse Files
                </div>
                {file ? (
                  <div className="text-sm text-cyan-300 mt-2">
                    Selected: <span className="font-mono">{file.name}</span>
                  </div>
                ) : null}
              </div>
            </div>

            <div className="mt-6 flex flex-col items-center gap-3">
              {isAnalyzing ? (
                <p className="text-sm text-cyan-300">Analyzing your uploaded file...</p>
              ) : file ? (
                <p className="text-sm text-slate-300">File uploaded. Detection started automatically.</p>
              ) : (
                <p className="text-sm text-slate-400">
                  Upload a file to automatically start deepfake detection.
                </p>
              )}
              {error ? <p className="text-sm text-red-300">{error}</p> : null}
              <p className="text-xs text-slate-500">
                Backend: <span className="font-mono">{API_BASE_URL}</span>
              </p>
            </div>
          </div>
        </div>
      </section>

      <section id="product" data-reveal className="reveal py-24 px-6 relative z-10">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-semibold tracking-tight text-white mb-4">
              Real-Time Analysis Interface
            </h2>
            <p className="text-base text-slate-400 max-w-2xl mx-auto">
              Upload and analyze media using the Python detection API, then inspect model-level scores.
            </p>
          </div>

          <div className="relative">
            <div
              className={`grid lg:grid-cols-5 gap-8 transition-all duration-300 ${
                isAnalysisLocked ? "blur-[3px] opacity-50 pointer-events-none select-none" : ""
              }`}
            >
            <div
              onMouseMove={applyInteractiveGlow}
              onMouseLeave={resetInteractiveGlow}
              className="interactive-card lg:col-span-3 glass-panel rounded-2xl p-2 relative overflow-hidden flex flex-col h-full"
            >
              <div className="flex items-center justify-between px-4 py-3 border-b border-white/5 mb-2">
                <span className="text-xs font-mono text-slate-400">
                  {file?.name ?? "No file uploaded"}
                </span>
                <div className="flex items-center gap-2">
                  <div
                    className={`w-2 h-2 rounded-full ${
                      isAnalyzing ? "bg-cyan-500 animate-pulse" : "bg-slate-500"
                    }`}
                  />
                  <span className="text-xs font-medium text-cyan-400 uppercase tracking-widest">
                    {isAnalyzing ? "Analyzing" : result ? "Completed" : "Standby"}
                  </span>
                </div>
              </div>
              <div className="relative bg-slate-950 rounded-xl overflow-hidden flex-1 min-h-[320px] md:min-h-[420px]">
                {previewUrl && mediaType === "image" ? (
                  <>
                    <img
                      src={previewUrl}
                      alt=""
                      aria-hidden="true"
                      className="absolute inset-0 w-full h-full object-cover object-center opacity-35 blur-2xl scale-110"
                    />
                    <img
                      src={previewUrl}
                      alt="Uploaded preview"
                      className="absolute inset-0 w-full h-full object-contain object-center opacity-95"
                    />
                  </>
                ) : null}
                {previewUrl && mediaType === "video" ? (
                  <video
                    src={previewUrl}
                    controls
                    className="absolute inset-0 w-full h-full object-contain opacity-95 bg-slate-950"
                  />
                ) : null}
                {previewUrl && mediaType === "audio" ? (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="glass-panel rounded-2xl p-6 border border-white/10 text-center">
                      <Icon icon="solar:music-note-2-linear" width={42} className="text-cyan-300 mx-auto mb-3" />
                      <p className="text-sm text-slate-200 mb-4">Audio preview loaded</p>
                      <audio src={previewUrl} controls />
                    </div>
                  </div>
                ) : null}
                {!previewUrl ? (
                  <div
                    className="absolute inset-0 opacity-40 mix-blend-luminosity bg-cover bg-center"
                    style={{
                      backgroundImage:
                        "url('https://images.unsplash.com/photo-1544717305-2782549b5136?q=80&w=1000&auto=format&fit=crop')"
                    }}
                  />
                ) : null}
                <div className="absolute inset-0 bg-gradient-to-tr from-red-500/20 via-transparent to-transparent mix-blend-overlay" />
                {isAnalyzing ? <div className="scan-line" /> : null}
              </div>
            </div>

            <div
              onMouseMove={applyInteractiveGlow}
              onMouseLeave={resetInteractiveGlow}
              className="interactive-card lg:col-span-2 glass-panel rounded-2xl p-6 flex flex-col"
            >
              <div className="mb-8">
                <h3 className="text-sm font-medium text-slate-400 mb-2">Overall Verdict</h3>
                <div className="flex items-end gap-3">
                  <span
                    className={`text-4xl font-semibold tracking-tight ${
                      isFake ? "text-red-400" : "text-cyan-300"
                    }`}
                  >
                  {!hasResult ? (isAnalyzing ? "Analyzing..." : "Awaiting Analysis") : isFake ? "Deepfake" : "Likely Real"}
                  </span>
                  <span className="text-base text-slate-500 mb-1">
                    {!hasResult ? (isAnalyzing ? "Processing" : "No result yet") : "Detected"}
                  </span>
                </div>
              </div>

              <div
                className={`mb-8 p-4 rounded-xl border ${
                  !hasResult
                    ? "bg-slate-500/5 border-slate-500/20"
                    : isFake
                      ? "bg-red-500/5 border-red-500/20"
                      : "bg-cyan-500/5 border-cyan-500/20"
                }`}
              >
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium text-white">Confidence Score</span>
                  <span
                    className={`text-sm font-mono ${
                      !hasResult ? "text-slate-400" : isFake ? "text-red-400" : "text-cyan-300"
                    }`}
                  >
                    {hasResult && result ? percent(result.confidence) : "--"}
                  </span>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-1.5">
                  <div
                    className={`h-1.5 rounded-full ${
                      !hasResult
                        ? "bg-slate-500"
                        : isFake
                          ? "bg-gradient-to-r from-orange-500 to-red-500"
                          : "bg-gradient-to-r from-cyan-500 to-blue-500"
                    }`}
                    style={{ width: `${hasResult && result ? Math.max(3, result.confidence * 100) : 0}%` }}
                  />
                </div>
              </div>

              <h4 className="text-sm font-medium text-slate-300 mb-4 pb-2 border-b border-white/5">
                Detection Breakdown
              </h4>

              {!hasResult ? (
                isAnalyzing ? (
                  <div className="rounded-xl border border-cyan-500/20 bg-cyan-500/[0.04] px-4 py-5">
                    <div className="flex items-center justify-center gap-2 mb-4">
                      <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse" />
                      <p className="text-sm text-cyan-300 font-medium">Running deepfake analysis...</p>
                    </div>
                    <div className="space-y-3">
                      <div className="h-2 rounded bg-slate-800 overflow-hidden">
                        <div className="h-full w-1/2 bg-gradient-to-r from-cyan-500 to-blue-500 animate-pulse" />
                      </div>
                      <div className="h-2 rounded bg-slate-800 overflow-hidden">
                        <div className="h-full w-2/3 bg-gradient-to-r from-cyan-500 to-blue-500 animate-pulse" />
                      </div>
                      <div className="h-2 rounded bg-slate-800 overflow-hidden">
                        <div className="h-full w-1/3 bg-gradient-to-r from-cyan-500 to-blue-500 animate-pulse" />
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="rounded-xl border border-white/5 bg-white/[0.02] px-4 py-6 text-center">
                    <p className="text-sm text-slate-400">Upload media to view the detection breakdown.</p>
                  </div>
                )
              ) : (
                <>
                  <div className="space-y-4">
                    {modelRows.map((row, index) => {
                      const modelValue = Math.max(0, Math.min(100, row.fake_probability * 100));
                      const warn = modelValue >= 50;
                      const prediction = modelPredictionLabel(modelValue);
                      const rowIcons = [
                        "solar:eye-linear",
                        "solar:magic-stick-3-linear",
                        "solar:volume-knob-linear",
                        "solar:shield-warning-linear",
                        "solar:chart-square-linear"
                      ];
                      return (
                        <div key={`${row.model_name}-${index}`}>
                          <div className="flex justify-between items-start mb-1 gap-3">
                            <div className="min-w-0">
                              <span className="text-xs text-slate-300 flex items-center gap-2">
                                <Icon icon={rowIcons[index % rowIcons.length]} width={14} />
                                <span className="truncate">{prettyModelName(row.model_name)}</span>
                              </span>
                              <p
                                className={`text-[11px] mt-1 ${prediction.textClass}`}
                              >
                                {prediction.label}
                              </p>
                            </div>
                            <span className={`text-xs font-mono ${warn ? "text-red-400" : "text-cyan-400"}`}>
                              {modelValue.toFixed(1)}%
                            </span>
                          </div>
                          <div className="w-full bg-slate-800 rounded-full h-1">
                            <div
                              className={`h-1 rounded-full ${prediction.barClass}`}
                              style={{ width: `${Math.max(2, modelValue)}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </>
              )}

              <button
                onClick={downloadReport}
                disabled={!result}
                className="mt-6 w-full py-3 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-sm font-medium text-white transition-colors flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Icon icon="solar:document-download-linear" width={18} />
                Download Full Report
              </button>
            </div>
            </div>
            {isAnalysisLocked ? (
              <div className="absolute inset-0 z-20 flex items-center justify-center px-6">
                <div className="glass-panel rounded-2xl border border-white/10 px-6 py-5 text-center max-w-md">
                  <p className="text-base text-white font-medium">Upload a file to unlock detection.</p>
                  <p className="text-sm text-slate-300 mt-2">
                    I can&apos;t run deepfake detection without an uploaded file.
                  </p>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      </section>

      <section id="how-it-works" data-reveal className="reveal py-24 px-6 border-y border-white/5 bg-slate-900/20">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-semibold tracking-tight text-white mb-4">How It Works</h2>
            <p className="text-base text-slate-400">A seamless process from upload to verifiable result.</p>
          </div>
          <div className="grid md:grid-cols-4 gap-8 relative">
            <div className="hidden md:block absolute top-1/2 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-y-1/2 z-0" />
            {[
              {
                icon: "solar:cloud-upload-linear",
                title: "Upload Media",
                body: "Securely submit images, videos, or audio in one drag-and-drop action."
              },
              {
                icon: "solar:scanner-linear",
                title: "Feature Extraction",
                body: "AI scans for biometric inconsistencies and rendering artifacts."
              },
              {
                icon: "solar:cpu-linear",
                title: "Neural Analysis",
                body: "Python ensemble models evaluate content against deepfake signatures."
              },
              {
                icon: "solar:file-check-linear",
                title: "Authenticity Report",
                body: "Receive confidence scores and model-level breakdown instantly."
              }
            ].map((item, idx) => (
              <div key={item.title} className="relative z-10 flex flex-col items-center text-center reveal" data-reveal>
                <div className="w-16 h-16 rounded-2xl glass-panel flex items-center justify-center mb-6 shadow-lg shadow-black/50 border border-white/10">
                  <Icon icon={item.icon} width={28} className="text-slate-300" />
                </div>
                <span className="text-xs font-mono text-cyan-400 mb-2">{`0${idx + 1}`}</span>
                <h3 className="text-base font-medium text-white mb-2">{item.title}</h3>
                <p className="text-sm text-slate-400">{item.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section data-reveal className="reveal py-24 px-6 relative">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-semibold tracking-tight text-white mb-12 text-center">
            Enterprise-Grade Detection
          </h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[
              ["solar:face-scan-linear", "Advanced Face Analysis", "text-blue-400", "bg-blue-500/10"],
              ["solar:video-frame-linear", "Frame-by-Frame Video", "text-purple-400", "bg-purple-500/10"],
              ["solar:code-file-linear", "Metadata Analysis", "text-cyan-400", "bg-cyan-500/10"],
              ["solar:chart-square-linear", "Confidence Scoring", "text-orange-400", "bg-orange-500/10"],
              ["solar:bolt-linear", "Fast Processing", "text-green-400", "bg-green-500/10"],
              ["solar:shield-check-linear", "Privacy Protection", "text-slate-400", "bg-slate-500/10"]
            ].map(([icon, title, tone, bg]) => (
              <div
                key={title}
                onMouseMove={applyInteractiveGlow}
                onMouseLeave={resetInteractiveGlow}
                className="interactive-card glass-panel p-6 rounded-2xl hover:bg-white/5 transition-colors border border-white/5 hover:border-white/10 group reveal"
                data-reveal
              >
                <div
                  className={`float-soft w-10 h-10 rounded-lg ${bg} flex items-center justify-center mb-4 ${tone} group-hover:scale-110 transition-transform`}
                >
                  <Icon icon={icon} width={24} />
                </div>
                <h3 className="text-base font-medium text-white mb-2">{title}</h3>
                <p className="text-sm text-slate-400">
                  Multi-model analysis detects synthetic artifacts with probabilistic scoring for better decisions.
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section id="pricing" data-reveal className="reveal py-24 px-6 bg-gradient-to-b from-transparent to-slate-900/50 border-t border-white/5">
        <div className="max-w-6xl mx-auto grid md:grid-cols-2 gap-16 items-center">
          <div>
            <h2 className="text-3xl font-semibold tracking-tight text-white mb-6">
              The Threat Landscape is Changing
            </h2>
            <p className="text-base text-slate-400 mb-6">
              Generative AI can produce realistic fake media at scale. Verification now needs to happen in real time.
            </p>
            <p className="text-base text-slate-400 mb-8">
              Ethical AI provides a trust layer powered by Python inference and a modern Next.js interface.
            </p>
            <div className="flex items-center gap-4 text-sm font-medium text-white">
              <Icon icon="solar:verified-check-linear" className="text-cyan-400" width={20} />
              Trusted by top media organizations
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="glass-panel p-6 rounded-2xl text-center border-t border-l border-white/10 bg-gradient-to-br from-white/5 to-transparent">
              <div className="text-4xl font-semibold tracking-tight text-cyan-400 mb-2">900%</div>
              <div className="text-xs text-slate-400 uppercase tracking-wider">Increase in deepfakes YoY</div>
            </div>
            <div className="glass-panel p-6 rounded-2xl text-center border-t border-l border-white/10 bg-gradient-to-br from-white/5 to-transparent mt-8">
              <div className="text-4xl font-semibold tracking-tight text-purple-400 mb-2">
                {result ? percent(result.confidence) : "99.2%"}
              </div>
              <div className="text-xs text-slate-400 uppercase tracking-wider">Detection Confidence</div>
            </div>
            <div className="glass-panel p-6 rounded-2xl text-center border-t border-l border-white/10 bg-gradient-to-br from-white/5 to-transparent -mt-8">
              <div className="text-4xl font-semibold tracking-tight text-blue-400 mb-2">&lt;1s</div>
              <div className="text-xs text-slate-400 uppercase tracking-wider">Average response time</div>
            </div>
            <div className="glass-panel p-6 rounded-2xl text-center border-t border-l border-white/10 bg-gradient-to-br from-white/5 to-transparent">
              <div className="text-4xl font-semibold tracking-tight text-white mb-2">0</div>
              <div className="text-xs text-slate-400 uppercase tracking-wider">Media files stored</div>
            </div>
          </div>
        </div>
      </section>

      <section data-reveal className="reveal py-20 px-6 border-t border-white/5">
        <div className="max-w-4xl mx-auto text-center">
          <div className="w-16 h-16 mx-auto rounded-full bg-slate-900 border border-white/10 flex items-center justify-center mb-6 shadow-[0_0_30px_rgba(6,182,212,0.15)]">
            <Icon icon="solar:lock-keyhole-linear" width={28} className="text-cyan-400" />
          </div>
          <h2 className="text-2xl font-semibold tracking-tight text-white mb-4">Zero Retention Privacy</h2>
          <p className="text-base text-slate-400">
            Uploaded media is sent to the Python API for inference and removed immediately after processing.
          </p>
        </div>
      </section>

      <section data-reveal className="reveal py-24 px-6 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-t from-blue-900/20 to-transparent" />
        <div className="max-w-4xl mx-auto text-center relative z-10 p-12 rounded-3xl glass-panel border border-white/10 bg-slate-900/50">
          <h2 className="text-4xl md:text-5xl font-semibold tracking-tight text-white mb-6">
            Verify Content Before You Believe It.
          </h2>
          <p className="text-lg text-slate-400 mb-10 max-w-2xl mx-auto">
            Join leading platforms in the fight against misinformation. Start using Ethical AI today.
          </p>
          <a
            href="#demo"
            className="magnetic-btn inline-flex px-8 py-3 rounded-full bg-white text-slate-950 text-base font-medium hover:bg-slate-200 transition-colors"
          >
            Try Detector Now
          </a>
        </div>
      </section>

      <footer data-reveal className="reveal border-t border-white/5 bg-slate-950 pt-16 pb-8 px-6">
        <div className="max-w-7xl mx-auto grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-8 mb-12">
          <div className="col-span-2 lg:col-span-2">
            <span className="text-xl font-semibold tracking-tighter text-white mb-4 block">EAI</span>
            <p className="text-sm text-slate-400 max-w-xs mb-6">
              Building the trust layer for the generative AI era with advanced deepfake detection models.
            </p>
            <div className="flex items-center gap-4 text-slate-400">
              <a href="#" className="hover:text-white transition-colors">
                <Icon icon="solar:twitter-linear" width={20} />
              </a>
              <a href="#" className="hover:text-white transition-colors">
                <Icon icon="solar:github-linear" width={20} />
              </a>
            </div>
          </div>
          <div>
            <h4 className="text-sm font-medium text-white mb-4">Product</h4>
            <ul className="space-y-3 text-sm text-slate-400">
              <li>
                <a href="#demo" className="hover:text-white transition-colors">
                  Detector Web App
                </a>
              </li>
              <li>
                <a href="#pricing" className="hover:text-white transition-colors">
                  Pricing
                </a>
              </li>
            </ul>
          </div>
          <div>
            <h4 className="text-sm font-medium text-white mb-4">Company</h4>
            <ul className="space-y-3 text-sm text-slate-400">
              <li>
                <a href="#" className="hover:text-white transition-colors">
                  About
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-white transition-colors">
                  Contact
                </a>
              </li>
            </ul>
          </div>
        </div>
        <div className="max-w-7xl mx-auto pt-8 border-t border-white/5 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-xs text-slate-500">© 2026 Ethical AI Inc. All rights reserved.</p>
          <div className="flex items-center gap-6 text-xs text-slate-500">
            <a href="#" className="hover:text-white transition-colors">
              Privacy Policy
            </a>
            <a href="#" className="hover:text-white transition-colors">
              Terms of Service
            </a>
            <a href="#" className="hover:text-white transition-colors">
              Cookie Policy
            </a>
          </div>
        </div>
      </footer>
    </main>
  );
}
