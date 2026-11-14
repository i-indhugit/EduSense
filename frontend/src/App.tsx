import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Header } from './components/Header';
import { MetricsGrid } from './components/MetricsGrid';
import { LiveClassroomView, DATASET_SAMPLES, DatasetSample, MediaSourceMode } from './components/LiveClassroomView';
import { TeacherRecommendations } from './components/TeacherRecommendations';
import { SpatialHeatmap } from './components/SpatialHeatmap';
import { AnalyticsCharts } from './components/AnalyticsCharts';
import { DebugPanel } from './components/DebugPanel';
import { StudentDetailModal } from './components/StudentDetailModal';
import { SingleStudentView } from './components/SingleStudentView';
import { GeminiInsightsPanel } from './components/GeminiInsightsPanel';
import { ToastContainer, ToastMessage } from './components/common/ToastContainer';
import { BrowserAiDiagnostics, BrowserAiPipeline, drawOverlayBoxes } from './ai/detector';
import { API_BASE_URL, getApiBaseUrl } from './config';
import { ClassroomMetrics, StudentAnalytics, TimelinePoint } from './types/analytics';
import { LayoutGrid, User, RefreshCw, Video, Cpu, Database } from 'lucide-react';

const initialMetrics: ClassroomMetrics = {
  eci_score: 0.0,
  eci_status: 'No Face Detected',
  avg_attention: 0.0,
  avg_engagement: 0.0,
  avg_participation: 0.0,
  avg_confusion: 0.0,
  drowsy_count: 0,
  hand_raise_count: 0,
  live_detected_persons: 0,
  valid_face_count: 0,
  active_tracks: 0,
  camera_connected: false,
  camera_source: 'BROWSER_MEDIAPIPE_AI',
  performance: { fps: 0.0, latency_ms: 0.0 },
  students: [],
  recommendations: [],
};

export function App() {
  const [metrics, setMetrics] = useState<ClassroomMetrics>(initialMetrics);
  const [backendOnline, setBackendOnline] = useState<boolean>(false);
  const [cameraConsentGranted, setCameraConsentGranted] = useState<boolean>(true);
  const [cameraFacingMode, setCameraFacingMode] = useState<'user' | 'environment'>('user');
  const [browserCamActive, setBrowserCamActive] = useState<boolean>(false);
  const [mediaSourceMode, setMediaSourceMode] = useState<MediaSourceMode>('camera');
  const [activeDatasetSampleId, setActiveDatasetSampleId] = useState<string>('phone-distraction');
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [isVideoUploaded, setIsVideoUploaded] = useState<boolean>(false);
  const [isMediaActive, setIsMediaActive] = useState<boolean>(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [cameraRetryTick, setCameraRetryTick] = useState<number>(0);
  const [modelRetryTick, setModelRetryTick] = useState<number>(0);
  const [showOverlays, setShowOverlays] = useState<boolean>(true);
  const [selectedStudentId, setSelectedStudentId] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'classroom' | 'focused'>('classroom');
  const [focusedStudentId, setFocusedStudentId] = useState<string | null>(null);
  const [timelineHistory, setTimelineHistory] = useState<TimelinePoint[]>([]);
  const [toasts, setToasts] = useState<ToastMessage[]>([]);
  const [modelDiag, setModelDiag] = useState<BrowserAiDiagnostics>({
    status: 'idle',
    delegate: 'GPU',
    error: null,
    loadedModels: [],
    browserSupported: typeof WebAssembly === 'object',
    memoryMb: null,
  });

  const lastTimelinePushRef = useRef<number>(0);
  const lastBackendSyncRef = useRef<number>(0);
  const lastUiUpdateRef = useRef<number>(0);
  const sessionIdRef = useRef<string>(`session-${Date.now()}`);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const imageCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const currentImageRef = useRef<HTMLImageElement | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const pipelineRef = useRef<BrowserAiPipeline>(new BrowserAiPipeline());

  const addToast = useCallback((toast: Omit<ToastMessage, 'id'>) => {
    const id = `toast-${Date.now()}-${Math.random().toString().slice(2, 6)}`;
    setToasts((prev) => [...prev, { ...toast, id }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4500);
  }, []);

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const applyAnalyticsUpdate = useCallback((analytics: ClassroomMetrics) => {
    setMetrics(analytics);

    const now = Date.now();
    if (now - lastTimelinePushRef.current >= 2000 && (analytics.valid_face_count ?? 0) > 0) {
      lastTimelinePushRef.current = now;
      const label = new Date().toLocaleTimeString([], {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      });
      setTimelineHistory((prev) => [
        ...prev.slice(-29),
        {
          time: label,
          attention: analytics.avg_attention ?? 0,
          engagement: analytics.avg_engagement ?? 0,
          confusion: analytics.avg_confusion ?? 0,
        },
      ]);
    }

    // Periodically send ONLY aggregated anonymized metrics (never raw frames or biometrics)
    if (now - lastBackendSyncRef.current >= 4000) {
      lastBackendSyncRef.current = now;
      fetch(`${getApiBaseUrl()}/api/analytics`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionIdRef.current,
          timestamp: new Date().toISOString(),
          eci_score: analytics.eci_score,
          eci_status: analytics.eci_status,
          avg_attention: analytics.avg_attention,
          avg_engagement: analytics.avg_engagement,
          avg_participation: analytics.avg_participation,
          avg_confusion: analytics.avg_confusion,
          drowsy_count: analytics.drowsy_count,
          hand_raise_count: analytics.hand_raise_count,
          live_detected_persons: analytics.live_detected_persons,
          valid_face_count: analytics.valid_face_count,
          active_tracks: analytics.active_tracks,
          fps: analytics.performance?.fps ?? 0,
          latency_ms: analytics.performance?.latency_ms ?? 0,
        }),
      })
        .then((res) => {
          if (res.ok) setBackendOnline(true);
        })
        .catch(() => {
          setBackendOnline(false);
        });
    }
  }, []);

  // Initialize Browser MediaPipe AI Models
  useEffect(() => {
    let active = true;
    const pipeline = pipelineRef.current;

    pipeline.initialize((diag) => {
      if (active) setModelDiag({ ...diag });
    }).then((ok) => {
      if (!active) return;
      if (ok) {
        addToast({
          type: 'success',
          title: 'Browser AI Ready',
          message: `MediaPipe Tasks loaded (${pipeline.diagnostics.delegate} delegate).`,
        });
      } else {
        addToast({
          type: 'error',
          title: 'Model Load Failed',
          message: pipeline.diagnostics.error || 'Could not initialize browser AI models.',
        });
      }
    });

    return () => {
      active = false;
    };
  }, [modelRetryTick, addToast]);

  // Cleanup models on unmount
  useEffect(() => {
    const pipeline = pipelineRef.current;
    return () => {
      pipeline.dispose();
    };
  }, []);

  // Browser MediaDevices getUserMedia camera management
  const stopBrowserCamera = useCallback(() => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((tr) => tr.stop());
      mediaStreamRef.current = null;
    }
    setBrowserCamActive(false);
  }, []);

  useEffect(() => {
    let cancelled = false;
    setCameraError(null);

    if (!cameraConsentGranted || mediaSourceMode !== 'camera') {
      stopBrowserCamera();
      return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      const msg = 'Browser camera access requires HTTPS or localhost.';
      setCameraError(msg);
      return;
    }

    const startCamera = async () => {
      const constraintsList: MediaStreamConstraints[] = [
        {
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: cameraFacingMode,
          },
          audio: false,
        },
        {
          video: {
            facingMode: cameraFacingMode,
          },
          audio: false,
        },
        {
          video: true,
          audio: false,
        },
      ];

      for (const constraints of constraintsList) {
        if (cancelled) return;
        try {
          const stream = await navigator.mediaDevices.getUserMedia(constraints);
          if (cancelled) {
            stream.getTracks().forEach((t) => t.stop());
            return;
          }
          mediaStreamRef.current = stream;
          stream.getVideoTracks().forEach((track) => {
            track.onended = () => {
              setBrowserCamActive(false);
              setCameraError('Camera disconnected. Tap Retry Camera or test with open dataset samples.');
              setMetrics((prev) => ({
                ...prev,
                camera_connected: false,
                live_detected_persons: 0,
                valid_face_count: 0,
                active_tracks: 0,
                students: [],
              }));
            };
          });
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            videoRef.current.play().catch(() => {});
          }
          setCameraError(null);
          setBrowserCamActive(true);
          return;
        } catch (err: any) {
          const errName = err?.name || '';
          if (errName === 'NotAllowedError' || errName === 'PermissionDeniedError') {
            setBrowserCamActive(false);
            setCameraError('Camera permission denied. Allow camera access in your phone/browser settings, or test with open dataset samples.');
            return;
          }
          // Continue to next constraint tier for mobile camera driver compatibility
        }
      }

      setBrowserCamActive(false);
      setCameraError('Could not start camera with requested constraints. Connect a camera, tap Retry, or test with open dataset samples.');
    };

    startCamera();

    return () => {
      cancelled = true;
      stopBrowserCamera();
    };
  }, [cameraRetryTick, cameraConsentGranted, cameraFacingMode, mediaSourceMode, stopBrowserCamera]);

  // Dataset Sample & Custom Upload Actions
  const handleSelectDatasetSample = useCallback(
    (sample: DatasetSample) => {
      stopBrowserCamera();
      setMediaSourceMode('dataset');
      setActiveDatasetSampleId(sample.id);
      setUploadedFileName(null);
      setIsVideoUploaded(false);

      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.src = sample.url;
      img.onload = () => {
        currentImageRef.current = img;
        const canvas = imageCanvasRef.current;
        if (canvas) {
          canvas.width = img.naturalWidth || 640;
          canvas.height = img.naturalHeight || 480;
          const ctx = canvas.getContext('2d');
          ctx?.drawImage(img, 0, 0);
        }
        setIsMediaActive(true);

        // Immediate single inference pass
        if (pipelineRef.current.diagnostics.status === 'ready' && canvas) {
          const res = pipelineRef.current.processVideoFrame(canvas, performance.now(), true);
          if (res) {
            if (overlayCanvasRef.current) {
              drawOverlayBoxes(overlayCanvasRef.current, canvas.width, canvas.height, res.students || []);
            }
            applyAnalyticsUpdate(res);
          }
        }
      };
    },
    [stopBrowserCamera, applyAnalyticsUpdate]
  );

  const handleFileUpload = useCallback(
    (file: File) => {
      stopBrowserCamera();
      setMediaSourceMode('upload');
      setUploadedFileName(file.name);

      if (file.type.startsWith('video/')) {
        setIsVideoUploaded(true);
        currentImageRef.current = null;
        const url = URL.createObjectURL(file);
        if (videoRef.current) {
          videoRef.current.srcObject = null;
          videoRef.current.src = url;
          videoRef.current.loop = true;
          videoRef.current.play().catch(() => {});
        }
        setIsMediaActive(true);
      } else {
        setIsVideoUploaded(false);
        const url = URL.createObjectURL(file);
        const img = new Image();
        img.src = url;
        img.onload = () => {
          currentImageRef.current = img;
          const canvas = imageCanvasRef.current;
          if (canvas) {
            canvas.width = img.naturalWidth || 640;
            canvas.height = img.naturalHeight || 480;
            const ctx = canvas.getContext('2d');
            ctx?.drawImage(img, 0, 0);
          }
          setIsMediaActive(true);

          if (pipelineRef.current.diagnostics.status === 'ready' && canvas) {
            const res = pipelineRef.current.processVideoFrame(canvas, performance.now(), true);
            if (res) {
              if (overlayCanvasRef.current) {
                drawOverlayBoxes(overlayCanvasRef.current, canvas.width, canvas.height, res.students || []);
              }
              applyAnalyticsUpdate(res);
            }
          }
        };
      }
    },
    [stopBrowserCamera, applyAnalyticsUpdate]
  );

  const handleSelectMediaSourceMode = useCallback(
    (mode: MediaSourceMode) => {
      if (mode === 'camera') {
        currentImageRef.current = null;
        setIsVideoUploaded(false);
        setIsMediaActive(false);
        setMediaSourceMode('camera');
        setCameraConsentGranted(true);
        setCameraRetryTick((t) => t + 1);
      } else if (mode === 'dataset') {
        const initial = DATASET_SAMPLES.find((s) => s.id === activeDatasetSampleId) || DATASET_SAMPLES[0];
        handleSelectDatasetSample(initial);
      } else {
        setIsMediaActive(false);
        setMediaSourceMode('upload');
      }
    },
    [activeDatasetSampleId, handleSelectDatasetSample]
  );

  // Browser-side real-time AI inference loop (throttled ~10 FPS for CPU/memory efficiency)
  useEffect(() => {
    const isRunning = browserCamActive || isMediaActive;
    if (!isRunning || modelDiag.status !== 'ready') return;

    let rafId = 0;
    let lastRun = 0;

    const tick = (now: number) => {
      rafId = requestAnimationFrame(tick);
      if (now - lastRun < 100) return; // ~10 FPS AI inference
      lastRun = now;

      const isImageSource = mediaSourceMode === 'dataset' || (mediaSourceMode === 'upload' && !isVideoUploaded);
      const canvas = overlayCanvasRef.current;
      let source: HTMLVideoElement | HTMLCanvasElement | null = null;
      let srcW = 0;
      let srcH = 0;

      if (isImageSource) {
        source = imageCanvasRef.current;
        if (source) {
          srcW = source.width;
          srcH = source.height;
          if (currentImageRef.current) {
            const ctx = source.getContext('2d');
            ctx?.drawImage(currentImageRef.current, 0, 0);
          }
        }
      } else {
        const vid = videoRef.current;
        if (vid && vid.readyState >= 2) {
          source = vid;
          srcW = vid.videoWidth;
          srcH = vid.videoHeight;
        }
      }

      if (!source || srcW === 0 || srcH === 0) return;

      const result = pipelineRef.current.processVideoFrame(source, now, true);
      if (result) {
        if (canvas) {
          drawOverlayBoxes(canvas, srcW, srcH, result.students || []);
        }

        // Throttle React state updates to ~3 times a second (300ms)
        // Completely eliminates DOM thrashing, card resizing, and UI shaking!
        if (now - lastUiUpdateRef.current >= 300) {
          lastUiUpdateRef.current = now;
          applyAnalyticsUpdate(result);
        }
      }
    };

    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [browserCamActive, isMediaActive, modelDiag.status, mediaSourceMode, isVideoUploaded, applyAnalyticsUpdate]);

  // Lightweight backend health check (handles Render free-tier sleeping/cold starts)
  const checkBackendHealth = useCallback(() => {
    fetch(`${API_BASE_URL}/health`)
      .then((res) => {
        setBackendOnline(res.ok);
      })
      .catch(() => {
        setBackendOnline(false);
      });
  }, []);

  useEffect(() => {
    checkBackendHealth();
    const timer = setInterval(checkBackendHealth, 15000);
    return () => clearInterval(timer);
  }, [checkBackendHealth]);

  const studentsList: StudentAnalytics[] = metrics.students || [];
  const focusedStudent = studentsList.find((s) => s.student_id === focusedStudentId) || studentsList[0];
  const perf = metrics.performance || { fps: 0.0, latency_ms: 0.0 };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col font-sans antialiased selection:bg-emerald-600 selection:text-white">
      <Header
        metrics={metrics}
        isConnected={backendOnline}
        cameraSourceLabel={browserCamActive ? 'Browser Camera' : 'Camera Off'}
        modelDiagnostics={modelDiag}
      />

      <main className="flex-1 max-w-[1600px] w-full mx-auto p-4 md:p-6 space-y-6">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-3 bg-zinc-900/60 p-3 rounded-xl border border-zinc-800/80">
          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={() => setViewMode('classroom')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer ${
                viewMode === 'classroom'
                  ? 'bg-zinc-800 text-white shadow-sm border border-zinc-700'
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              <LayoutGrid className="w-3.5 h-3.5" />
              Classroom View
            </button>
            <button
              onClick={() => {
                if (studentsList.length > 0 && !focusedStudentId) {
                  setFocusedStudentId(studentsList[0].student_id);
                }
                setViewMode('focused');
              }}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer ${
                viewMode === 'focused'
                  ? 'bg-zinc-800 text-white shadow-sm border border-zinc-700'
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              <User className="w-3.5 h-3.5" />
              Focused Student View
            </button>

            <span className="hidden sm:inline text-zinc-700">|</span>

            <button
              onClick={() => {
                if (browserCamActive) {
                  setCameraConsentGranted(false);
                  stopBrowserCamera();
                } else {
                  setCameraConsentGranted(true);
                  setCameraRetryTick((c) => c + 1);
                }
              }}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer border ${
                browserCamActive
                  ? 'bg-rose-950/60 text-rose-300 border-rose-800/80 hover:bg-rose-900/60'
                  : 'bg-emerald-950/70 text-emerald-300 border-emerald-800/80 hover:bg-emerald-900/60'
              }`}
            >
              <Video className="w-3.5 h-3.5" />
              {browserCamActive ? 'Stop Camera' : 'Start Camera'}
            </button>

            <button
              onClick={() => {
                setCameraFacingMode((prev) => (prev === 'user' ? 'environment' : 'user'));
                setCameraRetryTick((c) => c + 1);
              }}
              className="px-2.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 bg-zinc-900 hover:bg-zinc-800 text-zinc-300 border border-zinc-800 cursor-pointer transition-all"
              title="Flip between Front (Selfie) and Back (Classroom) Camera on mobile phones"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>{cameraFacingMode === 'user' ? 'Front Cam' : 'Back Cam'}</span>
            </button>

            <span className="px-2.5 py-1.5 rounded-lg text-xs font-mono flex items-center gap-1.5 bg-zinc-900 text-zinc-300 border border-zinc-800">
              Browser AI (Zero Cloud Video)
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-3 text-xs text-zinc-400 font-mono">
            <span>
              Tracks: <strong className="text-emerald-400">{metrics.active_tracks ?? studentsList.length}</strong>
            </span>
            <span className="text-zinc-600">•</span>
            <span>
              Latency: <strong className="text-zinc-200">{perf.latency_ms.toFixed(1)} ms</strong>
            </span>
            <span className="text-zinc-600">•</span>
            <span>
              FPS: <strong className="text-zinc-200">{perf.fps.toFixed(1)}</strong>
            </span>
            {modelDiag.memoryMb !== null && (
              <>
                <span className="text-zinc-600">•</span>
                <span>
                  JS Heap: <strong className="text-zinc-200">{modelDiag.memoryMb} MB</strong>
                </span>
              </>
            )}
            <button
              onClick={checkBackendHealth}
              className="p-1 rounded hover:bg-zinc-800 text-zinc-400 hover:text-white transition-all cursor-pointer"
              title="Ping Backend Health"
            >
              <RefreshCw className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {(cameraError || modelDiag.status === 'error' || !backendOnline) && (
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 bg-amber-950/40 border border-amber-800/60 text-amber-200 px-4 py-2.5 rounded-xl text-xs">
            <div className="space-y-1">
              {modelDiag.status === 'error' && (
                <div className="flex items-center gap-1.5 text-rose-300">
                  <Cpu className="w-3.5 h-3.5 shrink-0" />
                  <span>Browser AI model failure: {modelDiag.error}</span>
                </div>
              )}
              {cameraError && <div>{cameraError}</div>}
              {!backendOnline && (
                <div>
                  Metadata backend is sleeping or waking up (Render free-tier cold start). Browser AI continues running locally.
                </div>
              )}
            </div>
            <div className="flex items-center gap-2 shrink-0">
              {modelDiag.status === 'error' && (
                <button
                  onClick={() => setModelRetryTick((c) => c + 1)}
                  className="px-3 py-1.5 rounded-lg bg-rose-600/30 hover:bg-rose-600/50 border border-rose-500/50 text-rose-100 font-semibold cursor-pointer"
                >
                  Retry Models
                </button>
              )}
              {cameraError && mediaSourceMode === 'camera' && (
                <>
                  <button
                    onClick={() => setCameraRetryTick((c) => c + 1)}
                    className="px-3 py-1.5 rounded-lg bg-amber-600/30 hover:bg-amber-600/50 border border-amber-500/50 text-amber-100 font-semibold cursor-pointer"
                  >
                    Retry Camera
                  </button>
                  <button
                    onClick={() => handleSelectMediaSourceMode('dataset')}
                    className="px-3 py-1.5 rounded-lg bg-purple-600/40 hover:bg-purple-600/60 border border-purple-500/50 text-purple-100 font-semibold cursor-pointer flex items-center gap-1.5"
                  >
                    <Database className="w-3.5 h-3.5" />
                    <span>Test Open Dataset</span>
                  </button>
                </>
              )}
            </div>
          </div>
        )}

        <MetricsGrid metrics={metrics} />

        {viewMode === 'classroom' ? (
          <>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <LiveClassroomView
                  students={studentsList}
                  onSelectStudent={(id) => setSelectedStudentId(id)}
                  videoRef={videoRef}
                  overlayCanvasRef={overlayCanvasRef}
                  imageCanvasRef={imageCanvasRef}
                  browserCamActive={browserCamActive || isMediaActive}
                  modelStatus={modelDiag.status}
                  showOverlays={showOverlays}
                  onToggleOverlays={() => setShowOverlays((s) => !s)}
                  cameraFacingMode={cameraFacingMode}
                  onToggleFacingMode={() => setCameraFacingMode((m) => (m === 'user' ? 'environment' : 'user'))}
                  mediaSourceMode={mediaSourceMode}
                  onSelectMediaSourceMode={handleSelectMediaSourceMode}
                  activeDatasetSampleId={activeDatasetSampleId}
                  onSelectDatasetSample={handleSelectDatasetSample}
                  onFileUpload={handleFileUpload}
                  uploadedFileName={uploadedFileName}
                />
              </div>
              <div className="lg:col-span-1">
                <TeacherRecommendations recommendations={metrics.recommendations || []} />
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <SpatialHeatmap students={studentsList} onSelectStudent={(id) => setSelectedStudentId(id)} />
              <AnalyticsCharts metrics={metrics} timelineHistory={timelineHistory} />
            </div>

            <GeminiInsightsPanel metrics={metrics} backendOnline={backendOnline} />
          </>
        ) : (
          <div className="space-y-4">
            {studentsList.length > 0 ? (
              <div className="flex items-center gap-2 overflow-x-auto pb-2">
                <span className="text-xs text-zinc-400 font-semibold uppercase tracking-wider mr-1">
                  Select Student:
                </span>
                {studentsList.map((s) => (
                  <button
                    key={s.student_id}
                    onClick={() => setFocusedStudentId(s.student_id)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all cursor-pointer border ${
                      (focusedStudent?.student_id || '') === s.student_id
                        ? 'bg-emerald-950/70 text-emerald-300 border-emerald-700'
                        : 'bg-zinc-900 text-zinc-400 border-zinc-800 hover:text-zinc-200'
                    }`}
                  >
                    {s.student_id}
                  </button>
                ))}
              </div>
            ) : null}

            <SingleStudentView student={focusedStudent} />
          </div>
        )}

        <DebugPanel
          students={studentsList}
          personCount={metrics.live_detected_persons ?? 0}
          validFaceCount={metrics.valid_face_count ?? 0}
          activeTracks={metrics.active_tracks ?? studentsList.length}
        />
      </main>

      {selectedStudentId && (
        <StudentDetailModal
          studentId={selectedStudentId}
          studentData={studentsList.find((s) => s.student_id === selectedStudentId) || null}
          onClose={() => setSelectedStudentId(null)}
        />
      )}

      <ToastContainer toasts={toasts} onRemoveToast={removeToast} />
    </div>
  );
}
export default App;
