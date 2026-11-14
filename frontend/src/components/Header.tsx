import React, { useState } from 'react';
import { Camera, ShieldCheck, Activity, Users, ScanFace, Cpu, Globe, Check, AlertCircle } from 'lucide-react';
import { ClassroomMetrics } from '../types/analytics';
import { BrowserAiDiagnostics } from '../ai/detector';
import { getApiBaseUrl, setApiBaseUrl } from '../config';

interface HeaderProps {
  metrics: ClassroomMetrics;
  isConnected: boolean;
  cameraSourceLabel: string;
  modelDiagnostics?: BrowserAiDiagnostics;
}

export const Header: React.FC<HeaderProps> = ({
  metrics,
  isConnected,
  cameraSourceLabel,
  modelDiagnostics,
}) => {
  const [showBackendModal, setShowBackendModal] = useState<boolean>(false);
  const [backendUrlInput, setBackendUrlInput] = useState<string>(() => getApiBaseUrl());
  const [testing, setTesting] = useState<boolean>(false);
  const [testResult, setTestResult] = useState<{ success: boolean; message: string } | null>(null);

  const personCount = metrics.live_detected_persons ?? 0;
  const faceCount = metrics.valid_face_count ?? 0;
  const camOk = metrics.camera_connected !== false;
  const modelStatus = modelDiagnostics?.status ?? 'idle';

  const handleTestConnection = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      const cleanUrl = backendUrlInput.trim().replace(/\/+$/, '');
      const res = await fetch(`${cleanUrl}/health`);
      if (res.ok) {
        setTestResult({ success: true, message: 'Backend reachable and healthy (200 OK)!' });
      } else {
        setTestResult({ success: false, message: `Backend responded with HTTP ${res.status}` });
      }
    } catch (err) {
      setTestResult({
        success: false,
        message: 'Could not reach backend. Verify URL, check if Render container is starting (~45s), and ensure CORS is allowed.',
      });
    } finally {
      setTesting(false);
    }
  };

  const handleSaveBackendUrl = () => {
    setApiBaseUrl(backendUrlInput.trim());
    setShowBackendModal(false);
    window.location.reload();
  };

  return (
    <header className="sticky top-0 z-40 bg-zinc-950/90 backdrop-blur-md border-b border-zinc-800/80 px-4 sm:px-6 py-3">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 sm:gap-4">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-lg bg-zinc-900 border border-zinc-700 flex items-center justify-center text-zinc-100">
            <Activity className="h-5 w-5 text-emerald-400" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-semibold tracking-tight text-white">EduSense</h1>
              <span className="text-[11px] px-2 py-0.5 rounded bg-zinc-900 text-zinc-400 font-mono border border-zinc-800">
                v3.0 Live
              </span>
            </div>
            <p className="text-[11px] text-zinc-400">
              Observable Classroom Engagement &amp; Attention Estimation
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2.5 text-xs">
          <div className="flex items-center gap-1.5 text-zinc-300 bg-zinc-900 px-3 py-1.5 rounded-lg border border-zinc-800 font-mono text-[11px]">
            <Users className="h-3.5 w-3.5 text-cyan-400" />
            <span>Persons: <strong>{personCount}</strong></span>
          </div>

          <div className="flex items-center gap-1.5 text-zinc-300 bg-zinc-900 px-3 py-1.5 rounded-lg border border-zinc-800 font-mono text-[11px]">
            <ScanFace className="h-3.5 w-3.5 text-emerald-400" />
            <span>Valid Faces: <strong>{faceCount}</strong></span>
          </div>

          <div className="flex items-center gap-1.5 text-zinc-300 bg-zinc-900 px-3 py-1.5 rounded-lg border border-zinc-800 font-mono text-[11px]">
            <Camera className={`h-3.5 w-3.5 ${camOk ? 'text-emerald-400' : 'text-rose-400'}`} />
            <span>{camOk ? cameraSourceLabel : 'Camera Disconnected'}</span>
          </div>

          <div className="px-3 py-1.5 rounded-lg border border-zinc-800 bg-zinc-900 font-mono flex items-center gap-2">
            <span className="text-[11px] text-zinc-400 uppercase tracking-wider">ECI:</span>
            <span className="text-sm font-bold text-white">{metrics.eci_score.toFixed(1)}</span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-300">{metrics.eci_status}</span>
          </div>

          <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-zinc-900 border border-zinc-800 font-mono text-[11px] text-zinc-300">
            <Cpu
              className={`h-3.5 w-3.5 ${
                modelStatus === 'ready'
                  ? 'text-emerald-400'
                  : modelStatus === 'error'
                  ? 'text-rose-400'
                  : 'text-amber-400 animate-pulse'
              }`}
            />
            <span>
              {modelStatus === 'ready'
                ? `BROWSER AI (${modelDiagnostics?.delegate ?? 'GPU'})`
                : modelStatus === 'error'
                ? 'MODEL ERROR'
                : 'LOADING AI'}
            </span>
          </div>

          <button
            onClick={() => setShowBackendModal((s) => !s)}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 font-mono text-[11px] text-zinc-300 cursor-pointer transition-colors"
            title="Configure Backend API URL"
          >
            <span className={`h-2 w-2 rounded-full ${isConnected ? 'bg-emerald-400' : 'bg-amber-400 animate-pulse'}`}></span>
            <span>{isConnected ? 'API ONLINE' : 'API SLEEPING'}</span>
          </button>

          <div className="hidden xl:flex items-center gap-1.5 text-zinc-400 bg-zinc-900 px-2.5 py-1.5 rounded-lg border border-zinc-800 font-mono text-[11px]">
            <ShieldCheck className="h-3.5 w-3.5 text-zinc-300" />
            <span>Anonymous IDs</span>
          </div>
        </div>

        {showBackendModal && (
          <div className="absolute right-4 sm:right-6 top-16 z-50 w-80 p-4 rounded-xl bg-zinc-950 border border-zinc-700 shadow-2xl text-xs space-y-3">
            <div className="flex items-center justify-between">
              <span className="font-bold text-white uppercase text-[11px] flex items-center gap-1.5">
                <Globe className="w-4 h-4 text-emerald-400" /> Backend API Connection
              </span>
              <button
                onClick={() => setShowBackendModal(false)}
                className="text-zinc-400 hover:text-white cursor-pointer px-1.5 py-0.5 rounded hover:bg-zinc-800"
              >
                ✕
              </button>
            </div>
            <p className="text-[11px] text-zinc-400 leading-relaxed">
              If your backend is deployed on Render, paste its URL below (e.g. <code className="text-emerald-400 font-mono">https://edusense-api-xxxx.onrender.com</code>):
            </p>
            <input
              type="text"
              value={backendUrlInput}
              onChange={(e) => setBackendUrlInput(e.target.value)}
              placeholder="https://edusense-api.onrender.com"
              className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-2.5 py-1.5 text-xs text-white placeholder:text-zinc-500 focus:outline-none focus:border-emerald-500 font-mono"
            />
            <div className="flex items-center justify-between gap-2 pt-1">
              <button
                onClick={handleTestConnection}
                disabled={testing}
                className="px-3 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-xs font-semibold border border-zinc-700 cursor-pointer disabled:opacity-50"
              >
                {testing ? 'Testing...' : 'Test Health'}
              </button>
              <button
                onClick={handleSaveBackendUrl}
                className="px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold cursor-pointer transition-all shadow-sm"
              >
                Save &amp; Connect
              </button>
            </div>
            {testResult && (
              <div
                className={`p-2.5 rounded-lg text-[11px] flex items-start gap-1.5 leading-normal ${
                  testResult.success
                    ? 'bg-emerald-950/70 text-emerald-300 border border-emerald-800'
                    : 'bg-rose-950/70 text-rose-300 border border-rose-800'
                }`}
              >
                {testResult.success ? (
                  <Check className="w-3.5 h-3.5 shrink-0 text-emerald-400 mt-0.5" />
                ) : (
                  <AlertCircle className="w-3.5 h-3.5 shrink-0 text-rose-400 mt-0.5" />
                )}
                <span>{testResult.message}</span>
              </div>
            )}
          </div>
        )}
      </div>
    </header>
  );
};
