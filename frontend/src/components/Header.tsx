import React from 'react';
import { Camera, ShieldCheck, Activity, Users, ScanFace, Cpu } from 'lucide-react';
import { ClassroomMetrics } from '../types/analytics';
import { BrowserAiDiagnostics } from '../ai/detector';

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
  const personCount = metrics.live_detected_persons ?? 0;
  const faceCount = metrics.valid_face_count ?? 0;
  const camOk = metrics.camera_connected !== false;
  const modelStatus = modelDiagnostics?.status ?? 'idle';

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

          <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-zinc-900 border border-zinc-800 font-mono text-[11px] text-zinc-300">
            <span className={`h-2 w-2 rounded-full ${isConnected ? 'bg-emerald-400' : 'bg-amber-400 animate-pulse'}`}></span>
            <span>{isConnected ? 'API ONLINE' : 'API SLEEPING'}</span>
          </div>

          <div className="hidden xl:flex items-center gap-1.5 text-zinc-400 bg-zinc-900 px-2.5 py-1.5 rounded-lg border border-zinc-800 font-mono text-[11px]">
            <ShieldCheck className="h-3.5 w-3.5 text-zinc-300" />
            <span>Anonymous IDs</span>
          </div>
        </div>
      </div>
    </header>
  );
};
