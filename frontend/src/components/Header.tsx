import React from 'react';
import { Camera, ShieldCheck, Activity } from 'lucide-react';
import { ClassroomMetrics } from '../types/analytics';

interface HeaderProps {
  metrics: ClassroomMetrics;
  isConnected: boolean;
}

export const Header: React.FC<HeaderProps> = ({ metrics, isConnected }) => {
  return (
    <header className="sticky top-0 z-40 bg-zinc-950/90 backdrop-blur-md border-b border-zinc-800/80 px-6 py-3.5">
      <div className="flex flex-col md:flex-row items-center justify-between gap-4">
        {/* Brand Logo & Title */}
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-lg bg-zinc-900 border border-zinc-700 flex items-center justify-center text-zinc-100">
            <Activity className="h-5 w-5 text-zinc-200" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-semibold tracking-tight text-white">EduSense-X</h1>
              <span className="text-[11px] px-2 py-0.5 rounded bg-zinc-900 text-zinc-400 font-mono border border-zinc-800">
                Research v2.1
              </span>
            </div>
            <p className="text-[11px] text-zinc-400">Reliability-Aware Multimodal Classroom Intelligence</p>
          </div>
        </div>

        {/* Status Indicators */}
        <div className="flex items-center gap-3 text-xs">
          {/* Privacy Badge */}
          <div className="flex items-center gap-1.5 text-zinc-400 bg-zinc-900 px-3 py-1.5 rounded-lg border border-zinc-800 font-mono text-[11px]">
            <ShieldCheck className="h-3.5 w-3.5 text-zinc-300" />
            <span>Anonymous Tracking</span>
          </div>

          {/* Live Camera Indicator */}
          <div className="flex items-center gap-1.5 text-zinc-400 bg-zinc-900 px-3 py-1.5 rounded-lg border border-zinc-800 font-mono text-[11px]">
            <Camera className="h-3.5 w-3.5 text-zinc-300" />
            <span>Live Webcam</span>
          </div>

          {/* ECI Score Badge */}
          <div className="px-3.5 py-1.5 rounded-lg border border-zinc-800 bg-zinc-900 font-mono flex items-center gap-2">
            <span className="text-[11px] text-zinc-400 uppercase tracking-wider">ECI:</span>
            <span className="text-sm font-bold text-white">{metrics.eci_score.toFixed(1)}</span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-300">{metrics.eci_status}</span>
          </div>

          {/* Connection Status */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-zinc-900 border border-zinc-800 font-mono text-[11px] text-zinc-400">
            <span className={`h-2 w-2 rounded-full ${isConnected ? 'bg-emerald-400' : 'bg-amber-400'}`}></span>
            <span>{isConnected ? 'LIVE' : 'POLLING'}</span>
          </div>
        </div>
      </div>
    </header>
  );
};
