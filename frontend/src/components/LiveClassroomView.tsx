import React, { useState } from 'react';
import { Layers, ShieldCheck, User, Sparkles, SwitchCamera } from 'lucide-react';
import { StudentAnalytics } from '../types/analytics';

interface LiveClassroomViewProps {
  frame?: string;
  students: StudentAnalytics[];
  onSelectStudent: (studentId: string) => void;
  videoRef?: React.RefObject<HTMLVideoElement>;
  overlayCanvasRef?: React.RefObject<HTMLCanvasElement>;
  browserCamActive?: boolean;
  modelStatus?: string;
  showOverlays?: boolean;
  onToggleOverlays?: () => void;
  cameraFacingMode?: 'user' | 'environment';
  onToggleFacingMode?: () => void;
}

export const LiveClassroomView: React.FC<LiveClassroomViewProps> = ({
  frame,
  students,
  onSelectStudent,
  videoRef,
  overlayCanvasRef,
  browserCamActive = false,
  modelStatus = 'idle',
  showOverlays: externalShowOverlays,
  onToggleOverlays,
  cameraFacingMode = 'user',
  onToggleFacingMode,
}) => {
  const [internalShowOverlays, setInternalShowOverlays] = useState(true);
  const showOverlays = externalShowOverlays ?? internalShowOverlays;
  const toggleOverlays = onToggleOverlays ?? (() => setInternalShowOverlays((s) => !s));

  return (
    <div className="panel-card rounded-xl p-4 border border-zinc-800 flex flex-col h-full">
      {/* Video Feed Header Controls */}
      <div className="flex items-center justify-between mb-3 px-1">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-emerald-400"></span>
          <h2 className="text-xs font-bold text-white tracking-wide uppercase">Vision Feed</h2>
        </div>
        <div className="flex items-center gap-2">
          {onToggleFacingMode && (
            <button
              onClick={onToggleFacingMode}
              className="text-xs px-2.5 py-1 rounded border font-medium flex items-center gap-1.5 transition-all cursor-pointer bg-zinc-900 hover:bg-zinc-800 text-zinc-300 border-zinc-800"
              title="Flip between Front and Rear Camera"
            >
              <SwitchCamera className="h-3.5 w-3.5 text-emerald-400" />
              <span>{cameraFacingMode === 'user' ? 'Front' : 'Rear'} Cam</span>
            </button>
          )}
          <button
            onClick={toggleOverlays}
            className={`text-xs px-2.5 py-1 rounded border font-medium flex items-center gap-1 transition-all cursor-pointer ${
              showOverlays
                ? 'bg-zinc-800 text-white border-zinc-700'
                : 'bg-zinc-900 text-zinc-400 border-zinc-800'
            }`}
          >
            <Layers className="h-3 w-3" />
            Overlays {showOverlays ? 'ON' : 'OFF'}
          </button>
        </div>
      </div>

      {/* Main Video Frame Canvas (Browser-Side AI Direct Feed) */}
      <div className="relative w-full aspect-video bg-zinc-950 rounded-lg overflow-hidden border border-zinc-800 flex items-center justify-center">
        {/* Floating Quick State Tester & Guide Banner (Desktop/Tablet) */}
        <div className="absolute top-2 left-2 z-20 hidden sm:flex flex-wrap items-center gap-1.5 bg-zinc-950/85 backdrop-blur border border-zinc-800 px-2.5 py-1 rounded-lg text-[10px] font-mono pointer-events-none">
          <span className="text-zinc-400">Calibrated Triggers:</span>
          <span className="text-emerald-400 font-bold bg-emerald-950/70 px-1.5 py-0.5 rounded border border-emerald-800/60">
            ATTENTIVE: Frontal &ge;68%
          </span>
          <span className="text-amber-400 font-bold bg-amber-950/70 px-1.5 py-0.5 rounded border border-amber-800/60">
            MODERATE: 44&ndash;67% (Yaw/Glance)
          </span>
          <span className="text-rose-400 font-bold bg-rose-950/70 px-1.5 py-0.5 rounded border border-rose-800/60">
            DROWSY: Eyes Closed / Micro-Sleep
          </span>
        </div>
        {videoRef ? (
          <>
            <video
              ref={videoRef}
              playsInline
              muted
              autoPlay
              className={`w-full h-full object-contain ${browserCamActive ? 'block' : 'hidden'}`}
            />
            {overlayCanvasRef && (
              <canvas
                ref={overlayCanvasRef}
                className={`absolute inset-0 w-full h-full object-contain pointer-events-none ${
                  browserCamActive && showOverlays ? 'block' : 'hidden'
                }`}
              />
            )}
            {!browserCamActive && (
              <div className="flex flex-col items-center gap-2 text-zinc-500">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-zinc-400"></div>
                <span className="text-xs font-mono">
                  {modelStatus === 'loading'
                    ? 'Loading Browser MediaPipe Models...'
                    : 'Awaiting Browser Camera Permission...'}
                </span>
              </div>
            )}
          </>
        ) : frame ? (
          <img
            src={frame}
            alt="Live Classroom Feed"
            className="w-full h-full object-contain"
          />
        ) : (
          <div className="flex flex-col items-center gap-2 text-zinc-500">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-zinc-400"></div>
            <span className="text-xs font-mono">Initializing Vision Pipeline...</span>
          </div>
        )}
      </div>

      {/* Detected Students Selector Cards with AI Suggestions */}
      <div className="mt-3">
        <div className="flex items-center justify-between mb-2 px-1">
          <span className="text-[11px] font-medium text-zinc-400 uppercase tracking-wider">
            Active Detections ({students.length}) — Click to Inspect
          </span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
          {students.map((student) => {
            const isDrowsy = student.final_state === 'DROWSY' || student.is_drowsy;
            const isPhone = student.final_state === 'PHONE DISTRACTED' || student.phone_detected;
            const isAttentive = student.final_state === 'ATTENTIVE';
            const isModerate = student.final_state === 'MODERATE';

            const badgeClass = isDrowsy
              ? 'bg-rose-950/80 text-rose-300 border-rose-800'
              : isPhone
              ? 'bg-purple-950/80 text-purple-300 border-purple-800'
              : isAttentive
              ? 'bg-emerald-950/80 text-emerald-300 border-emerald-800'
              : isModerate
              ? 'bg-amber-950/80 text-amber-300 border-amber-800'
              : 'bg-zinc-900 text-zinc-400 border-zinc-800';

            return (
              <button
                key={student.student_id}
                onClick={() => onSelectStudent(student.student_id)}
                className="inner-card p-2.5 rounded-lg border border-zinc-800 hover:border-zinc-600 transition-all text-left group cursor-pointer flex flex-col justify-between h-[5.75rem] min-h-[5.75rem] overflow-hidden"
              >
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-1.5">
                      <User className="h-3.5 w-3.5 text-zinc-400" />
                      <span className="text-xs font-bold text-white group-hover:text-zinc-200">
                        {student.student_id}
                      </span>
                    </div>
                    {student.confidence_pct != null && (
                      <span className="text-[10px] font-mono text-zinc-400 flex items-center gap-0.5">
                        <ShieldCheck className="h-3 w-3 text-zinc-400" />
                        {student.confidence_pct.toFixed(0)}%
                      </span>
                    )}
                  </div>

                  <div className="flex items-center justify-between text-[11px] text-zinc-400 font-mono mb-1.5">
                    <span>
                      Attn: <strong className="text-white">{student.final_state === 'UNABLE TO DETERMINE' ? 'N/A' : `${student.attention_pct.toFixed(0)}%`}</strong>
                    </span>
                    <span className={`text-[10px] px-1.5 py-0.5 rounded border font-semibold font-mono ${badgeClass}`}>
                      {isPhone && !student.final_state?.includes('📱') ? '📱 ' : ''}{student.final_state || student.attention_status || 'ATTENTIVE'}
                    </span>
                  </div>
                </div>

                {student.ai_suggestion && (
                  <div className="mt-1 pt-1.5 border-t border-zinc-800/80 flex items-start gap-1 text-[10px]">
                    <Sparkles className="h-3 w-3 text-emerald-400 shrink-0 mt-0.5" />
                    <span className="text-zinc-300 truncate font-sans">
                      <strong className="text-white">{student.ai_suggestion.action}:</strong>{' '}
                      {student.ai_suggestion.pedagogical_tip}
                    </span>
                  </div>
                )}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};
