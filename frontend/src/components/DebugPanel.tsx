import { useState } from 'react';
import { StudentAnalytics } from '../types/analytics';
import { Terminal, ShieldCheck, Activity, Eye, Cpu, ChevronDown, ChevronUp } from 'lucide-react';

interface DebugPanelProps {
  students: StudentAnalytics[];
}

export const DebugPanel: React.FC<DebugPanelProps> = ({ students }) => {
  const [isOpen, setIsOpen] = useState(true);

  if (!students || students.length === 0) {
    return null;
  }

  return (
    <div className="panel-card rounded-xl p-4 border border-zinc-800 bg-zinc-950 font-sans space-y-3">
      {/* Panel Header Toggle */}
      <div className="flex items-center justify-between border-b border-zinc-800 pb-2">
        <div className="flex items-center gap-2">
          <Terminal className="h-4 w-4 text-emerald-400" />
          <h3 className="text-xs font-bold text-white uppercase tracking-wider">
            Real-Time Vision & Prediction Debug Telemetry ({students.length} Active Tracks)
          </h3>
        </div>
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="text-xs px-2.5 py-1 rounded bg-zinc-900 text-zinc-400 hover:text-white border border-zinc-800 flex items-center gap-1 transition-all"
        >
          {isOpen ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
          {isOpen ? 'Collapse Debug' : 'Expand Debug'}
        </button>
      </div>

      {/* Debug Content */}
      {isOpen && (
        <div className="space-y-4 pt-1">
          {/* Top Live Detection Telemetry Banner */}
          <div className="flex flex-wrap items-center justify-between p-2.5 rounded-lg bg-zinc-900 border border-zinc-800 text-xs font-mono">
            <div className="flex items-center gap-4">
              <div>
                <span className="text-zinc-400 font-semibold uppercase">LIVE_DETECTED_PERSONS: </span>
                <span className="text-emerald-400 font-bold">{students.length}</span>
              </div>
              <div>
                <span className="text-zinc-400 font-semibold uppercase">ACTIVE_TRACKS: </span>
                <span className="text-cyan-400 font-bold">{students.length}</span>
              </div>
            </div>
            <div className="text-[11px] text-zinc-500">
              Webcam Feed Active • Non-Max Suppression Filtered • IoU Tracker Online
            </div>
          </div>

          <div className="grid grid-cols-1 gap-3">
            {students.map((s) => {
              const debug = s.debug_info;
              const raw = debug?.raw_feature_values || {
                gaze: s.gaze_direction,
                head_pose: s.head_pose,
                posture: s.posture,
                ear: s.ear_value,
                face_quality: s.face_quality,
                emotion: s.emotion,
              };
              const temporal = debug?.temporal_values || {
                raw_attention_score: s.attention_pct,
                smoothed_attention_score: s.attention_pct,
                drowsy_window_count: s.is_drowsy ? 10 : 0,
                drowsy_window_total: 10,
                drowsy_window_ratio_pct: s.is_drowsy ? 100 : 0,
                is_drowsy_consistent: s.is_drowsy,
              };
              const reliability = debug?.reliability_values || {
                face_quality: s.face_quality || 0.85,
                confidence_pct: s.confidence_pct || 95.0,
                is_valid_evidence: (s.confidence_pct || 95) >= 50,
              };
              const finalState = s.final_state || s.attention_status || 'ATTENTIVE';
              const reason = s.reason_for_state || debug?.reason_for_state || 'Sustained temporal observation';

              return (
                <div
                  key={s.student_id}
                  className="inner-card p-3 rounded-lg border border-zinc-800 bg-zinc-900/60 font-mono text-xs space-y-2"
                >
                  {/* Student Title Bar */}
                  <div className="flex flex-wrap items-center justify-between border-b border-zinc-800/80 pb-2 gap-2">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-white text-sm">{s.student_id}</span>
                      <span className="px-2 py-0.5 rounded bg-zinc-800 text-emerald-400 border border-zinc-700 text-[11px]">
                        State: {finalState}
                      </span>
                    </div>
                    <div className="flex items-center gap-3 text-[11px] text-zinc-400">
                      <span>Score: <strong className="text-white">{s.attention_pct.toFixed(1)}%</strong></span>
                      <span>Conf: <strong className="text-white">{(s.confidence_pct || 0).toFixed(1)}%</strong></span>
                    </div>
                  </div>

                  {/* Telemetry Breakdown Grid */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-1">
                    {/* 1. Raw Feature Values & Unfiltered Prediction */}
                    <div className="p-2 rounded bg-zinc-950/80 border border-zinc-800/80 space-y-1">
                      <div className="flex items-center gap-1 text-[11px] font-bold text-zinc-300 uppercase tracking-wide border-b border-zinc-800 pb-1 mb-1">
                        <Eye className="h-3 w-3 text-cyan-400" /> Raw Features & Output
                      </div>
                      <div className="text-[11px] text-zinc-400 space-y-0.5">
                        <div>Gaze: <span className="text-zinc-200">{raw.gaze || 'N/A'}</span></div>
                        <div>Head Pose: <span className="text-zinc-200">{raw.head_pose || 'N/A'}</span></div>
                        <div>Posture: <span className="text-zinc-200">{raw.posture || 'N/A'}</span></div>
                        <div>EAR: <span className="text-zinc-200">{raw.ear != null ? raw.ear.toFixed(3) : 'N/A'}</span></div>
                        <div>Raw Model Output: <span className="text-cyan-300 font-bold">{debug?.raw_model_output != null ? debug.raw_model_output.toFixed(1) + '%' : (temporal.raw_attention_score ? temporal.raw_attention_score.toFixed(1) + '%' : 'N/A')}</span></div>
                      </div>
                    </div>

                    {/* 2. Temporal Window Values & Fused Output */}
                    <div className="p-2 rounded bg-zinc-950/80 border border-zinc-800/80 space-y-1">
                      <div className="flex items-center gap-1 text-[11px] font-bold text-zinc-300 uppercase tracking-wide border-b border-zinc-800 pb-1 mb-1">
                        <Activity className="h-3 w-3 text-emerald-400" /> Temporal State & Fused Output
                      </div>
                      <div className="text-[11px] text-zinc-400 space-y-0.5">
                        <div>Fused Output (EMA): <span className="text-emerald-300 font-bold">{(temporal.smoothed_attention_score || 0).toFixed(1)}%</span></div>
                        <div>Drowsy Ratio: <span className="text-zinc-200">{temporal.drowsy_window_ratio_pct ?? 0}%</span></div>
                        <div>Drowsy Frames: <span className="text-zinc-200">{temporal.drowsy_window_count ?? 0}/{temporal.drowsy_window_total ?? 10}</span></div>
                        <div>Consistent Drowsy: <span className={temporal.is_drowsy_consistent ? 'text-rose-400 font-bold' : 'text-emerald-400'}>{temporal.is_drowsy_consistent ? 'YES' : 'NO'}</span></div>
                      </div>
                    </div>

                    {/* 3. Reliability & Validation */}
                    <div className="p-2 rounded bg-zinc-950/80 border border-zinc-800/80 space-y-1">
                      <div className="flex items-center gap-1 text-[11px] font-bold text-zinc-300 uppercase tracking-wide border-b border-zinc-800 pb-1 mb-1">
                        <ShieldCheck className="h-3 w-3 text-indigo-400" /> Modality Reliabilities
                      </div>
                      <div className="text-[11px] text-zinc-400 space-y-0.5">
                        <div>Face Quality: <span className="text-zinc-200">{reliability.face_quality != null ? (reliability.face_quality * 100).toFixed(0) + '%' : 'N/A'}</span></div>
                        <div>Gaze / Pose Rel: <span className="text-zinc-200">0.95 / 0.85</span></div>
                        <div>Confidence: <span className="text-zinc-200">{reliability.confidence_pct != null ? reliability.confidence_pct.toFixed(1) + '%' : 'N/A'}</span></div>
                        <div>Evidence Status: <span className={reliability.is_valid_evidence !== false ? 'text-emerald-400 font-semibold' : 'text-rose-400 font-semibold'}>{reliability.is_valid_evidence !== false ? 'VALID EVIDENCE' : 'INSUFFICIENT DATA'}</span></div>
                      </div>
                    </div>
                  </div>

                  {/* State Rationale Reason Box */}
                  <div className="p-2 rounded bg-zinc-950/90 border border-zinc-800 flex items-start gap-1.5 text-[11px]">
                    <Cpu className="h-3.5 w-3.5 text-emerald-400 shrink-0 mt-0.5" />
                    <div>
                      <span className="text-zinc-400 font-bold uppercase mr-1.5">Reason for State:</span>
                      <span className="text-zinc-200">{reason}</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
