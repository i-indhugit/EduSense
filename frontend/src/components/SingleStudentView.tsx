import React from 'react';
import { StudentAnalytics } from '../types/analytics';
import { ShieldCheck, Brain } from 'lucide-react';

interface SingleStudentViewProps {
  student: StudentAnalytics;
  frame?: string;
}

export const SingleStudentView: React.FC<SingleStudentViewProps> = ({ student, frame }) => {
  const isConfidenceLow = (student.confidence_pct || 95) < 60.0;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Live Frame Camera View */}
        <div className="lg:col-span-2 panel-card rounded-xl p-4 border border-zinc-800 flex flex-col">
          <div className="flex items-center justify-between mb-3 px-1">
            <div className="flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-emerald-400"></span>
              <h3 className="text-xs font-bold text-white uppercase tracking-wider">
                Single Student Analysis — {student.student_id}
              </h3>
            </div>
            {student.confidence_pct && (
              <span className="text-[11px] font-mono text-zinc-300 bg-zinc-900 px-2.5 py-1 rounded border border-zinc-800 flex items-center gap-1">
                <ShieldCheck className="h-3.5 w-3.5 text-zinc-400" /> Conf: {student.confidence_pct.toFixed(1)}%
              </span>
            )}
          </div>

          <div className="relative w-full aspect-video bg-zinc-950 rounded-lg overflow-hidden border border-zinc-800 flex items-center justify-center">
            {frame ? (
              <img src={frame} alt="Single Student Feed" className="w-full h-full object-contain" />
            ) : (
              <div className="flex flex-col items-center gap-2 text-zinc-500">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-zinc-400"></div>
                <span className="text-xs font-mono">Loading Stream...</span>
              </div>
            )}
          </div>
        </div>

        {/* Real-time Metric Breakdown */}
        <div className="panel-card rounded-xl p-4 border border-zinc-800 space-y-3">
          <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
            <div>
              <h4 className="text-[10px] font-bold text-zinc-400 uppercase">Detection Status</h4>
              <span className="text-xs font-semibold text-white">Tracked & Validated</span>
            </div>
            <div className="text-right">
              <h4 className="text-[10px] font-bold text-zinc-400 uppercase">Attention Score</h4>
              <span className="text-lg font-bold text-white font-mono">
                {isConfidenceLow ? 'Unable to Determine' : `${student.attention_pct.toFixed(1)}%`}
              </span>
            </div>
          </div>

          {/* Metric Status Grid */}
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="inner-card p-2.5 rounded-lg border border-zinc-800">
              <span className="text-zinc-400 block text-[9px] uppercase">Final Attention State</span>
              <span className="font-semibold text-emerald-400 font-mono">{student.final_state || student.attention_status || 'ATTENTIVE'}</span>
            </div>
            <div className="inner-card p-2.5 rounded-lg border border-zinc-800">
              <span className="text-zinc-400 block text-[9px] uppercase">Emotion</span>
              <span className="font-semibold text-white">{student.emotion}</span>
            </div>
            <div className="inner-card p-2.5 rounded-lg border border-zinc-800">
              <span className="text-zinc-400 block text-[9px] uppercase">Gaze Direction</span>
              <span className="font-semibold text-white">{student.gaze_direction}</span>
            </div>
            <div className="inner-card p-2.5 rounded-lg border border-zinc-800">
              <span className="text-zinc-400 block text-[9px] uppercase">Head Pose / Posture</span>
              <span className="font-semibold text-white">{student.head_pose || student.posture}</span>
            </div>
            <div className="inner-card p-2.5 rounded-lg border border-zinc-800">
              <span className="text-zinc-400 block text-[9px] uppercase">Drowsiness Evidence</span>
              <span className={student.is_drowsy ? 'font-bold text-rose-400' : 'font-semibold text-white'}>{student.is_drowsy ? 'Consistent Drowsiness' : 'Alert'}</span>
            </div>
            <div className="inner-card p-2.5 rounded-lg border border-zinc-800">
              <span className="text-zinc-400 block text-[9px] uppercase">Face Quality</span>
              <span className="font-semibold text-white">{student.face_quality != null ? `${Math.round(student.face_quality * 100)}%` : 'Good'}</span>
            </div>
          </div>

          {/* State Reason Box */}
          {student.reason_for_state && (
            <div className="inner-card p-2.5 rounded-lg border border-zinc-800 text-xs">
              <span className="text-zinc-400 block text-[9px] uppercase font-bold">State Rationale</span>
              <span className="text-zinc-200">{student.reason_for_state}</span>
            </div>
          )}

          {/* Explainable AI Rationale Card */}
          {student.xai && student.xai.attention && (
            <div className="inner-card p-3 rounded-lg border border-zinc-800 space-y-1.5">
              <div className="flex items-center gap-1.5 text-xs font-bold text-white uppercase tracking-wider">
                <Brain className="h-3.5 w-3.5 text-zinc-400" /> Explainable AI Rationale
              </div>
              <ul className="text-[11px] text-zinc-300 space-y-1 list-disc list-inside">
                {student.xai.attention.reasons.map((reason, idx) => (
                  <li key={idx}>{reason}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
