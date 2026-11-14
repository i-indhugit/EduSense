import React from 'react';
import { StudentAnalytics } from '../types/analytics';
import { ShieldCheck, Brain, Sparkles, CheckCircle2, Compass } from 'lucide-react';

interface SingleStudentViewProps {
  student: StudentAnalytics;
  frame?: string;
}

export const SingleStudentView: React.FC<SingleStudentViewProps> = ({ student, frame }) => {
  const conf = student.confidence_pct ?? 0;
  const isFaceUnavailable = !student.face_detected || student.final_state === 'UNABLE TO DETERMINE' || conf < 50.0;

  const stateColor =
    student.final_state === 'DROWSY'
      ? 'text-rose-400 bg-rose-950/60 border-rose-800'
      : student.final_state === 'ATTENTIVE'
      ? 'text-emerald-400 bg-emerald-950/60 border-emerald-800'
      : student.final_state === 'MODERATE'
      ? 'text-amber-400 bg-amber-950/60 border-amber-800'
      : 'text-zinc-400 bg-zinc-900 border-zinc-800';

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 panel-card rounded-xl p-4 border border-zinc-800 flex flex-col">
          <div className="flex items-center justify-between mb-3 px-1">
            <div className="flex items-center gap-2">
              <span className={`h-2.5 w-2.5 rounded-full ${student.face_detected ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`}></span>
              <h3 className="text-xs font-bold text-white uppercase tracking-wider">
                Single Student Analysis — {student.student_id}
              </h3>
            </div>
            <span className="text-[11px] font-mono text-zinc-300 bg-zinc-900 px-2.5 py-1 rounded border border-zinc-800 flex items-center gap-1">
              <ShieldCheck className="h-3.5 w-3.5 text-zinc-400" /> Conf: {conf.toFixed(1)}%
            </span>
          </div>

          <div className="relative w-full aspect-video bg-zinc-950 rounded-lg overflow-hidden border border-zinc-800 flex items-center justify-center">
            {frame ? (
              <img src={frame} alt="Single Student Feed" className="w-full h-full object-contain" />
            ) : (
              <div className="flex flex-col items-center gap-2 text-zinc-400 p-4 text-center">
                <span className="text-xs font-mono text-emerald-400">Browser-Side AI Telemetry Active</span>
                <span className="text-[11px] text-zinc-500">
                  Switch to Classroom View to view the live camera overlay; all behavioral metrics on the right update in real time.
                </span>
              </div>
            )}
          </div>
        </div>

        <div className="panel-card rounded-xl p-4 border border-zinc-800 space-y-3">
          <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
            <div>
              <h4 className="text-[10px] font-bold text-zinc-400 uppercase">Face Visibility</h4>
              <span className={`text-xs font-semibold ${student.face_detected ? 'text-emerald-400' : 'text-amber-400'}`}>
                {student.face_detected ? 'Validated Face & Landmarks' : (student.face_status || 'Face Unavailable')}
              </span>
            </div>
            <div className="text-right">
              <h4 className="text-[10px] font-bold text-zinc-400 uppercase">Engagement Est.</h4>
              <span className="text-lg font-bold text-white font-mono">
                {isFaceUnavailable ? 'Unable to Determine' : `${student.attention_pct.toFixed(1)}%`}
              </span>
            </div>
          </div>

          {/* AI Pedagogical Suggestion Card */}
          {student.corrective_guidance?.is_fallback && (
            <div className="p-3 rounded-xl border border-amber-600/70 bg-amber-950/30 text-xs space-y-1.5 shadow-sm">
              <div className="flex items-center gap-1.5 font-bold text-amber-300 uppercase text-[10px] tracking-wider">
                <Compass className="h-3.5 w-3.5 text-amber-400 shrink-0" />
                Fallback Redirection: Path to High Accuracy
              </div>
              <p className="text-zinc-100 text-[11px] font-semibold">
                👉 {student.corrective_guidance.action_prompt}
              </p>
              <p className="text-[10px] text-zinc-400">
                {student.corrective_guidance.target_accuracy_path}
              </p>
            </div>
          )}

          {student.ai_suggestion && (
            <div className="inner-card p-3 rounded-xl border border-zinc-700 bg-gradient-to-br from-zinc-900/90 to-zinc-950/90 space-y-1.5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5 text-xs font-bold text-white uppercase tracking-wider">
                  <Sparkles className="h-3.5 w-3.5 text-emerald-400" />
                  AI Pedagogical Suggestion
                </div>
                <span className={`text-[10px] px-2 py-0.5 rounded font-mono font-bold uppercase border ${
                  student.ai_suggestion.priority === 'HIGH'
                    ? 'bg-rose-950/80 text-rose-300 border-rose-700/80'
                    : student.ai_suggestion.priority === 'MEDIUM'
                    ? 'bg-amber-950/80 text-amber-300 border-amber-700/80'
                    : 'bg-emerald-950/80 text-emerald-300 border-emerald-700/80'
                }`}>
                  {student.ai_suggestion.priority} Priority
                </span>
              </div>
              <div className="text-xs">
                <span className="font-bold text-white block text-xs">{student.ai_suggestion.action}</span>
                <p className="text-zinc-300 text-[11px] mt-0.5 leading-relaxed">{student.ai_suggestion.pedagogical_tip}</p>
              </div>
            </div>
          )}

          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="inner-card p-2.5 rounded-lg border border-zinc-800">
              <span className="text-zinc-400 block text-[9px] uppercase">Estimated State</span>
              <span className={`inline-block mt-0.5 text-xs font-bold px-1.5 py-0.5 rounded border font-mono ${stateColor}`}>
                {student.final_state || student.attention_status || 'UNABLE TO DETERMINE'}
              </span>
              {student.cautious_engagement_category && (
                <span className="text-[10px] text-zinc-400 block mt-1">
                  {student.cautious_engagement_category}
                </span>
              )}
            </div>
            <div className="inner-card p-2.5 rounded-lg border border-zinc-800">
              <span className="text-zinc-400 block text-[9px] uppercase">Expression</span>
              <span className="font-semibold text-white">{student.emotion}</span>
              {student.expression_probabilities && (
                <div className="text-[9px] text-zinc-400 mt-1 flex flex-wrap gap-1 font-mono">
                  {Object.entries(student.expression_probabilities).map(([k, v]) => (
                    <span key={k} className="bg-zinc-800/80 px-1 py-0.5 rounded">
                      {k.slice(0, 3)}: {Math.round(v * 100)}%
                    </span>
                  ))}
                </div>
              )}
            </div>
            <div className="inner-card p-2.5 rounded-lg border border-zinc-800">
              <span className="text-zinc-400 block text-[9px] uppercase">Gaze Direction</span>
              <span className="font-semibold text-white">{student.gaze_direction} ({student.gaze_vector || 'Unknown'})</span>
            </div>
            <div className="inner-card p-2.5 rounded-lg border border-zinc-800">
              <span className="text-zinc-400 block text-[9px] uppercase">Head Pose</span>
              <span className="font-semibold text-white">
                {student.head_pose || 'Unknown'}
                {student.pitch_deg != null && student.yaw_deg != null ? ` [P:${student.pitch_deg}° Y:${student.yaw_deg}°]` : ''}
              </span>
            </div>
            <div className="inner-card p-2.5 rounded-lg border border-zinc-800">
              <span className="text-zinc-400 block text-[9px] uppercase">Posture / Hand</span>
              <span className="font-semibold text-white">
                {student.posture} {student.hand_raised ? '• Hand Raised' : ''}
              </span>
            </div>
            <div className="inner-card p-2.5 rounded-lg border border-zinc-800">
              <span className="text-zinc-400 block text-[9px] uppercase">Drowsiness / Eyelids</span>
              <span className={student.is_drowsy ? 'font-bold text-rose-400' : 'font-semibold text-emerald-400'}>
                {student.is_drowsy ? 'Drowsy (Closed/Droop)' : 'Alert (Eyes Open)'}
              </span>
              <span className="text-[10px] text-zinc-400 block font-mono">
                EAR: {student.ear_value != null ? student.ear_value.toFixed(2) : 'N/A'} • PERCLOS: {student.perclos_pct != null ? `${student.perclos_pct.toFixed(0)}%` : '0%'}
              </span>
            </div>
          </div>

          {student.reason_for_state && (
            <div className="inner-card p-2.5 rounded-lg border border-zinc-800 text-xs">
              <span className="text-zinc-400 block text-[9px] uppercase font-bold">State Rationale</span>
              <span className="text-zinc-200">{student.reason_for_state}</span>
            </div>
          )}

          {student.xai?.attention && (
            <div className="inner-card p-3 rounded-lg border border-zinc-800 space-y-1.5">
              <div className="flex items-center gap-1.5 text-xs font-bold text-white uppercase tracking-wider">
                <Brain className="h-3.5 w-3.5 text-zinc-400" /> Explainable Cue Breakdown
              </div>
              <ul className="text-[11px] text-zinc-300 space-y-1">
                {student.xai.attention.reasons.map((reason, idx) => (
                  <li key={idx} className="flex items-start gap-1.5">
                    <CheckCircle2 className="h-3 w-3 text-emerald-400 shrink-0 mt-0.5" />
                    <span>{reason}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
