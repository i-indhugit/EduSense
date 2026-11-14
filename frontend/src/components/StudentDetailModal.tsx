import React, { useEffect, useState } from 'react';
import { X, User, Activity, Eye, Brain, CheckCircle2, Info, AlertCircle, Sparkles, Compass } from 'lucide-react';
import { API_BASE_URL } from '../config';
import { StudentAnalytics } from '../types/analytics';

interface StudentDetailModalProps {
  studentId: string | null;
  studentData?: StudentAnalytics | null;
  onClose: () => void;
}

export const StudentDetailModal: React.FC<StudentDetailModalProps> = ({ studentId, studentData, onClose }) => {
  const [details, setDetails] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [fetchError, setFetchError] = useState<string | null>(null);

  useEffect(() => {
    if (!studentId) return;
    if (studentData) {
      setDetails({
        found: true,
        student: studentData,
        recommendations: studentData.xai?.counterfactual?.explanation
          ? [studentData.xai.counterfactual.explanation]
          : [],
      });
      setLoading(false);
      setFetchError(null);
      return;
    }
    setLoading(true);
    setFetchError(null);

    fetch(`${API_BASE_URL}/student/${encodeURIComponent(studentId)}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        setDetails(data);
        setLoading(false);
      })
      .catch((err) => {
        setFetchError(err?.message || 'Failed to fetch student details');
        setLoading(false);
      });
  }, [studentId, studentData]);

  if (!studentId) return null;

  const student: StudentAnalytics | null = details?.student || null;
  const xai: any = student?.xai || null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-zinc-950/80 backdrop-blur-sm">
      <div className="panel-card w-full max-w-3xl rounded-2xl border border-zinc-800 shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        <div className="px-5 py-3.5 border-b border-zinc-800 flex items-center justify-between bg-zinc-900">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-lg bg-zinc-800 border border-zinc-700 flex items-center justify-center text-zinc-100 font-mono font-bold">
              <User className="h-4 w-4" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-base font-bold text-white">{studentId}</h2>
                <span className="text-[10px] text-zinc-400 font-mono">(Anonymous Track ID)</span>
              </div>
              <p className="text-[11px] text-zinc-400">Live Observable Cues &amp; Explainable AI Output</p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all cursor-pointer"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="p-5 overflow-y-auto space-y-4">
          {loading && (
            <div className="p-8 text-center text-xs font-mono text-zinc-400">Loading live student telemetry...</div>
          )}

          {fetchError && (
            <div className="p-4 rounded-xl border border-rose-800 bg-rose-950/40 text-rose-200 text-xs flex items-center gap-2">
              <AlertCircle className="h-4 w-4 shrink-0" />
              <span>Could not reach backend: {fetchError}</span>
            </div>
          )}

          {!loading && student && (
            <>
              <div className="inner-card p-4 rounded-xl border border-zinc-800 flex flex-col md:flex-row items-center justify-between gap-4">
                <div>
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
                    EduSense Student Index (ESI)
                  </span>
                  <div className="flex items-baseline gap-2 mt-0.5">
                    <span className="text-3xl font-extrabold text-white font-mono">
                      {student.face_detected ? student.esi_score.toFixed(1) : 'N/A'}
                    </span>
                    <span className={`text-[11px] px-2 py-0.5 rounded border font-semibold font-mono ${
                      student.final_state === 'DROWSY'
                        ? 'bg-rose-950 text-rose-300 border-rose-800'
                        : student.final_state === 'PHONE DISTRACTED' || student.phone_detected
                        ? 'bg-purple-950 text-purple-300 border-purple-800'
                        : student.final_state === 'ATTENTIVE'
                        ? 'bg-emerald-950 text-emerald-300 border-emerald-800'
                        : student.final_state === 'MODERATE'
                        ? 'bg-amber-950 text-amber-300 border-amber-800'
                        : 'bg-zinc-800 text-zinc-300 border-zinc-700'
                    }`}>
                      State: {student.final_state || student.attention_status || student.status}
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-4 gap-2 text-center text-xs">
                  <div className="bg-zinc-900 p-2 rounded-lg border border-zinc-800">
                    <span className="text-zinc-400 block text-[9px] uppercase">Attention Est.</span>
                    <span className="font-bold text-white font-mono">
                      {student.face_detected ? `${student.attention_pct.toFixed(0)}%` : 'N/A'}
                    </span>
                  </div>
                  <div className="bg-zinc-900 p-2 rounded-lg border border-zinc-800">
                    <span className="text-zinc-400 block text-[9px] uppercase">Expression</span>
                    <span className="font-bold text-white font-mono">{student.emotion}</span>
                  </div>
                  <div className="bg-zinc-900 p-2 rounded-lg border border-zinc-800">
                    <span className="text-zinc-400 block text-[9px] uppercase">Gaze</span>
                    <span className="font-bold text-white font-mono">{student.gaze_vector || student.gaze_direction}</span>
                  </div>
                  <div className="bg-zinc-900 p-2 rounded-lg border border-zinc-800">
                    <span className="text-zinc-400 block text-[9px] uppercase">Posture</span>
                    <span className="font-bold text-white font-mono">{student.posture}</span>
                  </div>
                </div>
              </div>

              {/* Fallback Redirection & Path to Accuracy */}
              {student.corrective_guidance?.is_fallback && (
                <div className="inner-card p-3.5 rounded-xl border border-amber-600/70 bg-amber-950/30 space-y-1.5 shadow-sm">
                  <div className="flex items-center gap-1.5 text-xs font-bold text-amber-300 uppercase tracking-wider">
                    <Compass className="h-4 w-4 text-amber-400" />
                    Corrective Path to Achieve High Accuracy
                  </div>
                  <p className="text-zinc-100 text-xs font-semibold">
                    👉 {student.corrective_guidance.action_prompt}
                  </p>
                  <p className="text-[11px] text-zinc-400">
                    {student.corrective_guidance.target_accuracy_path}
                  </p>
                </div>
              )}

              {/* Student-level AI Suggestion */}
              {student.ai_suggestion && (
                <div className="inner-card p-3.5 rounded-xl border border-zinc-700 bg-gradient-to-r from-zinc-900 via-zinc-900/90 to-zinc-950 space-y-1.5">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1.5 text-xs font-bold text-white uppercase tracking-wider">
                      <Sparkles className="h-4 w-4 text-emerald-400" />
                      Individualized Pedagogical Recommendation
                    </div>
                    <span className={`text-[10px] px-2 py-0.5 rounded font-mono font-bold uppercase border ${
                      student.ai_suggestion.priority === 'HIGH'
                        ? 'bg-rose-950/80 text-rose-300 border-rose-700'
                        : student.ai_suggestion.priority === 'MEDIUM'
                        ? 'bg-amber-950/80 text-amber-300 border-amber-700'
                        : 'bg-emerald-950/80 text-emerald-300 border-emerald-700'
                    }`}>
                      Priority: {student.ai_suggestion.priority}
                    </span>
                  </div>
                  <div className="text-xs">
                    <span className="font-bold text-white text-sm block">{student.ai_suggestion.action}</span>
                    <p className="text-zinc-300 text-xs mt-0.5">{student.ai_suggestion.pedagogical_tip}</p>
                  </div>
                </div>
              )}

              {xai ? (
                <>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div className="inner-card p-3.5 rounded-xl border border-zinc-800">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-1.5">
                          <Eye className="h-3.5 w-3.5 text-zinc-400" />
                          Observable Cue Rationale
                        </h3>
                        <span className="text-[9px] font-mono bg-zinc-800 text-zinc-300 px-1.5 py-0.5 rounded">
                          Conf: {student.confidence_pct != null ? `${student.confidence_pct.toFixed(1)}%` : 'N/A'}
                        </span>
                      </div>
                      <ul className="space-y-1 text-xs text-zinc-300">
                        {(xai.attention?.reasons || ['No rationale available']).map((r: string, idx: number) => (
                          <li key={idx} className="flex items-start gap-1.5">
                            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0 mt-0.5" />
                            <span>{r}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    <div className="inner-card p-3.5 rounded-xl border border-zinc-800">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-1.5">
                          <Activity className="h-3.5 w-3.5 text-zinc-400" />
                          Engagement Attributions
                        </h3>
                      </div>
                      <p className="text-xs text-zinc-400 mb-2">{xai.engagement?.summary_rationale || 'Unavailable'}</p>
                      <ul className="space-y-1 text-xs text-zinc-300 font-mono">
                        {(xai.engagement?.attributions || []).map((a: string, idx: number) => (
                          <li key={idx} className="flex items-start gap-1.5">
                            <Info className="h-3.5 w-3.5 text-zinc-400 shrink-0 mt-0.5" />
                            <span>{a}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>

                  {xai.counterfactual?.explanation && (
                    <div className="inner-card p-3.5 rounded-xl border border-zinc-800 bg-zinc-900/60">
                      <h3 className="text-xs font-bold text-white uppercase tracking-wider mb-1 flex items-center gap-1.5">
                        <Brain className="h-3.5 w-3.5 text-zinc-300" />
                        Counterfactual Modality Analysis
                      </h3>
                      <p className="text-xs font-mono text-zinc-300">{xai.counterfactual.explanation}</p>
                    </div>
                  )}
                </>
              ) : (
                <div className="inner-card p-4 rounded-xl border border-zinc-800 text-xs font-mono text-zinc-400">
                  {details?.recommendations?.[0] || 'Detailed XAI rationale unavailable for this track.'}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};
