import React, { useRef, useState } from 'react';
import {
  Sparkles,
  MessageSquare,
  Send,
  AlertTriangle,
  ShieldCheck,
  RefreshCw,
  ArrowRight,
  CheckCircle2,
  KeyRound,
  Compass,
} from 'lucide-react';
import { API_BASE_URL } from '../config';
import { ClassroomMetrics } from '../types/analytics';

export interface FallbackGuidance {
  is_fallback: boolean;
  fallback_reason?: string;
  redirect_action?: string;
  corrective_steps?: string[];
}

export interface GeminiReportData {
  available: boolean;
  source: string;
  headline: string;
  observations_summary: string;
  pedagogical_recommendations: string[];
  limitations_and_uncertainty: string;
  answer_to_query?: string | null;
  fallback_guidance?: FallbackGuidance | null;
}

interface GeminiInsightsPanelProps {
  metrics: ClassroomMetrics;
  backendOnline: boolean;
}

export const GeminiInsightsPanel: React.FC<GeminiInsightsPanelProps> = ({ metrics, backendOnline }) => {
  const [report, setReport] = useState<GeminiReportData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [query, setQuery] = useState<string>('');
  const [apiKeyInput, setApiKeyInput] = useState<string>(() => localStorage.getItem('edusense_gemini_key') || '');
  const [showKeyInput, setShowKeyInput] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleSaveKey = (key: string) => {
    setApiKeyInput(key);
    if (key.trim()) {
      localStorage.setItem('edusense_gemini_key', key.trim());
    } else {
      localStorage.removeItem('edusense_gemini_key');
    }
  };

  const fetchGeminiReport = async (userQuery?: string) => {
    if (!backendOnline) {
      setError('Backend service is offline or sleeping (Render cold-start).');
      return;
    }
    setLoading(true);
    setError(null);

    const students = metrics.students || [];
    const withFaces = students.filter((s) => s.face_detected);
    const drowsyCount = students.filter((s) => s.is_drowsy).length;
    const phoneCount = students.filter((s) => s.phone_detected).length;
    const handCount = students.filter((s) => s.hand_raised).length;
    const activeTracks = metrics.active_tracks ?? students.length;

    const studentSummaries = withFaces.map((s) => {
      const tip = s.ai_suggestion ? `[Action: ${s.ai_suggestion.action}]` : '';
      return `${s.student_id}: state=${s.final_state || 'ATTENTIVE'} (att=${Math.round(s.attention_pct)}%, cue=${s.behavioral_cue || 'normal'}, pose=${s.posture}) ${tip}`;
    });

    const payload = {
      total_students: activeTracks,
      valid_observations: withFaces.length,
      looking_toward_front: Math.max(
        0,
        withFaces.filter((s) => s.gaze_direction === 'Looking at Board' || s.attention_status?.includes('Note-Taking')).length
      ),
      looking_away: withFaces.filter((s) => s.gaze_direction !== 'Looking at Board').length,
      hand_raised: handCount,
      insufficient_evidence: Math.max(0, activeTracks - withFaces.length),
      drowsy_count: drowsyCount,
      phone_distracted: phoneCount,
      avg_attention_score: metrics.avg_attention ?? 0,
      avg_confusion_index: metrics.avg_confusion ?? 0,
      student_summaries: studentSummaries,
      api_key_override: apiKeyInput.trim() || null,
      user_query: userQuery?.trim() || null,
    };

    try {
      const res = await fetch(`${API_BASE_URL}/api/gemini/report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: GeminiReportData = await res.json();
      setReport(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to retrieve Gemini report.');
    } finally {
      setLoading(false);
    }
  };

  const handleAsk = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    fetchGeminiReport(query.trim());
    setQuery('');
  };

  return (
    <div className="panel-card p-4 rounded-xl border border-zinc-800 flex flex-col space-y-3 bg-zinc-900/40">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-emerald-950/80 border border-emerald-700/60 text-emerald-300">
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-1.5">
              Gemini Pedagogical Insights
            </h3>
            <p className="text-[10px] text-zinc-400">
              Aggregated anonymized reporting • Zero video/biometrics transmitted
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowKeyInput((s) => !s)}
            className="px-2.5 py-1 text-xs font-semibold rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-300 border border-zinc-700 transition-all cursor-pointer"
            title="Configure Gemini API Key"
          >
            {apiKeyInput ? 'API Key Configured' : 'Set Gemini Key'}
          </button>
          <button
            onClick={() => fetchGeminiReport()}
            disabled={loading || !backendOnline}
            className="px-2.5 py-1 text-xs font-semibold rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white border border-emerald-500 flex items-center gap-1.5 transition-all disabled:opacity-50 cursor-pointer"
            title="Generate fresh session report"
          >
            <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} />
            {loading ? 'Analyzing...' : 'Generate Insight'}
          </button>
        </div>
      </div>

      {showKeyInput && (
        <div className="p-2.5 rounded-lg bg-zinc-950/70 border border-zinc-800 flex items-center gap-2">
          <input
            type="password"
            value={apiKeyInput}
            onChange={(e) => handleSaveKey(e.target.value)}
            placeholder="Paste your Google Gemini API Key (starts with AIzaSy...)"
            className="flex-1 bg-zinc-900 border border-zinc-700 rounded px-2.5 py-1 text-xs text-zinc-200 placeholder:text-zinc-500 focus:outline-none"
          />
          <button
            onClick={() => setShowKeyInput(false)}
            className="px-2.5 py-1 text-xs font-semibold rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-200 border border-zinc-700 cursor-pointer"
          >
            Done
          </button>
        </div>
      )}

      {error && (
        <div className="text-[11px] text-amber-300 bg-amber-950/40 border border-amber-800/60 rounded-lg p-2.5 flex items-center gap-1.5">
          <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {report ? (
        <div className="space-y-3 text-xs">
          <div className="p-3 rounded-lg bg-zinc-900/80 border border-zinc-800 space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="font-semibold text-emerald-400">{report.headline}</span>
              <span className="text-[10px] font-mono text-zinc-400 bg-zinc-800 px-1.5 py-0.5 rounded">
                {report.source}
              </span>
            </div>
            <p className="text-zinc-300 leading-relaxed text-[11px]">{report.observations_summary}</p>
          </div>

          {report.fallback_guidance?.is_fallback && (
            <div className="p-3 rounded-lg bg-amber-950/30 border border-amber-800/60 space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5 text-amber-300 font-semibold text-[11px]">
                  <Compass className="w-3.5 h-3.5 text-amber-400" />
                  <span>Fallback Active: Correct Path Redirection</span>
                </div>
                {report.fallback_guidance.redirect_action === 'SET_GEMINI_API_KEY' && (
                  <button
                    onClick={() => setShowKeyInput(true)}
                    className="px-2 py-0.5 rounded bg-amber-500 hover:bg-amber-400 text-black font-bold text-[10px] flex items-center gap-1 cursor-pointer transition-colors shadow-sm"
                  >
                    <KeyRound className="w-3 h-3" />
                    <span>Connect Gemini Key</span>
                  </button>
                )}
              </div>

              <p className="text-[10px] text-amber-200/80 leading-normal">
                To upgrade from rule approximations to high-accuracy generative pedagogical analysis, follow these corrective steps:
              </p>

              {report.fallback_guidance.corrective_steps && report.fallback_guidance.corrective_steps.length > 0 && (
                <div className="space-y-1 pt-0.5">
                  {report.fallback_guidance.corrective_steps.map((step, idx) => (
                    <div key={idx} className="flex items-start gap-1.5 text-[10px] text-zinc-300">
                      <span className="flex items-center justify-center w-3.5 h-3.5 rounded-full bg-amber-900/60 text-amber-300 font-mono text-[9px] shrink-0 mt-0.5">
                        {idx + 1}
                      </span>
                      <span>{step}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {report.answer_to_query && (
            <div className="p-2.5 rounded-lg bg-zinc-800/70 border border-zinc-700 space-y-1">
              <div className="text-[10px] font-bold text-zinc-400 uppercase tracking-wider">Answer to Query:</div>
              <p className="text-zinc-200 text-[11px]">{report.answer_to_query}</p>
            </div>
          )}

          {report.pedagogical_recommendations?.length > 0 && (
            <div className="space-y-1.5">
              <span className="text-[10px] font-bold text-zinc-400 uppercase tracking-wider">
                Recommended Pedagogical Actions:
              </span>
              <ul className="space-y-1">
                {report.pedagogical_recommendations.map((rec, i) => (
                  <li
                    key={i}
                    className="text-[11px] text-zinc-300 bg-zinc-900/50 p-2 rounded border border-zinc-800/80 flex items-start gap-1.5"
                  >
                    <span className="text-emerald-400 font-bold shrink-0">•</span>
                    <span>{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="p-2 rounded bg-zinc-950/60 border border-zinc-800/80 flex items-start gap-1.5 text-[10px] text-zinc-400">
            <ShieldCheck className="w-3.5 h-3.5 text-zinc-400 shrink-0 mt-0.5" />
            <span>{report.limitations_and_uncertainty}</span>
          </div>
        </div>
      ) : (
        <div className="text-center py-6 px-4 border border-dashed border-zinc-800 rounded-lg text-zinc-400 text-xs">
          Click <strong className="text-zinc-200">Generate Insight</strong> or submit a natural-language question below
          to analyze current aggregate classroom engagement.
        </div>
      )}

      {/* Interactive Natural-Language Q&A */}
      <form onSubmit={handleAsk} className="flex items-center gap-2 pt-1">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask Gemini about current engagement (e.g., 'Are students participating?')"
          className="flex-1 bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-1.5 text-xs text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:border-emerald-600 transition-colors"
        />
        <button
          type="submit"
          disabled={loading || !query.trim() || !backendOnline}
          className="px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs flex items-center gap-1 transition-all disabled:opacity-50 cursor-pointer"
        >
          <Send className="w-3 h-3" />
          <span>Ask</span>
        </button>
      </form>
    </div>
  );
};

