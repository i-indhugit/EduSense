import { useEffect, useState } from 'react';
import { X, User, Activity, Eye, Brain, CheckCircle2, Info } from 'lucide-react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip } from 'recharts';
import { StudentAnalytics } from '../types/analytics';

interface StudentDetailModalProps {
  studentId: string | null;
  onClose: () => void;
}

export const StudentDetailModal: React.FC<StudentDetailModalProps> = ({ studentId, onClose }) => {
  const [details, setDetails] = useState<any>(null);

  useEffect(() => {
    if (!studentId) return;

    fetch(`http://localhost:8000/student/${studentId}`)
      .then((res) => res.json())
      .then((data) => {
        setDetails(data);
      })
      .catch((err) => {
        console.error('Error fetching student details:', err);
      });
  }, [studentId]);

  if (!studentId) return null;

  const student: StudentAnalytics = details?.student || {
    student_id: studentId,
    attention_pct: 85.0,
    gaze_direction: 'Looking at Board',
    emotion: 'Neutral',
    is_drowsy: false,
    is_yawning: false,
    hand_raised: false,
    posture: 'Upright',
    participation_pct: 75.0,
    engagement_pct: 82.0,
    confusion_index: 12.0,
    confusion_level: 'Low',
    esi_score: 84.0,
    status: 'Good',
  };

  const history = details?.history || [
    { time: '-14m', attention: 88, engagement: 82, confusion: 10, esi_score: 85 },
    { time: '-12m', attention: 86, engagement: 81, confusion: 12, esi_score: 84 },
    { time: '-10m', attention: 80, engagement: 77, confusion: 18, esi_score: 80 },
    { time: '-8m', attention: 75, engagement: 72, confusion: 25, esi_score: 75 },
    { time: '-6m', attention: 82, engagement: 79, confusion: 15, esi_score: 81 },
    { time: '-4m', attention: 87, engagement: 84, confusion: 11, esi_score: 86 },
    { time: 'Now', attention: student.attention_pct, engagement: student.engagement_pct, confusion: student.confusion_index, esi_score: student.esi_score },
  ];

  const xai: any = student.xai || {
    attention: {
      confidence_pct: 94.0,
      reasons: [
        "Eye gaze vector aligned with lecture board (angular deviation < 15°).",
        "Head pose orientation: Pitch -4.0°, Yaw 2.0°.",
        "Eyes open with regular blinking cycle (EAR >= 0.25).",
        "Mobile device absent from desk zone."
      ]
    },
    engagement: {
      summary_rationale: "Engagement score driven by Neutral sentiment, Upright posture, and 85% attention alignment.",
      attributions: [
        "+34.0% from High Sustained Attention (85%).",
        "+24.0% from Positive/Receptive Expression (Neutral).",
        "+15.0% from Upright Body Posture."
      ]
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-zinc-950/80 backdrop-blur-sm">
      <div className="panel-card w-full max-w-3xl rounded-2xl border border-zinc-800 shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="px-5 py-3.5 border-b border-zinc-800 flex items-center justify-between bg-zinc-900">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-lg bg-zinc-800 border border-zinc-700 flex items-center justify-center text-zinc-100 font-mono font-bold">
              <User className="h-4 w-4" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-base font-bold text-white">{student.student_id}</h2>
                <span className="text-[10px] text-zinc-400 font-mono">(Anonymous ID)</span>
              </div>
              <p className="text-[11px] text-zinc-400">Explainable AI (XAI) Rationale & Temporal Trends</p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Content Body */}
        <div className="p-5 overflow-y-auto space-y-4">
          {/* ESI Score Hero Card */}
          <div className="inner-card p-4 rounded-xl border border-zinc-800 flex flex-col md:flex-row items-center justify-between gap-4">
            <div>
              <span className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">EduSense Student Index (ESI)</span>
              <div className="flex items-baseline gap-2 mt-0.5">
                <span className="text-3xl font-extrabold text-white font-mono">{student.esi_score.toFixed(1)}</span>
                <span className="text-[11px] px-2 py-0.5 rounded border border-zinc-700 bg-zinc-800 text-zinc-300 font-semibold font-mono">
                  Status: {student.status}
                </span>
              </div>
            </div>

            {/* ESI Component Weights */}
            <div className="grid grid-cols-4 gap-2 text-center text-xs">
              <div className="bg-zinc-900 p-2 rounded-lg border border-zinc-800">
                <span className="text-zinc-400 block text-[9px] uppercase">Attention</span>
                <span className="font-bold text-white font-mono">{student.attention_pct.toFixed(0)}%</span>
              </div>
              <div className="bg-zinc-900 p-2 rounded-lg border border-zinc-800">
                <span className="text-zinc-400 block text-[9px] uppercase">Emotion</span>
                <span className="font-bold text-white font-mono">{student.emotion}</span>
              </div>
              <div className="bg-zinc-900 p-2 rounded-lg border border-zinc-800">
                <span className="text-zinc-400 block text-[9px] uppercase">Part.</span>
                <span className="font-bold text-white font-mono">{student.participation_pct.toFixed(0)}%</span>
              </div>
              <div className="bg-zinc-900 p-2 rounded-lg border border-zinc-800">
                <span className="text-zinc-400 block text-[9px] uppercase">Posture</span>
                <span className="font-bold text-white font-mono">{student.posture}</span>
              </div>
            </div>
          </div>

          {/* Explainable AI Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="inner-card p-3.5 rounded-xl border border-zinc-800">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-1.5">
                  <Eye className="h-3.5 w-3.5 text-zinc-400" />
                  Attention XAI Rationale
                </h3>
                <span className="text-[9px] font-mono bg-zinc-800 text-zinc-300 px-1.5 py-0.5 rounded">
                  Conf: {xai.attention?.confidence_pct || 94}%
                </span>
              </div>
              <ul className="space-y-1 text-xs text-zinc-300">
                {xai.attention?.reasons?.map((r: string, idx: number) => (
                  <li key={idx} className="flex items-start gap-1.5">
                    <CheckCircle2 className="h-3.5 w-3.5 text-zinc-400 shrink-0 mt-0.5" />
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
              <p className="text-xs text-zinc-400 mb-2">{xai.engagement?.summary_rationale}</p>
              <ul className="space-y-1 text-xs text-zinc-300 font-mono">
                {xai.engagement?.attributions?.map((a: string, idx: number) => (
                  <li key={idx} className="flex items-start gap-1.5">
                    <Info className="h-3.5 w-3.5 text-zinc-400 shrink-0 mt-0.5" />
                    <span>{a}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Counterfactual XAI Panel */}
          {xai.counterfactual && (
            <div className="inner-card p-3.5 rounded-xl border border-zinc-800 bg-zinc-900/60">
              <h3 className="text-xs font-bold text-white uppercase tracking-wider mb-1 flex items-center gap-1.5">
                <Brain className="h-3.5 w-3.5 text-zinc-300" />
                Counterfactual XAI Analysis
              </h3>
              <p className="text-xs font-mono text-zinc-300">
                {xai.counterfactual.explanation || "Removing top modality shifts prediction state."}
              </p>
            </div>
          )}

          {/* Timeline History Chart */}
          <div className="inner-card p-4 rounded-xl border border-zinc-800">
            <h3 className="text-xs font-bold text-white mb-2 uppercase tracking-wider">Individual Metric Timeline</h3>
            <div className="h-44 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={history}>
                  <XAxis dataKey="time" stroke="#52525b" fontSize={10} />
                  <YAxis domain={[0, 100]} stroke="#52525b" fontSize={10} />
                  <Tooltip contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', borderRadius: '8px', fontSize: '11px', color: '#fff' }} />
                  <Line type="monotone" dataKey="attention" name="Attention %" stroke="#e4e4e7" strokeWidth={1.5} dot={false} />
                  <Line type="monotone" dataKey="engagement" name="Engagement %" stroke="#a1a1aa" strokeWidth={1.5} dot={false} />
                  <Line type="monotone" dataKey="esi_score" name="ESI Score" stroke="#71717a" strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
