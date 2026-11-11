import { Lightbulb, ArrowRight } from 'lucide-react';
import { TeacherRecommendation } from '../types/analytics';

interface TeacherRecommendationsProps {
  recommendations: TeacherRecommendation[];
}

export const TeacherRecommendations: React.FC<TeacherRecommendationsProps> = ({ recommendations }) => {
  return (
    <div className="panel-card p-4 rounded-xl border border-zinc-800 h-full flex flex-col">
      <div className="flex items-center justify-between mb-3 px-1">
        <div>
          <h3 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-1.5">
            <Lightbulb className="h-3.5 w-3.5 text-zinc-300" />
            Pedagogical Recommendations
          </h3>
          <p className="text-[10px] text-zinc-400">Automated feedback from real-time classroom analytics</p>
        </div>
      </div>

      <div className="flex-1 space-y-2 overflow-y-auto max-h-[300px] pr-1">
        {recommendations.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center p-6 text-zinc-400">
            <p className="text-xs">No active pedagogical alerts.</p>
            <p className="text-[10px] text-zinc-500 mt-1">Classroom engagement metrics are optimal.</p>
          </div>
        ) : (
          recommendations.map((rec) => (
            <div
              key={rec.id}
              className="p-3 rounded-lg border border-zinc-800 bg-zinc-900/60 flex flex-col justify-between space-y-2 hover:border-zinc-700 transition-all"
            >
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-bold text-white">{rec.title}</span>
                  <span className="text-[9px] font-mono text-zinc-400 bg-zinc-800 px-1.5 py-0.5 rounded border border-zinc-700">
                    {rec.type}
                  </span>
                </div>
                <p className="text-xs text-zinc-300">{rec.message}</p>
              </div>

              <div className="flex items-center gap-1.5 text-[11px] font-semibold text-zinc-200 bg-zinc-800/80 px-2.5 py-1 rounded border border-zinc-700/60 self-start">
                <span>{rec.action}</span>
                <ArrowRight className="h-3 w-3 text-zinc-400" />
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
