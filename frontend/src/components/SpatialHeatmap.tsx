import React from 'react';
import { Grid, Sparkles } from 'lucide-react';
import { StudentAnalytics } from '../types/analytics';

interface SpatialHeatmapProps {
  students: StudentAnalytics[];
  onSelectStudent: (studentId: string) => void;
}

export const SpatialHeatmap: React.FC<SpatialHeatmapProps> = ({ students, onSelectStudent }) => {
  return (
    <div className="panel-card p-4 rounded-xl border border-zinc-800 mb-4">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-1.5">
            <Grid className="h-3.5 w-3.5 text-zinc-300" />
            Classroom Seating Spatial Matrix
          </h3>
          <p className="text-[10px] text-zinc-400">2D Spatial seating layout &amp; student status cards</p>
        </div>
      </div>

      {/* Front Podium Indicator */}
      <div className="w-full bg-zinc-900 border border-zinc-800 rounded py-1 text-center text-[10px] text-zinc-400 mb-3 tracking-widest uppercase font-mono">
        ▲ INSTRUCTIONAL PODIUM &amp; BOARD ▲
      </div>

      {/* Seating Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {students.map((student) => {
          const isDrowsy = student.final_state === 'DROWSY' || student.is_drowsy;
          const isAttentive = student.final_state === 'ATTENTIVE';
          const isModerate = student.final_state === 'MODERATE';

          const badgeClass = isDrowsy
            ? 'bg-rose-950/80 text-rose-300 border-rose-800'
            : isAttentive
            ? 'bg-emerald-950/80 text-emerald-300 border-emerald-800'
            : isModerate
            ? 'bg-amber-950/80 text-amber-300 border-amber-800'
            : 'bg-zinc-800 text-zinc-400 border-zinc-700';

          return (
            <button
              key={student.student_id}
              onClick={() => onSelectStudent(student.student_id)}
              className="inner-card p-3 rounded-lg border border-zinc-800 hover:border-zinc-600 transition-all text-left cursor-pointer flex flex-col justify-between min-h-[7.5rem]"
            >
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-xs font-bold font-mono text-white">{student.student_id}</span>
                  <span className={`text-[10px] font-bold font-mono px-2 py-0.5 rounded border ${badgeClass}`}>
                    {student.final_state || student.attention_status || 'ATTENTIVE'}
                  </span>
                </div>

                <div className="space-y-0.5 text-[11px] text-zinc-300 my-1">
                  <div className="flex justify-between">
                    <span className="text-zinc-400">Attention Est:</span>
                    <span className="font-mono font-semibold text-white">
                      {student.final_state === 'UNABLE TO DETERMINE' ? 'N/A' : `${student.attention_pct.toFixed(0)}%`}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-zinc-400">Expression:</span>
                    <span>{student.emotion}</span>
                  </div>
                </div>
              </div>

              {student.ai_suggestion && (
                <div className="flex items-center gap-1 text-[10px] pt-1.5 border-t border-zinc-800 text-zinc-300">
                  <Sparkles className="h-3 w-3 text-emerald-400 shrink-0" />
                  <span className="truncate">
                    <strong className="text-zinc-100">{student.ai_suggestion.action}</strong>
                  </span>
                </div>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
};
