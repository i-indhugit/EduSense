import React from 'react';
import { Grid } from 'lucide-react';
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
          <p className="text-[10px] text-zinc-400">2D Spatial seating layout & student status cards</p>
        </div>
      </div>

      {/* Front Podium Indicator */}
      <div className="w-full bg-zinc-900 border border-zinc-800 rounded py-1 text-center text-[10px] text-zinc-400 mb-3 tracking-widest uppercase font-mono">
        ▲ INSTRUCTIONAL PODIUM & BOARD ▲
      </div>

      {/* Seating Grid */}
      <div className="grid grid-cols-3 gap-3">
        {students.map((student) => (
          <button
            key={student.student_id}
            onClick={() => onSelectStudent(student.student_id)}
            className="inner-card p-3 rounded-lg border border-zinc-800 hover:border-zinc-600 transition-all text-left cursor-pointer flex flex-col justify-between h-28"
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold font-mono text-white">{student.student_id}</span>
              <span className="text-[10px] font-bold font-mono px-2 py-0.5 rounded bg-zinc-800 text-zinc-300">
                ESI {student.esi_score.toFixed(0)}
              </span>
            </div>

            <div className="space-y-0.5 text-[11px] text-zinc-300 my-1">
              <div className="flex justify-between">
                <span className="text-zinc-400">Attention:</span>
                <span className="font-mono font-semibold">{student.attention_pct.toFixed(0)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-400">Emotion:</span>
                <span>{student.emotion}</span>
              </div>
            </div>

            <div className="flex items-center justify-between text-[10px] pt-1.5 border-t border-zinc-800 text-zinc-400 font-mono">
              <span className="truncate">{student.gaze_direction}</span>
              {student.hand_raised && <span className="px-1.5 py-0.2 rounded bg-zinc-100 text-zinc-950 font-bold text-[9px]">HAND</span>}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};
