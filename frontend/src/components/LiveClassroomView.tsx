import React, { useState } from 'react';
import { Layers, ShieldCheck, User } from 'lucide-react';
import { StudentAnalytics } from '../types/analytics';

interface LiveClassroomViewProps {
  frame?: string;
  students: StudentAnalytics[];
  onSelectStudent: (studentId: string) => void;
}

export const LiveClassroomView: React.FC<LiveClassroomViewProps> = ({
  frame,
  students,
  onSelectStudent,
}) => {
  const [showOverlays, setShowOverlays] = useState(true);

  return (
    <div className="panel-card rounded-xl p-4 border border-zinc-800 flex flex-col h-full">
      {/* Video Feed Header Controls */}
      <div className="flex items-center justify-between mb-3 px-1">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-emerald-400"></span>
          <h2 className="text-xs font-bold text-white tracking-wide uppercase">Vision Feed & Real-Time Face Overlay</h2>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowOverlays(!showOverlays)}
            className={`text-xs px-2.5 py-1 rounded border font-medium flex items-center gap-1 transition-all ${
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

      {/* Main Video Frame Canvas */}
      <div className="relative w-full aspect-video bg-zinc-950 rounded-lg overflow-hidden border border-zinc-800 flex items-center justify-center">
        {frame ? (
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

      {/* Detected Students Selector Cards */}
      <div className="mt-3">
        <div className="flex items-center justify-between mb-2 px-1">
          <span className="text-[11px] font-medium text-zinc-400 uppercase tracking-wider">
            Active Detections ({students.length}) — Select for Deep-Dive
          </span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
          {students.map((student) => (
            <button
              key={student.student_id}
              onClick={() => onSelectStudent(student.student_id)}
              className="inner-card p-2.5 rounded-lg border border-zinc-800 hover:border-zinc-600 transition-all text-left group cursor-pointer"
            >
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-1.5">
                  <User className="h-3.5 w-3.5 text-zinc-400" />
                  <span className="text-xs font-bold text-white group-hover:text-zinc-200">
                    {student.student_id}
                  </span>
                </div>
                {student.confidence_pct && (
                  <span className="text-[10px] font-mono text-zinc-400 flex items-center gap-0.5">
                    <ShieldCheck className="h-3 w-3 text-zinc-400" />
                    {student.confidence_pct.toFixed(1)}%
                  </span>
                )}
              </div>

              <div className="flex items-center justify-between text-[11px] text-zinc-400 font-mono">
                <span>Score: <strong className="text-white">{student.final_state === 'UNABLE TO DETERMINE' ? 'N/A' : `${student.attention_pct.toFixed(0)}%`}</strong></span>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-900 text-emerald-400 border border-zinc-800 font-semibold">
                  {student.final_state || student.attention_status || 'ATTENTIVE'}
                </span>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
