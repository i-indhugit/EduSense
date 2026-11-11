import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Header } from './components/Header';
import { MetricsGrid } from './components/MetricsGrid';
import { LiveClassroomView } from './components/LiveClassroomView';
import { TeacherRecommendations } from './components/TeacherRecommendations';
import { SpatialHeatmap } from './components/SpatialHeatmap';
import { AnalyticsCharts } from './components/AnalyticsCharts';
import { DebugPanel } from './components/DebugPanel';
import { StudentDetailModal } from './components/StudentDetailModal';
import { SingleStudentView } from './components/SingleStudentView';
import { ToastContainer, ToastMessage } from './components/common/ToastContainer';
import { ClassroomMetrics, StudentAnalytics } from './types/analytics';
import { LayoutGrid, User, RefreshCw } from 'lucide-react';

const API_BASE = 'http://127.0.0.1:8000';
const WS_URL = 'ws://127.0.0.1:8000/ws/classroom';

const initialMetrics: ClassroomMetrics = {
  eci_score: 0.0,
  eci_status: 'No Face Detected',
  avg_attention: 0.0,
  avg_engagement: 0.0,
  avg_participation: 0.0,
  avg_confusion: 0.0,
  drowsy_count: 0,
  hand_raise_count: 0,
  live_detected_persons: 0,
  active_tracks: 0,
  students: [],
  recommendations: [],
};

export function App() {
  const [metrics, setMetrics] = useState<ClassroomMetrics>(initialMetrics);
  const [liveFrame, setLiveFrame] = useState<string>('');
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [selectedStudentId, setSelectedStudentId] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'classroom' | 'focused'>('classroom');
  const [focusedStudentId, setFocusedStudentId] = useState<string | null>(null);
  const [toasts, setToasts] = useState<ToastMessage[]>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const addToast = useCallback((toast: Omit<ToastMessage, 'id'>) => {
    const id = `toast-${Date.now()}-${Math.random().toString().slice(2, 6)}`;
    setToasts((prev) => [...prev, { ...toast, id }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 5000);
  }, []);

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  // WebSocket connection management with auto-reconnect
  useEffect(() => {
    let isMounted = true;

    const connectWebSocket = () => {
      try {
        const ws = new WebSocket(WS_URL);
        wsRef.current = ws;

        ws.onopen = () => {
          if (!isMounted) return;
          setIsConnected(true);
          addToast({
            type: 'success',
            title: 'Vision Pipeline Connected',
            message: 'Streaming real-time classroom analytics and webcam feed at ~20 FPS.',
          });
        };

        ws.onmessage = (event) => {
          if (!isMounted) return;
          try {
            const data = JSON.parse(event.data);
            if (data.frame) {
              setLiveFrame(data.frame);
            }
            if (data.analytics) {
              setMetrics(data.analytics);
            }
          } catch (err) {
            console.error('WebSocket payload parsing error:', err);
          }
        };

        ws.onerror = () => {
          if (!isMounted) return;
          setIsConnected(false);
        };

        ws.onclose = () => {
          if (!isMounted) return;
          setIsConnected(false);
          // Auto-reconnect after 2 seconds
          reconnectTimeoutRef.current = setTimeout(() => {
            if (isMounted) {
              connectWebSocket();
            }
          }, 2000);
        };
      } catch (err) {
        setIsConnected(false);
        reconnectTimeoutRef.current = setTimeout(connectWebSocket, 2000);
      }
    };

    connectWebSocket();

    // Fallback REST polling if WebSocket is not receiving data
    const fallbackPoll = setInterval(() => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        fetch(`${API_BASE}/dashboard`)
          .then((res) => res.json())
          .then((data) => {
            if (data.frame) setLiveFrame(data.frame);
            if (data.classroom) {
              setMetrics({
                ...data.classroom,
                students: data.students || [],
                recommendations: data.recommendations || [],
              });
            }
          })
          .catch(() => {});
      }
    }, 1500);

    return () => {
      isMounted = false;
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      clearInterval(fallbackPoll);
      if (wsRef.current) wsRef.current.close();
    };
  }, [addToast]);

  const handleSelectStudent = (studentId: string) => {
    setSelectedStudentId(studentId);
  };

  const studentsList: StudentAnalytics[] = metrics.students || [];
  const focusedStudent = studentsList.find((s) => s.student_id === focusedStudentId) || studentsList[0];

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col font-sans antialiased selection:bg-emerald-600 selection:text-white">
      {/* Top Header */}
      <Header metrics={metrics} isConnected={isConnected} />

      {/* Main Container */}
      <main className="flex-1 max-w-[1600px] w-full mx-auto p-4 md:p-6 space-y-6">
        {/* Navigation / Mode Switcher */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 bg-zinc-900/60 p-3 rounded-xl border border-zinc-800/80">
          <div className="flex items-center gap-2">
            <button
              onClick={() => setViewMode('classroom')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer ${
                viewMode === 'classroom'
                  ? 'bg-zinc-800 text-white shadow-sm border border-zinc-700'
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              <LayoutGrid className="w-3.5 h-3.5" />
              Classroom View
            </button>
            <button
              onClick={() => {
                if (studentsList.length > 0 && !focusedStudentId) {
                  setFocusedStudentId(studentsList[0].student_id);
                }
                setViewMode('focused');
              }}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer ${
                viewMode === 'focused'
                  ? 'bg-zinc-800 text-white shadow-sm border border-zinc-700'
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              <User className="w-3.5 h-3.5" />
              Focused Student View
            </button>
          </div>

          <div className="flex items-center gap-3 text-xs text-zinc-400 font-mono">
            <span>
              Live Tracks: <strong className="text-emerald-400">{studentsList.length}</strong>
            </span>
            <span className="text-zinc-600">•</span>
            <span>
              FPS: <strong className="text-zinc-200">{isConnected ? '20' : '0'}</strong>
            </span>
            <button
              onClick={() => {
                fetch(`${API_BASE}/dashboard`)
                  .then((r) => r.json())
                  .then((data) => {
                    if (data.frame) setLiveFrame(data.frame);
                    if (data.classroom) {
                      setMetrics({
                        ...data.classroom,
                        students: data.students || [],
                        recommendations: data.recommendations || [],
                      });
                    }
                  });
              }}
              className="p-1 rounded hover:bg-zinc-800 text-zinc-400 hover:text-white transition-all cursor-pointer"
              title="Force Refresh Data"
            >
              <RefreshCw className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* Global Classroom KPI Metrics Grid */}
        <MetricsGrid metrics={metrics} />

        {/* Dynamic View Modes */}
        {viewMode === 'classroom' ? (
          <>
            {/* Primary Grid: Video Stream + Pedagogical Recommendations */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <LiveClassroomView
                  frame={liveFrame}
                  students={studentsList}
                  onSelectStudent={handleSelectStudent}
                />
              </div>
              <div className="lg:col-span-1">
                <TeacherRecommendations recommendations={metrics.recommendations || []} />
              </div>
            </div>

            {/* Secondary Grid: Spatial Heatmap + Analytics Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <SpatialHeatmap students={studentsList} onSelectStudent={handleSelectStudent} />
              <AnalyticsCharts metrics={metrics} />
            </div>
          </>
        ) : (
          <div className="space-y-4">
            {/* Student Picker Pill Bar */}
            {studentsList.length > 0 ? (
              <div className="flex items-center gap-2 overflow-x-auto pb-2">
                <span className="text-xs text-zinc-400 font-semibold uppercase tracking-wider mr-1">
                  Select Student:
                </span>
                {studentsList.map((st) => (
                  <button
                    key={st.student_id}
                    onClick={() => setFocusedStudentId(st.student_id)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition-all cursor-pointer ${
                      focusedStudent?.student_id === st.student_id
                        ? 'bg-emerald-950 text-emerald-300 border border-emerald-700'
                        : 'bg-zinc-900 text-zinc-400 border border-zinc-800 hover:border-zinc-700'
                    }`}
                  >
                    {st.student_id} ({st.attention_pct.toFixed(0)}%)
                  </button>
                ))}
              </div>
            ) : (
              <div className="p-8 text-center text-zinc-500 bg-zinc-900/40 rounded-xl border border-zinc-800">
                No active students detected. Ensure webcam is active and face is visible.
              </div>
            )}

            {focusedStudent && (
              <SingleStudentView student={focusedStudent} frame={liveFrame} />
            )}
          </div>
        )}

        {/* Debug & Telemetry Information Panel */}
        <DebugPanel students={studentsList} />
      </main>

      {/* Deep-Dive Student Detail Modal (opens on student card click) */}
      <StudentDetailModal
        studentId={selectedStudentId}
        onClose={() => setSelectedStudentId(null)}
      />

      {/* Global Toast Alerts */}
      <ToastContainer toasts={toasts} onRemoveToast={removeToast} />
    </div>
  );
}

export default App;
