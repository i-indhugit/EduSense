import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { ClassroomMetrics, TimelinePoint } from '../types/analytics';

interface AnalyticsChartsProps {
  metrics: ClassroomMetrics;
  timelineHistory: TimelinePoint[];
}

export const AnalyticsCharts: React.FC<AnalyticsChartsProps> = ({ metrics, timelineHistory }) => {
  const students = metrics.students || [];

  const emotionCounts: Record<string, number> = {};
  students.forEach((s) => {
    const e = s.emotion || 'Unable to Determine';
    emotionCounts[e] = (emotionCounts[e] || 0) + 1;
  });

  const pieData = Object.entries(emotionCounts).map(([name, value]) => ({ name, value }));
  const COLORS = ['#34d399', '#e4e4e7', '#fbbf24', '#60a5fa', '#f87171', '#a1a1aa', '#52525b'];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
      <div className="panel-card p-4 rounded-xl border border-zinc-800">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h3 className="text-xs font-bold text-white uppercase tracking-wider">Live Session Trend</h3>
            <p className="text-[10px] text-zinc-400">Real-time observed attention &amp; engagement history</p>
          </div>
          <span className="text-[10px] font-mono text-zinc-400 bg-zinc-900 px-2 py-0.5 rounded border border-zinc-800">
            {timelineHistory.length} samples
          </span>
        </div>

        <div className="h-56 w-full flex items-center justify-center">
          {timelineHistory.length === 0 ? (
            <span className="text-xs font-mono text-zinc-500">Waiting for live classroom samples...</span>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={timelineHistory}>
                <defs>
                  <linearGradient id="colorAtt" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#34d399" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#34d399" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="colorEng" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#60a5fa" stopOpacity={0.25} />
                    <stop offset="95%" stopColor="#60a5fa" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="time" stroke="#52525b" fontSize={10} />
                <YAxis domain={[0, 100]} stroke="#52525b" fontSize={10} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', borderRadius: '8px', fontSize: '11px', color: '#fff' }}
                />
                <Area type="monotone" dataKey="attention" name="Attention Est. %" stroke="#34d399" strokeWidth={1.5} fillOpacity={1} fill="url(#colorAtt)" />
                <Area type="monotone" dataKey="engagement" name="Engagement Est. %" stroke="#60a5fa" strokeWidth={1.5} fillOpacity={1} fill="url(#colorEng)" />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      <div className="panel-card p-4 rounded-xl border border-zinc-800">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h3 className="text-xs font-bold text-white uppercase tracking-wider">Expression Breakdown</h3>
            <p className="text-[10px] text-zinc-400">Active frame facial expression distribution</p>
          </div>
          <span className="text-[10px] font-mono text-zinc-400 bg-zinc-900 px-2 py-0.5 rounded border border-zinc-800">
            {students.length} Track(s)
          </span>
        </div>

        <div className="h-56 w-full flex items-center justify-center">
          {pieData.length === 0 ? (
            <span className="text-xs font-mono text-zinc-500">No active student detections in frame.</span>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={75}
                  paddingAngle={4}
                  dataKey="value"
                >
                  {pieData.map((_, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', borderRadius: '8px', fontSize: '11px', color: '#fff' }} />
              </PieChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </div>
  );
};
