import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { ClassroomMetrics } from '../types/analytics';

interface AnalyticsChartsProps {
  metrics: ClassroomMetrics;
}

export const AnalyticsCharts: React.FC<AnalyticsChartsProps> = ({ metrics }) => {
  const students = metrics.students || [];

  // Timeline trend points
  const timelineData = [
    { time: '10:00', attention: 88, engagement: 82, confusion: 12 },
    { time: '10:05', attention: 85, engagement: 80, confusion: 15 },
    { time: '10:10', attention: 82, engagement: 78, confusion: 18 },
    { time: '10:15', attention: 79, engagement: 75, confusion: 22 },
    { time: '10:20', attention: 84, engagement: 81, confusion: 14 },
    { time: '10:25', attention: 89, engagement: 86, confusion: 10 },
    { time: '10:30', attention: metrics.avg_attention, engagement: metrics.avg_engagement, confusion: metrics.avg_confusion },
  ];

  // Emotion count aggregation
  const emotionCounts: Record<string, number> = {
    Neutral: 0,
    Happy: 0,
    Confused: 0,
    Bored: 0,
    Sleepy: 0,
    Frustrated: 0,
  };

  students.forEach((s) => {
    const e = s.emotion || 'Neutral';
    emotionCounts[e] = (emotionCounts[e] || 0) + 1;
  });

  const pieData = Object.entries(emotionCounts).map(([name, value]) => ({ name, value }));
  const COLORS = ['#e4e4e7', '#a1a1aa', '#71717a', '#52525b', '#3f3f46', '#27272a'];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
      {/* Chart 1: Attention & Engagement Timeline */}
      <div className="panel-card p-4 rounded-xl border border-zinc-800">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h3 className="text-xs font-bold text-white uppercase tracking-wider">Classroom Temporal Trend</h3>
            <p className="text-[10px] text-zinc-400">Attention & engagement temporal window</p>
          </div>
          <span className="text-[10px] font-mono text-zinc-400 bg-zinc-900 px-2 py-0.5 rounded border border-zinc-800">
            30m Window
          </span>
        </div>

        <div className="h-56 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={timelineData}>
              <defs>
                <linearGradient id="colorAtt" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#e4e4e7" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#e4e4e7" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="colorEng" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#71717a" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#71717a" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="time" stroke="#52525b" fontSize={10} />
              <YAxis domain={[0, 100]} stroke="#52525b" fontSize={10} />
              <Tooltip
                contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', borderRadius: '8px', fontSize: '11px', color: '#fff' }}
              />
              <Area type="monotone" dataKey="attention" name="Attention %" stroke="#e4e4e7" strokeWidth={1.5} fillOpacity={1} fill="url(#colorAtt)" />
              <Area type="monotone" dataKey="engagement" name="Engagement %" stroke="#71717a" strokeWidth={1.5} fillOpacity={1} fill="url(#colorEng)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Chart 2: Emotion Distribution */}
      <div className="panel-card p-4 rounded-xl border border-zinc-800">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h3 className="text-xs font-bold text-white uppercase tracking-wider">Facial Emotion Distribution</h3>
            <p className="text-[10px] text-zinc-400">Classroom expression breakdown</p>
          </div>
          <span className="text-[10px] font-mono text-zinc-400 bg-zinc-900 px-2 py-0.5 rounded border border-zinc-800">
            Active Frame
          </span>
        </div>

        <div className="h-56 w-full flex items-center justify-center">
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
              <Tooltip contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', borderRadius: '8px', fontSize: '11px' }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
