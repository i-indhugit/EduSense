import { Eye, Hand, Moon, Brain, HelpCircle, Smartphone } from 'lucide-react';
import { ClassroomMetrics } from '../types/analytics';

interface MetricsGridProps {
  metrics: ClassroomMetrics;
}

export const MetricsGrid: React.FC<MetricsGridProps> = ({ metrics }) => {
  const phoneCount = metrics.phone_distraction_count ?? 0;

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-7 gap-2.5 sm:gap-3 mb-4">
      {/* Average Attention Card */}
      <div className="inner-card p-3.5 rounded-xl border border-zinc-800">
        <div className="flex items-center justify-between text-zinc-400 mb-2">
          <span className="text-[11px] font-medium uppercase tracking-wider">Avg Attention</span>
          <Eye className="h-3.5 w-3.5 text-zinc-400" />
        </div>
        <div className="flex items-baseline gap-1">
          <span className="text-xl font-bold text-white font-mono">{metrics.avg_attention.toFixed(1)}%</span>
        </div>
        <div className="w-full bg-zinc-800 h-1 rounded mt-2.5 overflow-hidden">
          <div 
            className="bg-zinc-300 h-full rounded transition-all duration-300" 
            style={{ width: `${Math.min(100, metrics.avg_attention)}%` }}
          />
        </div>
      </div>

      {/* Average Engagement Card */}
      <div className="inner-card p-3.5 rounded-xl border border-zinc-800">
        <div className="flex items-center justify-between text-zinc-400 mb-2">
          <span className="text-[11px] font-medium uppercase tracking-wider">Engagement</span>
          <Brain className="h-3.5 w-3.5 text-zinc-400" />
        </div>
        <div className="flex items-baseline gap-1">
          <span className="text-xl font-bold text-white font-mono">{metrics.avg_engagement.toFixed(1)}%</span>
        </div>
        <div className="w-full bg-zinc-800 h-1 rounded mt-2.5 overflow-hidden">
          <div 
            className="bg-zinc-300 h-full rounded transition-all duration-300" 
            style={{ width: `${Math.min(100, metrics.avg_engagement)}%` }}
          />
        </div>
      </div>

      {/* Participation Rate Card */}
      <div className="inner-card p-3.5 rounded-xl border border-zinc-800">
        <div className="flex items-center justify-between text-zinc-400 mb-2">
          <span className="text-[11px] font-medium uppercase tracking-wider">Participation</span>
          <Hand className="h-3.5 w-3.5 text-zinc-400" />
        </div>
        <div className="flex items-baseline gap-1">
          <span className="text-xl font-bold text-white font-mono">{metrics.avg_participation.toFixed(1)}%</span>
        </div>
        <div className="w-full bg-zinc-800 h-1 rounded mt-2.5 overflow-hidden">
          <div 
            className="bg-zinc-300 h-full rounded transition-all duration-300" 
            style={{ width: `${Math.min(100, metrics.avg_participation)}%` }}
          />
        </div>
      </div>

      {/* Confusion Index Card */}
      <div className="inner-card p-3.5 rounded-xl border border-zinc-800">
        <div className="flex items-center justify-between text-zinc-400 mb-2">
          <span className="text-[11px] font-medium uppercase tracking-wider">Confusion Index</span>
          <HelpCircle className="h-3.5 w-3.5 text-zinc-400" />
        </div>
        <div className="flex items-baseline gap-1">
          <span className="text-xl font-bold text-white font-mono">{metrics.avg_confusion.toFixed(1)}</span>
        </div>
        <div className="w-full bg-zinc-800 h-1 rounded mt-2.5 overflow-hidden">
          <div 
            className="bg-zinc-400 h-full rounded transition-all duration-300" 
            style={{ width: `${Math.min(100, metrics.avg_confusion)}%` }}
          />
        </div>
      </div>

      {/* Raised Hands Count Card */}
      <div className="inner-card p-3.5 rounded-xl border border-zinc-800">
        <div className="flex items-center justify-between text-zinc-400 mb-2">
          <span className="text-[11px] font-medium uppercase tracking-wider">Raised Hands</span>
          <Hand className="h-3.5 w-3.5 text-zinc-400" />
        </div>
        <div className="flex items-baseline gap-1">
          <span className="text-xl font-bold text-white font-mono">{metrics.hand_raise_count}</span>
          <span className="text-[10px] text-zinc-400">students</span>
        </div>
        <div className="text-[10px] text-zinc-500 mt-2 font-mono">Active engagement</div>
      </div>

      {/* Sleep / Drowsiness Count Card */}
      <div className="inner-card p-3.5 rounded-xl border border-zinc-800">
        <div className="flex items-center justify-between text-zinc-400 mb-2">
          <span className="text-[11px] font-medium uppercase tracking-wider">Drowsiness</span>
          <Moon className="h-3.5 w-3.5 text-zinc-400" />
        </div>
        <div className="flex items-baseline gap-1">
          <span className="text-xl font-bold text-white font-mono">{metrics.drowsy_count}</span>
          <span className="text-[10px] text-zinc-400">flagged</span>
        </div>
        <div className="text-[10px] text-zinc-500 mt-2 font-mono">PERCLOS alert</div>
      </div>

      {/* Phone Distraction Card */}
      <div className={`inner-card p-3.5 rounded-xl border transition-colors ${phoneCount > 0 ? 'border-purple-800/80 bg-purple-950/20' : 'border-zinc-800'}`}>
        <div className="flex items-center justify-between text-zinc-400 mb-2">
          <span className="text-[11px] font-medium uppercase tracking-wider">Phone Use</span>
          <Smartphone className={`h-3.5 w-3.5 ${phoneCount > 0 ? 'text-purple-400 animate-pulse' : 'text-zinc-400'}`} />
        </div>
        <div className="flex items-baseline gap-1">
          <span className={`text-xl font-bold font-mono ${phoneCount > 0 ? 'text-purple-300' : 'text-white'}`}>
            {phoneCount}
          </span>
          <span className="text-[10px] text-zinc-400">flagged</span>
        </div>
        <div className={`text-[10px] mt-2 font-mono ${phoneCount > 0 ? 'text-purple-400 font-medium' : 'text-zinc-500'}`}>
          {phoneCount > 0 ? 'Secondary device active' : 'Zero phone distraction'}
        </div>
      </div>
    </div>
  );
};
