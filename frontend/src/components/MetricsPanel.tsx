import { Cpu, Gauge, MemoryStick, Zap, MonitorDot, Database, Clock, ThermometerSun, Cpu as CpuChip, Gauge as GaugeIcon } from 'lucide-react';
import type { Metrics, HardwareInfo } from '../types';

interface MetricsPanelProps {
  metrics?: Metrics | null;
  hardware?: HardwareInfo | null;
  history?: Metrics[];
  compact?: boolean;
}

function MetricCard({
  icon: Icon,
  label,
  value,
  unit,
  subtitle,
  accent = 'snap-blue',
  progress,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string | number;
  unit?: string;
  subtitle?: string;
  accent?: 'snap-blue' | 'green' | 'amber' | 'red' | 'purple';
  progress?: number;
}) {
  const accentClasses: Record<string, string> = {
    'snap-blue': 'bg-snap-blue/10 text-snap-blue',
    green: 'bg-green-50 text-green-700',
    amber: 'bg-amber-50 text-amber-700',
    red: 'bg-red-50 text-red-700',
    purple: 'bg-purple-50 text-purple-700',
  };

  return (
    <div className="bg-white border border-border rounded-lg p-4 hover:shadow-card-hover transition-shadow">
      <div className="flex items-start justify-between mb-3">
        <div className={`w-9 h-9 rounded-md flex items-center justify-center ${accentClasses[accent]}`}>
          <Icon className="w-4.5 h-4.5" />
        </div>
        {subtitle && (
          <span className="text-[10px] text-text-secondary font-mono uppercase tracking-wide">
            {subtitle}
          </span>
        )}
      </div>
      <div className="flex items-baseline gap-1 mb-2">
        <span className="text-2xl font-bold text-navy tabular-nums">{value}</span>
        {unit && <span className="text-sm text-text-secondary font-medium">{unit}</span>}
      </div>
      <p className="text-xs font-medium text-text-secondary uppercase tracking-wide">{label}</p>
      {progress !== undefined && (
        <div className="mt-3 w-full h-1.5 bg-slate-100 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${accent === 'snap-blue' ? 'bg-snap-blue' : accent === 'green' ? 'bg-green-500' : accent === 'amber' ? 'bg-amber-500' : accent === 'red' ? 'bg-red-500' : 'bg-purple-500'}`}
            style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
          />
        </div>
      )}
    </div>
  );
}

export default function MetricsPanel({ metrics, hardware, history = [], compact = false }: MetricsPanelProps) {
  const cpuAccent = metrics && metrics.cpu_usage_pct > 85 ? 'red' : metrics && metrics.cpu_usage_pct > 60 ? 'amber' : 'green';
  const memAccent = metrics && (metrics.memory_usage_mb / (hardware?.ram_gb || 8) / 128) > 0.85 ? 'red' : metrics && (metrics.memory_usage_mb / (hardware?.ram_gb || 8) / 128) > 0.6 ? 'amber' : 'green';
  const latencyAccent = metrics && metrics.latency_ms > 200 ? 'red' : metrics && metrics.latency_ms > 100 ? 'amber' : 'snap-blue';

  return (
    <div className="space-y-4">
      <div className={`grid gap-3 ${compact ? 'grid-cols-2 sm:grid-cols-3' : 'grid-cols-2 sm:grid-cols-3 lg:grid-cols-4'}`}>
        <MetricCard
          icon={Gauge}
          label="Frame Rate"
          value={metrics?.fps?.toFixed(1) || '0.0'}
          unit="FPS"
          subtitle="Real-time"
          accent={metrics && metrics.fps < 15 ? 'amber' : 'green'}
          progress={metrics ? (metrics.fps / 60) * 100 : 0}
        />
        <MetricCard
          icon={Clock}
          label="Inference Latency"
          value={metrics?.latency_ms?.toFixed(0) || '0'}
          unit="ms"
          subtitle="Per frame"
          accent={latencyAccent}
          progress={metrics ? Math.max(0, 100 - (metrics.latency_ms / 300) * 100) : 0}
        />
        <MetricCard
          icon={Cpu}
          label="CPU Usage"
          value={metrics?.cpu_usage_pct?.toFixed(1) || '0.0'}
          unit="%"
          subtitle="All cores"
          accent={cpuAccent}
          progress={metrics?.cpu_usage_pct || 0}
        />
        <MetricCard
          icon={MemoryStick}
          label="Memory Usage"
          value={metrics?.memory_usage_mb ? (metrics.memory_usage_mb / 1024).toFixed(2) : '0.00'}
          unit="GB"
          subtitle={hardware?.ram_gb ? `of ${hardware.ram_gb} GB` : 'RAM'}
          accent={memAccent}
          progress={metrics && hardware?.ram_gb ? (metrics.memory_usage_mb / (hardware.ram_gb * 1024)) * 100 : undefined}
        />
      </div>

      <div className="card rounded-lg p-4">
        <div className="flex items-center gap-2 mb-4">
          <MonitorDot className="w-4 h-4 text-snap-blue" />
          <h3 className="font-semibold text-sm text-navy">System Configuration</h3>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <div className="bg-surface-alt rounded-md p-3">
            <div className="flex items-center gap-2 mb-2">
              <Database className="w-4 h-4 text-snap-blue" />
              <span className="text-xs font-medium text-text-secondary uppercase tracking-wide">Backend</span>
            </div>
            <p className="text-sm font-semibold text-navy capitalize">{metrics?.backend_used || 'N/A'}</p>
          </div>
          <div className="bg-surface-alt rounded-md p-3">
            <div className="flex items-center gap-2 mb-2">
              <CpuChip className="w-4 h-4 text-snap-blue" />
              <span className="text-xs font-medium text-text-secondary uppercase tracking-wide">Model</span>
            </div>
            <p className="text-sm font-semibold text-navy">{metrics?.model_name || 'N/A'}</p>
          </div>
          <div className="bg-surface-alt rounded-md p-3">
            <div className="flex items-center gap-2 mb-2">
              <GaugeIcon className="w-4 h-4 text-snap-blue" />
              <span className="text-xs font-medium text-text-secondary uppercase tracking-wide">Input Resolution</span>
            </div>
            <p className="text-sm font-semibold text-navy font-mono">{metrics?.input_resolution || 'N/A'}</p>
          </div>
        </div>
      </div>

      {!compact && (
        <>
          {(metrics?.gpu_usage_pct !== undefined || metrics?.npu_usage_pct !== undefined || metrics?.power_watts !== undefined || metrics?.temperature_c !== undefined) && (
            <div className={`grid gap-3 ${compact ? 'grid-cols-2' : 'grid-cols-2 sm:grid-cols-4'}`}>
              {metrics?.gpu_usage_pct !== undefined && (
                <MetricCard
                  icon={MonitorDot}
                  label="GPU Usage"
                  value={metrics.gpu_usage_pct.toFixed(1)}
                  unit="%"
                  accent={metrics.gpu_usage_pct > 85 ? 'red' : 'purple'}
                  progress={metrics.gpu_usage_pct}
                />
              )}
              {metrics?.npu_usage_pct !== undefined && (
                <MetricCard
                  icon={Zap}
                  label="NPU Usage"
                  value={metrics.npu_usage_pct.toFixed(1)}
                  unit="%"
                  subtitle="Neural Processing"
                  accent={metrics.npu_usage_pct > 85 ? 'red' : 'snap-blue'}
                  progress={metrics.npu_usage_pct}
                />
              )}
              {metrics?.power_watts !== undefined && (
                <MetricCard
                  icon={Zap}
                  label="Power Draw"
                  value={metrics.power_watts.toFixed(1)}
                  unit="W"
                  subtitle="Total system"
                  accent={metrics.power_watts > 25 ? 'amber' : 'green'}
                />
              )}
              {metrics?.temperature_c !== undefined && (
                <MetricCard
                  icon={ThermometerSun}
                  label="Temperature"
                  value={metrics.temperature_c.toFixed(0)}
                  unit="°C"
                  subtitle="Die temp"
                  accent={metrics.temperature_c > 85 ? 'red' : metrics.temperature_c > 70 ? 'amber' : 'green'}
                />
              )}
            </div>
          )}

          {hardware && (
            <div className="card rounded-lg p-4">
              <div className="flex items-center gap-2 mb-4">
                <CpuChip className="w-4 h-4 text-snap-blue" />
                <h3 className="font-semibold text-sm text-navy">Hardware Information</h3>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-3 text-sm">
                <div className="flex justify-between py-2 border-b border-border last:border-0">
                  <span className="text-text-secondary">CPU</span>
                  <span className="text-navy font-medium text-right">{hardware.cpu}</span>
                </div>
                <div className="flex justify-between py-2 border-b border-border last:border-0">
                  <span className="text-text-secondary">GPU</span>
                  <span className="text-navy font-medium text-right">{hardware.gpu}</span>
                </div>
                <div className="flex justify-between py-2 border-b border-border last:border-0">
                  <span className="text-text-secondary">NPU</span>
                  <span className="text-navy font-medium text-right">{hardware.npu || 'N/A'}</span>
                </div>
                <div className="flex justify-between py-2 border-b border-border last:border-0">
                  <span className="text-text-secondary">RAM</span>
                  <span className="text-navy font-medium text-right">{hardware.ram_gb} GB</span>
                </div>
                <div className="flex justify-between py-2 border-b border-border last:border-0">
                  <span className="text-text-secondary">Platform</span>
                  <span className="text-navy font-medium text-right">{hardware.platform}</span>
                </div>
                <div className="flex justify-between py-2 border-b border-border last:border-0">
                  <span className="text-text-secondary">Device</span>
                  <span className="text-navy font-medium text-right">{hardware.vendor} {hardware.model}</span>
                </div>
              </div>
            </div>
          )}

          {history.length > 0 && (
            <div className="card rounded-lg p-4">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Gauge className="w-4 h-4 text-snap-blue" />
                  <h3 className="font-semibold text-sm text-navy">Performance History</h3>
                </div>
                <span className="text-xs text-text-secondary">Last {history.length} samples</span>
              </div>
              <div className="flex items-end gap-1 h-24">
                {history.map((m, i) => {
                  const h = Math.max(4, (m.fps / 60) * 100);
                  return (
                    <div
                      key={i}
                      className="flex-1 bg-snap-blue/80 rounded-t-sm transition-all hover:bg-snap-blue"
                      style={{ height: `${h}%` }}
                      title={`${m.fps.toFixed(1)} FPS`}
                    />
                  );
                })}
              </div>
              <div className="flex justify-between mt-2 text-[10px] text-text-secondary font-mono">
                <span>Oldest</span>
                <span>Frames</span>
                <span>Latest</span>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
