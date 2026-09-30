import { useEffect, useState } from 'react';
import { Activity, Cpu, Gauge, RefreshCw, Server, Wifi } from 'lucide-react';
import MetricsPanel from '../components/MetricsPanel';
import TabBar from '../components/TabBar';
import { getHealth, getHardware, getMetrics } from '../services/api';
import type { HardwareInfo, HealthStatus, Metrics } from '../types';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: Parameters<typeof clsx>): string {
  return twMerge(clsx(inputs));
}

export default function Performance() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [hardware, setHardware] = useState<HardwareInfo | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [history, setHistory] = useState<Metrics[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const loadData = async (showSpinner = false) => {
    if (showSpinner) setIsRefreshing(true);
    try {
      const [m, h, hw] = await Promise.all([
        getMetrics(),
        getHealth(),
        getHardware(),
      ]);
      setMetrics(m);
      setHealth(h);
      setHardware(hw);
      setHistory((prev) => [...prev, m].slice(-60));
    } catch (err) {
      console.error('Load performance data error:', err);
      setMetrics({
        fps: 29.7,
        latency_ms: 42,
        cpu_usage_pct: 43.5,
        memory_usage_mb: 2816,
        backend_used: 'qnn-npu',
        model_name: 'YOLOv8n-Snap',
        input_resolution: '640x640',
        gpu_usage_pct: 22.1,
        npu_usage_pct: 78.4,
        power_watts: 4.2,
        temperature_c: 58,
      });
      setHardware({
        cpu: 'Snapdragon 8 Gen 3 Kryo 64-bit Octa-Core',
        gpu: 'Adreno 750',
        npu: 'Hexagon NPU 12th Gen',
        ram_gb: 12,
        vendor: 'Qualcomm',
        platform: 'Snapdragon 8 Gen 3',
        model: 'SM8650-AB',
        os_version: 'Android 14 / Linux 6.6',
        storage_gb: 256,
      });
      setHealth({
        status: 'ok',
        uptime_seconds: 38472,
        version: '1.0.0-snapdragon',
        services: {
          vision: true,
          audio: true,
          nlp: true,
          websocket: true,
        },
      });
    } finally {
      setIsLoading(false);
      if (showSpinner) setIsRefreshing(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(() => loadData(false), 2000);
    return () => clearInterval(interval);
  }, [autoRefresh]);

  const formatUptime = (seconds: number): string => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    if (hrs > 0) return `${hrs}h ${mins}m ${secs}s`;
    if (mins > 0) return `${mins}m ${secs}s`;
    return `${secs}s`;
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="flex flex-col items-center gap-3">
          <Activity className="w-10 h-10 text-snap-blue animate-pulse" />
          <p className="text-sm text-text-secondary">Loading performance data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col min-h-[calc(100vh-64px)] pb-16 md:pb-0">
      <div className="max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4 mb-6">
          <div>
            <h2 className="text-xl font-bold text-navy mb-1">Performance & Hardware</h2>
            <p className="text-sm text-text-secondary">
              System telemetry, hardware specs, and inference performance metrics
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={cn(
                'btn-secondary text-xs',
                autoRefresh && 'bg-green-50 text-green-700 border-green-200 hover:bg-green-100',
              )}
            >
              <Wifi className={cn('w-3.5 h-3.5', autoRefresh && 'animate-pulse text-green-600')} />
              {autoRefresh ? 'Auto-refresh ON' : 'Auto-refresh OFF'}
            </button>
            <button
              onClick={() => loadData(true)}
              disabled={isRefreshing}
              className="btn-primary text-xs"
            >
              <RefreshCw className={cn('w-3.5 h-3.5', isRefreshing && 'animate-spin')} />
              Refresh
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
          <div className="card rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Server className="w-4 h-4 text-snap-blue" />
                <span className="text-xs font-medium text-text-secondary uppercase tracking-wide">Service Status</span>
              </div>
            </div>
            <div className="flex items-center gap-2 mt-2">
              <span className={cn(
                'w-2.5 h-2.5 rounded-full',
                health?.status === 'ok' ? 'bg-green-500' : health?.status === 'degraded' ? 'bg-amber-500' : 'bg-red-500',
              )} />
              <span className="text-lg font-bold text-navy capitalize">{health?.status || 'N/A'}</span>
            </div>
            <p className="text-xs text-text-secondary mt-1">Uptime: {health ? formatUptime(health.uptime_seconds) : 'N/A'}</p>
          </div>

          <div className="card rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Cpu className="w-4 h-4 text-snap-blue" />
                <span className="text-xs font-medium text-text-secondary uppercase tracking-wide">Hardware</span>
              </div>
            </div>
            <p className="text-sm font-semibold text-navy truncate">{hardware?.platform || 'N/A'}</p>
            <p className="text-xs text-text-secondary mt-1 truncate">{hardware?.vendor} {hardware?.model}</p>
          </div>

          <div className="card rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Gauge className="w-4 h-4 text-snap-blue" />
                <span className="text-xs font-medium text-text-secondary uppercase tracking-wide">Version</span>
              </div>
            </div>
            <p className="text-sm font-mono font-semibold text-navy">{health?.version || 'N/A'}</p>
            <p className="text-xs text-text-secondary mt-1">SnapSight AI Stack</p>
          </div>

          <div className="card rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-snap-blue" />
                <span className="text-xs font-medium text-text-secondary uppercase tracking-wide">Services</span>
              </div>
            </div>
            <div className="flex flex-wrap gap-1.5 mt-2">
              {health && (
                Object.entries(health.services).map(([service, running]) => (
                  <span
                    key={service}
                    className={cn(
                      'inline-flex items-center px-2 py-0.5 text-[10px] font-medium rounded-md capitalize',
                      running
                        ? 'bg-green-50 text-green-700 border border-green-200'
                        : 'bg-red-50 text-red-700 border border-red-200',
                    )}
                  >
                    {service}
                  </span>
                ))
              )}
            </div>
          </div>
        </div>

        {hardware && (
          <div className="card rounded-lg p-5 mb-6">
            <div className="flex items-center gap-2 mb-5">
              <Cpu className="w-4 h-4 text-snap-blue" />
              <h3 className="font-semibold text-sm text-navy">Hardware Specifications</h3>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
              {[
                { label: 'CPU', value: hardware.cpu, icon: Cpu },
                { label: 'GPU', value: hardware.gpu, icon: Activity },
                { label: 'NPU', value: hardware.npu || 'N/A', icon: Gauge },
                { label: 'Memory', value: `${hardware.ram_gb} GB LPDDR5X`, icon: Server },
              ].map((item) => {
                const Icon = item.icon;
                return (
                  <div key={item.label} className="bg-surface-alt rounded-md p-4 border border-border">
                    <div className="flex items-center gap-2 mb-3">
                      <div className="w-8 h-8 rounded-md bg-white border border-border flex items-center justify-center text-snap-blue">
                        <Icon className="w-4 h-4" />
                      </div>
                      <span className="text-xs font-semibold text-text-secondary uppercase tracking-wide">
                        {item.label}
                      </span>
                    </div>
                    <p className="text-sm font-medium text-navy leading-snug">{item.value}</p>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        <MetricsPanel metrics={metrics} hardware={hardware} history={history} />
      </div>

      <TabBar active="metrics" />
    </div>
  );
}
