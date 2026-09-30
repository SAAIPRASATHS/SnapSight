import { useState } from 'react';
import { ShieldAlert, AlertTriangle, PersonStanding, Triangle, Car as CarIcon, HelpCircle, CheckCircle, PhoneCall, Clock } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';
import type { SafetyAlert } from '../types';

function cn(...inputs: Parameters<typeof clsx>): string {
  return twMerge(clsx(inputs));
}

interface SafetyPanelProps {
  alerts?: SafetyAlert[];
  onAcknowledge?: (alertId: string) => void;
  onNeedHelp?: (alertId: string) => void;
}

const TYPE_CONFIG: Record<SafetyAlert['type'], { icon: React.ComponentType<{ className?: string }>; label: string; color: string; bgColor: string; borderColor: string }> = {
  fall: {
    icon: PersonStanding,
    label: 'Possible Fall',
    color: 'text-red-700',
    bgColor: 'bg-red-50',
    borderColor: 'border-red-200',
  },
  obstacle: {
    icon: Triangle,
    label: 'Obstacle Detected',
    color: 'text-amber-700',
    bgColor: 'bg-amber-50',
    borderColor: 'border-amber-200',
  },
  person: {
    icon: PersonStanding,
    label: 'Person Nearby',
    color: 'text-snap-blue',
    bgColor: 'bg-snap-blue/10',
    borderColor: 'border-snap-blue/20',
  },
  vehicle: {
    icon: CarIcon,
    label: 'Vehicle Approaching',
    color: 'text-orange-700',
    bgColor: 'bg-orange-50',
    borderColor: 'border-orange-200',
  },
  hazard: {
    icon: AlertTriangle,
    label: 'Potential Hazard',
    color: 'text-red-700',
    bgColor: 'bg-red-50',
    borderColor: 'border-red-200',
  },
  unknown: {
    icon: ShieldAlert,
    label: 'Safety Alert',
    color: 'text-purple-700',
    bgColor: 'bg-purple-50',
    borderColor: 'border-purple-200',
  },
};

function formatTimestamp(ts: number): string {
  const now = Date.now();
  const diff = Math.floor((now - ts) / 1000);
  if (diff < 5) return 'Just now';
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function confidenceColor(confidence: number): string {
  if (confidence >= 0.85) return 'text-red-600';
  if (confidence >= 0.6) return 'text-amber-600';
  return 'text-text-secondary';
}

export default function SafetyPanel({ alerts = [], onAcknowledge, onNeedHelp }: SafetyPanelProps) {
  const [pendingHelp, setPendingHelp] = useState<string | null>(null);

  const unacknowledged = alerts.filter((a) => !a.acknowledged);
  const critical = unacknowledged.filter((a) => a.type === 'fall' || a.type === 'hazard');

  const handleAcknowledge = (alertId: string) => {
    if (pendingHelp === alertId) {
      setPendingHelp(null);
    }
    onAcknowledge?.(alertId);
  };

  const handleNeedHelp = (alertId: string) => {
    setPendingHelp(alertId);
    onNeedHelp?.(alertId);
  };

  return (
    <div className="card rounded-lg overflow-hidden flex flex-col h-full">
      <div className="px-4 py-3 border-b border-border flex items-center justify-between">
        <div className="flex items-center gap-2">
          <ShieldAlert className="w-4 h-4 text-red-600" />
          <h3 className="font-semibold text-sm text-navy">Safety Monitoring</h3>
        </div>
        <div className="flex items-center gap-2">
          {critical.length > 0 && (
            <span className="status-pill bg-red-50 text-red-700 border-red-200 animate-pulse">
              {critical.length} CRITICAL
            </span>
          )}
          <span className="status-pill bg-amber-50 text-amber-700 border-amber-200">
            {unacknowledged.length} Active
          </span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-3 max-h-[480px]">
        {alerts.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-center px-4">
            <div className="w-16 h-16 rounded-full bg-green-50 border border-green-200 flex items-center justify-center mb-4">
              <CheckCircle className="w-8 h-8 text-green-600" />
            </div>
            <h4 className="font-semibold text-navy mb-1">All Clear</h4>
            <p className="text-sm text-text-secondary max-w-xs">
              No safety concerns detected. The scene appears safe.
            </p>
          </div>
        ) : (
          alerts.map((alert) => {
            const config = TYPE_CONFIG[alert.type] || TYPE_CONFIG.unknown;
            const Icon = config.icon;
            const needsResponse = !alert.acknowledged && (alert.type === 'fall' || alert.type === 'hazard');
            const showHelpButtons = needsResponse && pendingHelp !== alert.id;

            return (
              <div
                key={alert.id}
                className={cn(
                  'border rounded-md p-4 transition-all',
                  config.bgColor,
                  config.borderColor,
                  alert.acknowledged && 'opacity-60 grayscale-[30%]',
                  needsResponse && 'ring-2 ring-red-300 ring-offset-1',
                )}
              >
                <div className="flex items-start gap-3 mb-3">
                  <div className={cn(
                    'w-10 h-10 rounded-md flex items-center justify-center flex-shrink-0',
                    alert.type === 'fall' ? 'bg-red-100 text-red-600' :
                    alert.type === 'obstacle' ? 'bg-amber-100 text-amber-600' :
                    alert.type === 'vehicle' ? 'bg-orange-100 text-orange-600' :
                    'bg-snap-blue/20 text-snap-blue',
                  )}>
                    <Icon className="w-5 h-5" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between gap-2 mb-1">
                      <h4 className={cn('font-semibold text-sm', config.color)}>{config.label}</h4>
                      <span className={cn('text-xs font-mono font-bold flex-shrink-0', confidenceColor(alert.confidence))}>
                        {(alert.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                    <p className="text-sm text-navy leading-relaxed mb-1">{alert.message}</p>
                    <div className="flex items-center gap-1.5 text-xs text-text-secondary">
                      <Clock className="w-3 h-3" />
                      {formatTimestamp(alert.timestamp)}
                      {alert.acknowledged && (
                        <>
                          <span className="text-slate-300">•</span>
                          <CheckCircle className="w-3 h-3 text-green-600" />
                          <span className="text-green-700 font-medium">Acknowledged</span>
                        </>
                      )}
                    </div>
                  </div>
                </div>

                {showHelpButtons && (
                  <div className="mt-4 pt-3 border-t border-current/10">
                    <div className="flex items-center gap-2 mb-3">
                      <HelpCircle className="w-4 h-4 text-red-600 flex-shrink-0" />
                      <p className="text-sm font-semibold text-navy">Are you okay?</p>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <button
                        onClick={() => handleAcknowledge(alert.id)}
                        className="btn-success w-full py-2.5"
                      >
                        <CheckCircle className="w-4 h-4" />
                        YES, I'm Fine
                      </button>
                      <button
                        onClick={() => handleNeedHelp(alert.id)}
                        className="btn-danger w-full py-2.5"
                      >
                        <PhoneCall className="w-4 h-4" />
                        NEED HELP
                      </button>
                    </div>
                  </div>
                )}

                {pendingHelp === alert.id && (
                  <div className="mt-4 pt-3 border-t border-red-200">
                    <div className="flex items-center gap-2 text-red-700 bg-red-100 rounded-md px-3 py-2">
                      <PhoneCall className="w-4 h-4 animate-pulse" />
                      <div>
                        <p className="text-sm font-semibold">Help request sent</p>
                        <p className="text-xs">Emergency contacts have been notified</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
