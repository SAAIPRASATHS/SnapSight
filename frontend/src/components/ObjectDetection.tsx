import { Package, User, Car, TreePine, Smartphone, Building, Cat, Dog, AlertTriangle, Box, Coffee } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';
import type { DetectedObject } from '../types';

function cn(...inputs: Parameters<typeof clsx>): string {
  return twMerge(clsx(inputs));
}

interface ObjectDetectionProps {
  objects?: DetectedObject[];
  maxItems?: number;
}

const LABEL_ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  person: User,
  people: User,
  car: Car,
  vehicle: Car,
  truck: Car,
  bus: Car,
  phone: Smartphone,
  cellphone: Smartphone,
  tree: TreePine,
  plant: TreePine,
  building: Building,
  house: Building,
  cat: Cat,
  dog: Dog,
  box: Box,
  package: Package,
  cup: Coffee,
  bottle: Coffee,
};

function IconForLabel(label: string) {
  const lower = label.toLowerCase();
  for (const [key, Icon] of Object.entries(LABEL_ICONS)) {
    if (lower.includes(key)) {
      return <Icon className="w-4 h-4" />;
    }
  }
  return <Package className="w-4 h-4" />;
}

function confidenceColor(confidence: number): string {
  if (confidence >= 0.85) return 'bg-green-500';
  if (confidence >= 0.6) return 'bg-amber-500';
  return 'bg-red-500';
}

export default function ObjectDetection({ objects = [], maxItems = 20 }: ObjectDetectionProps) {
  const displayObjects = objects.slice(0, maxItems);

  return (
    <div className="card rounded-lg overflow-hidden flex flex-col h-full">
      <div className="px-4 py-3 border-b border-border flex items-center justify-between">
        <div className="flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-snap-blue" />
          <h3 className="font-semibold text-sm text-navy">Detected Objects</h3>
        </div>
        <span className="status-pill bg-snap-blue/10 text-snap-blue border border-snap-blue/20">
          {objects.length} items
        </span>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-2 max-h-[360px]">
        {displayObjects.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-10 text-center">
            <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center mb-3">
              <Package className="w-6 h-6 text-slate-400" />
            </div>
            <p className="text-sm text-text-secondary mb-1">No objects detected</p>
            <p className="text-xs text-slate-400">Objects will appear here when camera is running</p>
          </div>
        ) : (
          displayObjects.map((obj) => (
            <div
              key={obj.id}
              className="bg-surface-alt border border-border rounded-md p-3 hover:shadow-card-hover transition-shadow"
            >
              <div className="flex items-start gap-3">
                <div className="w-9 h-9 rounded-md bg-snap-blue/10 flex items-center justify-center text-snap-blue flex-shrink-0">
                  {IconForLabel(obj.label)}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2 mb-1.5">
                    <h4 className="font-medium text-sm text-navy truncate capitalize">
                      {obj.label}
                    </h4>
                    <span className="text-xs font-mono font-semibold text-navy flex-shrink-0">
                      {(obj.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full h-1.5 bg-slate-200 rounded-full overflow-hidden">
                    <div
                      className={cn('h-full rounded-full transition-all', confidenceColor(obj.confidence))}
                      style={{ width: `${Math.min(100, obj.confidence * 100)}%` }}
                    />
                  </div>
                  {obj.category && (
                    <p className="text-xs text-text-secondary mt-1.5">
                      Category: <span className="capitalize">{obj.category}</span>
                    </p>
                  )}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
