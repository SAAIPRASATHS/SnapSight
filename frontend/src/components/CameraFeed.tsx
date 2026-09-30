import { useCallback, useEffect, useRef, useState } from 'react';
import { Camera, CameraOff, Download, Loader2 } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';
import type { DetectedObject } from '../types';

function cn(...inputs: Parameters<typeof clsx>): string {
  return twMerge(clsx(inputs));
}

interface CameraFeedProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  canvasRef?: React.RefObject<HTMLCanvasElement>;
  isRunning: boolean;
  onStart: () => Promise<void> | void;
  onStop: () => void;
  onCapture?: () => Promise<Blob | null>;
  fps?: number;
  resolution?: string;
  detections?: DetectedObject[];
  error?: string | null;
}

const COLORS = [
  '#2563eb',
  '#10b981',
  '#f59e0b',
  '#ef4444',
  '#8b5cf6',
  '#ec4899',
  '#06b6d4',
  '#84cc16',
];

function colorForLabel(label: string): string {
  let hash = 0;
  for (let i = 0; i < label.length; i++) {
    hash = label.charCodeAt(i) + ((hash << 5) - hash);
  }
  return COLORS[Math.abs(hash) % COLORS.length];
}

export default function CameraFeed({
  videoRef,
  canvasRef,
  isRunning,
  onStart,
  onStop,
  onCapture,
  fps = 0,
  resolution,
  detections = [],
  error,
}: CameraFeedProps) {
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const animRef = useRef<number>(0);
  const [starting, setStarting] = useState(false);

  const handleStart = async () => {
    setStarting(true);
    try {
      await onStart();
    } finally {
      setStarting(false);
    }
  };

  const drawDetections = useCallback(() => {
    const video = videoRef.current;
    const overlay = overlayRef.current;
    if (!video || !overlay) return;

    const rect = video.getBoundingClientRect();
    overlay.width = rect.width;
    overlay.height = rect.height;

    const ctx = overlay.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, overlay.width, overlay.height);

    if (!video.videoWidth || !video.videoHeight) {
      animRef.current = requestAnimationFrame(drawDetections);
      return;
    }

    const scaleX = rect.width / video.videoWidth;
    const scaleY = rect.height / video.videoHeight;

    detections.forEach((obj) => {
      const x = obj.bbox.x * scaleX;
      const y = obj.bbox.y * scaleY;
      const w = obj.bbox.w * scaleX;
      const h = obj.bbox.h * scaleY;
      const color = colorForLabel(obj.label);

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);

      const label = `${obj.label} ${(obj.confidence * 100).toFixed(0)}%`;
      ctx.font = '600 12px Inter, system-ui, sans-serif';
      const textWidth = ctx.measureText(label).width;
      const textHeight = 18;

      ctx.fillStyle = color;
      ctx.fillRect(x, y - textHeight - 1, textWidth + 10, textHeight);

      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, x + 5, y - 5);
    });

    animRef.current = requestAnimationFrame(drawDetections);
  }, [videoRef, detections]);

  useEffect(() => {
    animRef.current = requestAnimationFrame(drawDetections);
    return () => cancelAnimationFrame(animRef.current);
  }, [drawDetections]);

  return (
    <div className="card rounded-lg overflow-hidden flex flex-col h-full">
      <div className="px-4 py-3 border-b border-border flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Camera className="w-4 h-4 text-snap-blue" />
          <h3 className="font-semibold text-sm text-navy">Live Camera Feed</h3>
        </div>
        <div className="flex items-center gap-3 text-xs">
          {resolution && (
            <span className="text-text-secondary font-mono">{resolution}</span>
          )}
          <span
            className={cn(
              'status-pill',
              isRunning
                ? 'bg-green-50 text-green-700 border-green-200'
                : 'bg-slate-50 text-slate-600 border-slate-200',
            )}
          >
            {isRunning ? 'LIVE' : 'STOPPED'}
          </span>
          {isRunning && (
            <span className="font-mono text-snap-blue font-semibold">{fps} FPS</span>
          )}
        </div>
      </div>

      <div
        ref={containerRef}
        className="relative flex-1 bg-slate-900 flex items-center justify-center min-h-[320px] aspect-video"
      >
        <video
          ref={videoRef}
          className="w-full h-full object-cover"
          playsInline
          muted
          autoPlay
        />
        <canvas
          ref={overlayRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
        />
        {canvasRef && (
          <canvas ref={canvasRef} className="hidden" />
        )}

        {!isRunning && !starting && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-900/80">
            <div className="w-16 h-16 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center mb-3">
              <CameraOff className="w-8 h-8 text-slate-400" />
            </div>
            <p className="text-slate-300 text-sm mb-1">Camera is off</p>
            <p className="text-slate-500 text-xs">Click Start Camera to begin</p>
          </div>
        )}

        {error && (
          <div className="absolute inset-x-4 bottom-4 bg-red-50 border border-red-200 rounded-md px-3 py-2">
            <p className="text-red-700 text-sm">{error}</p>
          </div>
        )}
      </div>

      <div className="px-4 py-3 border-t border-border flex items-center gap-2">
        {!isRunning ? (
          <button
            onClick={handleStart}
            disabled={starting}
            className="btn-primary flex-1"
          >
            {starting ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Starting...
              </>
            ) : (
              <>
                <Camera className="w-4 h-4" />
                Start Camera
              </>
            )}
          </button>
        ) : (
          <button
            onClick={onStop}
            className="btn-danger flex-1"
          >
            <CameraOff className="w-4 h-4" />
            Stop Camera
          </button>
        )}
        <button
          onClick={onCapture}
          disabled={!isRunning}
          className="btn-secondary"
          title="Capture Snapshot"
        >
          <Download className="w-4 h-4" />
          Snapshot
        </button>
      </div>
    </div>
  );
}
