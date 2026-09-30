import { useEffect, useState } from 'react';
import {
  Settings as SettingsIcon,
  Cpu,
  Camera,
  Gauge,
  SlidersHorizontal,
  Cloud,
  Save,
  RefreshCw,
  ShieldCheck,
  MonitorDot,
  Zap,
  Bell,
  Volume2,
} from 'lucide-react';
import TabBar from '../components/TabBar';
import { getModels, updateSettings } from '../services/api';
import type { ModelInfo } from '../types';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: Parameters<typeof clsx>): string {
  return twMerge(clsx(inputs));
}

type BackendType = 'auto' | 'cpu' | 'gpu' | 'npu' | 'qnn-npu' | 'tensorrt' | 'openvino';
type ResolutionType = '640x480' | '1280x720' | '1920x1080' | '3840x2160';

interface SettingsState {
  modelId: string;
  backend: BackendType;
  cameraResolution: ResolutionType;
  inferenceFps: number;
  confidenceThreshold: number;
  cloudApisEnabled: boolean;
  autoStartCamera: boolean;
  safetyMonitoringEnabled: boolean;
  ocrEnabled: boolean;
  audioEnabled: boolean;
  alertsEnabled: boolean;
  iouThreshold: number;
  maxDetections: number;
}

const DEFAULT_SETTINGS: SettingsState = {
  modelId: 'yolov8n-snap',
  backend: 'qnn-npu',
  cameraResolution: '1280x720',
  inferenceFps: 30,
  confidenceThreshold: 0.5,
  cloudApisEnabled: false,
  autoStartCamera: false,
  safetyMonitoringEnabled: true,
  ocrEnabled: true,
  audioEnabled: true,
  alertsEnabled: true,
  iouThreshold: 0.45,
  maxDetections: 100,
};

const BACKEND_OPTIONS: { value: BackendType; label: string; description: string }[] = [
  { value: 'auto', label: 'Auto Select', description: 'Best available hardware' },
  { value: 'qnn-npu', label: 'QNN NPU (Snapdragon)', description: 'Hexagon NPU acceleration' },
  { value: 'gpu', label: 'GPU (OpenCL)', description: 'Adreno GPU shader cores' },
  { value: 'tensorrt', label: 'TensorRT (NVIDIA)', description: 'NVIDIA CUDA GPUs' },
  { value: 'openvino', label: 'OpenVINO (Intel)', description: 'Intel CPUs & iGPUs' },
  { value: 'cpu', label: 'CPU Only', description: 'Fallback, no acceleration' },
];

const RESOLUTION_OPTIONS: { value: ResolutionType; label: string; aspect: string }[] = [
  { value: '640x480', label: 'VGA (640x480)', aspect: '4:3' },
  { value: '1280x720', label: 'HD 720p', aspect: '16:9' },
  { value: '1920x1080', label: 'Full HD 1080p', aspect: '16:9' },
  { value: '3840x2160', label: '4K UHD', aspect: '16:9' },
];

export default function Settings() {
  const [settings, setSettings] = useState<SettingsState>(DEFAULT_SETTINGS);
  const [models, setModels] = useState<ModelInfo[]>([
    {
      id: 'yolov8n-snap',
      name: 'YOLOv8n-Snap',
      task: 'detection',
      backend: 'qnn-npu',
      size_mb: 6.2,
      quantized: true,
    },
    {
      id: 'yolov8s-snap',
      name: 'YOLOv8s-Snap',
      task: 'detection',
      backend: 'qnn-npu',
      size_mb: 21.5,
      quantized: true,
    },
    {
      id: 'yolov8-seg',
      name: 'YOLOv8-Segment',
      task: 'segmentation',
      backend: 'gpu',
      size_mb: 50.1,
      quantized: true,
    },
    {
      id: 'paddleocr-snap',
      name: 'PaddleOCR-Mobile',
      task: 'ocr',
      backend: 'qnn-npu',
      size_mb: 11.8,
      quantized: true,
    },
    {
      id: 'mobilenetv3-cls',
      name: 'MobileNetV3-Large',
      task: 'classification',
      backend: 'qnn-npu',
      size_mb: 5.4,
      quantized: true,
    },
  ]);
  const [saved, setSaved] = useState(false);
  const [isResetting, setIsResetting] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    getModels().then((m) => {
      if (m && m.length > 0) setModels(m);
    }).catch(() => {
      // Use default models
    });
  }, []);

  const handleSave = async () => {
    setIsSaving(true);
    try {
      await updateSettings(settings as unknown as Record<string, unknown>);
    } catch (err) {
      console.error('Save settings error:', err);
    }
    setSaved(true);
    setTimeout(() => setSaved(false), 2500);
    setIsSaving(false);
  };

  const handleReset = () => {
    setIsResetting(true);
    setTimeout(() => {
      setSettings(DEFAULT_SETTINGS);
      setIsResetting(false);
    }, 400);
  };

  const update = <K extends keyof SettingsState>(key: K, value: SettingsState[K]) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
  };

  return (
    <div className="flex flex-col min-h-[calc(100vh-64px)] pb-16 md:pb-0">
      <div className="max-w-4xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4 mb-6">
          <div>
            <h2 className="text-xl font-bold text-navy mb-1 flex items-center gap-2">
              <SettingsIcon className="w-5 h-5 text-snap-blue" />
              Settings
            </h2>
            <p className="text-sm text-text-secondary">
              Configure models, hardware acceleration, camera, and inference parameters
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleReset}
              disabled={isResetting}
              className="btn-secondary text-xs"
            >
              <RefreshCw className={cn('w-3.5 h-3.5', isResetting && 'animate-spin')} />
              Reset Defaults
            </button>
            <button
              onClick={handleSave}
              disabled={isSaving || saved}
              className={cn(
                'btn-primary text-xs',
                saved && 'bg-green-600 hover:bg-green-600',
              )}
            >
              {saved ? (
                <>
                  <ShieldCheck className="w-3.5 h-3.5" />
                  Saved
                </>
              ) : (
                <>
                  <Save className={cn('w-3.5 h-3.5', isSaving && 'animate-pulse')} />
                  {isSaving ? 'Saving...' : 'Save Changes'}
                </>
              )}
            </button>
          </div>
        </div>

        <div className="space-y-6">
          <section className="card rounded-lg p-5">
            <div className="flex items-center gap-2 mb-5">
              <MonitorDot className="w-4 h-4 text-snap-blue" />
              <h3 className="font-semibold text-sm text-navy">Model Selection</h3>
            </div>
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-navy mb-2">Detection Model</label>
                <select
                  value={settings.modelId}
                  onChange={(e) => update('modelId', e.target.value)}
                  className="input-field"
                >
                  {models.filter((m) => m.task === 'detection' || m.task === 'classification').map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name} ({m.size_mb.toFixed(1)} MB) — {m.backend}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs font-medium text-navy mb-2">Inference Backend</label>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {BACKEND_OPTIONS.map((b) => {
                    const active = settings.backend === b.value;
                    return (
                      <button
                        key={b.value}
                        type="button"
                        onClick={() => update('backend', b.value)}
                        className={cn(
                          'text-left p-3 rounded-md border-2 transition-all',
                          active
                            ? 'border-snap-blue bg-snap-blue/5 shadow-sm'
                            : 'border-border bg-white hover:border-slate-300',
                        )}
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className={cn('text-sm font-semibold', active ? 'text-snap-blue' : 'text-navy')}>
                            {b.label}
                          </span>
                          {active && (
                            <span className="w-4 h-4 rounded-full bg-snap-blue text-white flex items-center justify-center">
                              <svg viewBox="0 0 20 20" fill="currentColor" className="w-2.5 h-2.5">
                                <path
                                  fillRule="evenodd"
                                  d="M16.704 5.29a1 1 0 010 1.42l-7.5 7.5a1 1 0 01-1.42 0l-3.5-3.5a1 1 0 011.42-1.42l2.79 2.79 6.79-6.79a1 1 0 011.42 0z"
                                  clipRule="evenodd"
                                />
                              </svg>
                            </span>
                          )}
                        </div>
                        <p className="text-xs text-text-secondary">{b.description}</p>
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>
          </section>

          <section className="card rounded-lg p-5">
            <div className="flex items-center gap-2 mb-5">
              <Camera className="w-4 h-4 text-snap-blue" />
              <h3 className="font-semibold text-sm text-navy">Camera & Inference</h3>
            </div>
            <div className="space-y-5">
              <div>
                <label className="block text-xs font-medium text-navy mb-2">Camera Resolution</label>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                  {RESOLUTION_OPTIONS.map((r) => {
                    const active = settings.cameraResolution === r.value;
                    return (
                      <button
                        key={r.value}
                        type="button"
                        onClick={() => update('cameraResolution', r.value)}
                        className={cn(
                          'p-2.5 rounded-md border-2 text-center transition-all',
                          active
                            ? 'border-snap-blue bg-snap-blue/5'
                            : 'border-border hover:border-slate-300',
                        )}
                      >
                        <p className={cn('text-sm font-semibold', active ? 'text-snap-blue' : 'text-navy')}>
                          {r.label.split(' (')[0]}
                        </p>
                        <p className="text-[10px] text-text-secondary font-mono mt-0.5">{r.aspect}</p>
                      </button>
                    );
                  })}
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="block text-xs font-medium text-navy">Target Inference FPS</label>
                    <span className="text-sm font-mono font-semibold text-snap-blue">{settings.inferenceFps} FPS</span>
                  </div>
                  <input
                    type="range"
                    min={1}
                    max={60}
                    step={1}
                    value={settings.inferenceFps}
                    onChange={(e) => update('inferenceFps', Number(e.target.value))}
                    className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-snap-blue"
                  />
                  <div className="flex justify-between mt-1 text-[10px] text-text-secondary font-mono">
                    <span>1</span>
                    <span>30</span>
                    <span>60</span>
                  </div>
                </div>

                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="block text-xs font-medium text-navy">Confidence Threshold</label>
                    <span className="text-sm font-mono font-semibold text-snap-blue">
                      {(settings.confidenceThreshold * 100).toFixed(0)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0.1}
                    max={0.95}
                    step={0.05}
                    value={settings.confidenceThreshold}
                    onChange={(e) => update('confidenceThreshold', Number(e.target.value))}
                    className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-snap-blue"
                  />
                  <div className="flex justify-between mt-1 text-[10px] text-text-secondary font-mono">
                    <span>10%</span>
                    <span>50%</span>
                    <span>95%</span>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="block text-xs font-medium text-navy">IoU Threshold</label>
                    <span className="text-sm font-mono font-semibold text-snap-blue">
                      {(settings.iouThreshold * 100).toFixed(0)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0.1}
                    max={0.9}
                    step={0.05}
                    value={settings.iouThreshold}
                    onChange={(e) => update('iouThreshold', Number(e.target.value))}
                    className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-snap-blue"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-navy mb-2">Max Detections per Frame</label>
                  <div className="flex items-center gap-3">
                    <input
                      type="number"
                      min={1}
                      max={500}
                      step={5}
                      value={settings.maxDetections}
                      onChange={(e) => update('maxDetections', Math.min(500, Math.max(1, Number(e.target.value))))}
                      className="input-field w-32"
                    />
                    <span className="text-xs text-text-secondary">objects</span>
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section className="card rounded-lg p-5">
            <div className="flex items-center gap-2 mb-5">
              <SlidersHorizontal className="w-4 h-4 text-snap-blue" />
              <h3 className="font-semibold text-sm text-navy">Features & Privacy</h3>
            </div>
            <div className="space-y-1">
              {[
                {
                  key: 'safetyMonitoringEnabled' as const,
                  label: 'Safety Monitoring',
                  description: 'Detect falls, obstacles, and potential hazards',
                  icon: ShieldCheck,
                },
                {
                  key: 'ocrEnabled' as const,
                  label: 'Text Recognition (OCR)',
                  description: 'Read and extract text from camera frames',
                  icon: Gauge,
                },
                {
                  key: 'audioEnabled' as const,
                  label: 'Audio & Speech Recognition',
                  description: 'Enable microphone input and transcription',
                  icon: Volume2,
                },
                {
                  key: 'alertsEnabled' as const,
                  label: 'Safety Alerts & Notifications',
                  description: 'Show urgent prompts when hazards are detected',
                  icon: Bell,
                },
                {
                  key: 'autoStartCamera' as const,
                  label: 'Auto-start Camera on Launch',
                  description: 'Begin processing immediately when app opens',
                  icon: Camera,
                },
                {
                  key: 'cloudApisEnabled' as const,
                  label: 'Enable Cloud API Fallback',
                  description: 'Use cloud services when local models are unavailable',
                  icon: Cloud,
                  danger: true,
                },
              ].map(({ key, label, description, icon: Icon, danger }) => {
                const value = settings[key];
                return (
                  <div
                    key={key}
                    className="flex items-start justify-between gap-4 py-3 border-b border-border last:border-0"
                  >
                    <div className="flex items-start gap-3 flex-1 min-w-0">
                      <div className={cn(
                        'w-9 h-9 rounded-md flex items-center justify-center flex-shrink-0',
                        danger ? 'bg-red-50 text-red-600' : 'bg-snap-blue/10 text-snap-blue',
                      )}>
                        <Icon className="w-4.5 h-4.5" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className={cn('text-sm font-semibold', danger && value ? 'text-red-700' : 'text-navy')}>
                          {label}
                        </p>
                        <p className="text-xs text-text-secondary mt-0.5 leading-relaxed">{description}</p>
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={() => update(key, !value)}
                      className={cn(
                        'relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-snap-blue/30',
                        value ? (danger ? 'bg-red-600' : 'bg-snap-blue') : 'bg-slate-300',
                      )}
                    >
                      <span
                        className={cn(
                          'pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out',
                          value ? 'translate-x-5' : 'translate-x-0',
                        )}
                      />
                    </button>
                  </div>
                );
              })}
            </div>
          </section>

          <section className="card rounded-lg p-5 border-2 border-snap-blue/20 bg-snap-blue/[0.02]">
            <div className="flex items-start gap-3">
              <div className="w-9 h-9 rounded-md bg-snap-blue flex items-center justify-center flex-shrink-0">
                <Zap className="w-4.5 h-4.5 text-white" />
              </div>
              <div className="flex-1">
                <h3 className="font-semibold text-sm text-navy mb-1">Processing Location</h3>
                <p className="text-xs text-text-secondary leading-relaxed mb-2">
                  All inference runs <strong className="text-navy">locally on your device</strong>. Raw camera frames never leave your device unless you explicitly enable Cloud API Fallback above. Your privacy is protected by design.
                </p>
                <div className="flex items-center gap-2">
                  <span className="status-pill bg-green-50 text-green-700 border-green-200">
                    <span className="w-1.5 h-1.5 rounded-full bg-green-500 inline-block mr-1" />
                    Edge Processing Active
                  </span>
                  <span className="status-pill bg-snap-blue/10 text-snap-blue border-snap-blue/20">
                    <Cpu className="w-3 h-3 inline-block mr-1" />
                    Snapdragon NPU
                  </span>
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>

      <TabBar />
    </div>
  );
}
