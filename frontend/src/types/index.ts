export interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Detection {
  label: string;
  confidence: number;
  bbox: BBox;
}

export interface DetectedObject {
  id: string;
  label: string;
  confidence: number;
  bbox: BBox;
  category?: string;
  timestamp: number;
}

export interface OCRResult {
  text: string;
  confidence: number;
  bbox: BBox;
}

export interface SafetyAlert {
  id: string;
  type: 'fall' | 'obstacle' | 'person' | 'vehicle' | 'hazard' | 'unknown';
  confidence: number;
  message: string;
  timestamp: number;
  acknowledged?: boolean;
  location?: BBox;
}

export interface Metrics {
  fps: number;
  latency_ms: number;
  cpu_usage_pct: number;
  memory_usage_mb: number;
  backend_used: string;
  model_name: string;
  input_resolution: string;
  gpu_usage_pct?: number;
  npu_usage_pct?: number;
  power_watts?: number;
  temperature_c?: number;
}

export interface HardwareInfo {
  cpu: string;
  gpu: string;
  npu: string;
  ram_gb: number;
  vendor: string;
  platform: string;
  model: string;
  os_version?: string;
  storage_gb?: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
}

export interface CameraStatus {
  is_running: boolean;
  resolution: string;
  current_fps: number;
  device_id?: string;
  facing_mode?: 'user' | 'environment';
}

export interface ModelInfo {
  id: string;
  name: string;
  task: 'detection' | 'ocr' | 'classification' | 'segmentation';
  backend: string;
  size_mb: number;
  quantized: boolean;
}

export interface HealthStatus {
  status: 'ok' | 'degraded' | 'error';
  uptime_seconds: number;
  version: string;
  services: {
    vision: boolean;
    audio: boolean;
    nlp: boolean;
    websocket: boolean;
  };
}

export interface SceneDescription {
  description: string;
  summary: string;
  objects: string[];
  scene_type: string;
  confidence: number;
}

export interface DetectionResult {
  objects: DetectedObject[];
  timestamp: number;
  processing_time_ms: number;
}

export interface OCRResponse {
  results: OCRResult[];
  full_text: string;
  processing_time_ms: number;
}

export type WebSocketStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

export interface WebSocketMessage {
  type: 'detections' | 'status' | 'metrics' | 'alert' | 'ocr' | 'scene' | 'error';
  data: unknown;
  timestamp: number;
}
