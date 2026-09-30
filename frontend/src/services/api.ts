import axios from 'axios';
import type {
  DetectionResult,
  HardwareInfo,
  HealthStatus,
  Metrics,
  ModelInfo,
  OCRResponse,
  SafetyAlert,
  SceneDescription,
} from '../types';

const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000,
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.message);
    return Promise.reject(error);
  },
);

export async function getHealth(): Promise<HealthStatus> {
  const response = await api.get<HealthStatus>('/health');
  return response.data;
}

export async function getHardware(): Promise<HardwareInfo> {
  const response = await api.get<HardwareInfo>('/hardware');
  return response.data;
}

export async function getModels(): Promise<ModelInfo[]> {
  const response = await api.get<ModelInfo[]>('/models');
  return response.data;
}

export async function detectObjects(image: Blob | File): Promise<DetectionResult> {
  const formData = new FormData();
  formData.append('image', image);

  const response = await api.post<DetectionResult>('/vision/detect', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
}

export async function ocrImage(image: Blob | File): Promise<OCRResponse> {
  const formData = new FormData();
  formData.append('image', image);

  const response = await api.post<OCRResponse>('/vision/ocr', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
}

export async function describeScene(image: Blob | File): Promise<SceneDescription> {
  const formData = new FormData();
  formData.append('image', image);

  const response = await api.post<SceneDescription>('/vision/describe', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
}

export async function chatQuery(message: string, context?: string): Promise<{ response: string }> {
  const response = await api.post<{ response: string }>('/nlp/chat', {
    message,
    context,
  });
  return response.data;
}

export async function transcribeAudio(audio: Blob | File): Promise<{ text: string }> {
  const formData = new FormData();
  formData.append('audio', audio);

  const response = await api.post<{ text: string }>('/audio/transcribe', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
}

export async function analyzeSafety(image: Blob | File): Promise<{ alerts: SafetyAlert[] }> {
  const formData = new FormData();
  formData.append('image', image);

  const response = await api.post<{ alerts: SafetyAlert[] }>('/vision/safety', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
}

export async function getMetrics(): Promise<Metrics> {
  const response = await api.get<Metrics>('/metrics');
  return response.data;
}

export async function updateSettings(settings: Record<string, unknown>): Promise<{ success: boolean }> {
  const response = await api.patch<{ success: boolean }>('/settings', settings);
  return response.data;
}

export default api;
