import { useCallback, useEffect, useRef, useState } from 'react';
import { getWebSocketManager, WebSocketManager } from '../services/websocket';
import type { DetectedObject, Metrics, OCRResult, SafetyAlert, WebSocketStatus } from '../types';

interface UseWebSocketReturn {
  status: WebSocketStatus;
  isConnected: boolean;
  detections: DetectedObject[];
  metrics: Metrics | null;
  alerts: SafetyAlert[];
  ocrResults: OCRResult[];
  cameraStatus: { is_running: boolean; fps: number; resolution: string } | null;
  connect: () => void;
  disconnect: () => void;
  send: (data: unknown) => void;
  manager: WebSocketManager | null;
}

export function useWebSocket(autoConnect = true): UseWebSocketReturn {
  const managerRef = useRef<WebSocketManager | null>(null);

  const [status, setStatus] = useState<WebSocketStatus>('disconnected');
  const [detections, setDetections] = useState<DetectedObject[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [alerts, setAlerts] = useState<SafetyAlert[]>([]);
  const [ocrResults, setOcrResults] = useState<OCRResult[]>([]);
  const [cameraStatus, setCameraStatus] = useState<{
    is_running: boolean;
    fps: number;
    resolution: string;
  } | null>(null);

  const connect = useCallback(() => {
    if (!managerRef.current) {
      managerRef.current = getWebSocketManager();
    }
    managerRef.current.connect();
  }, []);

  const disconnect = useCallback(() => {
    if (managerRef.current) {
      managerRef.current.disconnect();
    }
  }, []);

  const send = useCallback((data: unknown) => {
    if (managerRef.current) {
      managerRef.current.send(data);
    }
  }, []);

  useEffect(() => {
    const manager = getWebSocketManager();
    managerRef.current = manager;

    const cleanupStatus = manager.onStatus((s) => setStatus(s));
    const cleanupDetections = manager.onDetections((d) => setDetections(d));
    const cleanupMetrics = manager.onMetrics((m) => setMetrics(m));
    const cleanupAlerts = manager.onAlert((a) => {
      setAlerts((prev) => {
        const existingIds = new Set(prev.map((alert) => alert.id));
        const newAlerts = a.filter((alert) => !existingIds.has(alert.id));
        return [...prev, ...newAlerts].slice(-50);
      });
    });
    const cleanupOcr = manager.onOCR((o) => setOcrResults(o));
    const cleanupCamera = manager.onCameraStatus((s) => setCameraStatus(s));

    if (autoConnect) {
      manager.connect();
    }

    return () => {
      cleanupStatus();
      cleanupDetections();
      cleanupMetrics();
      cleanupAlerts();
      cleanupOcr();
      cleanupCamera();
      manager.disconnect();
    };
  }, [autoConnect]);

  return {
    status,
    isConnected: status === 'connected',
    detections,
    metrics,
    alerts,
    ocrResults,
    cameraStatus,
    connect,
    disconnect,
    send,
    manager: managerRef.current,
  };
}
