import type { DetectedObject, Metrics, OCRResult, SafetyAlert, WebSocketMessage, WebSocketStatus } from '../types';

type MessageHandler = (data: WebSocketMessage) => void;
type StatusHandler = (status: WebSocketStatus) => void;
type DetectionHandler = (detections: DetectedObject[]) => void;
type MetricsHandler = (metrics: Metrics) => void;
type AlertHandler = (alerts: SafetyAlert[]) => void;
type OCRHandler = (results: OCRResult[]) => void;
type StatusDataHandler = (status: { is_running: boolean; fps: number; resolution: string }) => void;

export class WebSocketManager {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private shouldReconnect = true;
  private reconnectTimeout: ReturnType<typeof setTimeout> | null = null;

  private messageHandlers: Set<MessageHandler> = new Set();
  private statusHandlers: Set<StatusHandler> = new Set();
  private detectionHandlers: Set<DetectionHandler> = new Set();
  private metricsHandlers: Set<MetricsHandler> = new Set();
  private alertHandlers: Set<AlertHandler> = new Set();
  private ocrHandlers: Set<OCRHandler> = new Set();
  private cameraStatusHandlers: Set<StatusDataHandler> = new Set();

  private _status: WebSocketStatus = 'disconnected';

  constructor(host?: string) {
    const wsHost = host || window.location.host;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    this.url = `${protocol}//${wsHost}/ws/live`;
  }

  get status(): WebSocketStatus {
    return this._status;
  }

  private setStatus(status: WebSocketStatus): void {
    this._status = status;
    this.statusHandlers.forEach((handler) => handler(status));
  }

  connect(): void {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    this.shouldReconnect = true;
    this.setStatus('connecting');

    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
        this.setStatus('connected');
        console.log('[WebSocket] Connected to', this.url);
      };

      this.ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          this.handleMessage(message);
        } catch (error) {
          console.error('[WebSocket] Failed to parse message:', error);
        }
      };

      this.ws.onerror = (error) => {
        console.error('[WebSocket] Error:', error);
        this.setStatus('error');
      };

      this.ws.onclose = (event) => {
        console.log('[WebSocket] Disconnected. Code:', event.code, 'Reason:', event.reason);
        this.setStatus('disconnected');

        if (this.shouldReconnect) {
          this.scheduleReconnect();
        }
      };
    } catch (error) {
      console.error('[WebSocket] Failed to create connection:', error);
      this.setStatus('error');
      if (this.shouldReconnect) {
        this.scheduleReconnect();
      }
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('[WebSocket] Max reconnect attempts reached');
      return;
    }

    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }

    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts);
    this.reconnectAttempts++;

    console.log(`[WebSocket] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

    this.reconnectTimeout = setTimeout(() => {
      this.connect();
    }, delay);
  }

  disconnect(): void {
    this.shouldReconnect = false;

    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    if (this.ws) {
      this.ws.close(1000, 'Client disconnecting');
      this.ws = null;
    }
  }

  send(data: unknown): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn('[WebSocket] Cannot send message: not connected');
      return;
    }

    try {
      this.ws.send(JSON.stringify(data));
    } catch (error) {
      console.error('[WebSocket] Failed to send message:', error);
    }
  }

  private handleMessage(message: WebSocketMessage): void {
    this.messageHandlers.forEach((handler) => handler(message));

    switch (message.type) {
      case 'detections':
        this.detectionHandlers.forEach((handler) => handler(message.data as DetectedObject[]));
        break;
      case 'metrics':
        this.metricsHandlers.forEach((handler) => handler(message.data as Metrics));
        break;
      case 'alert':
        this.alertHandlers.forEach((handler) => handler(message.data as SafetyAlert[]));
        break;
      case 'ocr':
        this.ocrHandlers.forEach((handler) => handler(message.data as OCRResult[]));
        break;
      case 'status':
        this.cameraStatusHandlers.forEach((handler) => {
          const d = message.data as { is_running: boolean; fps: number; resolution: string };
          handler(d);
        });
        break;
      case 'error':
        console.error('[WebSocket] Server error:', message.data);
        break;
    }
  }

  onMessage(handler: MessageHandler): () => void {
    this.messageHandlers.add(handler);
    return () => this.messageHandlers.delete(handler);
  }

  onStatus(handler: StatusHandler): () => void {
    this.statusHandlers.add(handler);
    return () => this.statusHandlers.delete(handler);
  }

  onDetections(handler: DetectionHandler): () => void {
    this.detectionHandlers.add(handler);
    return () => this.detectionHandlers.delete(handler);
  }

  onMetrics(handler: MetricsHandler): () => void {
    this.metricsHandlers.add(handler);
    return () => this.metricsHandlers.delete(handler);
  }

  onAlert(handler: AlertHandler): () => void {
    this.alertHandlers.add(handler);
    return () => this.alertHandlers.delete(handler);
  }

  onOCR(handler: OCRHandler): () => void {
    this.ocrHandlers.add(handler);
    return () => this.ocrHandlers.delete(handler);
  }

  onCameraStatus(handler: StatusDataHandler): () => void {
    this.cameraStatusHandlers.add(handler);
    return () => this.cameraStatusHandlers.delete(handler);
  }
}

let wsManagerInstance: WebSocketManager | null = null;

export function getWebSocketManager(): WebSocketManager {
  if (!wsManagerInstance) {
    wsManagerInstance = new WebSocketManager();
  }
  return wsManagerInstance;
}
