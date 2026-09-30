import { useCallback, useEffect, useRef, useState } from 'react';
import type { CameraStatus } from '../types';

interface UseCameraOptions {
  facingMode?: 'user' | 'environment';
  resolution?: '640x480' | '1280x720' | '1920x1080';
  autoStart?: boolean;
}

interface UseCameraReturn {
  videoRef: React.RefObject<HTMLVideoElement>;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  isRunning: boolean;
  status: CameraStatus;
  error: string | null;
  startCamera: () => Promise<void>;
  stopCamera: () => void;
  captureFrame: () => Promise<Blob | null>;
  captureFrameAsDataUrl: () => string | null;
}

export function useCamera(options: UseCameraOptions = {}): UseCameraReturn {
  const { facingMode = 'user', resolution = '1280x720', autoStart = false } = options;

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const frameCountRef = useRef(0);
  const lastFpsTimeRef = useRef(Date.now());

  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<CameraStatus>({
    is_running: false,
    resolution,
    current_fps: 0,
    facing_mode: facingMode,
  });

  const parseResolution = (res: string): { width: number; height: number } => {
    const [w, h] = res.split('x').map(Number);
    return { width: w || 1280, height: h || 720 };
  };

  const startCamera = useCallback(async () => {
    if (streamRef.current) {
      return;
    }

    setError(null);

    try {
      const { width, height } = parseResolution(resolution);

      const constraints: MediaStreamConstraints = {
        video: {
          facingMode,
          width: { ideal: width },
          height: { ideal: height },
        },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();

        const actualTrack = stream.getVideoTracks()[0];
        const settings = actualTrack.getSettings();
        const actualResolution = `${settings.width || width}x${settings.height || height}`;

        setStatus({
          is_running: true,
          resolution: actualResolution,
          current_fps: 0,
          device_id: actualTrack.id,
          facing_mode: facingMode as 'user' | 'environment',
        });
        setIsRunning(true);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to access camera';
      setError(message);
      console.error('Camera startCamera error:', err);
      setStatus((prev) => ({ ...prev, is_running: false, current_fps: 0 }));
      setIsRunning(false);
    }
  }, [facingMode, resolution]);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setStatus((prev) => ({ ...prev, is_running: false, current_fps: 0 }));
    setIsRunning(false);
  }, []);

  const captureFrame = useCallback(async (): Promise<Blob | null> => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !video.videoWidth || !video.videoHeight) {
      return null;
    }

    const drawCanvas = canvas || document.createElement('canvas');
    drawCanvas.width = video.videoWidth;
    drawCanvas.height = video.videoHeight;

    const ctx = drawCanvas.getContext('2d');
    if (!ctx) {
      return null;
    }

    ctx.drawImage(video, 0, 0, drawCanvas.width, drawCanvas.height);

    return new Promise((resolve) => {
      drawCanvas.toBlob(
        (blob) => resolve(blob),
        'image/jpeg',
        0.9,
      );
    });
  }, []);

  const captureFrameAsDataUrl = useCallback((): string | null => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !video.videoWidth || !video.videoHeight) {
      return null;
    }

    const drawCanvas = canvas || document.createElement('canvas');
    drawCanvas.width = video.videoWidth;
    drawCanvas.height = video.videoHeight;

    const ctx = drawCanvas.getContext('2d');
    if (!ctx) {
      return null;
    }

    ctx.drawImage(video, 0, 0, drawCanvas.width, drawCanvas.height);
    return drawCanvas.toDataURL('image/jpeg', 0.9);
  }, []);

  useEffect(() => {
    if (!isRunning) return;

    let animationId: number;

    const updateFps = () => {
      frameCountRef.current++;
      const now = Date.now();
      const elapsed = now - lastFpsTimeRef.current;

      if (elapsed >= 1000) {
        const fps = Math.round((frameCountRef.current * 1000) / elapsed);
        setStatus((prev) => ({ ...prev, current_fps: fps }));
        frameCountRef.current = 0;
        lastFpsTimeRef.current = now;
      }

      animationId = requestAnimationFrame(updateFps);
    };

    animationId = requestAnimationFrame(updateFps);

    return () => cancelAnimationFrame(animationId);
  }, [isRunning]);

  useEffect(() => {
    if (autoStart) {
      startCamera();
    }

    return () => {
      stopCamera();
    };
  }, [autoStart, startCamera, stopCamera]);

  return {
    videoRef,
    canvasRef,
    isRunning,
    status,
    error,
    startCamera,
    stopCamera,
    captureFrame,
    captureFrameAsDataUrl,
  };
}
