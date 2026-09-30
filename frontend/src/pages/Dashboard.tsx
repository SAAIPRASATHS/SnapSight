import { useEffect, useRef, useState } from 'react';
import CameraFeed from '../components/CameraFeed';
import AIInsights from '../components/AIInsights';
import ObjectDetection from '../components/ObjectDetection';
import ChatPanel from '../components/ChatPanel';
import SafetyPanel from '../components/SafetyPanel';
import TabBar, { type ActiveTab } from '../components/TabBar';
import { useCamera } from '../hooks/useCamera';
import { useWebSocket } from '../hooks/useWebSocket';
import { detectObjects, describeScene, ocrImage, analyzeSafety, chatQuery, transcribeAudio } from '../services/api';
import type { DetectedObject, OCRResult, SafetyAlert, SceneDescription, ChatMessage } from '../types';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: Parameters<typeof clsx>): string {
  return twMerge(clsx(inputs));
}

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('camera');
  const [detections, setDetections] = useState<DetectedObject[]>([]);
  const [sceneDescription, setSceneDescription] = useState<SceneDescription | null>(null);
  const [ocrResults, setOcrResults] = useState<OCRResult[]>([]);
  const [ocrFullText, setOcrFullText] = useState('');
  const [alerts, setAlerts] = useState<SafetyAlert[]>([]);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content: "Hi! I'm SnapSight AI assistant. I can help you understand what the camera sees, answer questions about detected objects, read text from images, and monitor for safety concerns. How can I help you today?",
      timestamp: Date.now(),
    },
  ]);
  const [isChatLoading, setIsChatLoading] = useState(false);
  const lastProcessRef = useRef<number>(0);

  const camera = useCamera({
    facingMode: 'environment',
    resolution: '1280x720',
    autoStart: false,
  });

  const ws = useWebSocket(false);

  useEffect(() => {
    if (ws.detections && ws.detections.length > 0) {
      setDetections(ws.detections);
    }
    if (ws.alerts && ws.alerts.length > 0) {
      setAlerts(ws.alerts);
    }
    if (ws.ocrResults && ws.ocrResults.length > 0) {
      setOcrResults(ws.ocrResults);
    }
  }, [ws.detections, ws.alerts, ws.ocrResults]);

  useEffect(() => {
    if (!camera.isRunning) return;

    const interval = setInterval(async () => {
      const now = Date.now();
      if (now - lastProcessRef.current < 1000) return;
      lastProcessRef.current = now;

      const frame = await camera.captureFrame();
      if (!frame) return;

      try {
        const [detResult, sceneResult, ocrResult, safetyResult] = await Promise.allSettled([
          detectObjects(frame),
          describeScene(frame),
          ocrImage(frame),
          analyzeSafety(frame),
        ]);

        if (detResult.status === 'fulfilled') {
          setDetections(detResult.value.objects);
        }
        if (sceneResult.status === 'fulfilled') {
          setSceneDescription(sceneResult.value);
        }
        if (ocrResult.status === 'fulfilled') {
          setOcrResults(ocrResult.value.results);
          setOcrFullText(ocrResult.value.full_text);
        }
        if (safetyResult.status === 'fulfilled') {
          setAlerts((prev) => {
            const existingIds = new Set(prev.map((a) => a.id));
            const newAlerts = safetyResult.value.alerts.filter((a) => !existingIds.has(a.id));
            return [...newAlerts, ...prev].slice(0, 50);
          });
        }
      } catch (err) {
        console.error('Processing error:', err);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [camera.isRunning, camera.captureFrame]);

  const handleCapture = async () => {
    const frame = await camera.captureFrame();
    if (!frame) return null;

    try {
      const [detResult, sceneResult, ocrResult, safetyResult] = await Promise.all([
        detectObjects(frame),
        describeScene(frame),
        ocrImage(frame),
        analyzeSafety(frame),
      ]);

      setDetections(detResult.objects);
      setSceneDescription(sceneResult);
      setOcrResults(ocrResult.results);
      setOcrFullText(ocrResult.full_text);
      if (safetyResult.alerts.length > 0) {
        setAlerts((prev) => {
          const existingIds = new Set(prev.map((a) => a.id));
          const newAlerts = safetyResult.alerts.filter((a) => !existingIds.has(a.id));
          return [...newAlerts, ...prev].slice(0, 50);
        });
      }
    } catch (err) {
      console.error('Capture processing error:', err);
    }

    return frame;
  };

  const handleSendMessage = async (message: string) => {
    setIsChatLoading(true);
    try {
      const userMsg: ChatMessage = {
        id: `user-${Date.now()}`,
        role: 'user',
        content: message,
        timestamp: Date.now(),
      };
      setChatMessages((prev) => [...prev, userMsg]);

      const result = await chatQuery(message, sceneDescription?.description);
      const assistantMsg: ChatMessage = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: result.response,
        timestamp: Date.now(),
      };
      setChatMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      const fallbackMsg: ChatMessage = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: "I'm analyzing the scene. I can see " +
          (detections.length > 0 ? `${detections.length} object${detections.length > 1 ? 's' : ''}: ${detections.slice(0, 3).map((d) => d.label).join(', ')}. ` : 'no objects yet. ') +
          (sceneDescription?.summary || 'Please start the camera for live analysis.'),
        timestamp: Date.now(),
      };
      setChatMessages((prev) => [...prev, fallbackMsg]);
    } finally {
      setIsChatLoading(false);
    }
  };

  const handleTranscribeAudio = async (audio: Blob): Promise<string | null> => {
    try {
      const result = await transcribeAudio(audio);
      return result.text;
    } catch (err) {
      console.error('Transcribe error:', err);
      return null;
    }
  };

  const handleAcknowledgeAlert = (alertId: string) => {
    setAlerts((prev) =>
      prev.map((a) => (a.id === alertId ? { ...a, acknowledged: true } : a)),
    );
  };

  const handleNeedHelp = (alertId: string) => {
    setAlerts((prev) =>
      prev.map((a) => (a.id === alertId ? { ...a, acknowledged: true } : a)),
    );
  };

  const activeAlerts = alerts.filter((a) => !a.acknowledged);

  return (
    <div className="flex flex-col min-h-[calc(100vh-64px)] pb-16 md:pb-0">
      <div className="max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-6">
        <div className="mb-6">
          <h2 className="text-xl font-bold text-navy mb-1">Live Dashboard</h2>
          <p className="text-sm text-text-secondary">
            Real-time object detection, scene understanding, and safety monitoring
          </p>
        </div>

        {activeAlerts.length > 0 && (
          <div className="mb-4 bg-red-50 border border-red-200 rounded-lg px-4 py-3 flex items-start gap-3">
            <div className="w-8 h-8 rounded-md bg-red-100 flex items-center justify-center flex-shrink-0 animate-pulse">
              <span className="text-red-600 font-bold">!</span>
            </div>
            <div className="flex-1">
              <p className="text-sm font-semibold text-red-800">
                {activeAlerts.length} active safety alert{activeAlerts.length > 1 ? 's' : ''}
              </p>
              <p className="text-xs text-red-700 mt-0.5">
                Check the Safety panel for details
              </p>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
          <div className="lg:col-span-7 space-y-5">
            <CameraFeed
              videoRef={camera.videoRef}
              canvasRef={camera.canvasRef}
              isRunning={camera.isRunning}
              onStart={camera.startCamera}
              onStop={camera.stopCamera}
              onCapture={handleCapture}
              fps={camera.status.current_fps}
              resolution={camera.status.resolution}
              detections={detections}
              error={camera.error}
            />
            <div className="hidden lg:block">
              <SafetyPanel
                alerts={alerts}
                onAcknowledge={handleAcknowledgeAlert}
                onNeedHelp={handleNeedHelp}
              />
            </div>
          </div>

          <div className="lg:col-span-5 space-y-5">
            {activeTab === 'camera' || activeTab === 'ocr' ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-1 gap-5">
                {activeTab === 'camera' && (
                  <ObjectDetection objects={detections} />
                )}
                <AIInsights
                  sceneDescription={sceneDescription}
                  ocrResults={ocrResults}
                  ocrFullText={ocrFullText}
                />
              </div>
            ) : (
              <SafetyPanel
                alerts={alerts}
                onAcknowledge={handleAcknowledgeAlert}
                onNeedHelp={handleNeedHelp}
              />
            )}
            <ChatPanel
              messages={chatMessages}
              onSendMessage={handleSendMessage}
              isLoading={isChatLoading}
              onTranscribeAudio={handleTranscribeAudio}
            />
          </div>
        </div>
      </div>

      <TabBar active={activeTab} onChange={setActiveTab} />

      <div className={cn(
        'fixed left-4 right-4 md:left-auto md:right-6 md:w-56',
        activeTab === 'safety' ? 'md:bottom-6' : 'md:bottom-6'
      )}>
      </div>
    </div>
  );
}
