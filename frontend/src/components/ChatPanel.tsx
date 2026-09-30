import { useState, useRef, useEffect } from 'react';
import { Send, Mic, MicOff, Bot, User, MessageSquare } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';
import type { ChatMessage } from '../types';

function cn(...inputs: Parameters<typeof clsx>): string {
  return twMerge(clsx(inputs));
}

interface ChatPanelProps {
  messages?: ChatMessage[];
  onSendMessage?: (message: string) => void | Promise<void>;
  isLoading?: boolean;
  onTranscribeAudio?: (audio: Blob) => Promise<string | null>;
}

export default function ChatPanel({
  messages: initialMessages,
  onSendMessage,
  isLoading = false,
  onTranscribeAudio,
}: ChatPanelProps) {
  const [input, setInput] = useState('');
  const [localMessages, setLocalMessages] = useState<ChatMessage[]>(
    initialMessages || [
      {
        id: 'welcome',
        role: 'assistant',
        content: "Hi! I'm SnapSight AI assistant. I can help you understand what the camera sees, answer questions about detected objects, read text from images, and monitor for safety concerns. How can I help you today?",
        timestamp: Date.now(),
      },
    ],
  );
  const [isRecording, setIsRecording] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const messages = initialMessages || localMessages;

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || isLoading || isSending) return;

    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: Date.now(),
    };

    setLocalMessages((prev) => [...prev, userMsg]);
    setInput('');
    setIsSending(true);

    textareaRef.current?.focus();

    try {
      if (onSendMessage) {
        await onSendMessage(text);
      } else {
        await new Promise((resolve) => setTimeout(resolve, 800));
        const responseMsg: ChatMessage = {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content: 'I understand. Based on the current scene, I can see several objects. Would you like me to describe anything specific about the detected objects, OCR text, or safety status?',
          timestamp: Date.now(),
        };
        setLocalMessages((prev) => [...prev, responseMsg]);
      }
    } finally {
      setIsSending(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        audioChunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        stream.getTracks().forEach((t) => t.stop());

        if (onTranscribeAudio) {
          const text = await onTranscribeAudio(audioBlob);
          if (text) {
            setInput(text);
          }
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error('Failed to start recording:', err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const formatTime = (ts: number) => {
    return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="card rounded-lg overflow-hidden flex flex-col h-full min-h-[340px]">
      <div className="px-4 py-3 border-b border-border flex items-center justify-between">
        <div className="flex items-center gap-2">
          <MessageSquare className="w-4 h-4 text-snap-blue" />
          <h3 className="font-semibold text-sm text-navy">AI Assistant Chat</h3>
        </div>
        <span className="status-pill bg-snap-blue/10 text-snap-blue border border-snap-blue/20">
          {messages.length} messages
        </span>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-3 bg-gradient-to-b from-surface-alt/40 to-white">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={cn(
              'flex gap-2.5 max-w-[90%]',
              msg.role === 'user' ? 'ml-auto flex-row-reverse' : 'mr-auto',
            )}
          >
            <div
              className={cn(
                'w-7 h-7 rounded-md flex items-center justify-center flex-shrink-0',
                msg.role === 'user'
                  ? 'bg-snap-blue text-white'
                  : 'bg-slate-100 text-slate-600 border border-border',
              )}
            >
              {msg.role === 'user' ? (
                <User className="w-3.5 h-3.5" />
              ) : (
                <Bot className="w-3.5 h-3.5" />
              )}
            </div>
            <div className="flex flex-col gap-1">
              <div
                className={cn(
                  'px-3 py-2.5 rounded-lg text-sm leading-relaxed',
                  msg.role === 'user'
                    ? 'bg-snap-blue text-white rounded-tr-sm'
                    : 'bg-white text-navy border border-border rounded-tl-sm',
                )}
              >
                {msg.content}
              </div>
              <span
                className={cn(
                  'text-[10px] text-text-secondary',
                  msg.role === 'user' ? 'text-right' : 'text-left',
                )}
              >
                {formatTime(msg.timestamp)}
              </span>
            </div>
          </div>
        ))}

        {(isLoading || isSending) && (
          <div className="flex gap-2.5 max-w-[90%] mr-auto">
            <div className="w-7 h-7 rounded-md bg-slate-100 text-slate-600 border border-border flex items-center justify-center flex-shrink-0">
              <Bot className="w-3.5 h-3.5" />
            </div>
            <div className="px-3 py-2.5 rounded-lg rounded-tl-sm bg-white text-navy border border-border">
              <div className="flex items-center gap-1">
                <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="px-4 py-3 border-t border-border bg-white">
        <div className="flex items-end gap-2">
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about the scene, objects, or safety..."
              rows={2}
              className="input-field resize-none pr-10"
              disabled={isLoading || isSending}
            />
          </div>
          <button
            onClick={isRecording ? stopRecording : startRecording}
            className={cn(
              'btn w-10 h-10 p-0 flex-shrink-0',
              isRecording
                ? 'bg-red-50 text-red-600 border border-red-200 hover:bg-red-100'
                : 'bg-surface-alt text-navy border border-border hover:bg-slate-100',
            )}
            title={isRecording ? 'Stop recording' : 'Voice input'}
          >
            {isRecording ? (
              <MicOff className="w-4 h-4" />
            ) : (
                <Mic className="w-4 h-4" />
              )}
          </button>
          <button
            onClick={handleSend}
            disabled={!input.trim() || isLoading || isSending}
            className="btn-primary w-10 h-10 p-0 flex-shrink-0"
            title="Send message"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
        {isRecording && (
          <div className="mt-2 flex items-center gap-2 text-xs text-red-600">
          <span className="relative flex h-2 w-2 flex-shrink-0">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500"></span>
          </span>
          Recording audio...
        </div>
        )}
      </div>
    </div>
  );
}
