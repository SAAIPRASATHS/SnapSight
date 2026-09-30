import { Eye, FileText, Sparkles, Copy, Check } from 'lucide-react';
import { useState } from 'react';
import type { OCRResult, SceneDescription } from '../types';

interface AIInsightsProps {
  sceneDescription?: SceneDescription | null;
  ocrResults?: OCRResult[];
  ocrFullText?: string;
}

export default function AIInsights({
  sceneDescription,
  ocrResults = [],
  ocrFullText = '',
}: AIInsightsProps) {
  const [copied, setCopied] = useState(false);

  const handleCopyText = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const fullText = ocrFullText || ocrResults.map((r) => r.text).join(' ');

  return (
    <div className="card rounded-lg overflow-hidden flex flex-col h-full">
      <div className="px-4 py-3 border-b border-border flex items-center gap-2">
        <Sparkles className="w-4 h-4 text-snap-blue" />
        <h3 className="font-semibold text-sm text-navy">AI Insights</h3>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        <div>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Eye className="w-3.5 h-3.5 text-snap-blue" />
              <h4 className="text-xs font-semibold text-navy uppercase tracking-wide">Scene Description</h4>
            </div>
            {sceneDescription && (
              <span className="text-xs text-text-secondary font-mono">
                {(sceneDescription.confidence * 100).toFixed(0)}%
              </span>
            )}
          </div>

          {sceneDescription ? (
            <div className="bg-gradient-to-br from-snap-blue/5 to-transparent border border-snap-blue/15 rounded-md p-3">
              <p className="text-sm text-navy leading-relaxed mb-2">
                {sceneDescription.description}
              </p>
              {sceneDescription.objects && sceneDescription.objects.length > 0 && (
                <div className="flex flex-wrap gap-1.5 mt-3">
                  {sceneDescription.objects.map((obj, idx) => (
                    <span
                      key={idx}
                      className="inline-flex items-center px-2 py-0.5 text-xs font-medium bg-white border border-border rounded-full text-navy"
                    >
                      {obj}
                    </span>
                  ))}
                </div>
              )}
              {sceneDescription.scene_type && (
                <div className="mt-2 pt-2 border-t border-snap-blue/10">
                  <span className="text-xs text-text-secondary">Scene type: </span>
                  <span className="text-xs font-medium text-snap-blue capitalize">{sceneDescription.scene_type}</span>
                </div>
              )}
            </div>
          ) : (
            <div className="bg-surface-alt border border-border border-dashed rounded-md p-4 text-center">
              <Eye className="w-6 h-6 text-slate-300 mx-auto mb-2" />
              <p className="text-xs text-text-secondary">No scene description available</p>
            </div>
          )}
        </div>

        <div>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <FileText className="w-3.5 h-3.5 text-snap-blue" />
              <h4 className="text-xs font-semibold text-navy uppercase tracking-wide">OCR Text</h4>
            </div>
            {fullText && (
              <button
                onClick={() => handleCopyText(fullText)}
                className="inline-flex items-center gap-1 text-xs text-text-secondary hover:text-snap-blue transition-colors"
              >
                {copied ? (
                  <>
                    <Check className="w-3 h-3 text-green-600" />
                    Copied
                  </>
                ) : (
                  <>
                    <Copy className="w-3 h-3" />
                    Copy
                  </>
                )}
              </button>
            )}
          </div>

          {fullText ? (
            <div className="bg-surface-alt border border-border rounded-md p-3">
            {ocrResults.length > 0 && (
              <div className="space-y-2 mb-3">
                {ocrResults.slice(0, 5).map((result, idx) => (
                  <div key={idx} className="flex items-start gap-2">
                    <div
                    className="w-1 h-full mt-1 flex-shrink-0 self-stretch bg-snap-blue rounded-full w-1" />
                    <div className="flex-1">
                      <p className="text-sm text-navy">{result.text}</p>
                      <span className="text-xs text-text-secondary font-mono">
                        {(result.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
            {ocrResults.length === 0 && fullText && (
              <p className="text-sm text-navy leading-relaxed">{fullText}</p>
            )}
          </div>
          ) : (
            <div className="bg-surface-alt border border-border border-dashed rounded-md p-4 text-center">
              <FileText className="w-6 h-6 text-slate-300 mx-auto mb-2" />
              <p className="text-xs text-text-secondary">No text detected</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
