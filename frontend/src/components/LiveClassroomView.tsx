import React, { useState, useRef } from 'react';
import { Layers, ShieldCheck, User, Sparkles, SwitchCamera, Video, Database, Upload, RefreshCw } from 'lucide-react';
import { StudentAnalytics } from '../types/analytics';

export type MediaSourceMode = 'camera' | 'dataset' | 'upload';

export interface DatasetSample {
  id: string;
  name: string;
  badge: string;
  url: string;
  description: string;
  badgeColor: string;
}

export const DATASET_SAMPLES: DatasetSample[] = [
  {
    id: 'phone-distraction',
    name: 'Student on Phone',
    badge: '📱 PHONE DISTRACTED',
    url: '/samples/phone_distraction_coco.jpg',
    description: 'COCO val2017 #332570: Active cell phone usage in student hand',
    badgeColor: 'bg-purple-950/90 text-purple-300 border-purple-800',
  },
  {
    id: 'classroom-group',
    name: 'Multi-Student Group',
    badge: '👥 MULTI-STUDENT',
    url: '/samples/classroom_students_coco.jpg',
    description: 'COCO val2017 #459272: Multi-person classroom cohort tracking',
    badgeColor: 'bg-emerald-950/90 text-emerald-300 border-emerald-800',
  },
  {
    id: 'phone-laptop-study',
    name: 'Phone & Laptop Study',
    badge: '💻 MULTI-DEVICE',
    url: '/samples/laptop_phone_study_coco.jpg',
    description: 'COCO val2017 #537991: Laptop distractor screen & secondary phone usage',
    badgeColor: 'bg-amber-950/90 text-amber-300 border-amber-800',
  },
  {
    id: 'attentive-student',
    name: 'Attentive Student',
    badge: '🧑‍🎓 ATTENTIVE',
    url: '/samples/attentive_student_coco.jpg',
    description: 'COCO val2017 #212166: Upright frontal student with board focus',
    badgeColor: 'bg-teal-950/90 text-teal-300 border-teal-800',
  },
];

interface LiveClassroomViewProps {
  frame?: string;
  students: StudentAnalytics[];
  onSelectStudent: (studentId: string) => void;
  videoRef?: React.RefObject<HTMLVideoElement>;
  overlayCanvasRef?: React.RefObject<HTMLCanvasElement>;
  imageCanvasRef?: React.RefObject<HTMLCanvasElement>;
  browserCamActive?: boolean;
  modelStatus?: string;
  showOverlays?: boolean;
  onToggleOverlays?: () => void;
  cameraFacingMode?: 'user' | 'environment';
  onToggleFacingMode?: () => void;
  mediaSourceMode?: MediaSourceMode;
  onSelectMediaSourceMode?: (mode: MediaSourceMode) => void;
  activeDatasetSampleId?: string;
  onSelectDatasetSample?: (sample: DatasetSample) => void;
  onFileUpload?: (file: File) => void;
  uploadedFileName?: string | null;
}

export const LiveClassroomView: React.FC<LiveClassroomViewProps> = ({
  frame,
  students,
  onSelectStudent,
  videoRef,
  overlayCanvasRef,
  imageCanvasRef,
  browserCamActive = false,
  modelStatus = 'idle',
  showOverlays: externalShowOverlays,
  onToggleOverlays,
  cameraFacingMode = 'user',
  onToggleFacingMode,
  mediaSourceMode = 'camera',
  onSelectMediaSourceMode,
  activeDatasetSampleId = 'phone-distraction',
  onSelectDatasetSample,
  onFileUpload,
  uploadedFileName,
}) => {
  const [internalShowOverlays, setInternalShowOverlays] = useState(true);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const showOverlays = externalShowOverlays ?? internalShowOverlays;
  const toggleOverlays = onToggleOverlays ?? (() => setInternalShowOverlays((s) => !s));

  const isImageMode = mediaSourceMode === 'dataset' || (mediaSourceMode === 'upload' && Boolean(uploadedFileName?.match(/\.(jpg|jpeg|png|webp)$/i)));
  const activeSample = DATASET_SAMPLES.find((s) => s.id === activeDatasetSampleId) || DATASET_SAMPLES[0];

  return (
    <div className="panel-card rounded-xl p-4 border border-zinc-800 flex flex-col h-full">
      {/* Video Feed Header Controls & Source Selector */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2.5 mb-3 px-1">
        <div className="flex items-center gap-2">
          <span className={`h-2 w-2 rounded-full ${mediaSourceMode === 'camera' ? 'bg-emerald-400' : 'bg-purple-400'} animate-pulse`}></span>
          <h2 className="text-xs font-bold text-white tracking-wide uppercase">Vision Feed</h2>
          <span className="text-[10px] font-mono text-zinc-400 px-1.5 py-0.5 rounded bg-zinc-900 border border-zinc-800">
            {mediaSourceMode === 'camera' ? 'Live Webcam' : mediaSourceMode === 'dataset' ? 'COCO Open Dataset' : 'Custom Upload'}
          </span>
        </div>

        {/* Input Source Segmented Switch */}
        <div className="flex items-center gap-1 bg-zinc-900/90 p-1 rounded-lg border border-zinc-800">
          <button
            onClick={() => onSelectMediaSourceMode?.('camera')}
            className={`text-[11px] px-2.5 py-1 rounded font-medium flex items-center gap-1 transition-all cursor-pointer ${
              mediaSourceMode === 'camera'
                ? 'bg-zinc-800 text-white shadow-sm border border-zinc-700'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
            title="Use Live Camera Stream"
          >
            <Video className="h-3 w-3 text-emerald-400" />
            <span>Camera</span>
          </button>

          <button
            onClick={() => onSelectMediaSourceMode?.('dataset')}
            className={`text-[11px] px-2.5 py-1 rounded font-medium flex items-center gap-1 transition-all cursor-pointer ${
              mediaSourceMode === 'dataset'
                ? 'bg-zinc-800 text-white shadow-sm border border-zinc-700'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
            title="Test with Free Online Open Dataset Samples"
          >
            <Database className="h-3 w-3 text-purple-400" />
            <span>Dataset (COCO)</span>
          </button>

          <button
            onClick={() => onSelectMediaSourceMode?.('upload')}
            className={`text-[11px] px-2.5 py-1 rounded font-medium flex items-center gap-1 transition-all cursor-pointer ${
              mediaSourceMode === 'upload'
                ? 'bg-zinc-800 text-white shadow-sm border border-zinc-700'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
            title="Upload any image or video file to analyze"
          >
            <Upload className="h-3 w-3 text-blue-400" />
            <span>Upload</span>
          </button>

          {mediaSourceMode === 'camera' && onToggleFacingMode && (
            <button
              onClick={onToggleFacingMode}
              className="text-[11px] px-2 py-1 rounded font-medium flex items-center gap-1 text-zinc-300 hover:text-white transition-all cursor-pointer"
              title="Flip between Front and Rear Camera"
            >
              <SwitchCamera className="h-3 w-3 text-emerald-400" />
            </button>
          )}

          <button
            onClick={toggleOverlays}
            className={`text-[11px] px-2 py-1 rounded font-medium flex items-center gap-1 transition-all cursor-pointer ${
              showOverlays ? 'text-zinc-200' : 'text-zinc-500'
            }`}
            title="Toggle Visual Overlays"
          >
            <Layers className="h-3 w-3" />
          </button>
        </div>
      </div>

      {/* Dataset Sample Selection Carousel (when in dataset mode) */}
      {mediaSourceMode === 'dataset' && (
        <div className="mb-2.5 p-2 rounded-lg bg-zinc-950/80 border border-purple-900/40">
          <div className="flex items-center justify-between text-[11px] text-zinc-400 mb-1.5 px-0.5">
            <span className="font-mono text-purple-300 font-semibold flex items-center gap-1">
              <Database className="h-3 w-3" /> Open Dataset Test Scenarios (1-Click Evaluation):
            </span>
            <span className="text-[10px] text-zinc-500 font-mono">Source: MS-COCO 2017 (CC-BY 4.0)</span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-1.5">
            {DATASET_SAMPLES.map((sample) => {
              const isSelected = activeDatasetSampleId === sample.id;
              return (
                <button
                  key={sample.id}
                  onClick={() => onSelectDatasetSample?.(sample)}
                  className={`text-left p-1.5 rounded border transition-all text-xs cursor-pointer flex flex-col justify-between ${
                    isSelected
                      ? 'border-purple-600 bg-purple-950/50 shadow-sm'
                      : 'border-zinc-800 bg-zinc-900/60 hover:border-zinc-700'
                  }`}
                >
                  <div className="font-semibold text-white truncate text-[11px]">{sample.name}</div>
                  <div className="flex items-center justify-between mt-1">
                    <span className={`text-[9px] font-mono px-1 py-0.5 rounded border ${sample.badgeColor}`}>
                      {sample.badge}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Custom File Upload Box (when in upload mode) */}
      {mediaSourceMode === 'upload' && (
        <div className="mb-2.5 p-2 rounded-lg bg-zinc-950/80 border border-blue-900/40 flex items-center justify-between gap-3 text-xs">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*,video/*"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) onFileUpload?.(file);
            }}
          />
          <div className="truncate">
            <span className="text-zinc-400 font-mono block text-[10px]">Active Upload:</span>
            <span className="text-white font-semibold truncate block">
              {uploadedFileName || 'No file selected — choose any image or video'}
            </span>
          </div>
          <button
            onClick={() => fileInputRef.current?.click()}
            className="shrink-0 px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-medium text-xs flex items-center gap-1.5 cursor-pointer transition-all"
          >
            <Upload className="h-3.5 w-3.5" />
            <span>Choose File</span>
          </button>
        </div>
      )}

      {/* Main Video/Image Frame Canvas (Browser-Side AI Direct Feed) */}
      <div className="relative w-full aspect-video bg-zinc-950 rounded-lg overflow-hidden border border-zinc-800 flex items-center justify-center">
        {/* Floating Source Indicator Banner */}
        <div className="absolute top-2 left-2 z-20 flex flex-wrap items-center gap-1.5 bg-zinc-950/90 backdrop-blur border border-zinc-800 px-2.5 py-1 rounded-lg text-[10px] font-mono pointer-events-none">
          <span className="text-zinc-400">Target:</span>
          {mediaSourceMode === 'dataset' ? (
            <span className="text-purple-300 font-bold bg-purple-950/80 px-1.5 py-0.5 rounded border border-purple-800/80">
              {activeSample.badge}
            </span>
          ) : mediaSourceMode === 'upload' ? (
            <span className="text-blue-300 font-bold bg-blue-950/80 px-1.5 py-0.5 rounded border border-blue-800/80">
              {uploadedFileName || 'CUSTOM MEDIA'}
            </span>
          ) : (
            <span className="text-emerald-300 font-bold bg-emerald-950/80 px-1.5 py-0.5 rounded border border-emerald-800/80">
              LIVE WEBCAM
            </span>
          )}
        </div>

        {/* Video Element for Camera and Uploaded Video */}
        {videoRef && (
          <video
            ref={videoRef}
            playsInline
            muted
            autoPlay
            className={`w-full h-full object-contain ${
              browserCamActive && !isImageMode ? 'block' : 'hidden'
            }`}
          />
        )}

        {/* Canvas Element for Dataset Samples and Uploaded Images */}
        {imageCanvasRef && (
          <canvas
            ref={imageCanvasRef}
            className={`w-full h-full object-contain ${
              browserCamActive && isImageMode ? 'block' : 'hidden'
            }`}
          />
        )}

        {/* Real-time Bounding Box & Behavioral Overlays Canvas */}
        {overlayCanvasRef && (
          <canvas
            ref={overlayCanvasRef}
            className={`absolute inset-0 w-full h-full object-contain pointer-events-none ${
              browserCamActive && showOverlays ? 'block' : 'hidden'
            }`}
          />
        )}

        {/* Fallback Static Frame if Provided */}
        {!videoRef && !imageCanvasRef && frame && (
          <img src={frame} alt="Classroom Feed" className="w-full h-full object-contain" />
        )}

        {/* Loading Spinner */}
        {!browserCamActive && (
          <div className="flex flex-col items-center gap-2 text-zinc-500">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-zinc-400"></div>
            <span className="text-xs font-mono">
              {modelStatus === 'loading'
                ? 'Loading Browser MediaPipe Vision Pipeline...'
                : 'Awaiting Media Stream / Sample Selection...'}
            </span>
          </div>
        )}
      </div>

      {/* Detected Students Selector Cards with AI Suggestions */}
      <div className="mt-3">
        <div className="flex items-center justify-between mb-2 px-1">
          <span className="text-[11px] font-medium text-zinc-400 uppercase tracking-wider">
            Active Detections ({students.length}) — Click to Inspect
          </span>
          {mediaSourceMode === 'dataset' && (
            <span className="text-[10px] text-purple-400 font-mono">
              Evaluating: {activeSample.description}
            </span>
          )}
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
          {students.map((student) => {
            const isDrowsy = student.final_state === 'DROWSY' || student.is_drowsy;
            const isPhone = student.final_state === 'PHONE DISTRACTED' || student.phone_detected;
            const isAttentive = student.final_state === 'ATTENTIVE';
            const isModerate = student.final_state === 'MODERATE';

            const badgeClass = isDrowsy
              ? 'bg-rose-950/80 text-rose-300 border-rose-800'
              : isPhone
              ? 'bg-purple-950/80 text-purple-300 border-purple-800'
              : isAttentive
              ? 'bg-emerald-950/80 text-emerald-300 border-emerald-800'
              : isModerate
              ? 'bg-amber-950/80 text-amber-300 border-amber-800'
              : 'bg-zinc-900 text-zinc-400 border-zinc-800';

            return (
              <button
                key={student.student_id}
                onClick={() => onSelectStudent(student.student_id)}
                className="inner-card p-2.5 rounded-lg border border-zinc-800 hover:border-zinc-600 transition-all text-left group cursor-pointer flex flex-col justify-between h-[5.75rem] min-h-[5.75rem] overflow-hidden"
              >
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-1.5">
                      <User className="h-3.5 w-3.5 text-zinc-400" />
                      <span className="text-xs font-bold text-white group-hover:text-zinc-200">
                        {student.student_id}
                      </span>
                    </div>
                    {student.confidence_pct != null && (
                      <span className="text-[10px] font-mono text-zinc-400 flex items-center gap-0.5">
                        <ShieldCheck className="h-3 w-3 text-zinc-400" />
                        {student.confidence_pct.toFixed(0)}%
                      </span>
                    )}
                  </div>

                  <div className="flex items-center justify-between text-[11px] text-zinc-400 font-mono mb-1.5">
                    <span>
                      Attn: <strong className="text-white">{student.final_state === 'UNABLE TO DETERMINE' ? 'N/A' : `${student.attention_pct.toFixed(0)}%`}</strong>
                    </span>
                    <span className={`text-[10px] px-1.5 py-0.5 rounded border font-semibold font-mono ${badgeClass}`}>
                      {isPhone && !student.final_state?.includes('📱') ? '📱 ' : ''}{student.final_state || student.attention_status || 'ATTENTIVE'}
                    </span>
                  </div>
                </div>

                {student.ai_suggestion && (
                  <div className="mt-1 pt-1.5 border-t border-zinc-800/80 flex items-start gap-1 text-[10px]">
                    <Sparkles className="h-3 w-3 text-emerald-400 shrink-0 mt-0.5" />
                    <span className="text-zinc-300 truncate font-sans">
                      <strong className="text-white">{student.ai_suggestion.action}:</strong>{' '}
                      {student.ai_suggestion.pedagogical_tip}
                    </span>
                  </div>
                )}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};